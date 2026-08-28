"""Read an MHR pose out of one picture and show the result.

Run:    python mhr_pose/predict.py --example 40
Writes: mhr_pose/data/example-<n>.png   the crop, the body, and the two together
        mhr_pose/data/example-<n>.html  the same body in a 3D scene you can spin

The network answers in three pieces -- a heading, 31 joint angles and a position.
This turns those back into an ordinary MHR parameter vector, the same 204 numbers
`analytic_ik.solve` produces, so the result can be handed to anything that already
speaks MHR. The conversion is checked, not assumed: the rebuilt vector has to put
the joints within a hundredth of a millimetre of where the network put them.

There is no person detector in this repository, so a crop has to come from
somewhere. By default this uses an example from the built dataset, which comes
with its box. Pass --image and --box to try a picture of your own.
"""

import argparse
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                      # noqa: E402
import torch                            # noqa: E402
from PIL import Image                   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "mhr_posing"), str(HERE)]

import analytic_ik as ik                # noqa: E402
from make_dataset import CROP_PIXELS, cut_out   # noqa: E402  crop exactly as training did
import mhr_scene as scene               # noqa: E402
import model as net                     # noqa: E402
from mhr_kit.model import faces                 # noqa: E402
from mhr_kit.render import Camera, render       # noqa: E402

DATA = HERE / "data"


def to_mhr_parameters(rig, base_pose, articulation, rotation, position):
    """The network's three answers -> the 204 numbers MHR takes.

    The heading is stored as a rotation matrix and MHR wants three angles, so the
    rig is asked to do that conversion (`set_rotation` knows about the fixed turn
    built into every joint). The position is easier than it looks: the hips sit at
    `root_at_rest` when the three translation parameters are zero, and each of
    them slides the body by 10 cm, whatever the heading. Both facts are measured.
    """
    pose = np.array(base_pose, dtype=float)
    for name, value in zip(net.ARTICULATION, articulation):
        pose[rig.parameter_index[name]] = value

    _, rotations = rig.forward_kinematics(pose)
    rig.set_rotation(pose, "root", ("root_rx", "root_ry", "root_rz"), rotation, rotations)

    at_rest = net.rig_constants(rig)["root_at_rest"].numpy()
    for name, amount in zip(("root_tx", "root_ty", "root_tz"), position - at_rest):
        pose[rig.parameter_index[name]] = amount / 10.0
    return pose


def load_network(rig):
    saved = torch.load(DATA / "network.pt", map_location="cpu", weights_only=False)
    network = net.PoseNet(torch.tensor(saved["mean"], dtype=torch.float32),
                          torch.tensor(saved["std"], dtype=torch.float32))
    network.load_state_dict(saved["network"])
    network.eval()
    print(f"network from epoch {saved['epoch']}")
    return network, torch.tensor(saved["base_pose"], dtype=torch.float32)[None]


def read_example(labels, number):
    """One crop from the built dataset, with the box it was cut from."""
    picture = Image.open(DATA / "crops" / str(labels["name"][number]))
    image = torch.from_numpy(np.array(picture, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return image[None], {key: torch.tensor(labels[key][number], dtype=torch.float32)[None]
                         for key in ("box", "focal", "centre")}


def read_picture(path, box, focal, centre):
    """A crop cut out of any picture, given where the person is in it.

    Cut with the very same function that built the training set, so the network
    sees what it was trained on rather than something a pixel or two different.
    """
    whole = np.array(Image.open(path).convert("RGB"))
    image = torch.from_numpy(cut_out(whole, box, CROP_PIXELS).astype(np.float32) / 255.0)
    return image.permute(2, 0, 1)[None], {
        "box": torch.tensor([box], dtype=torch.float32),
        "focal": torch.tensor([focal], dtype=torch.float32),
        "centre": torch.tensor([centre], dtype=torch.float32)}


def crop_camera(where):
    """The camera that took the crop, as `mhr_kit.render` wants it.

    The crop is a square cut out of the full picture and then resized, and both of
    those change the camera: shrinking a picture shortens its focal length by the
    same factor, and cutting a piece out moves the point the lens looks straight
    through. Getting this right is what makes the body land *on* the person rather
    than merely near them.
    """
    centre_x, centre_y, side = where["box"][0].tolist()
    focal_x, _ = where["focal"][0].tolist()
    lens_x, lens_y = where["centre"][0].tolist()

    scale = CROP_PIXELS / side
    rotation, translation = scene.camera_view()
    return Camera(focal=focal_x * scale,
                  center=((lens_x - (centre_x - side / 2)) * scale,
                          (lens_y - (centre_y - side / 2)) * scale),
                  size=(CROP_PIXELS, CROP_PIXELS),
                  rotation=rotation, translation=translation)


def draw(image, mesh, triangles, where, path):
    """The crop, the body the network read from it, and the two laid over each other."""
    camera = crop_camera(where)
    crop = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    body = render(mesh, triangles, camera, background=(1.0, 1.0, 1.0))   # 0-1, not 0-255
    both = render(mesh, triangles, camera, background=crop, alpha=0.55)

    figure, panels = plt.subplots(1, 3, figsize=(9, 3.3))
    for panel, picture, title in zip(panels, [crop, body, both],
                                     ["the photograph", "what the network read",
                                      "the two together"]):
        panel.imshow(picture)
        panel.set_title(title, fontsize=9)
        panel.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=int, default=0, help="which crop from the dataset")
    parser.add_argument("--image", help="a picture of your own instead")
    parser.add_argument("--box", type=float, nargs=3, metavar=("X", "Y", "SIDE"),
                        help="where the person is in that picture, in pixels")
    parser.add_argument("--focal", type=float, nargs=2, default=(4036.2, 3994.2))
    parser.add_argument("--centre", type=float, nargs=2, default=(1913.4, 1097.9))
    arguments = parser.parse_args()

    # Draw to a file, with no screen to draw on. This is set here rather than when
    # the file is imported, because importing it from a notebook would otherwise
    # switch that notebook's figures off too.
    matplotlib.use("Agg")

    rig = ik.Rig()
    constants = net.rig_constants(rig)
    network, base_pose = load_network(rig)
    labels = dict(np.load(DATA / "labels.npz", allow_pickle=False))

    if arguments.image:
        if arguments.box is None:
            parser.error("--image needs --box, because there is no person detector here")
        image, where = read_picture(arguments.image, arguments.box,
                                    arguments.focal, arguments.centre)
        name = pathlib.Path(arguments.image).stem
    else:
        image, where = read_example(labels, arguments.example)
        name = f"example-{arguments.example}"

    with torch.no_grad():
        rotation, articulation, camera = network(image, where["box"], where["focal"],
                                                 where["centre"])
        position = net.translation_from_camera(camera, where["box"], where["focal"],
                                               where["centre"])
        joints = net.place_body(
            net.body_in_own_frame(rig.model, base_pose, constants["indices"], articulation,
                                  constants["root"], constants["keypoints"]),
            rotation, position)

    pose = to_mhr_parameters(rig, base_pose[0].numpy(), articulation[0].numpy(),
                             rotation[0].numpy(), position[0].numpy())

    # The parameter vector must describe the same body the network just built.
    rebuilt = rig.forward_kinematics(pose)[0][constants["keypoints"].numpy()]
    difference = np.abs(rebuilt - joints[0].numpy()).max()
    assert difference < 1e-2, f"the parameter vector disagrees by {difference:.4f} cm"
    print(f"parameter vector agrees with the network to {difference:.5f} cm")
    print(scene.describe(rig, pose))

    mesh = ik.mesh_from_pose(rig, pose)
    triangles = faces(rig.model)
    draw(image, mesh, triangles, where, DATA / f"{name}.png")

    figure = scene.scene_figure([mesh], triangles, names=["the person"],
                                title=scene.describe(rig, pose), rig=rig, poses=[pose])
    figure.write_html(DATA / f"{name}.html")
    print(f"wrote {DATA / f'{name}.html'}")


if __name__ == "__main__":
    main()
