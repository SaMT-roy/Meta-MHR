"""Turn ASPset clips into training pairs: a person crop, and the MHR pose for it.

For every sampled video frame this does four things:

  1. solve MHR pose parameters from the frame's 3D keypoints  (mhr_posing/analytic_ik.py)
  2. move that pose into the frame of the camera that filmed it (mhr_posing/mhr_scene.py)
  3. cut the person out of the 4K frame
  4. write down the answer the network will be asked to predict

Step 2 is the one that is easy to get wrong, and getting it wrong is silent.
ASPset's world origin is the LEFT camera. A clip filmed by the mid or the right
camera is still described from the left camera's point of view, so its root
position and root rotation say where the person was relative to a camera that did
not take the picture. Two thirds of the clips here are mid or right, so skipping
this would poison two thirds of the labels.

Run:    python mhr_pose/make_dataset.py
Writes: mhr_pose/data/crops/*.jpg  and  mhr_pose/data/labels.npz
"""

import json
import pathlib
import sys
import warnings

import c3d
import cv2
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "mhr_posing"), str(HERE)]

import analytic_ik as ik           # noqa: E402  the solver, already written
import mhr_scene as scene          # noqa: E402  the camera placement, already written
from model import ARTICULATION, camera_from_translation      # noqa: E402

# ASPset arrived as four separately extracted archives, which macOS renamed. They
# are not copies: each holds one kind of file.
SPLIT = "train"                    # the only split present on disk
ASPSET = ROOT / "ASPset"
VIDEOS = ASPSET / "ASPset-510" / SPLIT / "videos"
CAMERAS = ASPSET / "ASPset-510 3" / SPLIT / "cameras"
JOINTS = ASPSET / "ASPset-510 4" / SPLIT / "joints_3d"

OUT = HERE / "data"
CROPS = OUT / "crops"

FRAME_STRIDE = 5        # 50 fps, and neighbouring frames are 1.3 cm apart: keep every 5th
CROP_PIXELS = 224       # 224 = 16 x 14, and DINOv2 reads the image in 14-pixel patches
BOX_MARGIN = 1.3        # how much room to leave around the person
BOX_JITTER = 0.08       # the box comes from the ground truth, so wobble it (see below)


# ---------------------------------------------------------------------------
# Reading ASPset
# ---------------------------------------------------------------------------

def read_camera(subject, view):
    """The camera's intrinsics K (3x3) and extrinsics E (4x4, world -> camera)."""
    with open(CAMERAS / subject / f"{subject}-{view}.json") as f:
        camera = json.load(f)
    intrinsic = np.array(camera["intrinsic_matrix"]).reshape(3, 4)[:3, :3]
    extrinsic = np.array(camera["extrinsic_matrix"]).reshape(4, 4)
    return intrinsic, extrinsic


def read_joints(subject, clip):
    """Every frame's 17 joints, in millimetres, in ASPset's world."""
    with open(JOINTS / subject / f"{subject}-{clip}.c3d", "rb") as f:
        frames = [frame[1][:, :3] for frame in c3d.Reader(f).read_frames()]
    return np.array(frames)


def to_pixels(points_in_camera, intrinsic):
    """Points measured from the camera -> pixels. Any length unit works, since
    projecting divides one length by another and the units cancel."""
    pixels = points_in_camera @ intrinsic.T
    return pixels[:, :2] / pixels[:, 2:3]


def project_measured(joints_mm, intrinsic, extrinsic):
    """ASPset's own 3D keypoints -> pixels. This is the ground truth in 2D."""
    return to_pixels(joints_mm @ extrinsic[:3, :3].T + extrinsic[:3, 3], intrinsic)


def project_solved(points_cm, intrinsic):
    """The solved body's joints (viewer frame: cm, y up, camera at the origin) -> pixels.

    The camera counts y downwards and z into the scene, so both flip.
    """
    return to_pixels(points_cm * np.array([1.0, -1.0, -1.0]), intrinsic)


# ---------------------------------------------------------------------------
# Cutting the person out
# ---------------------------------------------------------------------------

def square_box(points_2d, jitter, generator):
    """A square box around the person: its centre and its side, in pixels.

    The box is deliberately wobbled. Its exact position comes from the ground
    truth joints, and at inference time it will come from a person detector
    instead. Training on perfect boxes and testing on detector boxes teaches the
    network to rely on something it will not get, so the wobble is a small,
    honest down payment on that difference.
    """
    low, high = points_2d.min(axis=0), points_2d.max(axis=0)
    centre = 0.5 * (low + high)
    side = float(np.max(high - low)) * BOX_MARGIN

    centre = centre + generator.uniform(-jitter, jitter, size=2) * side
    side = side * float(np.exp(generator.uniform(-jitter, jitter)))
    return float(centre[0]), float(centre[1]), side


def cut_out(frame, box, size):
    """The square `box` of `frame`, resized to `size` x `size`.

    warpAffine does the move and the resize in one step, and fills in black if the
    box hangs over the edge of the picture, which happens when a limb leaves frame.
    """
    centre_x, centre_y, side = box
    scale = size / side
    transform = np.array([[scale, 0.0, -scale * (centre_x - side / 2)],
                          [0.0, scale, -scale * (centre_y - side / 2)]])
    return cv2.warpAffine(frame, transform, (size, size))


# ---------------------------------------------------------------------------
# Building the set
# ---------------------------------------------------------------------------

def solve_frame(rig, joints_mm, extrinsic):
    """One frame of 3D keypoints -> an MHR pose seen from the filming camera."""
    targets = ik.targets_from_aspset(joints_mm)
    pose = ik.solve(rig, targets)
    rotation, translation = scene.camera_from_aspset(extrinsic)
    return scene.place(rig, pose, rotation, translation)


def main():
    CROPS.mkdir(parents=True, exist_ok=True)
    rig = ik.Rig()
    root = rig.joint_index["root"]
    keypoint_joints = [rig.joint_index[joint] for joint in ik.TARGET_TO_RIG.values()]
    aspset_order = [ik.ASPSET_ORDER.index(name) for name in ik.TARGET_TO_RIG]
    generator = np.random.default_rng(0)

    records = {key: [] for key in ("name", "subject", "view", "clip", "frame", "pose",
                                   "rotation", "position", "box", "focal", "centre",
                                   "joints_3d", "joints_2d")}

    videos = sorted(VIDEOS.rglob("*.mkv"))
    for video_path in tqdm(videos, desc="clips"):
        subject, clip, view = video_path.stem.split("-")
        intrinsic, extrinsic = read_camera(subject, view)
        joints = read_joints(subject, clip)

        capture = cv2.VideoCapture(str(video_path))
        for frame_number in range(len(joints)):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % FRAME_STRIDE:
                continue

            pose = solve_frame(rig, joints[frame_number], extrinsic)
            positions, rotations = rig.forward_kinematics(pose)
            joints_3d = positions[keypoint_joints]                  # the solved body, cm
            joints_2d = project_measured(joints[frame_number][aspset_order],
                                         intrinsic, extrinsic)      # what was measured, px

            box = square_box(joints_2d, BOX_JITTER, generator)
            name = f"{video_path.stem}-{frame_number:04d}.jpg"
            cv2.imwrite(str(CROPS / name), cut_out(frame, box, CROP_PIXELS),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            records["name"].append(name)
            records["subject"].append(subject)
            records["view"].append(view)
            records["clip"].append(f"{subject}-{clip}-{view}")
            records["frame"].append(frame_number)
            records["pose"].append(pose)
            records["rotation"].append(rotations[root])             # the body's heading
            records["position"].append(positions[root])             # where it stands, cm
            records["box"].append(box)
            records["focal"].append([intrinsic[0, 0], intrinsic[1, 1]])
            records["centre"].append([intrinsic[0, 2], intrinsic[1, 2]])
            records["joints_3d"].append(joints_3d)
            records["joints_2d"].append(joints_2d)
        capture.release()

    data = {key: np.array(value) for key, value in records.items()}
    summarise(rig, data)
    np.savez_compressed(OUT / "labels.npz", **data)
    print(f"\n{len(data['name'])} examples -> {OUT/'labels.npz'}")


def summarise(rig, data):
    """Add the few numbers training needs, and report how good the labels are.

    * `base_pose` is the average body build. Limb lengths barely change between
      the two people in this dataset, and one photograph is a poor way to measure
      them, so the network is not asked to guess them: it gets this fixed body.
    * `mean`/`std` put every predicted number on the same footing, so no single
      parameter dominates the loss just by having larger numbers.
    * `low`/`high` are the range each parameter was actually seen in, used to
      discourage poses that never occur.
    """
    poses = data["pose"]
    indices = [rig.parameter_index[name] for name in ARTICULATION]

    base = np.median(poses, axis=0)                 # a typical build
    base[indices] = 0.0                             # ...held in a rest pose
    for name in ("root_rx", "root_ry", "root_rz", "root_tx", "root_ty", "root_tz"):
        base[rig.parameter_index[name]] = 0.0       # ...standing at the origin

    articulation = poses[:, indices]
    camera = camera_from_translation(*[torch.tensor(data[key], dtype=torch.float64)
                                       for key in ("position", "box", "focal", "centre")]).numpy()

    data["base_pose"] = base
    data["articulation"] = articulation
    data["camera"] = camera
    data["mean"] = np.concatenate([articulation.mean(0), camera.mean(0)])
    data["std"] = np.concatenate([articulation.std(0), camera.std(0)]) + 1e-3
    data["low"] = articulation.min(0)
    data["high"] = articulation.max(0)

    # How good are the labels? Put each solved body back into the picture it came
    # from and see how far its joints land from the ones that were measured. This
    # is the noise floor: no network trained on these can do better.
    camera_frame = data["joints_3d"] * np.array([1.0, -1.0, -1.0])
    pixels = camera_frame[..., :2] / camera_frame[..., 2:3]
    pixels = pixels * data["focal"][:, None, :] + data["centre"][:, None, :]
    error = np.linalg.norm(pixels - data["joints_2d"], axis=2)
    print(f"\nlabel check: the solved bodies land {error.mean():.1f} px on average "
          f"({np.percentile(error, 99):.0f} px at worst) from the measured joints, "
          f"in a 3840 x 2160 frame")


if __name__ == "__main__":
    main()
