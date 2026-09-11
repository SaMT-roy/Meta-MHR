"""The network: one person crop in, MHR parameters out.

    crop (3, 256, 256)  ->  backbone  ->  49 numbers
                                          31 joint angles
                                           9 build scales
                                           6 root rotation
                                           3 where the body is

Then the MHR rig itself turns those numbers back into a body, inside the loss.
That is the important part: the losses are not only "did you get the numbers
right" but "did the body end up in the right place", measured by running the
real rig and the real camera.

Why the root rotation is six numbers
------------------------------------
MHR wants three euler angles, but euler angles wrap round at +/-pi: measured on
one clip, `root_rz` jumps by a full 2*pi six times in 120 frames while the body
barely turns. No network can fit a target that teleports. So the network
predicts six numbers that are turned into a rotation matrix by
:func:`rot6d_to_matrix`, which is smooth everywhere. The euler angles are worked
out once, at the very end, when a pose vector has to be handed to MHR.

Why "where the body is" is three numbers and not a distance
-----------------------------------------------------------
A crop cannot tell you how far away someone is: a child up close and an adult
far away fill the same box. What a crop *can* tell you is how big the person is,
and distance follows from that and from how big they look. So the network
predicts

    u, v   where the hips sit inside the crop, in units of half a crop
    size   how many metres across the crop is, out at the person's distance

and :func:`translation_from` turns those, plus the crop's position in the
original photograph, back into a real 3D position. That is what puts the body in
the upper right of the 3D view when the person is in the upper right of the
picture.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.spatial.transform import Rotation

from data import ANGLE_PARAMS, KEYPOINTS, PELVIS, SCALE_PARAMS

import analytic_ik as ik   # noqa: E402  (data.py has already put mhr_posing on the path)

N_ANGLES, N_SCALES = len(ANGLE_PARAMS), len(SCALE_PARAMS)
N_OUTPUTS = N_ANGLES + N_SCALES + 6 + 3


# ---------------------------------------------------------------------------
# Part 1: rotations that do not jump
# ---------------------------------------------------------------------------

def rot6d_to_matrix(six):
    """Six numbers -> a rotation matrix (B, 3, 3).

    Read the six as two rough directions. Keep the first as the new x axis, make
    the second square to it, and take the cross product for the third. Every set
    of six numbers gives a valid rotation, and rotations that are close together
    always come from numbers that are close together -- which is exactly what
    euler angles fail to do.
    """
    x = torch.nn.functional.normalize(six[:, 0:3], dim=1)
    y = six[:, 3:6] - (x * six[:, 3:6]).sum(1, keepdim=True) * x
    y = torch.nn.functional.normalize(y, dim=1)
    return torch.stack([x, y, torch.cross(x, y, dim=1)], dim=2)


def rotation_error(predicted, wanted):
    """The angle in radians you would have to turn one rotation to get the other.

    For reading, not for training: `acos` has an infinite slope at zero error, so
    a loss built on it fights hardest exactly where it has already won. The loss
    compares the nine numbers of the matrices instead.
    """
    trace = (predicted.transpose(1, 2) @ wanted).diagonal(dim1=1, dim2=2).sum(1)
    return torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Part 2: the camera
# ---------------------------------------------------------------------------
#
# The viewer frame: the camera is at the origin, +x is its right, +y is up, and
# it looks along -z. So a body has negative z, and -z is how far away it is.

def project(points, intrinsics):
    """Viewer-frame points (B, N, 3) in cm -> pixels in the original photograph."""
    fx, fy, cx, cy = intrinsics.unbind(-1)
    away = torch.clamp(-points[..., 2], min=10.0)
    return torch.stack([fx[:, None] * points[..., 0] / away + cx[:, None],
                        fy[:, None] * -points[..., 1] / away + cy[:, None]], dim=-1)


def to_crop(pixels, box):
    """Pixels in the photograph -> pixels in the crop, as fractions of half a crop.

    0 is the middle of the crop and +/-1 are its edges, whatever size the crop
    happened to be cut at. Measuring the 2D error in these units means a badly
    placed arm counts the same whether the person was near or far.
    """
    return (pixels - box[:, None, :2]) / (box[:, None, 2:3] / 2)


def camera_target(root, box, intrinsics):
    """The (u, v, size) the network should predict, worked out from the truth."""
    fx = intrinsics[:, 0]
    pixels = project(root[:, None, :], intrinsics)[:, 0]
    away = torch.clamp(-root[:, 2], min=10.0)
    return torch.stack([(pixels[:, 0] - box[:, 0]) / (box[:, 2] / 2),
                        (pixels[:, 1] - box[:, 1]) / (box[:, 2] / 2),
                        away * box[:, 2] / fx / 100.0], dim=1)


def translation_from(camera, box, intrinsics):
    """(u, v, size) plus the crop's place in the photograph -> a 3D position in cm.

    The exact inverse of :func:`camera_target`.
    """
    fx, fy, cx, cy = intrinsics.unbind(-1)
    u, v, size = camera.unbind(-1)
    away = torch.clamp(size, min=0.1) * 100.0 * fx / box[:, 2]
    pixel_x = box[:, 0] + u * box[:, 2] / 2
    pixel_y = box[:, 1] + v * box[:, 2] / 2
    return torch.stack([(pixel_x - cx) / fx * away,
                        -(pixel_y - cy) / fy * away,
                        -away], dim=1)


# ---------------------------------------------------------------------------
# Part 3: the rig, as a differentiable layer
# ---------------------------------------------------------------------------

class Body:
    """Runs the real MHR rig on predicted parameters, and keeps the gradients.

    MHR is a plain torch module, so posing it is differentiable and the loss can
    ask "where did the knee actually end up" instead of only "was the knee angle
    close". It runs on the CPU (pymomentum has no Apple GPU backend), which costs
    about 6 ms for a batch of 24 -- next to nothing beside the backbone.
    """

    def __init__(self, rig):
        self.rig = rig
        self.angles_at = torch.tensor([rig.parameter_index[n] for n in ANGLE_PARAMS])
        self.scales_at = torch.tensor([rig.parameter_index[n] for n in SCALE_PARAMS])
        self.joints_at = torch.tensor([rig.joint_index[ik.TARGET_TO_RIG[n]] for n in KEYPOINTS])

    def joints(self, angles, scales):
        """The 17 keypoints (B, 17, 3) in cm, relative to the hips and unrotated.

        The root's own rotation and position are left at zero here, so this is
        the shape of the pose on its own. The predicted root rotation and
        position are applied afterwards, in :func:`posed_joints`.
        """
        batch = angles.shape[0]
        parameters = torch.zeros(batch, 204)
        parameters = parameters.index_copy(1, self.angles_at, angles)
        parameters = parameters.index_copy(1, self.scales_at, scales)

        _, state = self.rig.model(torch.zeros(batch, 45), parameters, None)
        positions = state[:, self.joints_at, :3]
        return positions - positions[:, PELVIS:PELVIS + 1]


def posed_joints(local, rotation, translation):
    """Turn a body-shaped pose into where it really is: (B, 17, 3) in the viewer frame."""
    return local @ rotation.transpose(1, 2) + translation[:, None, :]


# ---------------------------------------------------------------------------
# Part 4: the network
# ---------------------------------------------------------------------------

BACKBONES = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

def average_outputs(dataset):
    """The average of the 49 numbers over a dataset, used to start the head off.

    Reads the labels straight out of the cache, so it costs no picture decoding.
    Both the plain and the mirrored labels are counted, which is what makes the
    average left-right symmetric.
    """
    angles, scales, cameras = [], [], []
    for label in dataset.labels.values():
        intrinsics = torch.from_numpy(label["intrinsics"])[None].expand(len(label["box"]), 4)
        for side in ("", "_flip"):
            box = torch.from_numpy(label["box"]).clone()
            if side:                                   # the crop is mirrored too
                box[:, 0] = 2 * label["intrinsics"][2] - box[:, 0]
            angles.append(label["angles" + side])
            scales.append(label["scales" + side])
            root = torch.from_numpy(label["joints" + side][:, PELVIS])
            cameras.append(camera_target(root, box, intrinsics).numpy())

    average = torch.zeros(N_OUTPUTS)
    average[:N_ANGLES] = torch.tensor(np.concatenate(angles).mean(0))
    average[N_ANGLES:N_ANGLES + N_SCALES] = torch.tensor(np.concatenate(scales).mean(0))
    average[-3:] = torch.tensor(np.concatenate(cameras).mean(0))
    return average


class PoseNet(nn.Module):
    """A picture backbone, plus a small head that reads out MHR parameters.

    The head is also told where in the photograph the crop came from and how big
    it was, as three numbers. A crop on its own cannot know that: someone at the
    edge of a wide picture is seen from an angle, and looks leant over even when
    they are standing straight. Three numbers fix it, and they are free.
    """

    def __init__(self, backbone="resnet18", pretrained=True, average=None):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {list(BACKBONES)}")
        self.backbone = getattr(torchvision.models, backbone)(
            weights="DEFAULT" if pretrained else None)
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(BACKBONES[backbone] + 3, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, N_OUTPUTS))

        # Start the network off predicting the average pose in the training set,
        # rather than a random one. `average` is the mean of the 49 outputs; the
        # rotation part is left as "no rotation at all".
        nn.init.zeros_(self.head[-1].weight)
        start = torch.zeros(N_OUTPUTS)
        start[N_ANGLES + N_SCALES:N_ANGLES + N_SCALES + 6] = torch.tensor([1., 0., 0., 0., 1., 0.])
        if average is not None:
            start[:N_ANGLES + N_SCALES] = torch.as_tensor(average[:N_ANGLES + N_SCALES])
            start[-3:] = torch.as_tensor(average[-3:])
        self.head[-1].bias.data.copy_(start)

    def forward(self, image, box, intrinsics):
        fx, fy, cx, cy = intrinsics.unbind(-1)
        where = torch.stack([(box[:, 0] - cx) / fx, (box[:, 1] - cy) / fy, box[:, 2] / fx], dim=1)
        out = self.head(torch.cat([self.backbone(image), where], dim=1))
        return dict(angles=out[:, :N_ANGLES],
                    scales=out[:, N_ANGLES:N_ANGLES + N_SCALES],
                    rotation=rot6d_to_matrix(out[:, N_ANGLES + N_SCALES:N_ANGLES + N_SCALES + 6]),
                    camera=out[:, -3:])


# ---------------------------------------------------------------------------
# Part 5: the losses
# ---------------------------------------------------------------------------
#
# Six of them, in two families.
#
#   parameters   angles, scales, rotation, camera -- "predict the numbers the
#                solver wrote". Direct and easy to optimise, but a small error
#                in a shoulder angle and a small error in a wrist angle count
#                the same, even though one moves the hand ten times further.
#
#   geometry     3d and 2d -- "put the body where the person is". These run the
#                real rig and the real camera, so they weigh every parameter by
#                how much it actually moves the body. They are what stops the
#                network settling for a pose that is numerically close and
#                visibly wrong.
#
# Both are needed. Parameters alone give a body that drifts; geometry alone is
# slow to get going and has more than one answer (a bent-forward torso and a
# tilted pelvis can look identical from one camera).

WEIGHTS = dict(angles=1.0, scales=0.5, rotation=1.0, camera=1.0, joints3d=1.0, joints2d=1.0)


def compute_loss(prediction, batch, body, weights=WEIGHTS):
    """Total loss, plus each part on its own so training can be watched."""
    box, intrinsics = batch["box"], batch["intrinsics"]
    root = batch["joints"][:, PELVIS]

    # what the solver wrote, rearranged into the same shapes the network predicts
    wanted_local = (batch["joints"] - root[:, None, :]) @ batch["root_rot"]
    wanted_camera = camera_target(root, box, intrinsics)
    wanted_2d = to_crop(project(batch["joints"], intrinsics), box)

    # what the network says, run through the rig and the camera
    local = body.joints(prediction["angles"], prediction["scales"])
    translation = translation_from(prediction["camera"], box, intrinsics)
    view = posed_joints(local, prediction["rotation"], translation)

    parts = dict(
        angles=(prediction["angles"] - batch["angles"]).abs().mean(),
        scales=(prediction["scales"] - batch["scales"]).abs().mean(),
        rotation=(prediction["rotation"] - batch["root_rot"]).abs().mean(),
        camera=(prediction["camera"] - wanted_camera).abs().mean(),
        joints3d=(local - wanted_local).abs().mean() / 100.0,          # cm -> m
        joints2d=(to_crop(project(view, intrinsics), box) - wanted_2d).abs().mean(),
    )
    total = sum(weights[name] * value for name, value in parts.items())
    return total, parts


@torch.no_grad()
def metrics(prediction, batch, body):
    """Numbers you can read: millimetres, pixels, degrees."""
    prediction = {name: value.detach() for name, value in prediction.items()}
    box, intrinsics = batch["box"], batch["intrinsics"]
    local = body.joints(prediction["angles"], prediction["scales"])
    translation = translation_from(prediction["camera"], box, intrinsics)
    view = posed_joints(local, prediction["rotation"], translation)

    truth = batch["truth"]                                  # the real ASPset keypoints
    hips = truth[:, PELVIS:PELVIS + 1]
    mpjpe = (view - view[:, PELVIS:PELVIS + 1] - (truth - hips)).norm(dim=-1).mean()
    pixels = (project(view, intrinsics) - project(truth, intrinsics)).norm(dim=-1).mean()
    return dict(
        mpjpe_mm=float(mpjpe) * 10.0,                       # cm -> mm
        pixel_error=float(pixels),
        root_error_cm=float((view[:, PELVIS] - truth[:, PELVIS]).norm(dim=-1).mean()),
        rotation_deg=np.degrees(float(rotation_error(prediction["rotation"],
                                                     batch["root_rot"]).mean())),
    )


# ---------------------------------------------------------------------------
# Part 6: back to MHR
# ---------------------------------------------------------------------------

def pose_vector(rig, angles, scales, rotation, translation):
    """One prediction -> the (204,) parameter vector MHR takes, ready to be posed.

    This is the only place euler angles appear. `root_rx/ry/rz` are exactly the
    xyz euler angles of the root's world rotation (measured, to 5 decimal
    places), and `root_tx/ty/tz` slide the body along the world axes at 10 cm per
    unit however it is turned -- so the position is written after the rotation,
    by asking the rig where the root ended up and moving it the rest of the way.
    """
    pose = np.zeros(204)
    for name, value in zip(ANGLE_PARAMS, np.asarray(angles)):
        pose[rig.parameter_index[name]] = value
    for name, value in zip(SCALE_PARAMS, np.asarray(scales)):
        pose[rig.parameter_index[name]] = value
    for name, value in zip(("root_rx", "root_ry", "root_rz"),
                           Rotation.from_matrix(np.asarray(rotation)).as_euler("xyz")):
        pose[rig.parameter_index[name]] = value

    positions, _ = rig.forward_kinematics(pose)
    shift = np.asarray(translation) - positions[rig.joint_index["root"]]
    for name, value in zip(("root_tx", "root_ty", "root_tz"), shift):
        pose[rig.parameter_index[name]] = value / 10.0
    return pose


def predict(net, body, batch, device="cpu"):
    """Run the network on a batch and return everything in the viewer frame."""
    net.eval()
    with torch.no_grad():
        out = net(batch["image"].to(device), batch["box"].to(device),
                  batch["intrinsics"].to(device))
    out = {k: v.cpu() for k, v in out.items()}
    local = body.joints(out["angles"], out["scales"])
    out["translation"] = translation_from(out["camera"], batch["box"], batch["intrinsics"])
    out["joints"] = posed_joints(local, out["rotation"], out["translation"])
    return out
