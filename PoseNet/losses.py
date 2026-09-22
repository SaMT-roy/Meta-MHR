"""
losses.py — score the 49 numbers by posing the real MHR rig, in millimetres.

    python losses.py     # self-check: score the labels against themselves

    prepare.py   wrote data/<clip>/crops/*.jpg + labels.npz
    dataset.py   serves (image, context, target)
    model_*.py   image + context -> 49 numbers
    losses.py    how wrong are those 49                  <- you are here
    train.py     the loop

Inputs
------
    prediction   from model_*.PoseNet
                 vector (B,49) | angles (B,31) | scales (B,9)
                 rot6d (B,6)   | camera (B,3)  | rotation (B,3,3)
    batch        from dataset.PoseData
                 target (B,49) | joints (B,17,3) | box (B,3) | intrinsics (B,4)
    body         a Body — the MHR rig, built once

`batch["joints"]` is where the solver put the 17 keypoints, in the viewer frame
(camera at the origin, cm, +y up). The geometry losses compare against it — the
label's own geometry, so the loss cannot contradict the parameters it is also
supervising.

Why the 49 numbers are not enough
---------------------------------
They do not know what they do. Inject 0.1 rad into one joint and score it:

        joint perturbed    joints3d     angles
        l_upleg_rz (hip)    0.00276    0.00323
        l_uparm_rz (shldr)  0.00155    0.00323
        l_knee_bend         0.00116    0.00323
        l_elbow_bend        0.00075    0.00323
        head_bend           0.00043    0.00323

The angles column is identical everywhere: a loss on the raw numbers cannot
tell a hip mistake from a head mistake. The joints3d column spans 6.4x, ordered
as anatomy says — a hip drags the whole leg, a head drags nothing. Only the
skeleton knows this.

So we run it. MHR is a plain nn.Module, so posing is differentiable: predict
parameters, pose the body, measure where the joints actually ended up. Each
parameter is then weighted by how far it really moves the body. About 6 ms per
batch of 24, against ~100 ms for the backbone.

Six losses, two families
------------------------
    PARAMETER   angles, scales, rotation, camera — predict the numbers the
                solver wrote. Fast and unambiguous, but blind as above.
    GEOMETRY    joints3d, joints2d — put the body where the person is. Weights
                each parameter correctly, but slow to start and ambiguous: a
                bent torso and a tilted pelvis can look the same from one
                camera.

Both are needed. Parameters alone drift; geometry alone is slow and has more
than one answer.

Everything is L1
----------------
The labels contain the occasional bad solve. A squared loss lets a handful of
those dominate the batch gradient; L1 does not.
"""

import numpy as np
import torch

from dataset import PoseData, split_clips
from model_spatial import N_ANGLES, N_SCALES, rotation_from_6d
from prepare import ANGLE_PARAMS, KEYPOINTS, PELVIS, SCALE_PARAMS, ik

# Where each group sits inside the 49-vector — the same layout as dataset.target.
ANGLES = slice(0, N_ANGLES)                                  # [ 0:31]
SCALES = slice(N_ANGLES, N_ANGLES + N_SCALES)                # [31:40]
ROT6D = slice(N_ANGLES + N_SCALES, N_ANGLES + N_SCALES + 6)  # [40:46]
CAMERA = slice(N_ANGLES + N_SCALES + 6, None)                # [46:49]

N_MHR_PARAMETERS = 204      # MHR's full control panel; we write 40 and leave 164 at zero
N_IDENTITY = 45             # identity coefficients; all zero = the average build

WEIGHTS = dict(angles=1.0, scales=0.5, rotation=1.0, camera=1.0,
               joints3d=1.0, joints2d=1.0)


def project(points, intrinsics):
    """
    Viewer-frame points -> photograph pixels. The torch twin of prepare.project.

    Given a 3D point in front of the camera, which pixel does it land on?

        in   points (B, N, 3) cm,  intrinsics (B, 4)
        out  (B, N, 2) pixels

        pixel_x = fx * ( x / -z) + cx
        pixel_y = fy * (-y / -z) + cy

    The division is perspective: things look smaller further away. 

    The same 30 cm of sideways offset, covers a big fraction of the frame when you're 1 m away 
    and almost nothing at 20 m.

    -z because the camera looks along negative z, -y because world-up is image-down.
    """
    fx, fy, cx, cy = intrinsics.unbind(-1)                    # each (B,)
    away = torch.clamp(-points[..., 2], min=10.0)             # (B, N); 10 cm floor
    return torch.stack([fx[:, None] * points[..., 0] / away + cx[:, None],
                        fy[:, None] * -points[..., 1] / away + cy[:, None]], dim=-1)


def to_crop_units(pixels, box):
    """
    Photograph pixels -> crop units: 0 at the crop's centre, +/-1 at its edges.

    Given a pixel in the full 4K photograph, where is that inside the crop?

        in   pixels (B, N, 2),  box (B, 3)
        out  (B, N, 2)

    Subtract the centre. Now the numbers are "how far from the middle of the crop", 
    in pixels, with the middle at 0.

    Divide by half the side. Half, not the full side, because the distance from centre to edge 
    is half the side. That puts the edges at exactly ±1.
    """
    return (pixels - box[:, None, :2]) / (box[:, None, 2:3] / 2)


def translation_from(camera, box, intrinsics):
    """
    It is the inverse of project plus a depth recovery. 
    It turns the network's three camera numbers — (u, v, size) — into a real 3D hip position in cm.

    The problem it solves: a pixel alone doesn't fix a 3D point. 
    Every point along a ray from the camera lands on the same pixel. 
    You need one more piece of information to say how far along that ray, and that's what size carries.

        in   camera (B, 3),  box (B, 3),  intrinsics (B, 4)
        out  (B, 3) viewer frame

    1. away = size * 100 * fx / box[:, 2]

    Size means "the crop is this many metres across in the real world". 
    The crop is box[2] pixels across. 
    Something that's S metres wide and appears B pixels wide must be at distance fx * S / B

    2. pixel_x = box[0] + u * box[2] / 2,   pixel_y = box[1] + v * box[2] / 2

    3. project did fx * x / away + cx; this subtracts cx, divides by fx, multiplies by away.

    """
    fx, fy, cx, cy = intrinsics.unbind(-1)
    u, v, size = camera.unbind(-1)

    away = torch.exp(size) * 100.0 * fx / box[:, 2]      # was: torch.clamp(size, min=0.1) * 100.0 * ...    # cm; clamp stops a blow-up
    pixel_x = box[:, 0] + u * box[:, 2] / 2
    pixel_y = box[:, 1] + v * box[:, 2] / 2

    return torch.stack([(pixel_x - cx) / fx * away,               # sideways
                        -(pixel_y - cy) / fy * away,              # up (minus undoes the flip)
                        -away], dim=1)                            # in front of camera is -z


def place_in_view(local, rotation, translation):
    """
    Takes a pose that's sitting at the origin and puts it where the person actually is.

        in   local (B, 17, 3),  rotation (B, 3, 3),  translation (B, 3)
        out  (B, 17, 3) viewer frame

    Rotate. Which way the person is facing — the root rotation. 
    This spins the whole body about the hips.

    Translate. Where the hips are in front of the camera, from translation_from. 
    This slides the rotated body out to its real position.

    Order matters: rotate first, then shift. Rotation is always about the origin, 
    so if you shifted first, the body would swing around the camera instead of turning in place.
    """
    return local @ rotation.transpose(1, 2) + translation[:, None, :]


# ---------------------------------------------------------------------------
# The rig as a differentiable layer
# ---------------------------------------------------------------------------

class Body:
    """
    The real MHR rig, wrapped so the loss can call it and keep the gradients.
    It's a thin wrapper, not a model. It owns no weights and learns nothing.

    In train.py:

        body = Body(ik.Rig())               # built once, reused every step
        local = body.joints(angles, scales) # (B, 17, 3), hips at the origin

    Our project speaks in 31 angles, 9 scales, 17 keypoints. 
    MHR speaks in a 204-slot parameter vector and 127 joints.

    IT RUNS ON THE CPU. pymomentum has no Apple GPU backend. That is fine — the
    backbone runs on the GPU, the rig on the CPU, and autograd crosses between
    them by itself with no special handling.
    """

    def __init__(self, rig):
        self.rig = rig

        # where each of your 31 angle names sits in MHR's 204
        self.angles_at = torch.tensor([rig.parameter_index[n] for n in ANGLE_PARAMS])

        # where each of your 9 scale names sits
        self.scales_at = torch.tensor([rig.parameter_index[n] for n in SCALE_PARAMS])

        # Which of MHR's 127 joints each of our 17 keypoints is.
        self.joints_at = torch.tensor([rig.joint_index[ik.TARGET_TO_RIG[n]] for n in KEYPOINTS])

    def joints(self, angles, scales):
        """
        Pose the rig and return the 17 keypoints.

            in   angles (B, 31),  scales (B, 9)
            out  (B, 17, 3) in cm, HIPS AT THE ORIGIN, no root rotation

        """
        batch = angles.shape[0]

        # Start from an all-zero 204-vector and write our 40 numbers into it.
        # The other 164 stay at zero — those are parameters the project doesn't predict, left at their neutral value.
        
        # index_copy rather than parameters[:, idx] = angles because the result has to stay differentiable. 
        # In-place assignment into a tensor breaks the autograd graph; index_copy returns a new tensor 
        # with the operation recorded, so gradients flow back from the posed joints into angles and scales.
        parameters = torch.zeros(batch, N_MHR_PARAMETERS)
        parameters = parameters.index_copy(1, self.angles_at, angles)
        parameters = parameters.index_copy(1, self.scales_at, scales)

        # Run forward kinematics. The 45 zeros are identity coefficients — the average human build.
        # person's proportions come from `scales`, and no facial expression.
        _vertices, state = self.rig.model(torch.zeros(batch, N_IDENTITY), parameters, None)

        # Gather 17 out of the 127, keep xyz (drop whatever else state carries), 
        # then subtract the pelvis so the hips land at the origin.
        positions = state[:, self.joints_at, :3]              # (B, 17, 3)
        return positions - positions[:, PELVIS:PELVIS + 1]    # move the hips to the origin


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

def compute_loss(prediction, batch, body, weights=WEIGHTS):
    """
    It builds two versions of the same body — what the labels say and what the network says, 
    then measures the gap six ways.

        out  (total, {angles, scales, rotation, camera, joints3d, joints2d})

    In train.py:

        loss, parts = compute_loss(net(batch["image"], batch["context"]), batch, body)
        loss.backward()
    """
    target, box, intrinsics = batch["target"], batch["box"], batch["intrinsics"]

    # The target stores the root rotation as two columns. 
    # Gram-Schmidt is a no-op here since they're already orthonormal — 
    # it's only being used to supply the missing third column and hand back a full 3×3.
    _six, wanted_rotation = rotation_from_6d(target[:, ROT6D])

    # This is place_in_view run backwards. Subtract the hips to kill the translation, 
    # then multiply by R to kill the rotation — @ R is the inverse of @ Rᵀ because rotations are orthogonal. 
    # What's left is the label's pose with the root stripped, 
    # in exactly the form body.joints returns, so the two are comparable.
    root = batch["joints"][:, PELVIS]                         # (B, 3) the true hips
    wanted_local = (batch["joints"] - root[:, None, :]) @ wanted_rotation

    # Note this projects the unstripped joints. 
    # The 2D term is supposed to include position error, so the root stays on.
    wanted_2d = to_crop_units(project(batch["joints"], intrinsics), box)

    local = body.joints(prediction["angles"], prediction["scales"])        # the pose alone
    translation = translation_from(prediction["camera"], box, intrinsics)  # where it sits
    view = place_in_view(local, prediction["rotation"], translation)       # the two combined
    predicted_2d = to_crop_units(project(view, intrinsics), box)           # and in the picture

    parts = dict(

        angles=(prediction["angles"] - target[:, ANGLES]).abs().mean(),
        scales=(prediction["scales"] - target[:, SCALES]).abs().mean(),

        rotation=(prediction["rot6d"] - target[:, ROT6D]).abs().mean(),
        camera=(prediction["camera"] - target[:, CAMERA]).abs().mean(),

        joints3d=(local - wanted_local).abs().mean() / 100.0,
        joints2d=(predicted_2d - wanted_2d).abs().mean(),
    )

    total = sum(weights[name] * value for name, value in parts.items())
    return total, parts


# ---------------------------------------------------------------------------
# Scores a human can read
# ---------------------------------------------------------------------------

@torch.no_grad()
def metrics(prediction, batch, body):
    """
    It is the read-only twin of compute_loss. 
    Same geometry, but the numbers come out in units you can judge: millimetres, pixels, centimetres, degrees. 
    Nothing here trains — @torch.no_grad() and the .detach() on every prediction make sure of it.

        out  {mpjpe_mm, pixel_error, root_error_cm, rotation_deg}

    In train.py:

        scores = metrics(prediction, batch, body)
        print(f"{scores['mpjpe_mm']:.0f} mm")

    """
    prediction = {name: value.detach() for name, value in prediction.items()}
    box, intrinsics = batch["box"], batch["intrinsics"]

    # Rebuild the predicted body exactly as compute_loss does.
    local = body.joints(prediction["angles"], prediction["scales"])
    translation = translation_from(prediction["camera"], box, intrinsics)
    view = place_in_view(local, prediction["rotation"], translation)

    reference = batch.get("truth", batch["joints"])           # see the docstring
    _six, wanted_rotation = rotation_from_6d(batch["target"][:, ROT6D])

    predicted_hips = view[:, PELVIS:PELVIS + 1]
    reference_hips = reference[:, PELVIS:PELVIS + 1]

    # MPJPE, the standard measure. Slide both bodies onto the same hips first,
    # so this scores the POSE and ignores where the body is.
    mpjpe = (view - predicted_hips - (reference - reference_hips)).norm(dim=-1).mean()

    # Reprojection: the same comparison in pixels of the original 4K photograph,
    # and WITHOUT sliding anything — so this one does include position error.
    pixels = (project(view, intrinsics) - project(reference, intrinsics)).norm(dim=-1).mean()

    return dict(
        mpjpe_mm=float(mpjpe) * 10.0,                         # cm -> mm
        pixel_error=float(pixels),
        root_error_cm=float((view[:, PELVIS] - reference[:, PELVIS]).norm(dim=-1).mean()),
        rotation_deg=np.degrees(float(
            rotation_error(prediction["rotation"], wanted_rotation).mean())),
    )


def rotation_error(predicted, wanted):
    """
    How many radians you would have to turn one rotation to reach the other, 
    i.e answers how far apart are two orientations, as a single angle?

        in   (B, 3, 3) and (B, 3, 3)
        out  (B,) radians

    The network predicts which way the person is facing. The label says which way they're actually facing. 
    Both are 3×3 matrices. We want one number: how wrong is it, in degrees?

    You can't just subtract two matrices and call that an answer — nine numbers, no clear meaning. 
    But there's a fact about rotations that makes this easy.

    Any rotation in 3D, no matter how complicated it looks as a matrix, is just: turn by some angle, 
    around some axis. That's it. Turn 30° around a tilted axis. Turn 90° around vertical. 
    There's always exactly one axis and one angle.

    So "how wrong is the prediction" becomes: what's the angle of the rotation that takes you from 
    the prediction to the truth?
    """
    # Think of it like subtraction. With numbers, "how far from 5 to 8" is 8 − 5 = 3. 
    # With rotations, the equivalent of subtracting is multiplying by the inverse.

    # Result: the rotation you'd still have to perform to fix the mistake. 
    # If the prediction were perfect, this comes out as the identity matrix — the do-nothing rotation.
    leftover = predicted.transpose(1, 2) @ wanted

    # Now we have the leftover rotation as a matrix, and we want its angle.
    # The trace is the sum of the three diagonal entries.
    # It doesn't matter what the axis is. Only the angle affects the trace. 
    # So the diagonal quietly carries the answer.
    trace = leftover.diagonal(dim1=1, dim2=2).sum(dim=1)

    # cos(angle) = (trace − 1) / 2
    # angle      = acos( (trace − 1) / 2 )
    return torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))   

    # No error. Leftover is the identity:

    # [1 0 0]
    # [0 1 0]     trace = 1+1+1 = 3
    # [0 0 1]

    # (3−1)/2 = 1, and acos(1) = 0. Zero error. 

    # Completely backwards — turned 180°:

    # trace = −1
    # (−1−1)/2 = −1, 
    # acos(−1) = π = 180°

    # And 180° is the worst possible, because you can always reach any orientation by turning at 
    # most half a circle.

    # Slightly off: trace comes out 2.99 → (2.99−1)/2 = 0.995 → acos(0.995) ≈ 0.1 rad ≈ 5.7 degrees. 
    # That's the kind of number you actually see in the training log.



def labels_as_prediction(batch):
    """The batch's own labels, shaped like a prediction — for measuring the floor.

        out  a dict with the same keys compute_loss and metrics expect

    In train.py:

        floor = metrics(labels_as_prediction(batch), batch, body)

    Feed this to `metrics` and you get the score of a network that predicted the
    labels perfectly. Nothing trained on these labels can beat it, so it is the
    line to draw on every chart.
    """
    target = batch["target"]
    _six, rotation = rotation_from_6d(target[:, ROT6D])
    return dict(angles=target[:, ANGLES], scales=target[:, SCALES],
                rot6d=target[:, ROT6D], camera=target[:, CAMERA], rotation=rotation)


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    body = Body(ik.Rig())
    _train, val = split_clips()
    batch = next(iter(DataLoader(PoseData(val, augment=False), batch_size=8)))

    total, parts = compute_loss(labels_as_prediction(batch), batch, body)
    print(f"label vs itself — total loss {float(total):.6f}")
    for name, value in parts.items():
        print(f"  {name:9s} {float(value):.6f}")
    print()
    for name, value in metrics(labels_as_prediction(batch), batch, body).items():
        print(f"  {name:14s} {value:8.3f}")
