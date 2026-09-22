"""
model_spatial.py — one crop -> 49 numbers.

    prepare.py   wrote the crops and labels
    dataset.py   serves (image, context, target)
    model_*.py   image + context -> 49 numbers      <- you are here
    losses.py    compares those against target
    train.py     the loop

Two kinds of output, so two heads:

    (u, v)            where the hips sit in the crop      -> spatial
    the other 47      how the body is posed and built     -> holistic

    image (B,3,256,256)
        |
        v  ResNet-18, no pooling, no classifier
    features (B,C,8,8)
        |
        v  1x1 conv, ReLU, flatten, concat context (B,3), one Linear
    (B,49)  angles 31 | scales 9 | rot6d 6 | u, v, size 3
        |
        v  the six are read in the RAY frame and composed with
           ray_rotation(context) back into the viewer frame
    vector (B,49), in dataset.target's order and frame

WHERE CONTEXT ACTUALLY DOES ITS WORK. It is three numbers beside 2048 features,
so as a plain input it would barely register — but its first two are the tangents
of the crop's off-axis angles, and that is not something to be learnt. The
viewer-frame root rotation of an upright person depends on where in the
photograph they stand; the crop cannot see that. So context is used
geometrically, not as a feature: it builds the rotation that carries the optical
axis onto the crop's ray, and the network is left predicting only what the pixels
can support. See ray_rotation.

Layout of `vector`, which follows dataset.target:

    [ 0:46]   angles, scales, rot6d      holistic
    [46:48]   u, v                       spatial, from the position head
    [48]      size                       holistic

Note `size` is holistic ("how big is this person"), so `vector[:47]` is not the
global head's output — the two branches interleave at the end.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataset import PELVIS, hips_in_crop
from prepare import ANGLE_PARAMS, SCALE_PARAMS

N_ANGLES = len(ANGLE_PARAMS)                 # 31
N_SCALES = len(SCALE_PARAMS)                 # 9
N_ROT6D = 6
N_SIZE = 1
N_GLOBAL = N_ANGLES + N_SCALES + N_ROT6D + N_SIZE     # 47, everything but (u, v)
N_OUTPUTS = N_GLOBAL + 2                              # 49

BACKBONE_CHANNELS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}


def rotation_from_6d(six):
    """
    Six raw numbers -> a tidied six and a proper rotation matrix.

        in   (B, 6)
        out  (B, 6), (B, 3, 3)

    Gram-Schmidt on two rough directions: normalise the first, square the
    second against it, cross for the third. Any six numbers give a valid
    rotation, and nearby rotations come from nearby sixes — which is what euler
    angles fail at, since they wrap at +/-pi.

    The tidied six is the matrix's first two columns, matching
    dataset.target[40:46], so the loss can compare them directly.
    """
    x = F.normalize(six[:, 0:3], dim=1)
    y = F.normalize(six[:, 3:6] - (x * six[:, 3:6]).sum(dim=1, keepdim=True) * x, dim=1)
    z = torch.cross(x, y, dim=1)
    return torch.cat([x, y], dim=1), torch.stack([x, y, z], dim=2)   # dim=2 -> columns


def ray_direction(context):
    """
    Which way the crop's centre looks, in viewer axes.

        in   context (B, 3)   from dataset.crop_context
        out  (B, 3) unit vector

    context[:, 0] and context[:, 1] are (box_x - cx)/fx and (box_y - cy)/fy,
    which are not arbitrary features — they are the TANGENTS of the angles the
    crop centre sits off the optical axis. Undo prepare.project on the crop's
    own centre and the ray falls out:

        pixel_x = fx * ( x / away) + cx   =>   x =  context[0] * away
        pixel_y = fy * (-y / away) + cy   =>   y = -context[1] * away
        away    = -z                      =>   z = -away

    so the direction is proportional to (context[0], -context[1], -1), and a
    centred crop gives (0, 0, -1): straight down the optical axis, as it should.
    """
    tx, ty = context[:, 0], context[:, 1]
    return F.normalize(torch.stack([tx, -ty, -torch.ones_like(tx)], dim=1), dim=1)


def ray_rotation(context):
    """
    The rotation carrying the optical axis onto the crop's own ray.

        in   context (B, 3)
        out  (B, 3, 3)

    WHY THIS EXISTS. `dataset.target[40:46]` stores the root rotation in the
    VIEWER frame — relative to the camera's axes. A crop cannot see the camera's
    axes. It sees the person along the ray it was cut from. Two frames of
    someone standing identically upright, one centred and one near the edge of a
    wide photograph, have near-identical pixels and genuinely different
    viewer-frame rotations, differing by exactly this matrix. Ask the network
    for the viewer-frame rotation from pixels alone and part of the target is
    unpredictable in principle; it can only ever learn the conditional mean and
    eat the rest as error.

    So the network predicts the rotation in the RAY frame — "which way is this
    person facing relative to how I am looking at them", which is what the
    pixels actually show — and this matrix composes it back:

        R_viewer = ray_rotation(context) @ R_ray

    The off-axis angle reaches tens of degrees at the edge of a 3840-wide
    frame, which is an order of magnitude above the rotation error worth
    chasing, so this is not a refinement.

    THE MINIMAL ROTATION, via Rodrigues. Of the many rotations taking the
    optical axis a = (0, 0, -1) to the ray d, take the one about a x d, turning
    by the angle between them, and nothing else:

        R = I + [v]x + [v]x^2 / (1 + c),    v = a x d,   c = a . d

    Two reasons over a Gram-Schmidt against world-up: it is exactly the identity
    for a centred crop, so the composition is a no-op where it should be; and it
    adds no roll, so the ray frame does not spin as the crop moves sideways.
    The 1/(1 + c) form has no division by |v|, so a centred crop is not a
    special case — and c < 0 would need a 90-degree-off-axis crop, which no
    real camera produces.
    """
    d = ray_direction(context)

    axis = torch.zeros_like(d)
    axis[:, 2] = -1.0                                   # the optical axis

    v = torch.cross(axis, d, dim=1)                     # (B, 3)
    c = (axis * d).sum(dim=1)                           # (B,) cosine between them

    zero = torch.zeros_like(c)
    skew = torch.stack([                                # [v]x, the cross-product matrix
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=1),
    ], dim=1)                                           # (B, 3, 3)

    identity = torch.eye(3, dtype=d.dtype, device=d.device).expand(len(d), 3, 3)
    return identity + skew + skew @ skew / (1.0 + c)[:, None, None]


def starting_guess(average=None):
    """
    The 49 numbers to output before learning anything.

        average  (49,) mean target over the training set, or None for zeros
        out      (49,)

    Identity rotation always — averaging rotations entry-by-entry does not give
    a rotation — and the dataset mean for everything else.

    Since the head now predicts in the ray frame, identity here means "facing
    straight back down the ray I was cut from" rather than "facing down the
    optical axis". That is the better prior, and it is why the rotation slots of
    `average` are still ignored: their mean is a viewer-frame quantity and no
    longer even the right frame to start from.
    """
    start = torch.zeros(N_OUTPUTS)
    start[N_ANGLES + N_SCALES:N_ANGLES + N_SCALES + N_ROT6D] = torch.tensor(
        [1., 0., 0., 0., 1., 0.])
    if average is not None:
        average = torch.as_tensor(average, dtype=torch.float32)
        start[:N_ANGLES + N_SCALES] = average[:N_ANGLES + N_SCALES]   # angles, scales
        start[-3:] = average[-3:]                                     # u, v, size
    return start


class FeatureExtractor(nn.Module):
    """
    A torchvision ResNet with its pooling and classifier removed.

        in   (B, 3, 256, 256)
        out  (B, 512, 8, 8)      resnet18/34;  (B, 2048, 8, 8) for resnet50

    Stride 32 is where the 8x8 comes from: 256 / 32 = 8.

    Keep it unfrozen. ImageNet trains a network to be invariant to pose and
    viewpoint — precisely what we are trying to measure — so the later layers
    have to be retrained. If you do freeze it for a warm-up, call .eval() too,
    or BatchNorm keeps updating its running statistics.
    """

    def __init__(self, name="resnet18", pretrained=True):
        super().__init__()
        net = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = BACKBONE_CHANNELS[name]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SpatialHead(nn.Module):
    def __init__(self, in_channels, filters=32, grid=8, average=None):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, filters, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(filters * grid * grid + 3, N_OUTPUTS)

        nn.init.zeros_(self.fc.weight)                     
        self.fc.bias.data.copy_(starting_guess(average))

    def forward(self, features, context):
        x = self.relu(self.reduce(features))
        x = self.dropout(x.flatten(1))
        return self.fc(torch.cat([x, context], dim=1))

class PoseNet(nn.Module):
    """
    One crop -> 49 numbers.

        backbone    "resnet18" (default), "resnet34" or "resnet50"
        pretrained  start from ImageNet weights, unless loading a checkpoint
        average     (49,) mean target for the starting bias — see average_target

    In train.py:

        net = PoseNet(average=average_target(train_data))
        out = net(batch["image"], batch["context"])
        loss = (out["vector"] - batch["target"]).abs().mean()      # roughly

    Returns a dict, because the loss wants the pieces and the geometry wants a
    real matrix:

        vector    (B,49)    everything, in dataset.target order
        angles    (B,31)   -+
        scales    (B,9)     | the same numbers, split out
        rot6d     (B,6)     | (orthonormalised, and in the VIEWER frame — the
                            |  ray composition has already been applied, so the
                            |  loss can still subtract target[40:46] directly)
        camera    (B,3)    -+  (u, v, size)
        rotation  (B,3,3)   the same rotation as a matrix, for Part 2

    Nothing downstream changes for the ray frame: losses.py, dataset.py and
    train.py all see the same viewer-frame quantities they saw before.
    """

    def __init__(self, backbone="resnet18", pretrained=True, average=None):
        super().__init__()
        self.backbone = FeatureExtractor(backbone, pretrained)
        self.head = SpatialHead(self.backbone.out_channels, average=average)

    def forward(self, image, context):
        features = self.backbone(image)                       # (B, C, 8, 8)

        out = self.head(features, context)                    # (B, 49)

        angles = out[:, :N_ANGLES]
        scales = out[:, N_ANGLES:N_ANGLES + N_SCALES]

        # The head's six rotation numbers are read in the RAY frame, then turned
        # back into the viewer frame the label is written in. See ray_rotation.
        _six, in_ray_frame = rotation_from_6d(
            out[:, N_ANGLES + N_SCALES:N_ANGLES + N_SCALES + N_ROT6D])
        rotation = ray_rotation(context) @ in_ray_frame        # (B, 3, 3), viewer frame

        # Re-read the six off the composed matrix, NOT off `_six`, so what the
        # loss compares against target[40:46] is in the same frame as the label.
        # Columns again, matching dataset.target.
        rot6d = torch.cat([rotation[:, :, 0], rotation[:, :, 1]], dim=1)

        camera = out[:, N_ANGLES + N_SCALES + N_ROT6D:]       # (B, 3) — u, v, size

        # Assembled in dataset.target's order, so the loss is one subtraction.
        vector = torch.cat([angles, scales, rot6d, camera], dim=1)

        return {
            'vector': vector,
            'angles': angles,
            'scales': scales,
            'rot6d': rot6d,
            'camera': camera,
            'rotation': rotation,
        }


def average_target(data):
    """
    The mean of the 49 targets over a dataset — for PoseNet(average=...).

        in   a dataset.PoseData
        out  (49,) numpy array

    Reads the shards directly rather than iterating the dataset, so it decodes
    no JPEGs and takes about a second. The (u, v) slots come out near 0 because
    the crop is built around the person, which is the right starting guess.
    """
    rows = []
    for shard in data.shards.values():
        intrinsics = shard["intrinsics"]
        for row in range(len(shard["frame"])):
            for side in ("", "_flip"):
                box = shard["box"][row].copy()
                if side:
                    box[0] = 2 * intrinsics[2] - box[0]
                rotation = shard["root_rot" + side][row]
                rows.append(np.concatenate([
                    shard["angles" + side][row],
                    shard["scales" + side][row],
                    rotation[:, 0], rotation[:, 1],
                    hips_in_crop(shard["joints" + side][row][PELVIS], box, intrinsics),
                ]))
    return np.mean(rows, axis=0)


if __name__ == "__main__":
    net = PoseNet(pretrained=False)
    out = net(torch.zeros(2, 3, 256, 256), torch.zeros(2, 3))
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters")
    for name, value in out.items():
        print(f"  {name:9s} {tuple(value.shape)}")