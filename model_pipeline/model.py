"""The network: a picture of a person goes in, MHR pose parameters come out.

What it predicts
----------------
Three different kinds of number, because they behave differently:

* **where the body faces** — a full rotation. Angles are a bad way to ask a
  network for this: the same heading can be written as two very different sets of
  angles, and near certain headings tiny changes in the body cause huge changes in
  the angles. So the network predicts six numbers that get turned into a rotation
  matrix, which has neither problem.
* **how the body is bent** — 31 joint angles. These stay in a narrow, safe range
  (an elbow only bends so far), so plain numbers are fine. They are divided by
  how much they normally vary, so that a shoulder and a knee matter equally.
* **where the body stands** — three numbers describing how the person sits in
  their crop. Combined with where the crop was taken from, those give a real
  position in front of the camera. This is what makes a person in the top right of
  the picture come out in the top right of the 3D scene.

How it is put together
----------------------
DINOv2 turns the crop into 256 patch descriptions. Then fifteen "questions"
(one per joint group, plus one about the camera) each look over those patches and
pull out what they need. Asking one question per joint, rather than squeezing
everything through a single summary of the image, lets the elbow question look at
the elbow. It also means adding fingers later is adding questions, not rebuilding
the network.

The backbone is **frozen** - DINOv2 already knows what a person looks like, and with
about 1 300 training pictures of a single person, 22 million adjustable weights
would learn that person rather than the pose. Only the head is trained, which is
4.7 million weights.

DINOv2 is Apache-2.0, and at 22 million parameters it is small enough to deploy.
"""

import torch
from torch import nn

# The joints, in the order the network answers about them. The first group is the
# whole body's heading and is treated specially; the rest are ordinary angles.
ROTATION_GROUPS = [
    ("root", ("root_rx", "root_ry", "root_rz")),
    ("spine", ("spine0_rx_flexible", "spine0_ry_flexible", "spine0_rz_flexible")),
    ("neck", ("neck_twist", "neck_lean", "neck_bend")),
    ("head", ("head_twist", "head_lean", "head_bend")),
    ("left collarbone", ("l_clavicle_rx", "l_clavicle_ry", "l_clavicle_rz")),
    ("right collarbone", ("r_clavicle_rx", "r_clavicle_ry", "r_clavicle_rz")),
    ("left upper arm", ("l_uparm_twist", "l_uparm_ry", "l_uparm_rz")),
    ("right upper arm", ("r_uparm_twist", "r_uparm_ry", "r_uparm_rz")),
    ("left thigh", ("l_upleg_twist", "l_upleg_ry", "l_upleg_rz")),
    ("right thigh", ("r_upleg_twist", "r_upleg_ry", "r_upleg_rz")),
]

# Elbows and knees are hinges: MHR gives them one angle each, so there is no
# choice of representation to get wrong and nothing to normalise away.
HINGES = ("l_elbow_bend", "r_elbow_bend", "l_knee_bend", "r_knee_bend")

# Everything the network predicts as a plain angle: the nine non-root groups,
# then the four hinges. 31 numbers, in this exact order.
ARTICULATION = [name for _, names in ROTATION_GROUPS[1:] for name in names] + list(HINGES)

# The 17 joints the dataset measures, in the order analytic_ik.TARGET_TO_RIG lists
# them. train.py checks that this still matches.
KEYPOINT_JOINTS = ("root", "c_spine1", "c_neck", "c_head", "c_head_null",
                   "l_uparm", "l_lowarm", "l_wrist_twist",
                   "r_uparm", "r_lowarm", "r_wrist_twist",
                   "l_upleg", "l_lowleg", "l_foot",
                   "r_upleg", "r_lowleg", "r_foot")

QUERIES = len(ROTATION_GROUPS) + len(HINGES) + 1        # + 1 for the camera
WIDTH = 384                                             # DINOv2-Small's feature size

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def rotation_from_six(six):
    """Six numbers -> a rotation matrix.

    Take the six as two arbitrary vectors. Make the first one a unit vector: that
    is the first axis. Push the second one square to it and make it a unit vector
    too: the second axis. The third axis is forced, being at right angles to both.
    Any six numbers give a valid rotation, so the network cannot produce a
    nonsense answer, and nearby rotations come from nearby numbers.
    """
    first, second = six[..., :3], six[..., 3:]
    x = torch.nn.functional.normalize(first, dim=-1)
    y = torch.nn.functional.normalize(second - (x * second).sum(-1, keepdim=True) * x, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def rotation_difference(predicted, wanted):
    """How far apart two rotations are, as a plain difference between the matrices.

    Comparing the matrices entry by entry sounds crude, but it behaves far better
    than measuring the angle between them: the angle involves an arccos, whose
    slope becomes infinite exactly where the answer is right, which is the worst
    possible place for training. This is smooth everywhere and is smallest in the
    same place. Use `rotation_angle` when you want a number to report.
    """
    return (predicted - wanted).abs().mean(dim=(-2, -1))


def rotation_angle(predicted, wanted):
    """How far apart two rotations are, in radians. For reporting, not training."""
    product = predicted.transpose(-1, -2) @ wanted
    trace = product.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))


# ---------------------------------------------------------------------------
# From the crop back to the real world
# ---------------------------------------------------------------------------
#
# The network only ever sees a small square cut out of a big picture, so it can
# only say where the person is *within that square*. Where the square came from
# turns that into a real position, and this is the sum that does it.
#
# A person twice as far away looks half as tall, so the size of the person in the
# crop gives the distance; and the position of the crop in the picture gives the
# sideways and vertical offset. Both need the camera's focal length, which ASPset
# supplies exactly.

def translation_from_camera(camera, box, focal, centre):
    """(log size, sideways, vertical) in the crop -> a position in front of the camera, cm."""
    log_size, sideways, vertical = camera[:, 0], camera[:, 1], camera[:, 2]
    box_x, box_y, box_side = box[:, 0], box[:, 1], box[:, 2]
    focal_x, focal_y = focal[:, 0], focal[:, 1]
    centre_x, centre_y = centre[:, 0], centre[:, 1]

    distance = 2 * focal_x / (box_side * torch.exp(log_size))
    across = sideways + (box_x - centre_x) * distance / focal_x
    down = vertical + (box_y - centre_y) * distance / focal_y

    # The camera counts y downwards and z into the scene; we count y up and the
    # scene at negative z, so both flip.
    return torch.stack([across, -down, -distance], dim=1)


def camera_from_translation(position, box, focal, centre):
    """The exact opposite of `translation_from_camera`, used to make the labels."""
    across, down, distance = position[:, 0], -position[:, 1], -position[:, 2]
    log_size = torch.log(2 * focal[:, 0] / (box[:, 2] * distance))
    sideways = across - (box[:, 0] - centre[:, 0]) * distance / focal[:, 0]
    vertical = down - (box[:, 1] - centre[:, 1]) * distance / focal[:, 1]
    return torch.stack([log_size, sideways, vertical], dim=1)


def project(points, focal, centre):
    """Points in front of the camera (cm) -> pixels in the full picture."""
    across, down, distance = points[..., 0], -points[..., 1], -points[..., 2]
    pixels = torch.stack([across / distance, down / distance], dim=-1)
    return pixels * focal[:, None, :] + centre[:, None, :]


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class PoseNet(nn.Module):
    def __init__(self, mean, std, depth=2):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                       verbose=False)

        # The backbone is frozen: it is never trained, and its weights never move.
        # There are 22 million of them and only about 1 300 training pictures, all
        # of one person, so letting them move mostly teaches the network to
        # recognise that person rather than to read a pose. Only the 4.7 million
        # in the head below are learned.
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        layer = nn.TransformerDecoderLayer(WIDTH, nhead=6, dim_feedforward=4 * WIDTH,
                                           batch_first=True, norm_first=True,
                                           dropout=0.0)
        self.reader    = nn.TransformerDecoder(layer, depth)
        self.questions = nn.Parameter(torch.randn(QUERIES, WIDTH) * 0.02)

        # Where the crop sat in the full picture, as three numbers the size of
        # which does not depend on the camera. Without this the network cannot
        # tell a person at the edge of the frame from one in the middle, and they
        # are seen from genuinely different angles.
        self.where = nn.Linear(3, WIDTH)
        self.answer = nn.Linear(WIDTH, 6)

        # Fixed, not learned: the average and spread of every predicted number.
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("image_mean", IMAGE_MEAN)
        self.register_buffer("image_std", IMAGE_STD)

    def train(self, mode=True):
        """Put the head in training mode, but leave the frozen backbone in eval.

        Without this, `network.train()` would switch the backbone's own layers back
        into training behaviour even though its weights are fixed, so the same crop
        would describe itself differently from one epoch to the next.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image, box, focal, centre):
        # No gradient is kept for this part: nothing in the backbone can change,
        # so there is nothing to send a gradient back to.
        with torch.no_grad():
            image = (image - self.image_mean) / self.image_std
            patches = self.backbone.forward_features(image)["x_norm_patchtokens"]

        place = torch.stack([(box[:, 0] - centre[:, 0]) / focal[:, 0],
                             (box[:, 1] - centre[:, 1]) / focal[:, 1],
                             box[:, 2] / focal[:, 0]], dim=1)
        described = torch.cat([patches, self.where(place)[:, None, :]], dim=1)

        answers = self.answer(self.reader(self.questions.expand(len(image), -1, -1), described))

        rotation = rotation_from_six(answers[:, 0, :])                  # the heading
        angles = answers[:, 1:len(ROTATION_GROUPS), :3].flatten(1)      # 9 groups x 3
        hinges = answers[:, len(ROTATION_GROUPS):-1, 0]                 # 4 hinges
        camera = answers[:, -1, :3]                                     # where it stands

        # Undo the normalisation, so these come back out as real angles.
        scaled = torch.cat([angles, hinges, camera], dim=1) * self.std + self.mean
        return rotation, scaled[:, :len(ARTICULATION)], scaled[:, len(ARTICULATION):]


# ---------------------------------------------------------------------------
# Posing the body
# ---------------------------------------------------------------------------

def body_in_own_frame(mhr, base_pose, indices, articulation, root, keypoints):
    """Run MHR forwards: joint angles -> joint positions, with the hips at the origin.

    The heading and the position are deliberately left out here and applied
    afterwards by `place_body`. Doing it that way means the network never has to
    name the heading in angles, which is the naming problem `rotation_from_six`
    avoids. Checked: taking a real pose apart this way and putting it back
    together reproduces every joint to 0.0000 cm.

    Two things about MHR's joint list that are easy to trip over, both measured:
    the hips are joint 1, not joint 0 -- joint 0 is a fixed point at the world
    origin that the body hangs from, and it does *not* move with the body, so it
    must not be carried through `place_body`. Only the joints asked for in
    `keypoints` come back, which keeps it out. And the hips face straight along
    the world axes when every angle is zero, so the rotation predicted here really
    is the direction the hips face, with no fixed turn left over.
    """
    pose = base_pose.expand(len(articulation), -1).clone()
    pose[:, indices] = articulation
    identity = torch.zeros(len(pose), 45, dtype=pose.dtype, device=pose.device)
    _, skeleton = mhr(identity, pose, None)
    joints = skeleton[:, :, :3]
    return (joints - joints[:, root:root + 1, :])[:, keypoints]


def place_body(joints, rotation, position):
    """Turn the body to face where it should, then stand it where it should."""
    return joints @ rotation.transpose(-1, -2) + position[:, None, :]


def rig_constants(rig):
    """The handful of fixed numbers about the rig that the rest of this file needs."""
    positions, rotations = rig.forward_kinematics(rig.rest_pose())
    root = rig.joint_index["root"]
    assert abs(rotations[root] - torch.eye(3).numpy()).max() < 1e-6, "hips are not axis-aligned at rest"
    return {
        "root": root,
        "root_at_rest": torch.tensor(positions[root], dtype=torch.float32),
        "indices": torch.tensor([rig.parameter_index[name] for name in ARTICULATION]),
        "keypoints": torch.tensor([rig.joint_index[name] for name in KEYPOINT_JOINTS]),
    }
