"""Analytic inverse kinematics for the MHR body model.

Given a set of 3D target keypoints (a person's joint positions in world space),
this works out MHR pose parameters that put the rig into the same pose.

There is no optimiser here, and no iteration. The pose is built in a single pass
down the skeleton, and every rotation comes from a direct geometric construction:
"here is where the bone points now, here is where it should point, rotate it".

The plan
--------
1. Make the rig's bones the same length as the target's bones.
2. Put the rig's root at the target's pelvis, and turn it to face the same way.
3. Walk down the skeleton from the root. For each bone:
     a. work out which way the bone points right now (forward kinematics),
     b. work out which way it should point (from the target keypoints),
     c. build the rotation that carries one to the other,
     d. write that rotation into the pose parameters MHR expects,
     e. re-run forward kinematics, so the next bone down sees a corrected parent.

Because we always work from the root outwards, every bone is solved against a
parent that has already been fixed.

Conventions this code relies on (all measured, see the notebook)
---------------------------------------------------------------
* MHR joints form a tree; a joint's parent always has a smaller index.
* A joint's world rotation is  R_parent @ pre_rotation @ euler_xyz(rx, ry, rz).
  The pre-rotation is fixed by the rig; the three euler angles are ours to set.
* Bones point along the joint's local +x axis, so rotating about x is a twist.
* Elbows and knees are hinges: MHR only lets us set ONE angle there (rz).
* Positions are centimetres, +y is up. Root translation is in units of 10 cm.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mhr_kit.model import load_mhr, joint_names, parameter_names


# ---------------------------------------------------------------------------
# Part 1: small geometry helpers
# ---------------------------------------------------------------------------

def normalise(vector):
    """Return the vector scaled to length 1."""
    length = np.linalg.norm(vector)
    if length < 1e-9:
        raise ValueError("cannot normalise a zero-length vector")
    return np.asarray(vector, dtype=float) / length


def shortest_arc_rotation(from_direction, to_direction):
    """The smallest rotation that turns one direction into another.

    Think of spinning a globe so that one city ends up where another was: there
    is a single axis to spin about (perpendicular to both) and a single angle.
    """
    a = normalise(from_direction)
    b = normalise(to_direction)

    axis = np.cross(a, b)
    axis_length = np.linalg.norm(axis)

    # The two directions already agree, so there is nothing to do.
    if axis_length < 1e-9 and np.dot(a, b) > 0:
        return np.eye(3)

    # The two directions are exactly opposite. Any perpendicular axis turns one
    # into the other; pick one that is definitely not parallel to a.
    if axis_length < 1e-9:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = normalise(np.cross(a, helper))
        return Rotation.from_rotvec(axis * np.pi).as_matrix()

    angle = np.arctan2(axis_length, np.dot(a, b))
    return Rotation.from_rotvec(normalise(axis) * angle).as_matrix()


def frame_from_two_directions(main_direction, side_direction):
    """Build a set of three perpendicular axes from two rough directions.

    `main_direction` is kept exactly. `side_direction` only decides the roll
    around it, so we straighten it up first (this is Gram-Schmidt: subtract off
    the part that points along the main direction, keeping the rest).
    """
    x_axis = normalise(main_direction)
    side = np.asarray(side_direction, dtype=float)
    y_axis = normalise(side - np.dot(side, x_axis) * x_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def rotation_between_frames(from_main, from_side, to_main, to_side):
    """The rotation that carries one pair of directions onto another pair.

    Used where aiming a bone is not enough and we also need to control how it is
    rolled about its own length — for example an upper arm, where the roll
    decides which way the elbow bends.
    """
    source = frame_from_two_directions(from_main, from_side)
    target = frame_from_two_directions(to_main, to_side)
    return target @ source.T


def signed_angle_about_axis(from_direction, to_direction, axis):
    """How far to turn about `axis` to get from one direction to the other.

    Only the parts of the directions that lie across the axis matter, so we
    flatten both onto the plane the axis is perpendicular to, then measure the
    angle between them. The sign follows the right-hand rule about the axis.
    """
    axis = normalise(axis)
    a = np.asarray(from_direction, dtype=float)
    b = np.asarray(to_direction, dtype=float)

    flat_a = a - np.dot(a, axis) * axis
    flat_b = b - np.dot(b, axis) * axis
    if np.linalg.norm(flat_a) < 1e-9 or np.linalg.norm(flat_b) < 1e-9:
        return 0.0                      # a direction lies along the axis: no turn is defined

    flat_a = normalise(flat_a)
    flat_b = normalise(flat_b)
    cosine = np.dot(flat_a, flat_b)
    sine = np.dot(np.cross(flat_a, flat_b), axis)
    return float(np.arctan2(sine, cosine))


# ---------------------------------------------------------------------------
# Part 2: the rig, and forward kinematics
# ---------------------------------------------------------------------------

class Rig:
    """The MHR model plus the few lookup tables this solver needs."""

    def __init__(self, level_of_detail=6):
        self.model = load_mhr(lod=level_of_detail, device="cpu", correctives=False)
        skeleton = self.model.character.skeleton

        self.joint_names = joint_names(self.model)
        self.parameter_names = parameter_names(self.model)
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self.parameter_index = {name: i for i, name in enumerate(self.parameter_names)}

        self.parents = np.asarray(skeleton.joint_parents)
        self.pre_rotations = np.asarray(skeleton.pre_rotations)     # fixed xyzw quaternion per joint

        self.identity = torch.zeros(1, 45)                          # an average body shape

    def rest_pose(self):
        """A pose vector with every parameter at zero (MHR's T-pose)."""
        return np.zeros(len(self.parameter_names))

    def forward_kinematics(self, pose):
        """Where every joint is, and which way it is turned, for a given pose.

        Returns positions (127, 3) in cm and rotations (127, 3, 3) in world space.
        """
        parameters = torch.tensor(np.asarray(pose, dtype=np.float32)[None, :])
        with torch.no_grad():
            _, skeleton_state = self.model(self.identity, parameters, None)
        state = skeleton_state[0].numpy()

        positions = state[:, 0:3]
        rotations = Rotation.from_quat(state[:, 3:7]).as_matrix()
        return positions, rotations

    def set_rotation(self, pose, joint, parameters_xyz, world_rotation, rotations):
        """Write a world rotation for `joint` into the pose parameters.

        MHR does not take world rotations. It takes three euler angles that sit
        *after* the parent's rotation and the joint's fixed pre-rotation:

            world = parent_world @ pre_rotation @ euler_xyz(rx, ry, rz)

        so we undo those two to get the part that is ours to set.
        """
        j = self.joint_index[joint]
        parent = self.parents[j]
        parent_rotation = np.eye(3) if parent < 0 else rotations[parent]
        pre_rotation = Rotation.from_quat(self.pre_rotations[j]).as_matrix()

        local = pre_rotation.T @ parent_rotation.T @ world_rotation
        angles = Rotation.from_matrix(local).as_euler("xyz")

        for name, angle in zip(parameters_xyz, angles):
            pose[self.parameter_index[name]] = angle

    def hinge_axis(self, joint, rotations):
        """The one axis a hinge joint (elbow, knee) is allowed to turn about.

        MHR only exposes this joint's `rz`, so the hinge spins about its local z.
        The joint's own angle does not move that axis, which is why we can work
        it out before knowing the angle.
        """
        j = self.joint_index[joint]
        parent_rotation = rotations[self.parents[j]]
        pre_rotation = Rotation.from_quat(self.pre_rotations[j]).as_matrix()
        return normalise(parent_rotation @ pre_rotation @ np.array([0.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# Part 3: which target keypoint means which part of the rig
# ---------------------------------------------------------------------------
#
# The target keypoints we expect, by name. This is the 17-joint layout used by
# ASPset-510 and by most 3D pose datasets.

TARGET_NAMES = [
    "pelvis", "spine", "neck", "head", "head_top",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
]

# Bones we can aim with a single rotation. Each entry says: swing `joint` until
# the joint called `tip` points straight at the target keypoint `aim_at`.
#
# We aim at a *point* rather than copy a direction because the rig's joint and
# the matching target keypoint are not always in the same place. The spine is
# the clear case: MHR's lowest spine joint sits 3 cm above and behind the root,
# so copying the target's pelvis-to-neck direction would tilt the spine by
# nearly 4 degrees before we even started. Aiming at the neck avoids that.
SIMPLE_BONES = [
    # joint,        tip,           aim at,           parameters (rx, ry, rz)
    ("l_clavicle",  "l_uparm",     "left_shoulder",
     ("l_clavicle_rx", "l_clavicle_ry", "l_clavicle_rz")),
    ("r_clavicle",  "r_uparm",     "right_shoulder",
     ("r_clavicle_rx", "r_clavicle_ry", "r_clavicle_rz")),
    ("c_neck",    "c_head",      "head",
     ("neck_twist", "neck_lean", "neck_bend")),
    ("c_head",    "c_head_null", "head_top",
     ("head_twist", "head_lean", "head_bend")),
]

# Two-bone limbs. The upper bone is aimed *and* rolled, so that the hinge below
# it bends in the right plane; then the hinge angle is measured directly.
LIMBS = [
    # upper joint, hinge joint, hinge tip,   targets (upper, middle, lower),        upper parameters,                        hinge parameter
    ("l_uparm", "l_lowarm", "l_wrist_twist", ("left_shoulder", "left_elbow", "left_wrist"),
     ("l_uparm_twist", "l_uparm_ry", "l_uparm_rz"), "l_elbow_bend"),
    ("r_uparm", "r_lowarm", "r_wrist_twist", ("right_shoulder", "right_elbow", "right_wrist"),
     ("r_uparm_twist", "r_uparm_ry", "r_uparm_rz"), "r_elbow_bend"),
    ("l_upleg", "l_lowleg", "l_foot", ("left_hip", "left_knee", "left_ankle"),
     ("l_upleg_twist", "l_upleg_ry", "l_upleg_rz"), "l_knee_bend"),
    ("r_upleg", "r_lowleg", "r_foot", ("right_hip", "right_knee", "right_ankle"),
     ("r_upleg_twist", "r_upleg_ry", "r_upleg_rz"), "r_knee_bend"),
]

# Bone lengths we can stretch to match the target. Each MHR parameter moves its
# bone by a fixed number of centimetres per unit (measured, not guessed); the hip
# width parameter moves both sides at once, so it counts double. Where the target
# has a left and a right copy of a bone we average the two, because MHR gives
# both sides a single shared parameter.
#
# Every distance here is one MHR bone cannot change with pose. Shoulder width is
# deliberately absent for that reason: the distance between the two shoulders
# changes as the collarbones rotate, so measuring it would mix pose into build.
BONE_LENGTHS = [
    # parameter,            rig joints,                    target pairs to average,                                cm per unit
    ("scale_uplegs", ("l_upleg", "l_lowleg"),
     [("left_hip", "left_knee"), ("right_hip", "right_knee")], 10.0),
    ("scale_lowlegs", ("l_lowleg", "l_foot"),
     [("left_knee", "left_ankle"), ("right_knee", "right_ankle")], 10.0),
    ("scale_uparms", ("l_uparm", "l_lowarm"),
     [("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")], 10.0),
    ("scale_lowarms", ("l_lowarm", "l_wrist_twist"),
     [("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")], 10.0),
    ("scale_spine_length", ("root", "c_neck"),
     [("pelvis", "neck")], 10.0),
    ("scale_neck_length", ("c_neck", "c_head"),
     [("neck", "head")], 10.0),
    ("scale_hip_width", ("l_upleg", "r_upleg"),
     [("left_hip", "right_hip")], 20.0),
]

CENTIMETRES_PER_UNIT = 10.0     # the root's translation parameters use these units too


# ---------------------------------------------------------------------------
# Part 4: the solver
# ---------------------------------------------------------------------------

def match_bone_lengths(rig, pose, targets):
    """Stretch the rig's bones so they are as long as the target's bones.

    This is not part of the pose — it is the body's build. Doing it first means
    the joints can actually land on the targets instead of merely pointing at
    them.
    """
    rest_positions, _ = rig.forward_kinematics(rig.rest_pose())

    for parameter, (rig_a, rig_b), target_pairs, centimetres_per_unit in BONE_LENGTHS:

        rig_length = np.linalg.norm(rest_positions[rig.joint_index[rig_a]]
                                    - rest_positions[rig.joint_index[rig_b]])
        
        target_length = np.mean([np.linalg.norm(targets[a] - targets[b])
                                 for a, b in target_pairs])

        pose[rig.parameter_index[parameter]] = (target_length - rig_length) / centimetres_per_unit

    return pose


def body_frame(targets):
    """Three perpendicular axes describing which way the target's body faces.

    across = right hip to left hip, up = pelvis to neck, forward = across x up.
    """
    across = targets["left_hip"] - targets["right_hip"]
    up = targets["neck"] - targets["pelvis"]
    frame = frame_from_two_directions(across, up)
    return frame[:, 0], frame[:, 1], np.cross(frame[:, 0], frame[:, 1])


def match_hip_offset(rig, pose, targets):
    """Move the rig's hips so they sit relative to the pelvis the way the target's do.

    Different marker sets disagree about where "the pelvis" is. ASPset's pelvis
    point sits several centimetres above and in front of the hip joints, while
    MHR's root sits almost between them. That is a difference in build, not in
    pose, so no rotation can fix it — but two of MHR's shape parameters move the
    hips up and forward, and we can measure exactly how far to move them.
    """
    rest_positions, _ = rig.forward_kinematics(rig.rest_pose())
    rig_offset = (0.5 * (rest_positions[rig.joint_index["l_upleg"]]
                         + rest_positions[rig.joint_index["r_upleg"]])
                  - rest_positions[rig.joint_index["root"]])

    _, up, forward = body_frame(targets)
    target_offset = 0.5 * (targets["left_hip"] + targets["right_hip"]) - targets["pelvis"]

    # In the rest pose the rig's own up and forward axes are simply +y and +z.
    pose[rig.parameter_index["scale_hip_height"]] = (
        np.dot(target_offset, up) - rig_offset[1]) / CENTIMETRES_PER_UNIT
    pose[rig.parameter_index["scale_hip_depth"]] = (
        np.dot(target_offset, forward) - rig_offset[2]) / CENTIMETRES_PER_UNIT
    return pose


def solve_root(rig, pose, targets):
    """Place the root at the target's pelvis and turn it to face the same way.

    The root has no single bone to aim, so instead we match two directions at
    once: the line across the hips, and the line up the spine. Two directions
    pin down a rotation completely.
    """
    # Which way do the hips and the spine run on the rig as it stands?
    positions, rotations = rig.forward_kinematics(pose)
    rig_hips = (positions[rig.joint_index["l_upleg"]]
                - positions[rig.joint_index["r_upleg"]])
    rig_spine = (positions[rig.joint_index["c_neck"]]
                 - positions[rig.joint_index["root"]])

    # And in the target?
    target_hips = targets["left_hip"] - targets["right_hip"]
    target_spine = targets["neck"] - targets["pelvis"]

    # Turn the rig's pair of directions onto the target's pair.
    turn = rotation_between_frames(rig_hips, rig_spine, target_hips, target_spine)
    new_world_rotation = turn @ rotations[rig.joint_index["root"]]
    rig.set_rotation(pose, "root", ("root_rx", "root_ry", "root_rz"),
                     new_world_rotation, rotations)

    # Now slide the whole body so the root sits on the target's pelvis.
    positions, _ = rig.forward_kinematics(pose)
    shift = targets["pelvis"] - positions[rig.joint_index["root"]]
    for name, amount in zip(("root_tx", "root_ty", "root_tz"), shift):
        pose[rig.parameter_index[name]] = amount / CENTIMETRES_PER_UNIT

    return pose


SPINE_PARAMETERS = ("spine0_rx_flexible", "spine0_ry_flexible", "spine0_rz_flexible")


def solve_spine(rig, pose, targets):
    """Swing the spine until the neck lands right, and twist it so the shoulders line up.

    Aiming on its own is not enough here. A bone that is merely aimed is still
    free to spin about its own length, and spinning the spine carries the
    shoulders round with it. The line across the shoulders pins that last
    freedom down, so we match two directions at once, exactly as we did at the
    root.
    """
    positions, rotations = rig.forward_kinematics(pose)
    here = positions[rig.joint_index["c_spine0"]]

    current_up = positions[rig.joint_index["c_neck"]] - here
    current_across = (positions[rig.joint_index["l_uparm"]]
                      - positions[rig.joint_index["r_uparm"]])

    wanted_up = targets["neck"] - here
    wanted_across = targets["left_shoulder"] - targets["right_shoulder"]

    turn = rotation_between_frames(current_up, current_across, wanted_up, wanted_across)
    new_world_rotation = turn @ rotations[rig.joint_index["c_spine0"]]
    rig.set_rotation(pose, "c_spine0", SPINE_PARAMETERS, new_world_rotation, rotations)
    return pose


def solve_simple_bone(rig, pose, targets, joint, tip, aim_at, parameters):
    """Swing one bone until its tip points at a target keypoint.

    Both directions are measured from where the rig's joint actually is, so this
    stays correct even when the rig joint and the target keypoint do not sit in
    quite the same place.
    """
    positions, rotations = rig.forward_kinematics(pose)
    here = positions[rig.joint_index[joint]]

    current_direction = positions[rig.joint_index[tip]] - here
    wanted_direction = targets[aim_at] - here

    turn = shortest_arc_rotation(current_direction, wanted_direction)
    new_world_rotation = turn @ rotations[rig.joint_index[joint]]
    rig.set_rotation(pose, joint, parameters, new_world_rotation, rotations)
    return pose


def solve_limb(rig, pose, targets, upper, hinge, tip, target_keys, upper_parameters, hinge_parameter):
    """Solve a two-bone limb: shoulder-elbow-wrist, or hip-knee-ankle.

    Aiming the upper bone is not enough on its own. A hinge can only bend in one
    plane, so before bending it we have to roll the upper bone until that plane
    contains the target. The axis the elbow bends about must stand perpendicular
    to both bones — that is exactly the cross product of the two target bones,
    and it gives us the roll for free.
    """
    upper_key, middle_key, lower_key = target_keys

    wanted_upper = targets[middle_key] - targets[upper_key]
    wanted_lower = targets[lower_key] - targets[middle_key]

    # --- the upper bone: aim it, and roll it so the hinge lines up ------------
    positions, rotations = rig.forward_kinematics(pose)
    current_upper = positions[rig.joint_index[hinge]] - positions[rig.joint_index[upper]]
    current_axis = rig.hinge_axis(hinge, rotations)

    wanted_axis = np.cross(wanted_upper, wanted_lower)

    if np.linalg.norm(wanted_axis) < 1e-6 * np.linalg.norm(wanted_upper) * np.linalg.norm(wanted_lower):
        # The limb is dead straight, so no plane is defined and no roll is
        # needed either. Just aim the upper bone.
        turn = shortest_arc_rotation(current_upper, wanted_upper)
    else:
        turn = rotation_between_frames(current_upper, current_axis, wanted_upper, wanted_axis)

    rig.set_rotation(pose, upper, upper_parameters, turn @ rotations[rig.joint_index[upper]], rotations)

    # --- the hinge: measure the bend directly ---------------------------------
    positions, rotations = rig.forward_kinematics(pose)
    current_lower = positions[rig.joint_index[tip]] - positions[rig.joint_index[hinge]]
    axis = rig.hinge_axis(hinge, rotations)

    pose[rig.parameter_index[hinge_parameter]] = signed_angle_about_axis(
        current_lower, wanted_lower, axis)
    return pose


def solve(rig, targets):
    """Work out MHR pose parameters that match a set of 3D target keypoints.

    `targets` is a dictionary of name -> (3,) position in centimetres, +y up.
    Returns the (204,) pose parameter vector.
    """
    missing = [name for name in TARGET_NAMES if name not in targets]
    if missing:
        raise ValueError(f"missing target keypoints: {missing}")

    pose = rig.rest_pose()

    # 1. the body's build, then where it stands and which way it faces
    pose = match_bone_lengths(rig, pose, targets)
    pose = match_hip_offset(rig, pose, targets)
    pose = solve_root(rig, pose, targets)

    # 2. up the spine, then the neck and head
    pose = solve_spine(rig, pose, targets)
    for joint, tip, aim_at, parameters in SIMPLE_BONES:
        pose = solve_simple_bone(rig, pose, targets, joint, tip, aim_at, parameters)

    # 3. the four limbs
    for upper, hinge, tip, target_keys, upper_parameters, hinge_parameter in LIMBS:
        pose = solve_limb(rig, pose, targets, upper, hinge, tip,
                          target_keys, upper_parameters, hinge_parameter)

    return pose


# ---------------------------------------------------------------------------
# Part 5: reading the answer back out
# ---------------------------------------------------------------------------
#
# Which rig joint stands in for which target keypoint, so we can measure how
# close we got. `head_top` has no rig joint of its own, so we use the tip of the
# head bone.

TARGET_TO_RIG = {
    "pelvis": "root", "spine": "c_spine1", "neck": "c_neck", "head": "c_head",
    "head_top": "c_head_null",
    "left_shoulder": "l_uparm", "left_elbow": "l_lowarm", "left_wrist": "l_wrist_twist",
    "right_shoulder": "r_uparm", "right_elbow": "r_lowarm", "right_wrist": "r_wrist_twist",
    "left_hip": "l_upleg", "left_knee": "l_lowleg", "left_ankle": "l_foot",
    "right_hip": "r_upleg", "right_knee": "r_lowleg", "right_ankle": "r_foot",
}


def posed_keypoints(rig, pose):
    """The rig's own version of the 17 keypoints, after posing."""
    positions, _ = rig.forward_kinematics(pose)
    return {name: positions[rig.joint_index[rig_joint]]
            for name, rig_joint in TARGET_TO_RIG.items()}


def position_errors(rig, pose, targets):
    """Distance in cm between each target keypoint and where the rig put it."""
    got = posed_keypoints(rig, pose)
    return {name: float(np.linalg.norm(got[name] - targets[name])) for name in TARGET_TO_RIG}


# The bones whose direction the solver actually controls. Getting these right is
# what the solver is for; joint positions also depend on bone lengths.
MEASURED_BONES = [
    ("pelvis", "neck"), ("neck", "head"), ("head", "head_top"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def direction_errors(rig, pose, targets):
    """Angle in degrees between each target bone and the rig's version of it."""
    got = posed_keypoints(rig, pose)
    errors = {}
    for a, b in MEASURED_BONES:
        wanted = normalise(targets[b] - targets[a])
        actual = normalise(got[b] - got[a])
        errors[f"{a} -> {b}"] = float(np.degrees(np.arccos(np.clip(np.dot(wanted, actual), -1.0, 1.0))))
    return errors


def to_model_parameters(rig, pose):
    """Wrap a pose vector as the (1, 204) tensor MHR's forward pass wants."""
    return torch.tensor(np.asarray(pose, dtype=np.float32)[None, :])


def mesh_from_pose(rig, pose):
    """The posed body mesh, (vertices, 3) in cm."""
    with torch.no_grad():
        vertices, _ = rig.model(rig.identity, to_model_parameters(rig, pose), None)
    return vertices[0].numpy()


# ---------------------------------------------------------------------------
# Part 6: turning ASPset keypoints into what the solver expects
# ---------------------------------------------------------------------------

# ASPset stores its 17 joints in this order, in millimetres with +y pointing down.
ASPSET_ORDER = [
    "right_ankle", "right_knee", "right_hip", "right_wrist", "right_elbow", "right_shoulder",
    "left_ankle", "left_knee", "left_hip", "left_wrist", "left_elbow", "left_shoulder",
    "head_top", "head", "neck", "spine", "pelvis",
]


def targets_from_aspset(joints_mm):
    """Convert one frame of ASPset joints into the dictionary `solve` expects.

    ASPset works in millimetres with +y down; MHR works in centimetres with
    +y up. Flipping y alone would turn the body inside out (a left hand would
    become a right hand), so we flip z as well to keep the handedness.
    """
    converted = 0.1 * np.asarray(joints_mm, dtype=float) * np.array([1.0, -1.0, -1.0])
    return {name: converted[i] for i, name in enumerate(ASPSET_ORDER)}
