"""Re-place a solved MHR body in its own coordinate system.

`analytic_ik.py` gives you a body sitting wherever the dataset's world happens to
be — for ASPset that is 18 metres from the origin, because the origin is a camera
that was standing somewhere in a field. That is correct but awkward to look at.

This module moves the body into a tidy frame of its own **without changing
anything about it**. The pose, the way the body is turned relative to the camera,
and how far away it was all stay exactly as they were. Only the root moves.

The idea in one line
--------------------
Instead of moving the body, move the *whole scene* so that the camera ends up at
the origin. Everything keeps its shape and its spacing, because sliding and
turning a scene never changes any distance inside it.

The frame we land in
--------------------
    origin      the camera / the viewer
    +x          the viewer's right
    +y          up
    -z          the direction the viewer is looking

so the body sits at negative z, exactly as far away as it really was. This is
MHR's own convention: a body standing at rest faces +z, which is straight back
towards a viewer at the origin. So a person who was facing the camera comes out
facing +z, the same way the rig faces in its rest pose.

Only the root changes
---------------------
Two facts about MHR make this simple, and both are checked in the notebook:

* the root's rotation does not move the root, and
* `root_tx/ty/tz` slide the body along the world axes at 10 cm per unit,
  whatever the rotation is.

So the new place and the new heading can be written independently, and every
other pose parameter is left completely alone.
"""

import numpy as np

# Camera axes are x right, y **down**, z into the scene (the usual computer-vision
# convention). Our frame is y up. Turning the camera upside down gets us there:
# this flips y and z together, which is a real rotation rather than a mirror. A
# mirror would swap the body's left and right hands.
CAMERA_TO_VIEWER = np.diag([1.0, -1.0, -1.0])

ROOT_ROTATION = ("root_rx", "root_ry", "root_rz")
ROOT_TRANSLATION = ("root_tx", "root_ty", "root_tz")
CENTIMETRES_PER_UNIT = 10.0

# A body at rest faces this way, in MHR's own world. Measured: the root's rotation
# is the identity in the rest pose, and the model faces +z.
BODY_FACES = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Part 1: describing where the camera is
# ---------------------------------------------------------------------------

def camera_from_aspset(extrinsic_matrix):
    """Read an ASPset camera in the units the solved body uses.

    ASPset's own camera maps millimetres with +y down. The body is in centimetres
    with +y up. Both differences fold neatly into the rotation and the
    translation, so the caller never has to think about them again.
    """
    flip = np.diag([1.0, -1.0, -1.0])                 # cm/y-up  ->  mm/y-down axes
    rotation = np.asarray(extrinsic_matrix)[:3, :3] @ flip
    translation = np.asarray(extrinsic_matrix)[:3, 3] / 10.0      # mm -> cm
    return rotation, translation


def camera_already_at_origin():
    """The camera for keypoints that were already measured from the camera.

    Monocular estimators usually hand you 3D joints measured from the camera
    rather than from some room corner. If you fed the solver keypoints like that
    -- in the same y-up centimetres it works in -- then the body is already in the
    viewer's frame and :func:`place_in_viewer_frame` has nothing to do.

    Note this is *not* the identity. The identity would mean "the camera's axes are
    the same as the body's", and they are not: camera axes point y down. What we
    want is the transform that :data:`CAMERA_TO_VIEWER` undoes, which is
    :data:`CAMERA_TO_VIEWER` itself, because flipping twice gets you back.
    Re-placing an already-placed body is then a genuine no-op.
    """
    return CAMERA_TO_VIEWER, np.zeros(3)


# ---------------------------------------------------------------------------
# Part 2: the move itself
# ---------------------------------------------------------------------------

def place_in_viewer_frame(rig, pose, camera_rotation, camera_translation):
    """Return the same pose, with the body re-placed in the viewer's frame.

    `camera_rotation` and `camera_translation` describe where the camera was:
    together they turn a point in the body's world into a point measured from the
    camera. Use :func:`camera_from_aspset` or :func:`camera_at_origin` to get them.
    """
    positions, rotations = rig.forward_kinematics(pose)
    root = rig.joint_index["root"]

    # Step 1: where is the body, as seen from the camera?
    seen_from_camera_position = camera_rotation @ positions[root] + camera_translation
    seen_from_camera_rotation = camera_rotation @ rotations[root]

    # Step 2: say the same thing with y pointing up.
    wanted_position = CAMERA_TO_VIEWER @ seen_from_camera_position
    wanted_rotation = CAMERA_TO_VIEWER @ seen_from_camera_rotation

    # Step 3: write both into the root, and nothing else.
    new_pose = np.array(pose, dtype=float)
    rig.set_rotation(new_pose, "root", ROOT_ROTATION, wanted_rotation, rotations)

    shift = wanted_position - positions[root]
    for name, amount in zip(ROOT_TRANSLATION, shift):
        new_pose[rig.parameter_index[name]] += amount / CENTIMETRES_PER_UNIT

    return new_pose


def viewer_camera():
    """Where the camera ends up once the body has been moved.

    At the origin, looking along -z. Feed these to :class:`mhr_kit.render.Camera`
    as its `rotation` and `translation` and you get the viewer's own view of the
    body -- the same view the real photograph had.
    """
    return CAMERA_TO_VIEWER, np.zeros(3)


# ---------------------------------------------------------------------------
# Part 3: reading the result back
# ---------------------------------------------------------------------------

def distance_to_viewer(rig, pose):
    """How far the body's root is from the viewer, in centimetres.

    In the viewer's frame the viewer is the origin, so this is just the length of
    the root's position.
    """
    positions, _ = rig.forward_kinematics(pose)
    return float(np.linalg.norm(positions[rig.joint_index["root"]]))


def facing_direction(rig, pose):
    """The unit vector the body's hips point along."""
    _, rotations = rig.forward_kinematics(pose)
    direction = rotations[rig.joint_index["root"]] @ BODY_FACES
    return direction / np.linalg.norm(direction)


def turn_towards_viewer(rig, pose):
    """How far the body is turned away from the viewer, in degrees.

    0 means facing the viewer squarely and 180 means facing straight away. The
    sign says which way the person turned: positive is towards the viewer's
    right. Only the turn on the level is measured, so a bow or a lean does not
    count as turning away.
    """
    positions, _ = rig.forward_kinematics(pose)
    root = positions[rig.joint_index["root"]]

    up = np.array([0.0, 1.0, 0.0])
    facing = facing_direction(rig, pose)
    towards_viewer = -root                       # the viewer sits at the origin

    # Flatten both onto the level ground, so only the turn is left.
    facing = facing - np.dot(facing, up) * up
    towards_viewer = towards_viewer - np.dot(towards_viewer, up) * up
    facing /= np.linalg.norm(facing)
    towards_viewer /= np.linalg.norm(towards_viewer)

    cosine = np.dot(facing, towards_viewer)
    sine = np.dot(np.cross(facing, towards_viewer), up)
    return float(np.degrees(np.arctan2(sine, cosine)))


def describe_turn(degrees):
    """Put the turn into words, the way you would describe a photograph."""
    size = abs(degrees)
    side = "the viewer's right" if degrees > 0 else "the viewer's left"

    if size < 20:
        return "facing the viewer"
    if size > 160:
        return "turned away from the viewer"
    if size < 70:
        return f"turned towards {side}"
    if size < 110:
        return f"side-on, facing {side}"
    return f"turned away, over {side} shoulder"


# ---------------------------------------------------------------------------
# Part 4: checks
# ---------------------------------------------------------------------------
#
# Moving a scene must never change anything inside it. These two functions make
# that claim testable rather than a promise.
#
# Both skip `body_world`. That is not a joint of the body: it is a fixed marker
# sitting at the origin of MHR's world, and the whole point of this module is to
# move the body relative to it. Including it would report the move itself as an
# error.

def body_joints(rig):
    """Every joint that really belongs to the body, i.e. all but `body_world`."""
    return [j for j in range(len(rig.parents)) if rig.joint_names[j] != "body_world"]


def largest_shape_change(rig, pose_before, pose_after):
    """The biggest change in the distance between any two joints, in centimetres.

    Sliding and turning a body cannot change how far its joints are from each
    other, so this should come out at zero. If it does not, the pose was altered.
    """
    keep = body_joints(rig)
    before, _ = rig.forward_kinematics(pose_before)
    after, _ = rig.forward_kinematics(pose_after)
    before, after = before[keep], after[keep]

    gaps_before = np.linalg.norm(before[:, None, :] - before[None, :, :], axis=2)
    gaps_after = np.linalg.norm(after[:, None, :] - after[None, :, :], axis=2)
    return float(np.abs(gaps_before - gaps_after).max())


# MHR has a handful of joints that sit on top of their parent, or within a few
# millimetres of it -- the ankle has one, and so do the twist joints and the eye
# and jaw markers. Two reasons to leave them out: a bone of no length has no
# direction at all, and a very short one has an unreliable direction, because MHR
# computes forward kinematics in 32-bit floats. Twenty metres from the origin that
# leaves about a thousandth of a centimetre of noise on every joint, which is
# nothing on a thigh and everything on a two-millimetre bone.
SHORTEST_REAL_BONE = 1.0        # centimetres


def largest_direction_change(rig, pose_before, pose_after, camera_rotation):
    """The biggest change in any bone's direction *as the camera sees it*, in degrees.

    The body is turned into a new frame, so its bones point somewhere new in world
    terms. What must not change is how they look from the camera. We therefore
    measure both in camera-facing terms before comparing them.
    """
    before, _ = rig.forward_kinematics(pose_before)
    after, _ = rig.forward_kinematics(pose_after)

    biggest = 0.0
    for joint, parent in enumerate(rig.parents):
        if parent < 0 or rig.joint_names[parent] == "body_world":
            continue                              # the root's "bone" is the placement itself
        bone_before = camera_rotation @ (before[joint] - before[parent])
        bone_after = CAMERA_TO_VIEWER.T @ (after[joint] - after[parent])
        if np.linalg.norm(bone_before) < SHORTEST_REAL_BONE:
            continue
        cosine = np.dot(bone_before, bone_after) / (np.linalg.norm(bone_before)
                                                    * np.linalg.norm(bone_after))
        biggest = max(biggest, float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return biggest
