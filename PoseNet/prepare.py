"""
Part 1a — run once: videos + keypoints  ->  a folder of person-crops and their answers.

    python prepare.py --limit 3     # a few clips, to eyeball the crops
    python prepare.py               # all 60, ~5 min, resumable

For each kept frame, four things happen:

    1. read the frame                      a 3840x2160 picture
    2. solve that frame's 17 keypoints     -> 46 MHR numbers (analytic_ik)
    3. project the keypoints to pixels     -> where the person is -> cut a square
    4. write the square + the numbers

What it writes
--------------
    data/<clip>/crops/00000.jpg     288x288 crop of the person, ~23 KB
    data/<clip>/labels.npz          that clip's answers, written LAST

One folder per clip ("a shard"). Nothing in it refers to any other clip, which
buys three things: a crashed run resumes (a clip with a labels.npz is skipped),
memory stays bounded (one clip at a time, at any dataset size), and clips could
be built in parallel later.

The four frames a point can live in
-----------------------------------
    1. ASPSET WORLD   3D, mm, +y down.  Origin is a fixed spot in the field,
                      shared by every camera. This is what the .c3d files hold.
    2. VIEWER FRAME   3D, cm, +y up.  Origin is THE CAMERA, looking along -z,
                      so a body in front of it has negative z and its distance
                      away is -z.
    3. PHOTOGRAPH     2D, pixels in the 3840x2160 frame, +y down.
    4. CROP           2D, pixels in the 288x288 square around the person.

VIEWER FRAME is the hub — both the keypoints and the posed MHR body live there, so
they can be compared. It has to be the CAMERA and not the field, because at
inference there is no field: a photograph comes with no mocap rig and no world
origin. ASPset ships frame 1 plus the calibration, and we derive frame 2.

Why this runs once instead of during training
---------------------------------------------
Measured here: reading one frame at a RANDOM position in a 4K clip costs 301 ms
(the decoder must seek to a keyframe and roll forward); reading IN ORDER costs
5.5 ms. Solving the IK costs 14.6 ms. Training needs random order, so doing this
per step would cost ~29 min an epoch with the GPU idle, against ~1 min reading
prepared crops — and it would buy nothing, since a frame's keypoints never
change and the solver returns the identical numbers every epoch.

Only `analytic_ik` is imported. Everything else here is written out in full.
"""

import argparse
import json
import pathlib
import sys

import c3d
import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))                       # mhr_kit, which analytic_ik needs
sys.path.insert(0, str(ROOT / "mhr_posing"))

import analytic_ik as ik

DATA = HERE / "data"
ASPSET = ROOT / "ASPset train"

CROP_PIXELS = 288      # size of the JPEG kept on disk
CROP_PADDING = 1.5     # it covers 1.5x the person's box, leaving room for random zoom/shift
BOX_PADDING = 1.25     # the person's box itself is 25% wider than their keypoints


# ---------------------------------------------------------------------------
# Which rig controls make up a label
# ---------------------------------------------------------------------------
#
# All four below are LISTS OF NAMES, not numbers. They exist so that "the 31
# angles" can be pulled out of MHR's 204-long parameter vector by name.
#
#                 length   contents
#   ANGLE_PARAMS    31     rig controls for joint angles
#   SCALE_PARAMS     9     rig controls for bone lengths
#   KEYPOINTS       17     ASPset keypoint names
#   PELVIS       int = 0   index of "pelvis" in that list
#
#   31 + 9 = 40 fixed numbers, plus 6 rotation + 3 camera assembled in
#   dataset.py = the 49 the network predicts.
#
# They are READ OUT of the solver's own tables rather than typed here. If
# someone adds a bone to the solver these follow automatically; typed out, they
# would silently drift, and a label wrong in one slot raises no error anywhere.

# 3 names: spine0_rx_flexible, spine0_ry_flexible, spine0_rz_flexible
ANGLE_PARAMS = list(ik.SPINE_PARAMETERS)

# neck, head, clavicles — 4 bones x 3 angles = 12
#   l_clavicle_rx/ry/rz, r_clavicle_rx/ry/rz, neck_twist/lean/bend, head_twist/lean/bend
for _joint, _tip, _aim, _params in ik.SIMPLE_BONES:
    ANGLE_PARAMS += list(_params)

# arms and legs — 4 limbs x 4 = 16.  Four, not six: an elbow or knee is a HINGE,
# so MHR exposes exactly one angle for it.
#   l_uparm_twist/ry/rz + l_elbow_bend        r_uparm_twist/ry/rz + r_elbow_bend
#   l_upleg_twist/ry/rz + l_knee_bend         r_upleg_twist/ry/rz + r_knee_bend
for _up, _hinge, _tip, _keys, _up_params, _hinge_param in ik.LIMBS:
    ANGLE_PARAMS += list(_up_params) + [_hinge_param]

# 7 from the solver's bone table, plus 2 it sets elsewhere (in match_hip_offset,
# not by measuring a bone), so those two are named by hand:
#   scale_uplegs, scale_lowlegs, scale_uparms, scale_lowarms,
#   scale_spine_length, scale_neck_length, scale_hip_width | hip_height, hip_depth
SCALE_PARAMS = [p for p, *_ in ik.BONE_LENGTHS] + ["scale_hip_height", "scale_hip_depth"]

# list() of a dict gives its KEYS, so this is the 17 ASPset names in the order
# TARGET_TO_RIG declares them:
#   pelvis, spine, neck, head, head_top, left/right_shoulder, elbow, wrist,
#   left/right_hip, knee, ankle
KEYPOINTS = list(ik.TARGET_TO_RIG)
PELVIS = KEYPOINTS.index("pelvis")       # 0 — the joint everything is measured relative to


# ---------------------------------------------------------------------------
# Finding and reading the dataset
# ---------------------------------------------------------------------------

def find_clips(aspset=ASPSET):
    """
    Every clip on disk, as a list of dicts:

        {"name":   "1e28-0028-right",     the clip id, used for the shard folder
         "video":  .../1e28-0028-right.mkv,
         "joints": .../1e28-0028.c3d,     keypoints are per (subject, take)...
         "camera": .../1e28-right.json}   ...cameras are per (subject, view)

    In prepare():

        for clip in find_clips()[:limit]:
            prepare_clip(rig, clip, stride)

    ASPset ships as four archives that unpacked into four SAME-NAMED folders,
    each holding one kind of file. So each kind is searched for separately
    rather than assumed to live beside the others.
    """
    def folder(kind):
        found = sorted(aspset.glob(f"ASPset-510*/*/{kind}"))    # the * matches all four copies
        if not found:
            raise FileNotFoundError(f"no '{kind}' folder under {aspset}")
        return found[0]

    videos, joints, cameras = folder("videos"), folder("joints_3d"), folder("cameras")

    clips = []
    for video in sorted(videos.rglob("*.mkv")):            # sorted: a stable, reproducible order
        subject, take, view = video.stem.split("-")        # "1e28-0028-right" -> the three parts
        clips.append(dict(name=video.stem, video=video,
                          joints=joints / subject / f"{subject}-{take}.c3d",
                          camera=cameras / subject / f"{subject}-{view}.json"))
    return clips


def read_keypoints(path):
    """
    One clip's ground-truth 3D keypoints, straight off disk.

        returns (frames, 17, 3) — millimetres, +y DOWN, ASPset's world frame

    In prepare_clip():

        all_keypoints = read_keypoints(clip["joints"])
        ... solve_frame(rig, all_keypoints[index], ...)     # one (17, 3) row per frame

    A .c3d is the standard motion-capture format. Each of its rows carries five
    numbers per marker — x, y, z, then a residual and a camera mask describing
    how confident the capture was. We keep the first three and drop the rest.

    The whole clip is read at once: a few hundred KB, so there is no reason to
    be clever about it.
    """
    with open(path, "rb") as handle:
        frames = c3d.Reader(handle).read_frames()          # yields (index, points, analog)
        return np.array([points for _, points, _ in frames])[:, :, :3]   # drop residual + mask


def read_camera(path):
    """
    Read one camera file exactly as it is written — no conversion.

        returns intrinsics  (4,)    fx, fy, cx, cy, in pixels
                rotation    (3,3)  ─┐ ASPset's world -> the camera,
                translation (3,)   ─┘ still in mm and +y down

    In prepare_clip():

        intrinsics, rotation, translation = read_camera(clip["camera"])
        rotation, translation = in_solver_axes(rotation, translation)   # the conversion

    A calibrated camera is two separate things, and ASPset stores both:

        K, the INTRINSICS (3x4)   camera's 3D frame -> pixels    "what lens it has"
        E, the EXTRINSICS (4x4)   ASPset's world    -> camera    "where it stands"

    From K only four numbers matter: fx, fy (focal length in pixels) and cx, cy
    (the principal point — the pixel the camera's axis passes through, near the
    middle of the picture). The rest of K describes skew and a projective row we
    do not need. From E we take its rotation block and its translation column.
    """
    data = json.loads(pathlib.Path(path).read_text())
    K = np.array(data["intrinsic_matrix"]).reshape(3, 4)
    E = np.array(data["extrinsic_matrix"]).reshape(4, 4)

    intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])   # fx, fy off the diagonal; cx, cy
    return intrinsics, E[:3, :3], E[:3, 3]                        # rotation block, translation column


def in_solver_axes(rotation, translation):
    """
    Rewrite the camera's world->camera recipe in the units and axes MHR uses.

        in   rotation (3,3), translation (3,)   mm, +y down   (from read_camera)
        out  rotation (3,3), translation (3,)   cm, +y up

    Where the rule comes from:

        MHR (Meta's body model) works in cm, +y up
            v  so analytic_ik must speak that, since it drives MHR
            v  so our camera must too, since we hand it to analytic_ik

    In solve_frame():

        world   = ik.targets_from_aspset(keypoints_mm)   # already cm, +y up
        targets = {n: rotation @ p + translation for n, p in world.items()}

    That second line is the reason. The POINTS have already changed units — the
    solver did it, one line earlier. So the camera must meet them there, or you
    are rotating a cm/+y-up point with a mm/+y-down matrix.

    WHY THE ROTATION IS SANDWICHED AND THE TRANSLATION IS NOT

    The camera file gives a recipe written wholly in ASPset's vocabulary:
    "hand me a point in ASPSET words, I return the answer in ASPSET words." We
    want one that speaks ours. We cannot rewrite it, so we wrap it:

        1. our point      -> ASPset words
        2. run their recipe
        3. their answer   -> our words

    Translate in, do the thing, translate back. Read right to left, the way the
    point actually travels:

        flip @ rotation @ flip
        --+--   ---+----  --+-
          |        |        `- 3rd: our point -> ASPset words
          |        `---------- 2nd: ASPset's own rotation
          `------------------- 1st: the answer -> back to our words

    The translation is not a recipe. It is just a point — the camera's own
    offset — so nothing is ever handed IN to it. Only the way out needs
    converting. Hence one flip, not two.

    AND WHY THIS IS SAFE: we flip TWO axes, which is a 180-degree turn about x —
    a real rotation, something a body can physically be put through. Flipping
    ONE axis would be a mirror, and would give the person two left hands.

    IF YOU SKIP THIS: the keypoints get converted and the camera does not, the
    two disagree, and every body comes out mirrored. From the front that looks
    perfectly normal, which is what makes it dangerous.
    """
    # Flip y and z. The same matrix serves both directions: flipping twice undoes it.
    flip = np.diag([1.0, -1.0, -1.0])

    rotation = flip @ rotation @ flip          # translate in, rotate, translate back
    translation = flip @ translation / 10.0    # one flip (output only), then mm -> cm

    return rotation, translation


# ---------------------------------------------------------------------------
# Geometry: 3D -> the photograph -> a crop
# ---------------------------------------------------------------------------

def project(points, intrinsics):
    """
    Viewer-frame points (..., 3) in cm  ->  photograph pixels (..., 2).

        pixel_x = fx * ( x / -z) + cx
        pixel_y = fy * (-y / -z) + cy

    THE INTUITION: things look smaller further away. Two similar triangles — one
    from the camera out to the point, one from the camera to a screen a unit in
    front of it — give "across the screen = sideways / away". That DIVISION is
    the whole of perspective. Double the distance, half the size.

    The three fiddly parts:

        -z      the camera looks along negative z, so distance away is -z.
                A point at z = -1200 is 12 metres in front.
        -y      the world says +y is up; pixel rows count DOWNWARDS. Head above
                hips must mean a smaller row number, hence the minus.
        fx cx   the division alone gives a fraction of a screen. fx, fy scale
                that to pixels; cx, cy move the origin from the middle of the
                picture to the top-left corner, where pixel (0,0) lives.

    Used in exactly two places, both about finding the person in 2D:

        prepare_clip()   person_box(project(truth, intrinsics))   where to cut
        dataset.py       hips_in_crop()                           the (u, v) label

    It is never part of the network.
    """
    fx, fy, cx, cy = intrinsics

    # How far in front of the camera each point is. Clipped so nothing sits at
    # the lens and makes the division explode.
    away = np.clip(-points[..., 2], 1e-3, None)

    return np.stack([fx * points[..., 0] / away + cx,      #  x / away, scaled and shifted
                     fy * -points[..., 1] / away + cy],    # -y / away: world up = image down
                    axis=-1)


def person_box(points_2d):
    """
    A square box round some 2D points.

        in   points_2d (17, 2)   the person's keypoints, photograph pixels
        out  (3,)                centre_x, centre_y, side — also pixels

    In prepare_clip():

        box = person_box(project(label["truth"], intrinsics))
        picture = crop(frame, box[0], box[1], box[2] * CROP_PADDING, CROP_PIXELS)

    SQUARE, not rectangular, and that is deliberate: a square survives a
    left-right flip and a random zoom with no extra bookkeeping, and it means a
    "box" is three numbers rather than four everywhere else in the codebase.

    Take the longer of the two sides so nothing is cut off, then pad it by
    BOX_PADDING (25%) so fingers and feet are not sitting on the border.
    """
    low, high = points_2d.min(axis=0), points_2d.max(axis=0)   # the bounding rectangle
    centre = 0.5 * (low + high)                                # its middle
    side = float((high - low).max()) * BOX_PADDING             # the LONGER side, padded
    return np.array([centre[0], centre[1], side], np.float32)


def crop(image, centre_x, centre_y, side, pixels):
    """
    Cut a square window out of an image and resize it to pixels x pixels.

        in   image      (H, W, 3)
             centre_x, centre_y, side   the window, in the image's own pixels
             pixels                     the output size
        out  (pixels, pixels, 3)

    Called twice in the pipeline, at different scales:

        prepare_clip()   out of the 4K frame  -> 288x288 saved to disk
        dataset.py       out of that 288      -> 256x256 fed to the network

    WHY A WARP RATHER THAN SLICE-THEN-RESIZE: the window is allowed to hang off
    the edge of the picture — someone standing at the border — and a slice there
    would come back the wrong size or raise. `warpAffine` with BORDER_REPLICATE
    handles it in one step by repeating the edge pixels.

    An affine map is pinned down exactly by where three corners land; the fourth
    then follows. So we say where the window's top-left, top-right and
    bottom-left should end up, and let OpenCV work out the matrix.
    """
    half = side / 2.0
    source = np.float32([[centre_x - half, centre_y - half],    # window top-left
                         [centre_x + half, centre_y - half],    # top-right
                         [centre_x - half, centre_y + half]])   # bottom-left
    target = np.float32([[0, 0], [pixels, 0], [0, pixels]])     # where those three must land
    return cv2.warpAffine(image, cv2.getAffineTransform(source, target), (pixels, pixels),
                          flags=cv2.INTER_LINEAR,               # smooth resampling
                          borderMode=cv2.BORDER_REPLICATE)      # off the edge -> repeat the edge


# ---------------------------------------------------------------------------
# Solving one frame
# ---------------------------------------------------------------------------

def solve_frame(rig, keypoints_mm, rotation, translation):
    """
    One frame of ASPset keypoints -> its answer.

        angles   (31,)  joint angles, radians
        scales   (9,)   how long this person's bones are
        root_rot (3,3)  which way the body faces
        joints   (17,3) where the posed rig put the keypoints, viewer frame
        truth    (17,3) where ASPset says they are — kept for scoring only

    In dataset.py :

    target = np.concatenate([
    shard["angles"][row],                            # 31
    shard["scales"][row],                            # 9
    rotation[:, 0], rotation[:, 1],                  # 6   <- from root_rot
    hips_in_crop(joints[PELVIS], box, intrinsics),   # 3   <- from joints
    ])

    NOT here: the root's POSITION. It is implicit in `joints`, and dataset.py
    turns it into (u, v, size) at sample time — because those three depend on
    the random crop, which has not been cut yet.
    """

    return solve_targets(rig, viewer_targets(keypoints_mm, rotation, translation))


def viewer_targets(keypoints_mm, rotation, translation):
    """One frame of ASPset keypoints -> {name: 3D point} in the viewer frame."""
    # keypoints_mm : ASPset world   mm, +y down,   a spot in the field
    # world :        ASPset world   cm, +y up,     a spot in the field
    #                ^ same place, only relabelled
    world = ik.targets_from_aspset(keypoints_mm)

    # targets :      VIEWER         cm, +y up,     the camera
    #                ^ this line is the actual change of frame.
    # A dict, not an array, because ik.solve looks parts up by name: it wants
    # "left_elbow", not "row 6", and errors if any of the 17 is missing.
    return {name: rotation @ point + translation
            for name, point in world.items()}


def swap_side(name):
    """'left_elbow' <-> 'right_elbow'; anything without a side unchanged."""
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


def mirrored(targets):
    """The same person seen in a mirror standing on the camera's optical axis.

    Two things must happen together: negate x (the world flips left-to-right)
    AND swap the left_/right_ names (their left side is now on the picture's
    right). Doing only the first would give the person two LEFT hands — a shape
    no human has and the rig cannot reach.

    Solving THESE keypoints gives exact labels for the flipped picture. The
    usual alternative is a hand-written table of which parameters to negate,
    which is fiddly and approximate; the solver is exactly mirror-symmetric, so
    this is not.
    """
    return {swap_side(name): point * np.array([-1.0, 1.0, 1.0])
            for name, point in targets.items()}


def solve_targets(rig, targets):
    """Solve one frame's viewer-frame targets into the five label arrays."""
    # (204,) numpy array. MHR's full control panel: every knob on the rig.
    # The solver sets 46 of them and leaves 158 at zero.
    # This is the answer — the thing the network is being trained to produce.
    # From it we pull out 31 angles and 9 scales by name.
    pose = ik.solve(rig, targets)

    # positions, rotations — forward_kinematics takes the pose and runs the skeleton,
    # i.e. actually bends every joint and reports where everything ended up.
    #   positions (127, 3)     where each rig joint is, cm, viewer frame
    #   rotations (127, 3, 3)  which way each rig joint faces
    # Needed because `pose` holds ANGLES, and angles are not positions: the
    # (u,v,size) label and Part 2's losses both need real 3D locations.
    positions, rotations = rig.forward_kinematics(pose)

    # Each line below is "look my names up, grab those slots":
    #   rig.parameter_index[name]  -> which of the 204 knobs
    #   rig.joint_index[name]      -> which of the 127 joints
    #   ik.TARGET_TO_RIG[name]     -> ASPset's name -> MHR's ("pelvis" -> "root")
    return dict(
        angles=np.array([pose[rig.parameter_index[n]] for n in ANGLE_PARAMS], np.float32),
        scales=np.array([pose[rig.parameter_index[n]] for n in SCALE_PARAMS], np.float32),
        root_rot=rotations[rig.joint_index["root"]].astype(np.float32),
        # where the RIG ended up — self-consistent with the 46 numbers above
        joints=np.array([positions[rig.joint_index[ik.TARGET_TO_RIG[n]]] for n in KEYPOINTS],
                        np.float32),
        # where ASPSET says they are — a few cm off, because MHR's joint centres
        # sit inside the skin and ASPset's markers sit on it. The honest target
        # to score against, which is why Part 2's error floor is ~16 mm, not 0.
        truth=np.array([targets[n] for n in KEYPOINTS], np.float32),
    )


# ---------------------------------------------------------------------------
# Building one shard
# ---------------------------------------------------------------------------

def prepare_clip(rig, clip, stride, data=DATA):
    """
    One clip -> one shard on disk. Walk the video in order, solve every Nth
    frame, save a crop and a label for each.

    Writes  data/<clip>/crops/00000.jpg   288x288, one per kept frame
            data/<clip>/labels.npz        N rows, written LAST

    Inside labels.npz, where N is the number of frames kept:

        frame      (N,)      which video frame each row came from
        box        (N,3)     centre_x, centre_y, side — photograph pixels
        intrinsics (4,)      fx, fy, cx, cy — ONE camera for the whole clip
        angles     (N,31)    -+
        scales     (N,9)      | solve_frame's answer, stacked
        root_rot   (N,3,3)    |
        joints     (N,17,3)   |
        truth      (N,17,3)  -+

    In dataset.py :

        shard = np.load(data / name / "labels.npz")
        image = cv2.imread(data / name / "crops" / f"{shard['frame'][row]:05d}.jpg")

    Returns  None  the shard already existed, nothing done
             0     the video would not decode
             N     frames written
    """
    # A shard is finished only if labels.npz is there, because it is written
    # LAST — after every JPEG. A run killed halfway leaves JPEGs and no .npz,
    # so the clip is simply redone. That is the whole resume mechanism.
    out = data / clip["name"]
    if (out / "labels.npz").exists():
        return None

    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Read once per CLIP, not per frame — one camera filmed the whole thing.
    #   read_camera     as written: mm, +y down, ASPset's axes
    #   in_solver_axes  converted:  cm, +y up,   what the solver expects
    intrinsics, rotation, translation = read_camera(clip["camera"])
    rotation, translation = in_solver_axes(rotation, translation)

    # (frames, 17, 3) in mm — the whole clip's keypoints, small enough to hold.
    all_keypoints = read_keypoints(clip["joints"])

    rows = []
    video = cv2.VideoCapture(str(clip["video"]))

    # Walk the video IN ORDER. That is the point of doing this here rather than
    # during training: in order costs ~5.5 ms a frame, seeking to a random one
    # costs ~301 ms, because the decoder has to find a keyframe and roll
    # forward. Never seek; just read from start to end and take what we want.
    for index in range(len(all_keypoints)):
        if index % stride:
            video.grab()          # step past it WITHOUT decoding the pixels: ~0.2 ms
            continue

        ok, frame = video.read()  # (2160, 3840, 3) BGR
        if not ok:
            break                 # video ran out before the keypoints did; keep what we have

        # Solve twice: as seen, and mirrored. The mirrored answer is the exact
        # label for a left-right flipped picture, which dataset.py uses as
        # augmentation — the one thing that genuinely adds pose variety when the
        # dataset has only two subjects.
        targets = viewer_targets(all_keypoints[index], rotation, translation)
        label = solve_targets(rig, targets)
        label.update({key + "_flip": value
                      for key, value in solve_targets(rig, mirrored(targets)).items()})

        # Now find the person in the picture. The keypoints are 3D, so project
        # them to pixels and take the square they fall inside.
        box = person_box(project(label["truth"], intrinsics))

        # Cut CROP_PADDING (1.5x) WIDER than that box. The network only ever
        # sees the middle of it; the margin exists so the random zoom and shift
        # in dataset.py have real pixels to move into instead of invented ones.
        picture = crop(frame, box[0], box[1], box[2] * CROP_PADDING, CROP_PIXELS)
        cv2.imwrite(str(crops_dir / f"{index:05d}.jpg"), picture,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Named by the VIDEO frame number, not by position in `rows`, so the
        # JPEG and its label stay matched however the stride changes.
        rows.append(dict(frame=index, box=box, **label))

    video.release()

    if not rows:
        return 0          # no .npz written, so a later run will try this clip again

    # `rows` is a list of per-frame dicts. Turn it inside out: stack each field
    # across frames, so one array per field instead of one dict per frame.
    #
    #   [{angles: (31,)}, {angles: (31,)}, ...]   ->   angles: (N, 31)
    #
    # That is the shape a Dataset wants — "give me field X of row i" is then one
    # index, not a lookup through N dicts.
    #
    # `intrinsics` stays a single (4,) because the camera never moved during the
    # clip; storing it N times would be N-1 copies of the same four numbers.
    #
    # Written last, on purpose: its existence is what marks the clip as done.
    np.savez(out / "labels.npz",
             frame=np.array([r["frame"] for r in rows]),
             intrinsics=intrinsics.astype(np.float32),
             **{key: np.stack([r[key] for r in rows])
                for key in ("box", "angles", "scales", "root_rot", "joints", "truth",
                            "angles_flip", "scales_flip", "root_rot_flip",
                            "joints_flip", "truth_flip")})

    return len(rows)


def prepare(stride=2, limit=None, data=DATA):
    """
    Build a shard for every clip. Safe to re-run: finished clips are skipped.

        stride   keep every Nth frame. The videos run at 50 fps, so neighbouring
                 frames are nearly the same picture and 2 halves the work for
                 almost no loss of variety.
        limit    only the first N clips — for a quick look before committing.

    Roughly 5 minutes and 180 MB for all 60 clips.
    """
    # The rig is built ONCE and reused for every clip and every frame: loading
    # MHR is the slow part, solving with it is not. Default level of detail,
    # because only the SKELETON is needed here — the mesh is never touched.
    rig = ik.Rig()

    clips = find_clips()[:limit]

    total = 0
    for number, clip in enumerate(clips, 1):
        kept = prepare_clip(rig, clip, stride, data)
        if kept is None:                       # None means "shard already there"
            print(f"[{number:3d}/{len(clips)}] {clip['name']:22s} already done")
        else:
            total += kept
            print(f"[{number:3d}/{len(clips)}] {clip['name']:22s} {kept:4d} frames")

    # Size of everything on disk, not just this run's additions — so re-running
    # reports the true total rather than 0 MB.
    megabytes = sum(f.stat().st_size for f in data.rglob("*.jpg")) / 1e6
    print(f"\n{total} new frames, {megabytes:.0f} MB total in {data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build the crop + label shards")
    parser.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips")
    args = parser.parse_args()
    prepare(stride=args.stride, limit=args.limit)
