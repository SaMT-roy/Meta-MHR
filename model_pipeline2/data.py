"""ASPset-510 -> (person crop, MHR parameters) training pairs.

The labels are made by the analytic solver in `mhr_posing/analytic_ik.py`: it
turns ASPset's 17 ground-truth 3D keypoints into MHR parameters. So the teacher
is geometry, not a human annotator, and every label is reproducible from the
dataset.

Everything lives in the *viewer frame*: the camera sits at the origin, +x is its
right, +y is up, and it looks along -z. A body's z is therefore negative, and -z
is how far away it really is. That is the frame the network predicts in, and it
is what makes "put the 3D body where the person is in the picture" mean
something.

Three steps, in order:

    clips()        find the videos, keypoints and cameras on disk
    build_cache()  walk each video once, solve every frame, save small crops
    PoseData       serve (crop, labels) pairs to the training loop

Why the cache exists
--------------------
The videos are 3840x2160. Reading one frame *in a random order* takes 0.31 s,
because the decoder has to seek; reading them *in order* takes 0.006 s. Training
needs random order, so a dataset that read the videos directly would spend all
its time decoding. So we walk each video once, in order, and keep a small JPEG
crop of the person. That is the only thing this code writes to disk: two files
per clip in `mhr_pose_net/cache/`, about 400 MB in total, built in around five
minutes.

The crop is saved with more room round the person than the network sees, so that
random zoom and shift at training time can still be cut out of it.
"""

import json
import pathlib
import sys
from collections import namedtuple

import c3d
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))                    # for mhr_kit
sys.path.insert(0, str(ROOT / "mhr_posing"))     # for analytic_ik

import analytic_ik as ik   # noqa: E402

CACHE = HERE / "cache"


# ---------------------------------------------------------------------------
# Part 1: what a label is
# ---------------------------------------------------------------------------
#
# The solver writes 46 of MHR's 204 parameters. We split them into three groups,
# because the network has to treat each group differently.
#
#   31 joint angles   plain numbers, predicted directly
#    9 build scales   plain numbers -- how long this person's bones are
#    6 root pose      where the body is and which way it faces; see below
#
# The two lists are read out of the solver's own tables, so they cannot drift
# apart from it.

ANGLE_PARAMS = list(ik.SPINE_PARAMETERS)
for _joint, _tip, _aim, _params in ik.SIMPLE_BONES:
    ANGLE_PARAMS += list(_params)
for _upper, _hinge, _tip, _keys, _upper_params, _hinge_param in ik.LIMBS:
    ANGLE_PARAMS += list(_upper_params) + [_hinge_param]

SCALE_PARAMS = [p for p, *_ in ik.BONE_LENGTHS] + ["scale_hip_height", "scale_hip_depth"]

KEYPOINTS = list(ik.TARGET_TO_RIG)        # the 17 names, `pelvis` first
PELVIS = KEYPOINTS.index("pelvis")        # `pelvis` is the rig's root joint

# The root is stored as a 3x3 rotation matrix, not as the three euler angles MHR
# takes. Measured reason: over one clip `root_rz` jumps by a full 2*pi six times
# in 120 frames, because euler angles wrap round at +/-pi, and a network cannot
# learn a target that teleports. A rotation matrix never jumps. The other 31
# angles were measured too: they move by at most 0.21 radians between
# neighbouring frames, so they are safe exactly as they are.

CROP_PIXELS = 288        # size of the crop kept on disk
BOX_PADDING = 1.25       # room left round the person's keypoints
CACHE_PADDING = 1.5      # extra room in the saved crop, for zoom and shift to use


# ---------------------------------------------------------------------------
# Part 2: finding the data
# ---------------------------------------------------------------------------
#
# macOS unpacked the four ASPset archives into four folders with the same name,
# so each kind of file lives in a different one. This finds them by content.

Clip = namedtuple("Clip", "name subject view video joints camera")


def _modality_dir(aspset, modality):
    """The folder holding one kind of ASPset file, whichever archive copy it is in."""
    found = sorted(aspset.glob(f"ASPset-510*/*/{modality}"))
    if not found:
        raise FileNotFoundError(f"no '{modality}' folder under {aspset}")
    return found[0]


def clips(aspset=ROOT / "ASPset"):
    """Every clip on disk, as (video, keypoints, camera) file paths."""
    videos = _modality_dir(aspset, "videos")
    joints = _modality_dir(aspset, "joints_3d")
    cameras = _modality_dir(aspset, "cameras")

    found = []
    for video in sorted(videos.rglob("*.mkv")):
        subject, number, view = video.stem.split("-")
        found.append(Clip(name=video.stem, subject=subject, view=view, video=video,
                          joints=joints / subject / f"{subject}-{number}.c3d",
                          camera=cameras / subject / f"{subject}-{view}.json"))
    return found


def read_joints(path):
    """One clip's ground-truth keypoints, (frames, 17, 3) in mm with +y down."""
    with open(path, "rb") as handle:
        return np.array([points for _, points, _ in c3d.Reader(handle).read_frames()])[:, :, :3]


def read_camera(path):
    """A camera as (intrinsics, rotation, translation).

    `intrinsics` is (fx, fy, cx, cy) in pixels. The rotation and translation move
    a point from ASPset's world into the viewer frame, in centimetres. ASPset
    works in mm with +y down, so the flip `analytic_ik` applies to the keypoints
    has to be applied to the camera as well, or the body comes out mirrored.
    """
    data = json.loads(pathlib.Path(path).read_text())
    K = np.array(data["intrinsic_matrix"]).reshape(3, 4)
    E = np.array(data["extrinsic_matrix"]).reshape(4, 4)

    flip = np.diag([1.0, -1.0, -1.0])
    rotation = flip @ E[:3, :3] @ flip        # world (cm, y-up) -> viewer
    translation = flip @ E[:3, 3] / 10.0      # mm -> cm
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]), rotation, translation


def project(points, intrinsics):
    """Viewer-frame points (..., 3) in cm -> pixels (..., 2).

    The camera looks along -z, so a point in front of it has negative z and its
    distance away is -z. The camera's own axes have +y down, hence the minus.
    """
    fx, fy, cx, cy = intrinsics
    away = np.clip(-points[..., 2], 1e-3, None)
    return np.stack([fx * points[..., 0] / away + cx,
                     fy * -points[..., 1] / away + cy], axis=-1)


# ---------------------------------------------------------------------------
# Part 3: solving one frame
# ---------------------------------------------------------------------------

def viewer_targets(joints_mm, rotation, translation):
    """One frame of ASPset keypoints, moved into the viewer frame."""
    world = ik.targets_from_aspset(joints_mm)
    return {name: rotation @ point + translation for name, point in world.items()}


def _swap_side(name):
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def mirrored(targets):
    """The same person, seen in a mirror standing on the camera's optical axis.

    Flipping the picture left-to-right flips the world's x, and turns the
    person's left side into their right. Doing both keeps a real human: flipping
    x alone would give them two left hands. Solving *these* keypoints gives
    labels for the flipped picture that are exact rather than guessed -- the
    solver was measured to be perfectly mirror-symmetric, to 0.000 cm.
    """
    return {_swap_side(name): point * np.array([-1.0, 1.0, 1.0])
            for name, point in targets.items()}


def solve_frame(rig, targets):
    """Solve one frame and pull out everything the network is trained against."""
    pose = ik.solve(rig, targets)
    positions, rotations = rig.forward_kinematics(pose)

    joints = np.array([positions[rig.joint_index[ik.TARGET_TO_RIG[n]]] for n in KEYPOINTS])
    return dict(
        angles=np.array([pose[rig.parameter_index[n]] for n in ANGLE_PARAMS], np.float32),
        scales=np.array([pose[rig.parameter_index[n]] for n in SCALE_PARAMS], np.float32),
        root_rot=rotations[rig.joint_index["root"]].astype(np.float32),
        joints=joints.astype(np.float32),                              # where the rig put them
        truth=np.array([targets[n] for n in KEYPOINTS], np.float32),   # where they really are
    )


# ---------------------------------------------------------------------------
# Part 4: building the cache
# ---------------------------------------------------------------------------

def person_box(joints_2d):
    """A square box round the projected keypoints: (centre x, centre y, side)."""
    low, high = joints_2d.min(axis=0), joints_2d.max(axis=0)
    centre = 0.5 * (low + high)
    side = float((high - low).max()) * BOX_PADDING
    return np.array([centre[0], centre[1], side], np.float32)


def cut_out(image, centre_x, centre_y, side, pixels):
    """A square window of the image, resized. Parts off the edge are repeated."""
    half = side / 2.0
    source = np.array([[centre_x - half, centre_y - half],
                       [centre_x + half, centre_y - half],
                       [centre_x - half, centre_y + half]], np.float32)
    target = np.array([[0, 0], [pixels, 0], [0, pixels]], np.float32)
    transform = cv2.getAffineTransform(source, target)
    return cv2.warpAffine(image, transform, (pixels, pixels), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def build_cache(rig, stride=2, clip_list=None, cache=CACHE, verbose=True):
    """Solve every clip once and save its crops and labels. Safe to re-run.

    `stride` keeps every Nth frame. The videos run at 50 fps, so neighbouring
    frames are nearly the same picture and 2 halves the work for almost nothing.

    Each clip becomes two files: `<clip>.crops`, all its JPEGs glued end to end,
    and `<clip>.npz`, the labels plus the offset of each JPEG in that file. The
    dataset then memory-maps the big one and reads a few kilobytes per sample.
    """
    cache.mkdir(parents=True, exist_ok=True)
    todo = list(clip_list or clips())
    for number, clip in enumerate(todo, 1):
        out = cache / f"{clip.name}.npz"
        if out.exists():
            continue        # the .npz is written last, so its presence means both files are whole

        intrinsics, rotation, translation = read_camera(clip.camera)
        all_joints = read_joints(clip.joints)

        jpegs, plains, flips, boxes = [], [], [], []
        video = cv2.VideoCapture(str(clip.video))
        for index in range(len(all_joints)):
            if index % stride:
                video.grab()                    # skip it without decoding the pixels
                continue
            ok, frame = video.read()
            if not ok:
                break

            targets = viewer_targets(all_joints[index], rotation, translation)
            plain = solve_frame(rig, targets)
            flipped = solve_frame(rig, mirrored(targets))

            box = person_box(project(plain["truth"], intrinsics))
            crop = cut_out(frame, box[0], box[1], box[2] * CACHE_PADDING, CROP_PIXELS)
            jpegs.append(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
            boxes.append(box)
            plains.append(plain)
            flips.append(flipped)
        video.release()

        if not jpegs:
            continue
        (cache / f"{clip.name}.crops").write_bytes(b"".join(jpegs))
        labels = {}
        for key in ("angles", "scales", "root_rot", "joints", "truth"):
            labels[key] = np.stack([p[key] for p in plains])
            labels[key + "_flip"] = np.stack([f[key] for f in flips])
        np.savez(out, offset=np.cumsum([0] + [len(j) for j in jpegs]).astype(np.int64),
                 box=np.stack(boxes), intrinsics=intrinsics.astype(np.float32), **labels)
        if verbose:
            print(f"[{number:3d}/{len(todo)}] {clip.name:22s} {len(jpegs):4d} frames")

    if verbose:
        megabytes = sum(f.stat().st_size for f in cache.iterdir()) / 1e6
        print(f"cache ready: {len(list(cache.glob('*.npz')))} clips, {megabytes:.0f} MB in {cache}")


# ---------------------------------------------------------------------------
# Part 5: the dataset
# ---------------------------------------------------------------------------
#
# Held-out clips, not held-out frames: at 50 fps two neighbouring frames are
# almost the same picture, so splitting by frame would let the network study the
# answers to its own test.

def split_clips(cache=CACHE, every=6):
    """(training clips, validation clips). Every 6th clip is kept back."""
    names = sorted(p.stem for p in cache.glob("*.npz"))
    if not names:
        raise FileNotFoundError(f"{cache} is empty -- run build_cache() first")
    validation = names[::every]
    return [n for n in names if n not in set(validation)], validation


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class PoseData(Dataset):
    """(person crop, MHR parameters) pairs, read one at a time from the cache.

    With `augment=True` each sample gets a random zoom, shift, left-right flip
    and brightness change. The flip swaps in the mirrored labels that
    `build_cache` already solved, so it is exact rather than approximated by a
    table of which parameter to negate.
    """

    def __init__(self, clip_names, cache=CACHE, pixels=256, augment=False):
        self.cache = pathlib.Path(cache)
        self.pixels = pixels
        self.augment = augment
        self.labels = {name: dict(np.load(self.cache / f"{name}.npz")) for name in clip_names}
        self.crops = {}          # memory maps, opened on first use in each worker
        self.index = [(name, i) for name in clip_names
                      for i in range(len(self.labels[name]["box"]))]

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        # Memory maps must not travel to a DataLoader worker: pickling one would
        # copy the whole file. Each worker opens its own on first use.
        return {**self.__dict__, "crops": {}}

    def _jpeg(self, name, frame):
        if name not in self.crops:
            self.crops[name] = np.memmap(self.cache / f"{name}.crops", dtype=np.uint8, mode="r")
        offset = self.labels[name]["offset"]
        return self.crops[name][offset[frame]:offset[frame + 1]]

    def __getitem__(self, i):
        name, frame = self.index[i]
        label = self.labels[name]
        intrinsics = label["intrinsics"]
        original = label["box"][frame]

        flip = self.augment and np.random.rand() < 0.5
        side = "_flip" if flip else ""

        image = cv2.imdecode(self._jpeg(name, frame), cv2.IMREAD_COLOR)[:, :, ::-1]
        centre_x, centre_y, box_side = original
        if flip:
            image = image[:, ::-1]
            centre_x = 2 * intrinsics[2] - centre_x      # mirror about the principal point

        # --- random zoom and shift, cut out of the roomier cached crop --------
        # `scale` converts image pixels to cached-crop pixels. The cached crop is
        # centred on the person, so a shift of `dx` image pixels is a shift of
        # `dx * scale` pixels away from the middle of the cached crop.
        zoom, shift_x, shift_y = 1.0, 0.0, 0.0
        if self.augment:
            zoom = float(np.random.uniform(0.9, 1.15))
            shift_x, shift_y = np.random.uniform(-0.06, 0.06, 2) * box_side

        scale = CROP_PIXELS / (box_side * CACHE_PADDING)
        image = cut_out(np.ascontiguousarray(image),
                        CROP_PIXELS / 2 + shift_x * scale,
                        CROP_PIXELS / 2 + shift_y * scale,
                        box_side * zoom * scale, self.pixels)
        box = np.array([centre_x + shift_x, centre_y + shift_y, box_side * zoom], np.float32)

        image = image.astype(np.float32) / 255.0
        if self.augment:
            image = np.clip(image * np.random.uniform(0.75, 1.3), 0.0, 1.0)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        return dict(
            image=torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            angles=torch.from_numpy(label["angles" + side][frame]),
            scales=torch.from_numpy(label["scales" + side][frame]),
            root_rot=torch.from_numpy(label["root_rot" + side][frame]),
            joints=torch.from_numpy(label["joints" + side][frame]),
            truth=torch.from_numpy(label["truth" + side][frame]),
            box=torch.from_numpy(box),
            intrinsics=torch.from_numpy(intrinsics),
        )


# ---------------------------------------------------------------------------
# Part 6: one picture at a time, with no cache
# ---------------------------------------------------------------------------
#
# What inference looks like: a photograph, a box round a person, and the
# camera's focal length. Nothing here touches the cache or the labels.

def read_frame(video, index):
    """One frame of a video, as RGB. Takes about 0.3 s: the decoder has to seek."""
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise IndexError(f"{video} has no frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_batch(photo, box, intrinsics, pixels=256):
    """One photograph and one person box -> a batch of one, ready for the network."""
    crop = cut_out(np.asarray(photo), box[0], box[1], box[2], pixels)
    crop = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return dict(
        image=torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))[None],
        box=torch.tensor(np.asarray(box, np.float32))[None],
        intrinsics=torch.tensor(np.asarray(intrinsics, np.float32))[None],
    )
