"""
Part 1b — every training step: one prepared frame -> the tensors the model wants.

    python dataset.py        # prints the shapes of one sample

    prepare.py   ran ONCE, wrote data/<clip>/crops/*.jpg + labels.npz
    dataset.py   runs MILLIONS of times, one sample per call        <- you are here

What one sample is
------------------
    image    (3, 256, 256)   the crop, randomly zoomed, shifted and brightened
    context  (3,)            where that crop sits in the photograph
    target   (49,)           the answer the network must learn

    joints   (17, 3)  -+
    box      (3,)      | not used in Part 1. The geometry losses need them in
    intrinsics (4,)   -+ Part 2. Ignore for now.

Why this file exists separately from prepare.py
-----------------------------------------------
The split is not arbitrary. It is one rule:

    FIXED per frame, expensive  ->  prepare.py, computed once
    RANDOM per sample, cheap    ->  here, computed every time

A frame's keypoints never change, so its 46 solver numbers never change —
computing them per epoch would burn 14.6 ms a sample to get the same answer
back. But the zoom, the shift and the brightness must differ every epoch, or the
network memorises exact pixels. Hence two files.

The 49 numbers, in order
------------------------
    [ 0:31]  joint angles    radians, straight from the shard
    [31:40]  bone scales     straight from the shard
    [40:46]  root rotation   the first TWO COLUMNS of the 3x3 rotation matrix
    [46:49]  (u, v, size)    where the hips sit in the crop, and how many metres
                             across the crop is

The first 40 are copied out of the shard untouched. The last 9 are built here.

Why the last 9 cannot be precomputed
------------------------------------
    [40:46]  could be, in principle — it is just a reshape of root_rot. It is
             done here only to keep the whole 49-vector assembled in one place.

    [46:49]  genuinely cannot be. (u, v, size) says where the body sits INSIDE
             THE CROP, and the random zoom and shift move the crop. Precompute
             it and the label would describe a crop the network is not looking
             at — it would be punished for not predicting a lie.

Two encodings worth understanding
---------------------------------
ROTATION AS SIX NUMBERS. Only two columns of the 3x3 are stored and nothing is
lost: the third is their cross product. Six rather than three euler angles
because euler angles wrap at +/-pi — measured, `root_rz` jumps a full 2*pi six
times in 120 consecutive frames while the person barely turns. No network can
fit a target that teleports while the picture stays still.

POSITION AS (u, v, size). Not (x, y, z), because a crop cannot see distance: a
child up close and an adult far away fill the same square. What a crop CAN see
is how big the person is — and distance follows from that plus how big they
look. See `hips_in_crop`.

How PyTorch uses this
---------------------
A Dataset needs only two methods: `__len__` (how many samples) and
`__getitem__(i)` (give me sample i). A DataLoader then calls `__getitem__` with
shuffled indices and stacks the results into batches:

    loader = DataLoader(PoseData(train_clips, augment=True), batch_size=24, shuffle=True)
    for batch in loader:
        batch["image"]    # (24, 3, 256, 256)
        batch["target"]   # (24, 49)
"""

import pathlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# From our own prepare.py — the same constants and geometry that built the data,
# so the two files cannot disagree about what a crop or a projection means.
from prepare import CROP_PADDING, CROP_PIXELS, DATA, PELVIS, crop, project

# The backbone is a ResNet pretrained on ImageNet. It learnt its filters on
# pictures scaled this way — each colour channel shifted by the dataset's mean
# and divided by its standard deviation — so pictures must arrive the same way
# or every feature it knows is slightly wrong.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# How hard the augmentation pushes. Deliberately small, for a reason specific to
# this task: the crop's geometry is PART OF THE LABEL ((u, v, size) is measured
# against the box). A violent crop jitter does not make the network robust, it
# makes the label harder to predict from the same pixels.
ZOOM_RANGE = (0.90, 1.15)       # multiply the box's side by this
SHIFT_FRACTION = 0.06           # slide the box by up to +/-6% of its side
BRIGHTNESS_RANGE = (0.75, 1.30) # multiply every pixel by this


def split_clips(data=DATA, every=10):
    """
    Decide which clips train and which are held back.

        returns (training clip names, validation clip names)
                e.g. (["1e28-0006-right", ...], ["1e28-0001-left", ...])

    In train.py:

        train_clips, val_clips = split_clips()
        train_data = PoseData(train_clips, augment=True)
        val_data   = PoseData(val_clips,   augment=False)

    WHOLE SUBJECTS, NOT FRAMES OR CLIPS. At 50 fps two neighbouring frames are
    almost the same picture, so a frame split would let the network study the
    answers to its own test. Clips are not enough either: ASPset films each take
    from three cameras at once, named "<subject>-<take>-left/mid/right". Hold
    back a clip and its two sibling views — the same person, the same instant,
    another angle — stay in training. Measured on the trainval split, every one
    of the 225 clips an every-6th clip split held back had both siblings in
    training. So the whole SUBJECT goes to one side.

    WHAT THIS MEASURES. With whole subjects held back this is the honest
    question: can it handle a person it has never seen. trainval has 15
    subjects, so every=6 keeps 3 of them back, about 20% of the frames.

    A shard counts only if its labels.npz exists — same completeness rule
    prepare.py uses, so a half-built clip is never trained on.
    """
    names = sorted(shard.name for shard in pathlib.Path(data).iterdir()
                   if (shard / "labels.npz").exists())
    if not names:
        raise FileNotFoundError(f"no shards in {data} — run prepare.py first")

    subject = lambda name: name.split("-")[0]                   # "04ac-0003-left" -> "04ac"
    held_back = set(sorted({subject(n) for n in names})[::every])   # every 6th SUBJECT
    #           ^ a set, so the two lookups below are one step rather than a scan
    return ([n for n in names if subject(n) not in held_back],
            [n for n in names if subject(n) in held_back])


class PoseData(Dataset):
    """
    Prepared frames of the given clips, served one at a time.

        clip_names   which clips this dataset covers (from split_clips)
        pixels       the size the network wants — 256
        augment      True for training; False for validation and scoring, so
                     the score is not noise from a different random zoom

    Only the named clips' shards are opened, so memory is bounded by the SPLIT,
    not by the size of the whole dataset. That matters at scale: 60 GB of video
    would be ~190 MB of labels, and there is no reason for the training set to
    hold the validation set's copy as well.
    """

    def __init__(self, clip_names, data=DATA, pixels=256, augment=False):
        self.data = pathlib.Path(data)
        self.pixels = pixels
        self.augment = augment

        # One entry per clip, each the contents of that clip's labels.npz:
        #   {"1e28-0006-right": {"frame": (N,), "box": (N,3), "angles": (N,31), ...}}
        # Labels are small — a few hundred KB a clip — so they live in memory.
        # The JPEGs do not; those are read from disk per sample.
        self.shards = {name: dict(np.load(self.data / name / "labels.npz"))
                       for name in clip_names}

        # Sample i has to mean something, but the data is split across clips. So
        # flatten: a list of (clip name, row within that clip's arrays).
        #   [("1e28-0006-right", 0), ("1e28-0006-right", 1), ..., ("1e28-0008-left", 0), ...]
        self.index = [(name, row) for name in clip_names
                      for row in range(len(self.shards[name]["frame"]))]

    def __len__(self):
        """How many samples exist. DataLoader draws its random indices from this."""
        return len(self.index)

    def __getitem__(self, i):
        """
        Sample i: read the prepared crop, disturb it randomly, build the 49 numbers.

        The order below matters. The box is updated AFTER the zoom and shift are
        chosen and BEFORE (u, v, size) is computed from it — because those three
        numbers describe the window we actually cut, not the one on disk.
        """
        name, row = self.index[i]
        shard = self.shards[name]

        intrinsics = shard["intrinsics"]                  # (4,) fx, fy, cx, cy — one per clip
        centre_x, centre_y, box_side = shard["box"][row]  # the person's box, photograph pixels

        # The JPEG is named by VIDEO frame number, which is why the shard stores it.
        # [:, :, ::-1] is BGR -> RGB: OpenCV reads in BGR, everything else wants RGB.
        frame = shard["frame"][row]
        image = cv2.imread(str(self.data / name / "crops" / f"{frame:05d}.jpg"))[:, :, ::-1]

        # --- 0. mirror half the training samples ------------------------------
        # prepare.py already SOLVED the mirrored keypoints, so the flipped label
        # is exact rather than a guess at which parameters to negate. `side`
        # selects that label set below.
        flip = self.augment and np.random.rand() < 0.5
        side = "_flip" if flip else ""
        if flip:
            image = image[:, ::-1]
            # Mirroring the picture reflects every pixel about the principal
            # point, so a column at x lands at 2*cx - x.
            centre_x = 2 * intrinsics[2] - centre_x

        # --- 1. choose the random disturbance --------------------------------
        # Identity values when not augmenting, so the code below has one path.
        zoom, shift_x, shift_y = 1.0, 0.0, 0.0
        if self.augment:
            zoom = float(np.random.uniform(*ZOOM_RANGE))
            shift_x, shift_y = np.random.uniform(-SHIFT_FRACTION, SHIFT_FRACTION, 2) * box_side
            #                  ^ a FRACTION OF THE BOX, not a fixed pixel count, so a
            #                    near person and a far one are disturbed by the same
            #                    amount relative to their own size.

        # --- 2. cut the network's view out of the prepared crop --------------
        # Two nested crops, and this is the fiddliest arithmetic in the file.
        #
        # prepare.py saved a 288-pixel JPEG covering box_side * 1.5 photograph
        # pixels. We now want a window of box_side * zoom photograph pixels out
        # of it. So we need a conversion rate between the two rulers:
        #
        #     scale = 288 prepared pixels / (box_side * 1.5) photograph pixels
        #
        # Worked example, box_side = 600 photograph pixels, zoom = 1.1:
        #
        #     the JPEG covers   600 * 1.5 = 900 photograph pixels, stored as 288
        #     scale           = 288 / 900 = 0.32 prepared pixels per photo pixel
        #     window wanted   = 600 * 1.1 = 660 photo px = 660 * 0.32 = 211 prepared px
        #     shift of 36 photo px        =  36 * 0.32 =  11.5 prepared px off-centre
        #
        #     -> cut a 211-pixel square at (144 + 11.5, 144) out of the 288, resize to 256.
        #
        #     +---------- 288 prepared px = 900 photo px -----------+
        #     |                                                     |
        #     |        +------ 211 px window ------+                |
        #     |        |                           |                |
        #     |        |      the person           |   <- margin the zoom/shift eats into
        #     |        +---------------------------+                |
        #     +-----------------------------------------------------+
        #
        # The margin is why CROP_PADDING is 1.5. Worst case needs
        # half a window plus a full shift = 1.15*600/2 + 0.06*600 = 381 px,
        # against the 450 available. It fits, with room to spare.
        scale = CROP_PIXELS / (box_side * CROP_PADDING)
        image = crop(np.ascontiguousarray(image),        # ::-1 above left a reversed view;
                     #                                    warpAffine needs real contiguous memory
                     CROP_PIXELS / 2 + shift_x * scale,  # centre of the prepared crop, then shifted
                     CROP_PIXELS / 2 + shift_y * scale,
                     box_side * zoom * scale,            # window width, in prepared-crop pixels
                     self.pixels)                        # resized to 256x256

        # --- 3. report the window we ACTUALLY cut ----------------------------
        # Back in photograph pixels, because that is the ruler the camera and the
        # label share. This must reflect the zoom and shift: the box is part of
        # the label, and (u, v, size) below is measured against it. Leave it
        # stale and the network is told the body sits somewhere it does not.
        box = np.array([centre_x + shift_x, centre_y + shift_y, box_side * zoom], np.float32)

        # --- 4. turn the pixels into a normalised tensor ---------------------
        image = image.astype(np.float32) / 255.0         # 0..255 ints -> 0..1 floats
        if self.augment:
            # Brightness only. No hue or heavy colour jitter: clothing and skin
            # tone are how the network finds limbs in the first place.
            image = np.clip(image * np.random.uniform(*BRIGHTNESS_RANGE), 0.0, 1.0)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD   # the scaling the backbone expects

        # --- 5. assemble the 49 numbers --------------------------------------
        joints = shard["joints" + side][row]              # (17, 3) viewer frame, cm
        rotation = shard["root_rot" + side][row]          # (3, 3) which way the body faces

        target = np.concatenate([
            shard["angles" + side][row],                     # 31  straight from the shard
            shard["scales" + side][row],                     # 9   straight from the shard
            rotation[:, 0], rotation[:, 1],                  # 6   COLUMNS, not rows: the matrix's
            #                                                      first two axes. The third is
            #                                                      their cross product, so dropping
            #                                                      it loses nothing.
            hips_in_crop(joints[PELVIS], box, intrinsics),   # 3   depends on the box above
        ]).astype(np.float32)

        return dict(
            # (H, W, 3) -> (3, H, W): torch puts the colour channel first.
            image=torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            context=torch.from_numpy(crop_context(box, intrinsics)),
            target=torch.from_numpy(target),
            # Part 2 only, from here down.
            joints=torch.from_numpy(joints),
            truth=torch.from_numpy(shard["truth" + side][row]),
            box=torch.from_numpy(box),
            intrinsics=torch.from_numpy(intrinsics),
        )


def hips_in_crop(root, box, intrinsics):
    """
    The hips' 3D position -> (u, v, size), the last three of the 49 numbers.

        in   root (3,)        the hips, viewer frame, centimetres
             box  (3,)        the window actually cut, photograph pixels
             intrinsics (4,)  fx, fy, cx, cy
        out  (3,)             u, v, size

        u, v   where the hips sit in the crop. 0 is its middle, +/-1 its edges.
        size   how many METRES across the crop is, out at the person's distance.

    WHY NOT JUST PREDICT z. Because a crop physically cannot see distance. A
    child two metres away and an adult ten metres away fill the same square.
    Asking the network for z is asking it to guess.

    WHAT A CROP CAN SEE is how big the person is — their build, read off the
    picture. That is what `size` encodes: "the square I am looking at is about
    1.8 metres across". Distance then follows from the pinhole formula, because
    we know how many PIXELS across the crop is and now also how many METRES:

        box_side = fx * (size_in_cm / away)   =>   away = size_in_cm * fx / box_side

    So the network predicts something learnable, and geometry supplies the rest.

    u and v are the same idea applied sideways: the hips project to some pixel,
    and u, v say where that pixel falls inside the crop. Dividing by HALF the
    side is what puts the edges at exactly +/-1.

    This is the exact inverse of `translation_from` in Part 2, which turns the
    network's predicted (u, v, size) back into a real 3D position.
    """
    fx = intrinsics[0]
    pixels = project(root, intrinsics)              # (2,) which pixel the hips land on
    half_side = box[2] / 2                          # half the crop's width, in photograph pixels
    away = max(-root[2], 10.0)                      # distance in front = -z; floor stops a blow-up

    return np.array([(pixels[0] - box[0]) / half_side,    # u: across the crop, -1..+1
                     (pixels[1] - box[1]) / half_side,    # v: down the crop,   -1..+1
                     np.log(away * box[2] / fx / 100.0)])      # was: away * box[2] / fx / 100.0
                    #  away * box[2] / fx / 100.0])         # size: crop width in metres (/100: cm->m)


def crop_context(box, intrinsics):
    """
    Where the crop sits in the photograph, as three scale-free numbers.

        in   box (3,), intrinsics (4,)
        out  (3,)   fed to the network ALONGSIDE the picture

    WHY THE NETWORK NEEDS THIS. A crop cannot know where it was cut from.
    Someone standing at the edge of a wide-angle photograph is viewed from an
    angle, so they look leant over in their crop even when standing perfectly
    straight. The picture alone is genuinely ambiguous; these three numbers
    resolve it, and they cost nothing because we already know them.

    WHY DIVIDED BY FOCAL LENGTH, not image width. It turns pixels into ANGLES —
    "how many focal lengths off the optical axis" — so a 4K camera and a phone
    camera produce the same three numbers for the same geometry. Divide by width
    instead and the network would have to learn each camera separately.
    """
    fx, fy, cx, cy = intrinsics
    return np.array([(box[0] - cx) / fx,       # how far right of the optical axis the crop is
                     (box[1] - cy) / fy,       # how far below it
                     box[2] / fx],             # how wide the crop is, as an angle
                    np.float32)


if __name__ == "__main__":
    # A quick look: build the training split and print one sample's shapes.
    train_clips, val_clips = split_clips()
    data = PoseData(train_clips, augment=True)
    print(f"{len(train_clips)} training clips, {len(val_clips)} held back, {len(data)} frames")
    for name, value in data[0].items():
        print(f"  {name:11s} {tuple(value.shape)}")
