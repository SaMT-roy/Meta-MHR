"""Train the network to read MHR poses out of pictures.

Run:    python mhr_pose/train.py
Writes: mhr_pose/data/network.pt

What is being asked of the network, and how each answer is marked
----------------------------------------------------------------
Six things are measured, because getting the parameters roughly right and getting
the body geometrically right are not the same job:

  heading    how far the predicted hips face from the real ones
  angles     the 31 joint angles, each divided by how much it normally varies
  joints     where the joints end up in space, once the body has actually been
             built -- this is the one that tracks whether the pose *looks* right,
             because a small error at the shoulder throws the wrist a long way
  picture    where those joints land in the photograph. This is what pins down
             how far away the person is: nothing else can tell a big person far
             away from a small person close up
  standing   the position itself, with distance measured as a ratio, because
             being 50 cm out at 3 m is a bad miss and at 20 m is not
  sensible   a small nudge away from angles no human in the data ever reached

The first two grade the answer sheet; the next three grade the body it produces.
A network can score well on one and badly on the other, so both are needed.

Splitting the data honestly
---------------------------
There are two people in this dataset. If frames of both appear in training and in
validation, the score mostly says "this is that man in that field again", which
is not what we want to know. So one whole person is held out. That leaves very
little to train on, and the honest reading of any result here is that this proves
the pipeline runs end to end, not that the network is any good yet.
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "mhr_posing"), str(HERE)]

import analytic_ik as ik           # noqa: E402
import model as net                # noqa: E402
from PIL import Image              # noqa: E402

DATA = HERE / "data"

# How much each of the six marks counts. They are in different units -- radians,
# centimetres, fractions of a crop -- so these numbers bring them to a similar
# size rather than expressing an opinion about which matters most.
WEIGHTS = {"heading": 1.0, "angles": 1.0, "joints": 0.02,
           "picture": 1.0, "standing": 1.0, "sensible": 1.0}


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------

class Crops(Dataset):
    """The crops and answers made by make_dataset.py, for one set of clips."""

    def __init__(self, labels, keep, jitter_colour=False):
        self.keep = np.flatnonzero(keep)
        self.labels = labels
        self.jitter_colour = jitter_colour

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, position):
        i = self.keep[position]
        picture = Image.open(DATA / "crops" / str(self.labels["name"][i]))
        image = torch.from_numpy(np.array(picture, dtype=np.float32) / 255.0)
        image = image.permute(2, 0, 1)                      # (height, width, colour) -> (colour, height, width)

        if self.jitter_colour:
            # The only augmentation here. Moving or flipping the crop is not safe:
            # the answers are tied to exactly where the crop was taken from, and
            # mirroring a person means renaming their left and right joints.
            image = (image * float(np.random.uniform(0.7, 1.3))).clamp(0, 1)

        example = {"image": image}
        for key in ("box", "focal", "centre", "rotation", "position",
                    "articulation", "camera", "joints_3d", "joints_2d"):
            example[key] = torch.tensor(self.labels[key][i], dtype=torch.float32)
        return example


def load_data(held_out, batch_size):
    labels = dict(np.load(DATA / "labels.npz", allow_pickle=False))
    is_held_out = labels["subject"] == held_out
    print(f"training on {(~is_held_out).sum()} crops, "
          f"validating on {is_held_out.sum()} crops of person {held_out}")

    train = DataLoader(Crops(labels, ~is_held_out, jitter_colour=True),
                       batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    validate = DataLoader(Crops(labels, is_held_out), batch_size=batch_size, num_workers=4)
    return labels, train, validate


# ---------------------------------------------------------------------------
# The six marks
# ---------------------------------------------------------------------------

def build_body(predicted, wanted, constants, base_pose, mhr):
    """Turn a prediction into an actual body: joints in space, and in the picture."""
    rotation, articulation, camera = predicted
    position = net.translation_from_camera(camera, wanted["box"], wanted["focal"],
                                           wanted["centre"])
    body = net.body_in_own_frame(mhr, base_pose, constants["indices"], articulation,
                                 constants["root"], constants["keypoints"])
    joints = net.place_body(body, rotation, position)
    pixels = net.project(joints, wanted["focal"], wanted["centre"])
    return joints, pixels, position


def hips_at_origin(joints):
    """The same joints measured from the hips, so only the shape of the pose is left."""
    return joints - joints[:, :1]


def score(predicted, wanted, constants, base_pose, mhr, low, high):
    """Compare a prediction with the truth. Returns every mark separately."""
    rotation, articulation, camera = predicted
    joints, pixels, _ = build_body(predicted, wanted, constants, base_pose, mhr)

    # In the picture, distances are counted in crop-widths rather than pixels, so
    # a miss on a distant person counts the same as the same miss up close.
    box_side = wanted["box"][:, 2:3, None]

    return {
        "heading": net.rotation_difference(rotation, wanted["rotation"]).mean(),
        "angles": ((articulation - wanted["articulation"]).abs() / constants["spread"]).mean(),
        "joints": (hips_at_origin(joints) - hips_at_origin(wanted["joints_3d"])).abs().mean(),
        "picture": ((pixels - wanted["joints_2d"]).abs() / box_side).mean(),
        "standing": (camera - wanted["camera"]).abs().mean(),
        "sensible": ((articulation - high).clamp(min=0)
                     + (low - articulation).clamp(min=0)).mean(),
    }


def report(predicted, wanted, constants, base_pose, mhr):
    """The four numbers a person actually wants to see, in real units."""
    rotation = predicted[0]
    joints, pixels, position = build_body(predicted, wanted, constants, base_pose, mhr)
    off_by = hips_at_origin(joints) - hips_at_origin(wanted["joints_3d"])
    return {
        "pose error (cm)": off_by.norm(dim=-1).mean(),
        "heading error (deg)": torch.rad2deg(net.rotation_angle(rotation, wanted["rotation"])).mean(),
        "distance error (%)": 100 * (position[:, 2] / wanted["position"][:, 2]).log().abs().mean(),
        "in-picture error (px)": (pixels - wanted["joints_2d"]).norm(dim=-1).mean(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def guessing_the_average(labels, trained_on, validate, constants, base_pose, mhr):
    """How well you would do by ignoring the picture and always saying the same thing.

    Worth knowing before reading any other number. A network that scores worse
    than this has learned nothing from the picture at all, and one that scores a
    little better than this has learned very little. The average is taken from
    the training person only, so it is a fair comparison.
    """
    average = torch.tensor(labels["articulation"][trained_on].mean(0), dtype=torch.float32)
    camera = torch.tensor(labels["camera"][trained_on].mean(0), dtype=torch.float32)

    # The average of several rotation matrices is not itself a rotation, so take
    # the nearest one to it.
    left, _, right = np.linalg.svd(labels["rotation"][trained_on].mean(0))
    turn = left @ np.diag([1.0, 1.0, float(np.linalg.det(left @ right))]) @ right
    turn = torch.tensor(turn, dtype=torch.float32)

    totals, counted = {}, 0
    for batch in validate:
        size = len(batch["box"])
        guess = (turn.expand(size, 3, 3), average.expand(size, -1), camera.expand(size, -1))
        for name, value in report(guess, batch, constants, base_pose, mhr).items():
            totals[name] = totals.get(name, 0.0) + float(value) * size
        counted += size
    return {name: value / counted for name, value in totals.items()}


def predict(network, batch, device):
    """Run the network, and bring the answer back to the CPU beside the truth.

    MHR's forward pass is a TorchScript program that only runs on the CPU, so the
    body-building that follows always happens there. On a graphics card PyTorch
    carries the gradients back across the move by itself.

    A warning from measuring it: this does **not** work on an Apple graphics card.
    One step runs in 1.5 s and the next never finishes. Plain CPU takes about two
    seconds for a batch of 24 and is what `--device` defaults to; a real CUDA card
    would be much quicker.
    """
    predicted = network(batch["image"].to(device), batch["box"].to(device),
                        batch["focal"].to(device), batch["centre"].to(device))
    return tuple(value.cpu() for value in predicted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--held-out", default="1e28", help="the person kept for validation")
    parser.add_argument("--device", default="cpu", help="cpu, or cuda if you have it")
    args = parser.parse_args()

    labels, train, validate = load_data(args.held_out, args.batch_size)

    rig = ik.Rig()
    assert list(net.KEYPOINT_JOINTS) == list(ik.TARGET_TO_RIG.values()), \
        "model.KEYPOINT_JOINTS no longer matches analytic_ik.TARGET_TO_RIG"
    constants = net.rig_constants(rig)
    constants["spread"] = torch.tensor(labels["std"][:len(net.ARTICULATION)], dtype=torch.float32)
    base_pose = torch.tensor(labels["base_pose"], dtype=torch.float32)[None]
    low = torch.tensor(labels["low"], dtype=torch.float32)
    high = torch.tensor(labels["high"], dtype=torch.float32)

    device = torch.device(args.device)
    network = net.PoseNet(torch.tensor(labels["mean"], dtype=torch.float32),
                          torch.tensor(labels["std"], dtype=torch.float32)).to(device)

    # Only the head is trained. The backbone is frozen in `model.PoseNet`, so
    # asking for the parameters that still want a gradient gives exactly the head.
    learning = [parameter for parameter in network.parameters() if parameter.requires_grad]
    print(f"training {sum(p.numel() for p in learning)/1e6:.2f}M weights; the backbone's "
          f"{sum(p.numel() for p in network.backbone.parameters())/1e6:.1f}M are frozen")
    optimiser = torch.optim.AdamW(learning, lr=3e-4, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs * len(train))

    bar = guessing_the_average(labels, labels["subject"] != args.held_out, validate,
                               constants, base_pose, rig.model)
    print("guessing the average: " + "  ".join(f"{n} {v:.2f}" for n, v in bar.items()))
    print("anything worse than that has learned nothing from the picture.\n")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        network.train()
        started, totals = time.time(), {}
        for batch in train:
            marks = score(predict(network, batch, device), batch,
                          constants, base_pose, rig.model, low, high)
            loss = sum(WEIGHTS[name] * mark for name, mark in marks.items())

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimiser.step()
            schedule.step()

            for name, mark in marks.items():
                totals[name] = totals.get(name, 0.0) + float(mark.detach())
        marks = " ".join(f"{name} {value/len(train):.3f}" for name, value in totals.items())
        print(f"epoch {epoch:3d}  {marks}  [{time.time()-started:.0f}s]")

        network.eval()
        totals = {}
        with torch.no_grad():
            for batch in validate:
                for name, value in report(predict(network, batch, device), batch,
                                          constants, base_pose, rig.model).items():
                    totals[name] = totals.get(name, 0.0) + float(value)
        held = {name: value / len(validate) for name, value in totals.items()}
        print("           held-out: " + "  ".join(f"{n} {v:.2f}" for n, v in held.items()))

        if held["pose error (cm)"] < best:
            best = held["pose error (cm)"]
            torch.save({"network": network.state_dict(),
                        "mean": labels["mean"], "std": labels["std"],
                        "base_pose": labels["base_pose"], "epoch": epoch},
                       DATA / "network.pt")
            print(f"           saved (best so far: {best:.2f} cm)")


if __name__ == "__main__":
    main()
