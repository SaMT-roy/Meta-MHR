# mhr_pose — a picture of a person in, a posed MHR body out

Three steps, three files, in this order.

```bash
python mhr_pose/make_dataset.py    # ASPset clips  -> crops + answers   (~5 min)
python mhr_pose/train.py           # crops         -> a trained network
python mhr_pose/predict.py --example 40      # a picture -> a posed body
```

Or open a notebook:

* `pipeline.ipynb` — all three steps in order, with the results drawn.
* `infer_image.ipynb` — **one photograph in, a posed 3D body out.** Finds the people with a
  lightweight detector, crops, predicts, and shows the picture, the body drawn back onto it,
  and an interactive 3D scene. Needs no keypoints and no camera calibration, though it has to
  assume a lens — see the warning at the top of it.

Nothing outside this folder is changed. The solver in `../mhr_posing/` does the
hard geometry and is used as it stands.

## What each file does

| file | what it does |
|---|---|
| `make_dataset.py` | For every fifth video frame: solve the MHR pose from ASPset's 3D keypoints, move it into the frame of the camera that filmed it, cut the person out of the 4K picture, and write down the answer. |
| `model.py` | The network. A **frozen** DINOv2 looks at the crop; fifteen questions — one per joint group, one about the camera — each read what they need out of it. Also holds the sums that convert between "where the person is in their crop" and "where the person is in front of the camera". |
| `train.py` | The six marks the network is graded on, and the loop. Only the head's 4.7M weights are trained; the backbone's 22.1M are frozen. |
| `infer_image.ipynb` | Inference on any photograph: detect, crop, predict, draw. `fasterrcnn_mobilenet_v3_large_320_fpn` finds the people (19M weights, ~0.15 s, BSD). |
| `pipeline.ipynb` | All of the above in order, with the pictures: the answers drawn back onto their own photographs, the training curves, and predictions beside the truth. |
| `predict.py` | Runs the trained network on one picture and turns its answer back into an ordinary 204-number MHR parameter vector. |

Everything lands in `mhr_pose/data/`: `crops/`, `labels.npz`, `network.pt`, and
the pictures `predict.py` draws.

## The three answers the network gives

Splitting the pose into three pieces is the one design decision worth
understanding, because everything else follows from it.

1. **A heading** — which way the hips face, as a rotation matrix built from six
   numbers. Not as angles: the same heading can be written as very different sets
   of angles, and near some headings a hair's movement of the body swings the
   angles wildly. A network asked for angles would be punished for correct
   answers.
2. **31 joint angles** — how the body is bent. These stay in a narrow range, so
   plain numbers are fine here.
3. **Three numbers about the camera** — where the person sits inside their crop.
   Combined with where the crop was cut from, those give a real position in front
   of the camera. This is what makes a person in the top right of the photograph
   come out in the top right of the 3D scene, at the right distance.

Taking a real pose apart this way and putting it back together reproduces every
joint to **0.000008 cm**, so nothing is lost in the split. `predict.py` checks
the same thing on its own output every time it runs.

## Measured facts this code depends on

All checked rather than assumed; several of them are silent if wrong.

- **ASPset's world origin is the left camera.** A clip filmed by the mid or right
  camera is still described from the left camera's point of view, so its position
  and heading refer to a camera that did not take the picture. Two thirds of the
  clips here are mid or right. `make_dataset.py` converts every label into the
  frame of the camera that actually filmed it.
- **MHR's joint 0 is not the body.** It is a fixed point at the world origin that
  the body hangs from, and it does not move with the body. The hips are joint 1.
  Subtracting joint 0 leaves the body floating 92 cm too high.
- **The hips face straight along the world axes when every angle is zero**, so the
  heading the network predicts really is the heading of the hips.
- **The hips sit at `(0, 92.3987, 0)` cm when the three translation parameters are
  zero, and each parameter slides the body 10 cm**, whatever the heading. That is
  what lets `predict.py` write a position straight into a parameter vector.
- **MHR's forward pass can be trained through.** Gradients reach the joint angles,
  so "where do the joints end up" and "where do they land in the photograph" are
  real losses, not just things to measure afterwards.
- **Training runs on the CPU, not on an Apple graphics card.** MHR's forward pass
  is a TorchScript program that only runs on the CPU, so the gradient has to cross
  between devices. On Metal that stalls: the first step takes 1.5 s and the second
  never finishes. On the CPU a batch of 24 takes about 2 seconds, which is roughly
  three minutes an epoch here.
- **MHR's own joint limits cannot be used as a plausibility prior.** Several are
  degenerate — `spine0_rx_flexible` is listed as `[0, 0]` — and the solver writes
  values outside them. The limits used here are the range each angle was actually
  seen in.

## How to read a score

Three reference numbers, all measured, all printed by the code. Any result only
means something between the first and the last.

- **The bar.** Ignore the picture entirely and always answer with the average
  pose of the training person: **29.8 cm** of joint error on the held-out person.
  `train.py` prints this before it starts. A network scoring worse than this has
  learned nothing at all from the picture.

- **The labels are not perfect.** Put a solved body back into the picture
  it came from and its joints land **2.7 px** from the measured ones on average,
  in a 3840 x 2160 frame. `make_dataset.py` prints this.
- **Every body is built with the same limb lengths** — the average of the dataset —
  because limb lengths are a poor thing to read off a single photograph. Feeding
  the network's grader the *perfect* answers still leaves **2.05 cm** of joint
  error for that reason alone. That is the best score this pipeline can reach.

## What this cannot tell you yet

The honest framing: this proves the pipeline runs end to end. It does not show
that the network is any good, and it cannot.

- **Two people, one camera angle each.** 13 708 frames sound like a lot, but
  neighbouring frames are 1.3 cm apart; at a five-centimetre threshold there are
  only about **2 400 genuinely different poses**, of two people, in one field.
  `train.py` holds one person out entirely, which is the only honest split
  available and leaves very little to train on.
- **The two people are not the same size**, which puts a floor under the distance
  error specifically. The held-out person has 6.4% longer legs and a 15.6% longer
  spine than the training person. Distance is read from how large someone looks,
  so a body assumed to be the wrong size is read as being at the wrong distance —
  and a network that has only ever seen one build has no way around it. Measured:
  every other score keeps improving while the distance error sits near 11%.
  This is the same fixed-build limitation as the 2.05 cm ceiling above, showing up
  in a different number.
- **The crop comes from the ground truth.** There is no person detector in this
  repository, so the box is worked out from the true joints, wobbled a little to
  soften the difference. At inference a detector would supply it.
- **No mirroring, no moving the crop.** Both are the obvious next augmentations.
  Mirroring means renaming left and right joints, which on a mirrored rig is easy
  to get subtly wrong; moving the crop means recomputing the box the answers were
  tied to.

## The next things worth doing, in order

1. **Re-fetch ASPset's `videos` archive.** The calibration for all three camera
   views is already on disk; only two thirds of the video files are missing.
   Getting them triples the pictures and, more usefully, shows the same pose from
   three angles.
2. **Pretrain on bodies rendered from MHR itself.** MHR is Apache-2.0 and
   `mhr_kit/render.py` draws a frame in about a tenth of a second, so poses,
   builds, cameras and backgrounds can be varied freely with perfect answers.
   That is what would carry the geometry; ASPset then only has to bridge the gap
   to real photographs.
3. **Add a second head that predicts the 2D keypoints.** It costs nothing at
   inference — throw it away when exporting — and it sharpens the features the
   pose head reads, which is the known weak spot of a self-supervised backbone.
4. **Predict the body build**, once there are more than two people to learn it
   from. That is the 2.05 cm ceiling above.

## Licensing

DINOv2 is Apache-2.0, and so is MHR. ASPset ships no licence file in this
checkout; worth confirming upstream before anything ships.
