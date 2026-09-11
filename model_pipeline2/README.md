# mhr_pose_net

One picture of a person in, a posed MHR body out.

```
ASPset-510 3D keypoints ──(mhr_posing/analytic_ik.py)──► MHR parameters   = the labels
person crop ────────────────────(this network)─────────► MHR parameters   = the prediction
```

The labels are not hand-annotated. `mhr_posing/analytic_ik.py` already solves 3D
keypoints into MHR parameters in one geometric pass; this trains a network to do
the same job from a picture, without the 3D keypoints.

## Files

| | |
|---|---|
| `data.py` | reads ASPset, solves the labels, caches crops, serves training pairs |
| `model.py` | the network, the MHR rig as a differentiable layer, and the losses |
| `train.ipynb` | run this first |
| `infer.ipynb` | run this second: 2D overlay, 3D scene, scores |
| `cache/` | built by `train.ipynb`, ~180 MB, delete it any time |
| `checkpoint.pt` | written by `train.ipynb` |

Nothing outside this folder is modified. `analytic_ik.py`, `mhr_scene.py` and
`mhr_kit` are imported as they are.

## What the network predicts

49 numbers, from a 256×256 crop plus three numbers saying where in the
photograph that crop came from:

| | |
|---|---|
| 31 | joint angles — spine, neck, head, clavicles, shoulders, elbows, hips, knees |
| 9 | build scales — how long this person's bones are |
| 6 | root rotation, as two vectors rather than three euler angles |
| 3 | where the body sits in the crop (2) and how big the crop is in metres (1) |

That is exactly the set of parameters the analytic solver writes: 46 of MHR's
204, with the root's three euler angles replaced by six numbers that do not wrap
round at ±π, and its three translations replaced by a crop-relative position that
a crop can actually know.

## What the losses are

Four on the numbers themselves (angles, scales, rotation, camera) and two on the
geometry: the predicted parameters are **run through the real MHR rig and the
real camera**, and compared in 3D centimetres and in crop pixels. The geometry
losses are what stop the network settling for a pose that is numerically close
and visibly wrong — a shoulder angle moves the hand ten times further than a
wrist angle does, and only the rig knows that.

## What it scores

25 epochs of `resnet18`, 27 minutes on an M-series CPU+GPU. 50 clips / 5 559
frames to train, 10 clips / 1 310 frames held back.

| | predicted | floor |
|---|---|---|
| joint error (root-relative) | **91 mm** | 16 mm |
| reprojection, 4K pixels | **12.9 px** | 2.9 px |
| root position | **24 cm** at 10–25 m | 0 |
| facing direction | **12°** | 0 |

The validation loss was still falling slowly at epoch 25 — more epochs, or
`resnet34`, will improve on this. The "floor" column is the same measurement
applied to the labels themselves.

## Measured facts this design rests on

All measured in this workspace, not assumed:

- Seeking one frame of a 4K ASPset video takes **0.31 s**; reading in order takes
  **0.006 s**. Hence the cache.
- `root_rz` jumps a full 2π **six times in 120 consecutive frames**. Hence the
  six-number rotation. The other 31 angles move at most 0.21 rad between
  neighbouring frames, so they are predicted as plain numbers.
- Posing MHR is differentiable and costs **~6 ms for a batch of 24** on the CPU,
  next to nothing beside the backbone. Hence the geometry losses.
- The analytic solver is **exactly mirror-symmetric (0.000 cm)**. Hence
  left-right flip augmentation is done by solving the mirrored keypoints, not by
  a hand-written table of which parameter to negate.
- Feeding the ground-truth labels into the scoring gives **≈24 mm** joint error,
  not 0: MHR's joint centres and ASPset's skin markers are a few cm apart by
  definition. That is the floor, and `train.ipynb` draws it on the charts.

## Caveats worth knowing

- ASPset-510 has **two subjects**. The clip split measures generalisation to new
  actions and camera angles, not to new people.
- The person box comes from the ground-truth keypoints. In real use it comes from
  a detector; `data.make_batch(photo, box, intrinsics)` is the entry point.
- Focal length is taken from the dataset's calibration. Without it, guess
  `f ≈ 1.2 × image width` — every distance then scales with that guess.
