# MHR workspace — a documented, runnable setup for the Momentum Human Rig

A self-contained environment for **MHR** (Momentum Human Rig), Meta's parametric 3D
human body model — [paper](https://arxiv.org/abs/2511.15586),
[repository](https://github.com/facebookresearch/MHR), Apache-2.0.

This folder does not fork MHR. It installs the official `mhr` package from PyPI and
adds the pieces the upstream repository leaves out: an asset downloader that fetches
only what you need, a GPU-free renderer, named-parameter helpers, documented demos
for images and video, and this guide.

```bash
./setup.sh                      # venv + dependencies + ~27 MB of model assets
source .venv/bin/activate
python demos/demo_01_basic.py   # first mesh, first render
python examples.py --all        # twelve short worked examples
```

---

## Contents

1. [What MHR is (and is not)](#1-what-mhr-is-and-is-not)
2. [How the model works](#2-how-the-model-works)
3. [Installation](#3-installation)
4. [Assets: downloading only what you need](#4-assets-downloading-only-what-you-need)
5. [The API in one page](#5-the-api-in-one-page)
6. [Parameter reference](#6-parameter-reference)
7. [What's in this folder](#7-whats-in-this-folder)
8. [Demos](#8-demos)
9. [Examples](#9-examples)
10. [Common use cases](#10-common-use-cases)
11. [Performance and memory](#11-performance-and-memory)
12. [Troubleshooting](#12-troubleshooting)
13. [Licences and citation](#13-licences-and-citation)

---

## 1. What MHR is (and is not)

MHR is a **parametric body model**: parameters in, a posed 3D mesh out. It is the
same category of thing as SMPL / SMPL-X, with three differences that matter in
practice.

| | SMPL / SMPL-X | MHR |
|---|---|---|
| Skeleton | joint centres regressed *from* the surface | independent skeleton, 127 joints, decoupled from the surface |
| Pose input | flat axis-angle vector | 204 **named** rig controls (`l_elbow_bend`, `spine_twist0`, `scale_uplegs`, …) |
| Pose correctives | linear, dense | sparse and **non-linear** (a small per-joint network) |
| Expressions | FLAME-style PCA basis (SMPL-X) | 72 artist-sculpted FACS blendshapes |
| Resolutions | one | 7 levels of detail, 595 → 73 639 vertices |

**What MHR does not do: it never looks at an image.** Recovering parameters from a
photograph or video is a separate estimator's job, and the one built on MHR is
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body). This workspace covers
both directions:

* *parameters → mesh → picture*: demos 1, 2, 3, 5 and 7;
* *image → parameters → mesh*: demos 4 and 6, which consume SAM 3D Body predictions
  (four ship with MHR, and `mhr_kit/sam3d.py` documents the format so you can produce
  your own).

## 2. How the model works

From the paper, with the corresponding code in `mhr` and in this workspace:

```
X(β, θ) = M( X̄ + Bs(βs) + Bf(βf) + Bp(θ),  Bk(βk), θ, ω )
          └──────────── unposed template ────────────┘  └ skinning ┘
```

1. **Template and identity.** A neutral template `X̄` plus identity blendshapes
   `Bs(βs)` (45 coefficients) and expression blendshapes `Bf(βf)` (72). Identity is
   split into three disjoint groups — 20 body, 20 head, 5 hands — trained on separate
   scan datasets (7 110 filtered full-body scans, plus dedicated head and hand sets).
2. **Pose correctives** `Bp(θ)` are added **in the unposed frame, before skinning**.
   For each joint a small MLP consumes the 6D rotation deviations of that joint *and
   its immediate neighbours*, and its output is multiplied by a **sparse per-vertex
   mask** initialised from inverse geodesic distance. That sparsity is the point: it
   gives non-linear expressivity without letting a wrist rotation move a knee.
   In the released code this is exactly:
   `SparseLinear(125×6 → 125×24) → ReLU → Linear(125×24 → V×3)`,
   where 125 = 127 joints minus the two global ones, 6 = the 6D rotation
   representation, 24 = the per-joint embedding width, and the sparse mask encodes
   joint adjacency.
3. **Skeleton and skinning.** Momentum joints carry 3 translations, 3 XYZ-Euler
   rotations and 1 uniform scale. A fixed pre-rotation aligns each joint's x-axis
   with its bone, so *x rotations are twists* and the other axes are anatomically
   symmetric. A linear **parameter transform** `Θj = Tp · Θp` maps the 204 model
   parameters onto the 127×7 joint parameters. That indirection is what lets one
   parameter drive several joints (fractional twist joints, which suppress the
   candy-wrapper artefact) and several parameters drive one joint (upper and lower
   spine bend overlap). Skinning is plain LBS with artist-authored weights, capped at
   4 joint influences per vertex (8 at LOD 0).
4. **Levels of detail.** Identity, expression and correctives are trained at LOD 1
   (18 439 vertices) and transferred to the other LODs — barycentric mapping down,
   subdivision up. **All LODs share the same parameter vectors**, so you can fit on a
   coarse mesh and evaluate on a fine one.

Evaluation in the paper: on 3DBodyTex (200 subjects, 2 poses each) MHR reaches a lower
scan-to-model distance than SMPL and SMPL-X with fewer shape components, and is
visibly better at elbows, knees and shoulders. Stated limitations: no eyeballs, no
teeth/tongue, and correctives and expressions do not yet vary with body shape.

## 3. Installation

**Requirements**

* **Python 3.12 or 3.13.** `pymomentum`, the C++ rig/skinning library MHR is built
  on, publishes no wheels for 3.11 or older. This is the single most common install
  failure.
* ~1 GB of disk for the Python packages, plus assets (see below).
* No GPU needed. Everything in this workspace runs on CPU; the renderer needs no
  OpenGL context or display.

**Automatic**

```bash
./setup.sh                    # LOD 1 (the default)
./setup.sh --lod 4            # a lighter model: 2 461 vertices, 89 MB of correctives
./setup.sh --torchscript      # also fetch mhr_model.pt (26 MB download, 696 MB on disk)
```

`setup.sh` picks a suitable interpreter, creates `.venv`, installs
`requirements.txt`, and downloads the assets. Re-running it is safe.

**Manual**

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m mhr_kit.assets --lod 1
```

**CUDA machines.** Replace `pymomentum-cpu` with `pymomentum-gpu` in
`requirements.txt` and pass `device="cuda"` to `load_mhr`. On Apple Silicon stay on
CPU — there is no MPS backend for pymomentum, and CPU inference is a few milliseconds
per mesh anyway.

**Upstream alternative.** The official repository recommends [pixi](https://pixi.sh)
(`pixi install && pixi run download-assets`), which resolves `pymomentum` from
conda-forge. That works too; this workspace uses a plain venv so it needs no extra
tooling.

## 4. Assets: downloading only what you need

MHR's release ships one 199 MB `assets.zip` containing rig and correctives for all
seven LODs plus the TorchScript export. `mhr_kit/assets.py` reads the archive's ZIP
central directory over HTTP range requests and pulls **only the members you ask for**,
inflating them in memory and verifying each CRC-32. Nothing is staged on disk.

```bash
python -m mhr_kit.assets --lod 1                  # 27 MB instead of 199 MB
python -m mhr_kit.assets --lod 4 --torchscript
python -m mhr_kit.assets --help
```

| LOD | vertices | download (rig + correctives) | on disk | correctives in RAM |
|----:|---------:|-----------------------------:|--------:|-------------------:|
| 0 | 73 639 | 113 MB | 2.7 GB | 2 651 MB |
| 1 | 18 439 | **26 MB** | 672 MB | 664 MB |
| 2 | 10 661 | 18 MB | 388 MB | 384 MB |
| 3 | 4 899 | 8 MB | 179 MB | 176 MB |
| 4 | 2 461 | 4 MB | 90 MB | 89 MB |
| 5 | 971 | 2 MB | 36 MB | 35 MB |
| 6 | 595 | 1 MB | 22 MB | 21 MB |

Every LOD also needs two small shared files (`compact_v6_1.model`, 31 KB, and
`corrective_activation.npz`, 3.3 MB); the downloader fetches them once.

The same command downloads the four **SAM 3D Body example predictions** into `data/`
(210 KB each) — real MHR parameters estimated from a photograph, used by demos 4 and 6.

`load_mhr()` calls the downloader itself when something is missing, so you can also
just run a demo and let it fetch what it needs.

## 5. The API in one page

```python
import torch
from mhr_kit.model import load_mhr, faces, joint_positions, save_mesh
from mhr_kit.params import pose
from mhr_kit import render

model = load_mhr(lod=1, device="cpu")        # downloads assets on first use

identity   = torch.zeros(1, 45)              # who: the average body
parameters = pose(model, "relaxed")          # how: a (1, 204) named pose
expression = None                            # face: None means neutral

with torch.no_grad():
    vertices, skeleton_state = model(identity, parameters, expression)

mesh = vertices[0].numpy()                                   # (18439, 3) in cm
camera = render.orbit_camera(mesh, azimuth=25, size=(480, 600))
render.save_image(render.render(mesh, faces(model), camera), "person.png")
save_mesh(vertices, model, "person.ply")
```

**Inputs**

| argument | shape | notes |
|---|---|---|
| `identity_coeffs` | `(1 or B, 45)` | a single row is broadcast over the batch |
| `model_parameters` | `(B, 204)` | radians; root translation in units of 10 cm |
| `face_expr_coeffs` | `(B, 72)` or `None` | `None` = neutral face |
| `apply_correctives` | `bool` | `False` skips the corrective network |

**Outputs**

| output | shape | notes |
|---|---|---|
| `vertices` | `(B, V, 3)` | centimetres, +y up, +z forward; feet at y ≈ 0 |
| `skeleton_state` | `(B, 127, 8)` | per joint `tx ty tz qx qy qz qw scale`, world space |

`joint_positions(skeleton_state)` and `joint_rotations(skeleton_state)` are the two
slices you normally want. Everything is a plain `torch.nn.Module`, so autograd works
through the whole model (see example 9).

**Conventions, measured on the LOD 1 rig** (`demos/demo_03_pose_and_correctives.py`
and example 4 reproduce these):

* All parameters zero = a **T-pose**, standing on `y = 0`, facing `+z`.
* Rotations are radians. The rig is mirrored: the same sign means the same anatomical
  motion on both sides (`l_uparm_rz` and `r_uparm_rz` both raise their arm).
* `root_tx/ty/tz` are in **units of 10 cm** (`root_ty = -4.5` drops the pelvis 45 cm),
  while vertices come out in cm.
* Positive `*_rz` on a leg swings it **backwards**, and `*_twist` parameters rotate
  about the bone axis — both consequences of the pre-rotation described in §2.

## 6. Parameter reference

**Identity — 45 coefficients**, roughly zero-mean and unit-variance; ±3 is extreme.

| coefficients | controls |
|---|---|
| 0–19 | body: build, mass distribution, proportions |
| 20–39 | head / skull |
| 40–44 | hands |

Identity changes the **surface only**. Stature is a *skeleton* property, driven by the
`scale_*` model parameters, so sweeping identity coefficient 0 moves the body from 45
to 133 litres of volume at a constant 173 cm (`demos/demo_02_identity_and_expression.py`).

**Model parameters — 204**, split by the paper into 136 pose and 68 skeleton
transformation parameters:

| columns | family | examples |
|---|---|---|
| 0–5 | rigid transform | `root_tx`, `root_ty`, `root_tz`, `root_rx`, `root_ry`, `root_rz` |
| 6–29 | spine and neck | `spine_twist0`, `spine_lean0`, `spine_bend0`, `neck_bend`, `head_twist` |
| 30–49 | arms | `l_clavicle_rz`, `l_uparm_ry`, `l_uparm_rz`, `l_elbow_bend`, `l_wrist_rz` |
| 50–67 | legs | `l_upleg_rz`, `l_knee_bend`, `l_foot_bend`, `l_ball_bend` |
| 68–121 | fingers | `r_thumb1_rz`, `l_index2_rz`, … (all five digits, both hands) |
| 122–129 | ankle and foot detail | `l_talocrural_rx_flexible`, `r_subtalar_rz_flexible`, `l_ball_rx_flexible` |
| 130–135 | bone lengths | `spine_length_flexible`, `arm_length_flexible`, `leg_length_flexible`, … |
| 136–203 | skeleton scales | `scale_uplegs`, `scale_shoulder_width`, `scale_r_hands`, … |

Set them by name; nothing else needs to be remembered:

```python
from mhr_kit.params import model_parameters, pose, POSES
parameters = model_parameters(model, {"l_elbow_bend": 1.4, "spine_twist0": 0.3, "root_ry": 0.5})
parameters = pose(model, "walk_stride")   # POSES: t_pose a_pose relaxed arms_up wave
                                          #        sitting walk_stride squat
```

**Expressions — 72 coefficients**, one FACS-style artist-sculpted blendshape each,
usually driven in `[0, 1]`. `demos/demo_02_identity_and_expression.py` renders the
basis one coefficient at a time.

## 7. What's in this folder

```
mhr_env/
├── README.md                 this guide
├── requirements.txt          pinned dependency list, with the reason for each
├── setup.sh                  venv + dependencies + assets, idempotent
├── examples.py               12 short worked examples (task-sized snippets)
├── mhr_kit/                  the thin layer this workspace adds
│   ├── assets.py             partial-download of the release archive (range requests)
│   ├── model.py              loading, summaries, skeleton readers, mesh export
│   ├── params.py             named parameters, pose library, keyframe interpolation
│   ├── render.py             GPU-free renderer, cameras, PNG/MP4/GIF/contact sheets
│   └── sam3d.py              SAM 3D Body predictions: the image → MHR direction
├── demos/                    seven runnable demos, see below
├── assets/    (downloaded)   lod*.fbx, corrective_blendshapes_lod*.npz, …
├── data/      (downloaded)   SAM 3D Body example predictions
└── outputs/   (generated)    everything the demos write
```

`assets/`, `data/`, `outputs/` and `.venv/` are git-ignored; they are all reproducible
from `setup.sh`.

## 8. Demos

Each demo is a standalone script with `--help`, writes into `outputs/<name>/`, and
prints what it measured.

| demo | shows | typical runtime |
|---|---|---|
| `demo_01_basic.py` | loading, the three parameter blocks, both outputs, mesh export, turntable stills | ~5 s |
| `demo_02_identity_and_expression.py` | identity sweep, per-group randomisation, the 72-blendshape expression basis as face close-ups | ~6 s |
| `demo_03_pose_and_correctives.py` | the pose library, a pose written by name, and a heat map of what the non-linear correctives contribute | ~5 s |
| `demo_04_image_reconstruction.py` | **image → MHR**: SAM 3D Body predictions rebuilt and verified, camera recovered exactly, everyone rendered back into the photo | ~3 s |
| `demo_05_video_animation.py` | **video**: keyframe interpolation → one batched forward pass → MP4 | ~16 s |
| `demo_06_video_from_image.py` | **video from a still**: 360° orbit of a person reconstructed from a photo; `--mode sequence` for per-frame tracking output | ~9 s |
| `demo_07_torchscript.py` | the traced model: PyTorch only, no pymomentum, plus a parity check | ~4 s |

Highlights of what they print:

* Demo 4 rebuilds each person from `mhr_model_params` and agrees with the estimator's
  own vertices to **0.0008 mm**, and recovers the camera (focal 2535.3 px, principal
  point 960, 445 → a 1920×890 image) to a **1.1 × 10⁻⁴ px** reprojection error.
* Demo 3 measures the correctives: 3.9 mm mean and 45 mm maximum vertex displacement in
  a squat, concentrated at hips, glutes, thighs and shoulders — and 0.00 mm in the rest
  pose, as designed.

## 9. Examples

`examples.py` holds twelve independent snippets, each a few lines and printing its
result:

```bash
python examples.py            # list them
python examples.py 9          # run one
python examples.py --all      # run all (~15 s once the assets are there)
```

1. the smallest complete program • 2. batching • 3. levels of detail •
4. finding parameters by name • 5. skeleton measurements (bone lengths) •
6. mesh export • 7. what the correctives contribute • 8. identity groups •
9. **fitting parameters with autograd** • 10. image → mesh via SAM 3D Body •
11. rendering stills and clips • 12. TorchScript inference

## 10. Common use cases

**Reconstruct people from a photograph.** Run SAM 3D Body to get one `.npz` per
person, then:

```python
from mhr_kit.sam3d import load_predictions, mhr_inputs, fit_camera, to_camera_space

predictions = load_predictions("data")               # or your own folder
identity, parameters, expression = mhr_inputs(predictions)
with torch.no_grad():
    vertices, _ = model(identity, parameters, expression)   # one pass for everybody

camera, residual = fit_camera(predictions)           # exact intrinsics from the keypoints
in_camera = [to_camera_space(vertices[i].numpy(), p) for i, p in enumerate(predictions)]
```

Units and axes are the two traps: SAM 3D Body works in metres with y down and z into
the scene, MHR in centimetres with y up. `to_camera_space` handles it; the conversion
and every `.npz` key are documented in `mhr_kit/sam3d.py`.

**Animate, and write a video.** Interpolate keyframes, skin the whole clip in one
batch — the corrective network is what costs, and batching amortises it:

```python
from mhr_kit.params import interpolate_keyframes, pose
animation = interpolate_keyframes([pose(model, "relaxed"), pose(model, "arms_up")], 20, loop=True)
with torch.no_grad():
    vertices, _ = model(torch.zeros(1, 45), animation, None)   # (frames, V, 3)

frames = [render.render(v, faces(model), camera) for v in vertices.numpy()]
render.save_video(frames, "clip.mp4", fps=24)                  # GIF fallback if no ffmpeg
```

**Render without a GPU.** `mhr_kit/render.py` projects the mesh through a pinhole
camera, culls backfaces, sorts triangles back to front and draws them with
Matplotlib's Agg backend. No OpenGL, no display, ~0.1 s per 480×600 frame at LOD 1.
Pass `background=<HxWx3 array>` to composite over a photograph and `alpha=0.85` to
see through the mesh.

**Fit parameters to something.** The model is differentiable end to end, so fitting is
an ordinary optimisation (example 9). Fit at LOD 4 and evaluate at LOD 1 — the
parameters are identical across LODs and the coarse mesh is ~5× cheaper. For
production-grade fitting, pymomentum also ships a fast C++ Gauss-Newton solver, which
the upstream `tools/mhr_smpl_conversion` uses.

**Take a mesh into Blender / MeshLab.** `save_mesh(vertices, model, "person.obj")`
writes any format trimesh supports. Vertex order and triangles are fixed per LOD, so
per-vertex data stays comparable between poses and bodies.

**Convert to or from SMPL / SMPL-X.** Not reimplemented here — the upstream repository
has it in [`tools/mhr_smpl_conversion`](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion)
(both directions, PyTorch or pymomentum backends). It needs the SMPL/SMPL-X model
files, which require registering on their own websites and so cannot be downloaded
automatically.

**Ship MHR without pymomentum.** Use the TorchScript export (demo 7): one 696 MB file,
PyTorch only. Caveats: LOD 1 only, no triangle indices or joint names, and it does not
broadcast a single identity row over the batch — pass one row per pose.

## 11. Performance and memory

Measured on an Apple Silicon Mac, CPU only, LOD 1:

| operation | cost |
|---|---|
| `load_mhr(lod=1)` | 1.6 s, ~1.4 GB resident |
| forward, batch 1 | 15 ms |
| forward, batch 8 | 6.8 ms/mesh |
| forward, batch 64 | 2–3 ms/mesh |
| forward without correctives, batch 8 | 3.4 ms/mesh (and no 664 MB matrix to load) |
| render 480×600 | ~0.10 s/frame |
| render 1920×890, four people | ~0.4 s |

Practical notes:

* Batch aggressively. The corrective network is a dense `(125×24) × (V×3)` matrix
  multiply; per-frame calls waste most of the work.
* Memory is dominated by the corrective blendshapes (664 MB at LOD 1, 2.7 GB at
  LOD 0). Use a coarser LOD, or `correctives=False`, when you only need silhouettes
  or rough geometry.
* `load_mhr(lod=4)` loads in 0.2 s and is plenty for fitting, previews and tests.

## 12. Troubleshooting

| symptom | cause and fix |
|---|---|
| `No matching distribution found for pymomentum-cpu` | Python 3.11 or older, or an unsupported platform. Use Python 3.12/3.13; `setup.sh` checks this for you. |
| `FileNotFoundError: missing assets for LOD n` | run `python -m mhr_kit.assets --lod n` (or let `load_mhr` do it). |
| `server did not honour range request` | a proxy or mirror stripping `Range`. Fall back to the official whole-archive downloader: `mhr-download-assets`. |
| Download hangs before the first byte | the GitHub release CDN handshake; the transfer itself is quick once connected. All reads share one connection, so you pay it once. |
| MP4 missing, GIF written instead | `imageio-ffmpeg` is not installed or its encoder failed; the renderer falls back automatically. |
| Killed / swapping while loading LOD 0 or 1 | 2.7 GB / 664 MB of correctives. Use `--lod 4`, or `load_mhr(correctives=False)`. |
| `Sizes of tensors must match` from the TorchScript model | it does not broadcast identity; pass `identity.repeat(batch, 1)`. |
| CUDA is ignored | you installed `pymomentum-cpu`; swap in `pymomentum-gpu` and pass `device="cuda"`. |

## 13. Licences and citation

* **MHR** — model, code and release assets: Apache-2.0 (Meta). The asset archive
  carries its own `LICENSE.txt`, downloaded alongside the model files.
* **pymomentum / Momentum** — Meta, open source; installed from PyPI.
* **SAM 3D Body** — separate repository and licence; only its *output* files are used
  here, and the four examples come from the MHR repository itself.
* **This workspace** (`mhr_kit/`, `demos/`, `examples.py`) — Apache-2.0, to match MHR.
* **SMPL / SMPL-X** model files are *not* included and cannot be redistributed; get
  them from their own websites if you need the conversion tools.

```bibtex
@misc{MHR:2025,
      title={MHR: Momentum Human Rig},
      author={Aaron Ferguson and Ahmed A. A. Osman and Berta Bescos and Carsten Stoll
              and Chris Twigg and Christoph Lassner and others},
      year={2025},
      eprint={2511.15586},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2511.15586},
}
```
