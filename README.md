<div align="center">

# Meta-MHR

**A working bench for Meta's Momentum Human Rig — the environment, the utilities, the demos, and the pipelines built on top of it.**

[![Model](https://img.shields.io/badge/model-MHR%20(Meta)-blue)](https://github.com/facebookresearch/MHR)
[![Paper](https://img.shields.io/badge/paper-arXiv%202511.15586-b31b1b)](https://arxiv.org/abs/2511.15586)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://www.apache.org/licenses/LICENSE-2.0)
[![Runs on](https://img.shields.io/badge/runs%20on-CPU%20only-lightgrey)]()

</div>

---

## Why this repository exists

[MHR](https://github.com/facebookresearch/MHR) is an excellent model with a thin on-ramp. The official release gives you a PyPI package, a 199 MB asset archive, and not much else — no small runnable examples, no renderer, no way to find a parameter by name, and no story for the part most people actually want, which is *going from a photograph to a posed body*.

This repository is what accumulated while closing that gap. It **does not fork MHR**. It installs the official `mhr` package and wraps it in the things that were missing:

- an asset downloader that pulls **only the level of detail you asked for** (27 MB instead of 199 MB),
- a **GPU-free renderer** that needs no OpenGL context and no display,
- **named** parameter helpers, so you write `l_elbow_bend=1.4` instead of indexing into a 204-vector,
- demos and notebooks that are short, runnable, and print what they measured,
- and pipelines that take the model somewhere past "hello, mesh".

If you want the one-line version: **this is the repo I wish had existed on the day I first tried to pose an MHR body.**

---

## Quickstart

```bash
./setup.sh                      # venv + dependencies + ~27 MB of model assets
source .venv/bin/activate

python demos/demo_01_basic.py   # first mesh, first render
python examples.py --all        # twelve short worked examples (~15 s)
```

`setup.sh` picks a suitable interpreter, creates `.venv`, installs `requirements.txt`, and downloads assets. Re-running it is safe.

```bash
./setup.sh --lod 4              # lighter model: 2 461 vertices, 89 MB of correctives
./setup.sh --torchscript        # also fetch mhr_model.pt (26 MB download, 696 MB on disk)
```

> **The single most common install failure:** Python 3.11 or older. `pymomentum`, the C++ rig/skinning library MHR sits on, publishes no wheels below 3.12. `setup.sh` checks this before it does anything else.

---

## Repository map

| Path | What it is |
| --- | --- |
| **`mhr_kit/`** | The thin layer this repo adds: assets, model loading, named parameters, rendering, SAM 3D Body I/O. Everything else imports from here. |
| **`demos/`** | Seven standalone scripts, each with `--help`, each writing into `outputs/<name>/`. The guided tour. |
| **`demos ipynb/`** | The same ground in notebook form — for reading, poking at intermediate tensors, and running on Colab without a local venv. |
| **`mhr_posing/`** | Posing work: building, editing and interpolating poses, and the conventions that make the 204-parameter rig navigable by hand. |
| **`model_pipeline/`** | First end-to-end pipeline built on MHR — the working version. |
| **`model_pipeline2/`** | The second pass at that pipeline. Kept alongside rather than replacing it, so the two are comparable. |
| `requirements.txt` | Pinned dependencies, with the reason for each. |
| `GUIDE(dummy).md` | Scratch / working notes. Not authoritative; this README is. |
| `assets/` `data/` `outputs/` | Downloaded or generated, all git-ignored, all reproducible from `setup.sh`. |

<details>
<summary><b>Inside <code>mhr_kit/</b></code></summary>

```
mhr_kit/
├── assets.py    partial download of the release archive over HTTP range requests
├── model.py     loading, summaries, skeleton readers, mesh export
├── params.py    named parameters, pose library, keyframe interpolation
├── render.py    GPU-free renderer, cameras, PNG / MP4 / GIF / contact sheets
└── sam3d.py     SAM 3D Body predictions — the image → MHR direction
```

</details>

---

## The two directions

MHR is a **parametric body model**: parameters in, posed mesh out. Crucially, **it never looks at an image.** Recovering parameters from a photo is a separate estimator's job — that's [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body). This repo covers both directions:

```
  parameters ──────────────▶ mesh ──────────────▶ picture      demos 1, 2, 3, 5, 7
   (identity, pose,          MHR                  mhr_kit.render
    expression)

  image ──▶ parameters ──▶ mesh ──▶ back into the photo        demos 4, 6
   SAM 3D Body            MHR      mhr_kit.sam3d
```

Four SAM 3D Body example predictions ship with MHR and are downloaded automatically; `mhr_kit/sam3d.py` documents the `.npz` format so you can produce your own.

---

## How the model works

```
X(β, θ) = M( X̄ + Bs(βs) + Bf(βf) + Bp(θ),  Bk(βk), θ, ω )
          └──────────── unposed template ────────────┘  └ skinning ┘
```

**1 · Template and identity.** A neutral template plus identity blendshapes (45 coefficients) and expression blendshapes (72). Identity splits into three disjoint groups — 20 body, 20 head, 5 hands — trained on separate scan datasets.

**2 · Pose correctives**, added *in the unposed frame, before skinning*. Per joint, a small MLP consumes the 6D rotation deviations of that joint and its immediate neighbours, multiplied by a **sparse per-vertex mask** initialised from inverse geodesic distance. The sparsity is the whole point: non-linear expressivity, without a wrist rotation being able to move a knee.

```
SparseLinear(125×6 → 125×24) → ReLU → Linear(125×24 → V×3)
```

**3 · Skeleton and skinning.** 127 joints, each with 3 translations, 3 XYZ-Euler rotations and 1 uniform scale. A fixed pre-rotation aligns each joint's x-axis with its bone, so *x rotations are twists*. A linear parameter transform maps 204 model parameters onto 127×7 joint parameters — that indirection is what lets one parameter drive several joints (fractional twist joints, which kill the candy-wrapper artefact) and several parameters drive one joint. Skinning is plain LBS, capped at 4 influences per vertex (8 at LOD 0).

**4 · Levels of detail.** Trained at LOD 1 and transferred outward. **All LODs share the same parameter vectors** — so you can fit on a coarse mesh and evaluate on a fine one.

<details>
<summary><b>How MHR differs from SMPL / SMPL-X</b></summary>

| | SMPL / SMPL-X | MHR |
| --- | --- | --- |
| Skeleton | joint centres regressed *from* the surface | independent skeleton, 127 joints, decoupled from the surface |
| Pose input | flat axis-angle vector | 204 **named** rig controls (`l_elbow_bend`, `spine_twist0`, …) |
| Pose correctives | linear, dense | sparse and **non-linear** (small per-joint network) |
| Expressions | FLAME-style PCA basis | 72 artist-sculpted FACS blendshapes |
| Resolutions | one | 7 levels of detail, 595 → 73 639 vertices |

On 3DBodyTex the paper reports a lower scan-to-model distance than SMPL and SMPL-X with fewer shape components, and visibly better elbows, knees and shoulders. Stated limitations: no eyeballs, no teeth or tongue, and correctives and expressions don't yet vary with body shape.

</details>

---

## The API in one page

```python
import torch
from mhr_kit.model import load_mhr, faces, joint_positions, save_mesh
from mhr_kit.params import pose
from mhr_kit import render

model = load_mhr(lod=1, device="cpu")        # downloads assets on first use

identity   = torch.zeros(1, 45)              # who:  the average body
parameters = pose(model, "relaxed")          # how:  a (1, 204) named pose
expression = None                            # face: None means neutral

with torch.no_grad():
    vertices, skeleton_state = model(identity, parameters, expression)

mesh   = vertices[0].numpy()                                  # (18439, 3) in cm
camera = render.orbit_camera(mesh, azimuth=25, size=(480, 600))
render.save_image(render.render(mesh, faces(model), camera), "person.png")
save_mesh(vertices, model, "person.ply")
```

**Inputs**

| argument | shape | notes |
| --- | --- | --- |
| `identity_coeffs` | `(1 or B, 45)` | a single row is broadcast over the batch |
| `model_parameters` | `(B, 204)` | radians; root translation in units of 10 cm |
| `face_expr_coeffs` | `(B, 72)` or `None` | `None` = neutral face |
| `apply_correctives` | `bool` | `False` skips the corrective network |

**Outputs**

| output | shape | notes |
| --- | --- | --- |
| `vertices` | `(B, V, 3)` | centimetres, +y up, +z forward; feet at y ≈ 0 |
| `skeleton_state` | `(B, 127, 8)` | per joint `tx ty tz qx qy qz qw scale`, world space |

It's a plain `torch.nn.Module`, so autograd runs through the whole thing (example 9).

**Conventions, measured on the LOD 1 rig** — `demos/demo_03_pose_and_correctives.py` and example 4 reproduce all of these:

- All parameters zero = a **T-pose**, standing on `y = 0`, facing `+z`.
- The rig is **mirrored**: the same sign means the same anatomical motion on both sides (`l_uparm_rz` and `r_uparm_rz` both raise their arm).
- `root_tx/ty/tz` are in **units of 10 cm** (`root_ty = -4.5` drops the pelvis 45 cm), while vertices come out in cm. This one bites.
- Positive `*_rz` on a leg swings it **backwards**, and `*_twist` rotates about the bone axis — both fall out of the pre-rotation above.

---

## Parameter reference

**Identity — 45 coefficients.** Roughly zero-mean, unit-variance; ±3 is extreme.

| coefficients | controls |
| --- | --- |
| 0–19 | body: build, mass distribution, proportions |
| 20–39 | head / skull |
| 40–44 | hands |

Identity changes the **surface only**. Stature is a *skeleton* property driven by the `scale_*` parameters — sweeping identity coefficient 0 moves the body from 45 to 133 litres at a constant 173 cm.

**Model parameters — 204**, split by the paper into 136 pose and 68 skeleton-transformation parameters.

| columns | family | examples |
| --- | --- | --- |
| 0–5 | rigid transform | `root_tx`, `root_ty`, `root_rz`, … |
| 6–29 | spine and neck | `spine_twist0`, `spine_bend0`, `neck_bend`, `head_twist` |
| 30–49 | arms | `l_clavicle_rz`, `l_uparm_ry`, `l_elbow_bend`, `l_wrist_rz` |
| 50–67 | legs | `l_upleg_rz`, `l_knee_bend`, `l_foot_bend`, `l_ball_bend` |
| 68–121 | fingers | `r_thumb1_rz`, `l_index2_rz`, … (five digits, both hands) |
| 122–129 | ankle and foot detail | `l_talocrural_rx_flexible`, `r_subtalar_rz_flexible` |
| 130–135 | bone lengths | `spine_length_flexible`, `arm_length_flexible`, … |
| 136–203 | skeleton scales | `scale_uplegs`, `scale_shoulder_width`, `scale_r_hands` |

Set them by name and forget the indices:

```python
from mhr_kit.params import model_parameters, pose, POSES

parameters = model_parameters(model, {"l_elbow_bend": 1.4, "spine_twist0": 0.3, "root_ry": 0.5})
parameters = pose(model, "walk_stride")
# POSES: t_pose  a_pose  relaxed  arms_up  wave  sitting  walk_stride  squat
```

**Expressions — 72 coefficients**, one FACS-style blendshape each, usually driven in `[0, 1]`.

---

## Assets: download only what you need

The official release ships one 199 MB `assets.zip` covering all seven LODs plus the TorchScript export. `mhr_kit/assets.py` reads the archive's ZIP central directory over HTTP **range requests** and pulls only the members you ask for, inflating them in memory and verifying each CRC-32. Nothing is staged on disk.

```bash
python -m mhr_kit.assets --lod 1                  # 27 MB instead of 199 MB
python -m mhr_kit.assets --lod 4 --torchscript
python -m mhr_kit.assets --help
```

| LOD | vertices | download | on disk | correctives in RAM |
| --- | --- | --- | --- | --- |
| 0 | 73 639 | 113 MB | 2.7 GB | 2 651 MB |
| **1** | **18 439** | **26 MB** | 672 MB | 664 MB |
| 2 | 10 661 | 18 MB | 388 MB | 384 MB |
| 3 | 4 899 | 8 MB | 179 MB | 176 MB |
| 4 | 2 461 | 4 MB | 90 MB | 89 MB |
| 5 | 971 | 2 MB | 36 MB | 35 MB |
| 6 | 595 | 1 MB | 22 MB | 21 MB |

Every LOD also needs two small shared files (`compact_v6_1.model`, 31 KB, and `corrective_activation.npz`, 3.3 MB), fetched once. The same command pulls the four SAM 3D Body example predictions into `data/`. `load_mhr()` calls the downloader itself when something is missing, so running a demo cold just works.

---

## Demos

Each is standalone, has `--help`, writes to `outputs/<name>/`, and prints what it measured.

| demo | shows | runtime |
| --- | --- | --- |
| `demo_01_basic.py` | loading, the three parameter blocks, both outputs, mesh export, turntable stills | ~5 s |
| `demo_02_identity_and_expression.py` | identity sweep, per-group randomisation, the 72-blendshape basis as face close-ups | ~6 s |
| `demo_03_pose_and_correctives.py` | the pose library, a pose written by name, and a heat map of what the correctives contribute | ~5 s |
| `demo_04_image_reconstruction.py` | **image → MHR**: SAM 3D Body predictions rebuilt and verified, camera recovered, everyone rendered back into the photo | ~3 s |
| `demo_05_video_animation.py` | **video**: keyframe interpolation → one batched forward pass → MP4 | ~16 s |
| `demo_06_video_from_image.py` | **video from a still**: 360° orbit of a person reconstructed from a photo; `--mode sequence` for per-frame output | ~9 s |
| `demo_07_torchscript.py` | the traced model: PyTorch only, no pymomentum, plus a parity check | ~4 s |

Two numbers worth calling out:

- Demo 4 rebuilds each person from `mhr_model_params` and agrees with the estimator's own vertices to **0.0008 mm**, and recovers the camera (focal 2535.3 px, principal point 960, 445 → a 1920×890 image) to a **1.1 × 10⁻⁴ px** reprojection error.
- Demo 3 measures the correctives: **3.9 mm mean, 45 mm max** vertex displacement in a squat — concentrated at hips, glutes, thighs and shoulders — and **0.00 mm** in the rest pose, exactly as designed.

The `demos ipynb/` folder mirrors these as notebooks, for when you'd rather step through than run.

### Examples

`examples.py` holds twelve independent snippets, each a few lines, each printing its result:

```bash
python examples.py            # list them
python examples.py 9          # run one
python examples.py --all      # run all
```

> 1 smallest complete program · 2 batching · 3 levels of detail · 4 finding parameters by name · 5 skeleton measurements · 6 mesh export · 7 what the correctives contribute · 8 identity groups · **9 fitting parameters with autograd** · 10 image → mesh via SAM 3D Body · 11 stills and clips · 12 TorchScript inference

---

## Posing and pipelines

Beyond the guided tour, three folders carry the actual project work.

**`mhr_posing/`** — everything about *getting a body into a pose you meant*. The 204-parameter vector is expressive but unfriendly; this is where poses get authored, named, interpolated and sanity-checked against the sign and unit conventions listed above.

**`model_pipeline/` and `model_pipeline2/`** — two end-to-end passes at the same problem, kept side by side on purpose. `model_pipeline2` is the later rework; the first is retained so the two can be compared rather than silently replaced. If you're picking one to read, start with `model_pipeline2`.

> These three folders move faster than this README. Treat the code as the source of truth, and `GUIDE(dummy).md` as working notes rather than documentation.

---

## Common use cases

<details open>
<summary><b>Reconstruct people from a photograph</b></summary>

Run SAM 3D Body to get one `.npz` per person, then:

```python
from mhr_kit.sam3d import load_predictions, mhr_inputs, fit_camera, to_camera_space

predictions = load_predictions("data")
identity, parameters, expression = mhr_inputs(predictions)

with torch.no_grad():
    vertices, _ = model(identity, parameters, expression)   # one pass for everybody

camera, residual = fit_camera(predictions)                  # exact intrinsics from keypoints
in_camera = [to_camera_space(vertices[i].numpy(), p) for i, p in enumerate(predictions)]
```

Units and axes are the two traps: SAM 3D Body works in **metres, y down, z into the scene**; MHR in **centimetres, y up**. `to_camera_space` handles it, and every `.npz` key is documented in `mhr_kit/sam3d.py`.

</details>

<details>
<summary><b>Animate, and write a video</b></summary>

Interpolate keyframes and skin the whole clip in one batch — the corrective network is what costs, and batching amortises it.

```python
from mhr_kit.params import interpolate_keyframes, pose

animation = interpolate_keyframes([pose(model, "relaxed"), pose(model, "arms_up")], 20, loop=True)
with torch.no_grad():
    vertices, _ = model(torch.zeros(1, 45), animation, None)   # (frames, V, 3)

frames = [render.render(v, faces(model), camera) for v in vertices.numpy()]
render.save_video(frames, "clip.mp4", fps=24)                  # GIF fallback if no ffmpeg
```

</details>

<details>
<summary><b>Render without a GPU</b></summary>

`mhr_kit/render.py` projects the mesh through a pinhole camera, culls backfaces, sorts triangles back to front, and draws them with Matplotlib's Agg backend. No OpenGL, no display, ~0.1 s per 480×600 frame at LOD 1. Pass `background=<HxWx3 array>` to composite over a photograph, `alpha=0.85` to see through the mesh.

</details>

<details>
<summary><b>Fit parameters to something</b></summary>

The model is differentiable end to end, so fitting is an ordinary optimisation (example 9). **Fit at LOD 4, evaluate at LOD 1** — parameters are identical across LODs and the coarse mesh is ~5× cheaper. For production-grade fitting, pymomentum also ships a fast C++ Gauss-Newton solver.

</details>

<details>
<summary><b>Take a mesh into Blender / MeshLab</b></summary>

`save_mesh(vertices, model, "person.obj")` writes any format trimesh supports. Vertex order and triangles are fixed per LOD, so per-vertex data stays comparable across poses and bodies.

</details>

<details>
<summary><b>Convert to or from SMPL / SMPL-X</b></summary>

Not reimplemented here — upstream has it in [`tools/mhr_smpl_conversion`](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion), both directions. It needs the SMPL/SMPL-X model files, which require registering on their own sites and so can't be downloaded automatically.

</details>

<details>
<summary><b>Ship MHR without pymomentum</b></summary>

Use the TorchScript export (demo 7): one 696 MB file, PyTorch only. Caveats: LOD 1 only, no triangle indices or joint names, and it does **not** broadcast a single identity row over the batch — pass one row per pose.

</details>

---

## Performance and memory

Apple Silicon Mac, CPU only, LOD 1:

| operation | cost |
| --- | --- |
| `load_mhr(lod=1)` | 1.6 s, ~1.4 GB resident |
| forward, batch 1 | 15 ms |
| forward, batch 8 | 6.8 ms/mesh |
| forward, batch 64 | 2–3 ms/mesh |
| forward without correctives, batch 8 | 3.4 ms/mesh (and no 664 MB matrix to load) |
| render 480×600 | ~0.10 s/frame |
| render 1920×890, four people | ~0.4 s |

Three habits that pay off: **batch aggressively** (the correctives are a dense `(125×24) × (V×3)` matmul, and per-frame calls waste most of it); **watch the correctives for memory**, not the mesh (664 MB at LOD 1, 2.7 GB at LOD 0); and **prototype at LOD 4**, which loads in 0.2 s and is plenty for fitting, previews and tests.

**CUDA:** swap `pymomentum-cpu` for `pymomentum-gpu` in `requirements.txt` and pass `device="cuda"`. On Apple Silicon stay on CPU — there's no MPS backend for pymomentum, and CPU inference is milliseconds anyway.

---

## Troubleshooting

| symptom | cause and fix |
| --- | --- |
| `No matching distribution found for pymomentum-cpu` | Python 3.11 or older, or an unsupported platform. Use 3.12/3.13 — `setup.sh` checks. |
| `FileNotFoundError: missing assets for LOD n` | `python -m mhr_kit.assets --lod n`, or let `load_mhr` fetch it. |
| `server did not honour range request` | a proxy or mirror stripping `Range`. Fall back to `mhr-download-assets`. |
| Download hangs before the first byte | the GitHub release CDN handshake. All reads share one connection, so you pay it once. |
| MP4 missing, GIF written instead | `imageio-ffmpeg` absent or its encoder failed; the renderer falls back automatically. |
| Killed / swapping while loading LOD 0 or 1 | 2.7 GB / 664 MB of correctives. Use `--lod 4`, or `load_mhr(correctives=False)`. |
| `Sizes of tensors must match` from TorchScript | it doesn't broadcast identity; pass `identity.repeat(batch, 1)`. |
| CUDA ignored | you installed `pymomentum-cpu`. |

---

## Data, licences, citation

**Further datasets** used or worth knowing about: [ASPset-510](https://archive.org/download/aspset510) · [Microsoft DAViD](https://github.com/microsoft/DAViD).

**Licences.** MHR — model, code and release assets — is Apache-2.0 (Meta); the asset archive carries its own `LICENSE.txt`. pymomentum / Momentum is Meta, open source, from PyPI. SAM 3D Body is a separate repository and licence, and only its *output* files are used here — the four examples come from the MHR repo itself. Everything original in this repository (`mhr_kit/`, `demos/`, `mhr_posing/`, the pipelines, `examples.py`) is Apache-2.0, to match. SMPL / SMPL-X model files are **not** included and cannot be redistributed.

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

<div align="center"><sub>Not affiliated with Meta. This is a workspace built around their open model.</sub></div>
