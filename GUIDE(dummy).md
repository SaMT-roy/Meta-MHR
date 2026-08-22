# MHR Toolkit — Complete Guide

A practical guide to **MHR (Momentum Human Rig)**, Meta's parametric 3D human
body model, and to `mhr_toolkit` — the documented wrapper in this repository
that adds asset management, named parameters, headless rendering, video output
and SAM 3D Body integration on top of it.

- Paper: [arXiv:2511.15586](https://arxiv.org/abs/2511.15586)
- Upstream code: [facebookresearch/MHR](https://github.com/facebookresearch/MHR)
- Licence: Apache 2.0 (code), see `assets/LICENSE.txt` for the model assets

---

## Contents

1. [What MHR is (and is not)](#1-what-mhr-is-and-is-not)
2. [Installation](#2-installation)
3. [Assets: downloading only what you need](#3-assets-downloading-only-what-you-need)
4. [Quick start](#4-quick-start)
5. [The parameter spaces](#5-the-parameter-spaces)
6. [Posing the rig](#6-posing-the-rig)
7. [Rendering](#7-rendering)
8. [Animation and video](#8-animation-and-video)
9. [Images and video in: SAM 3D Body](#9-images-and-video-in-sam-3d-body)
10. [Levels of detail](#10-levels-of-detail)
11. [The TorchScript backend](#11-the-torchscript-backend)
12. [Common use cases](#12-common-use-cases)
13. [Performance](#13-performance)
14. [Troubleshooting](#14-troubleshooting)
15. [API reference](#15-api-reference)
16. [Project layout](#16-project-layout)
17. [Licensing](#17-licensing)
18. [Citation](#18-citation)

---

## 1. What MHR is (and is not)

MHR is a **parametric body model** — a differentiable function

```
(identity[45], pose[204], expression[72])  ->  (vertices[V,3], skeleton[127,8])
```

It is **not** a pose estimator. Nothing in MHR or in this toolkit looks at
pixels. If you want MHR parameters *from* a photograph or a video, that is
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)'s job; MHR is the
model whose parameters SAM 3D Body predicts. Section 9 covers that pipeline,
and it works out of the box against the example predictions bundled here.

### How it differs from SMPL / SMPL-X

The paper positions MHR as combining **ATLAS**'s decoupled skeleton/shape
parameterisation with a production-style rig and **Momentum**-inspired
non-linear pose correctives. Concretely:

| | SMPL/SMPL-X | MHR |
|---|---|---|
| Pose parameterisation | axis-angle per joint | decomposed rig channels (`bend`/`twist`/`lean`) |
| Skeleton dimensions | implied by shape betas | independent `scale_*` channels |
| Pose-dependent deformation | linear pose blendshapes | non-linear MLP correctives |
| Joints | 24 / 55 | 127 |
| Levels of detail | one | seven (LOD 0–6) |

### The forward pass, step by step

Reading `mhr/mhr.py` upstream, `MHR.forward` does four things:

1. **Shape.** `identity` and `expression` are concatenated and run through the
   linear blendshape basis, giving a rest-pose mesh.
2. **Rig solve.** `pose` goes through the *parameter transform* (loaded from
   `compact_v6_1.model`) into per-joint parameters, then forward kinematics
   produces a 127-joint skeleton state.
3. **Non-linear correctives.** Each joint's local Euler rotation is converted to
   a 6-D rotation representation, and the 750-dimensional result
   (125 joints × 6) is fed to a `SparseLinear → ReLU → Linear` network that
   predicts a per-vertex offset. That offset is added to the rest mesh. This is
   MHR's headline feature: it captures muscle bulging and skin sliding that a
   linear model cannot.
4. **Skinning.** The corrected rest mesh is linear-blend-skinned by the skeleton
   state.

You can turn step 3 off with `forward(..., apply_correctives=False)` and measure
what it contributes — `demos/08_lod_and_correctives.py` does exactly that.

### Coordinate conventions

Measured on the released LOD-1 rig, and worth pinning up somewhere:

- **Units are centimetres** for vertices and joint positions. A default-shape
  adult is 172.7 cm tall. The one exception is `root_tx/ty/tz`, which are in
  **decimetres** — `root_ty = 1.0` raises the body by 10 cm.
- **+X is the model's own left, +Y is up, +Z is the direction it faces.** A
  camera at azimuth 0 looks the model in the face.
- The origin sits on the ground between the feet.
- Vertices are `[batch, V, 3]`; the skeleton state is `[batch, 127, 8]`, each row
  `(tx, ty, tz, qx, qy, qz, qw, scale)`.

---

## 2. Installation

### Requirements

| | |
|---|---|
| Python | **3.12 or 3.13** — `pymomentum` publishes cp312/cp313 wheels only |
| OS | macOS (arm64), Linux (x86-64), Windows (x86-64) |
| GPU | optional; everything here runs on CPU |
| Disk | ~700 MB for a LOD-1 setup (27 MB downloaded, the corrective basis unpacks large) |

### One command

```bash
./setup.sh
```

That creates `.venv313`, installs the dependencies, downloads the LOD-1 assets,
and verifies a forward pass. Useful variants:

```bash
./setup.sh --torchscript          # also fetch mhr_model.pt
./setup.sh --lods 1 4 6           # several levels of detail
./setup.sh --no-assets            # dependencies only
./setup.sh --python python3.12    # pick an interpreter
```

### Manual

```bash
python3.13 -m venv .venv313
source .venv313/bin/activate
pip install -r requirements.txt
python -m mhr_toolkit --lods 1
```

### Minimal install (no `pymomentum`)

If `pymomentum` will not install on your platform, the TorchScript backend needs
only PyTorch:

```bash
pip install torch numpy pillow
python -m mhr_toolkit --lods 1 --torchscript
python -c "from mhr_toolkit import load_torchscript_mhr; print(load_torchscript_mhr().forward().vertices.shape)"
```

See [section 11](#11-the-torchscript-backend) for what you give up.

---

## 3. Assets: downloading only what you need

Upstream distributes one `assets.zip` (190 MB compressed, 4.8 GB unpacked)
containing all seven LODs plus the TorchScript export. `mhr-download-assets`
fetches the whole thing.

`mhr_toolkit.assets` instead performs a **ranged download**. A ZIP stores its
central directory at the end of the file and each member as an independently
addressable byte range, and GitHub's release CDN honours HTTP `Range`, so the
downloader reads the last 64 KiB, works out which byte ranges it needs, and
fetches only those.

```bash
python -m mhr_toolkit --list                    # every member, with sizes
python -m mhr_toolkit --dry-run --lods 0 1      # what would this cost?
python -m mhr_toolkit --lods 1                  # ~27 MB
python -m mhr_toolkit --lods 1 --torchscript    # ~53 MB
python -m mhr_toolkit --lods 1 --no-correctives # ~8 MB, linear model only
```

What a LOD-1 setup actually pulls:

| Member | Download | On disk | Needed for |
|---|---:|---:|---|
| `compact_v6_1.model` | 0.03 MB | 0.03 MB | parameter transform, skeleton |
| `lod1.fbx` | 7.4 MB | 7.9 MB | rig, mesh, identity + expression blendshapes |
| `corrective_blendshapes_lod1.npz` | 19.0 MB | 664 MB | non-linear pose correctives |
| `corrective_activation.npz` | 0.23 MB | 3.3 MB | sparse activation mask of the corrective MLP |
| `LICENSE.txt` | — | 0.01 MB | asset licence |
| **Total** | **26.6 MB** | **675 MB** | |

Compare: the LOD-0 corrective basis alone is 84 MB compressed and 2.6 GB
unpacked, and `mhr_model.pt` is another 26 MB. Skipping what you do not use is
worth it.

In code, every entry point downloads on demand and is idempotent:

```python
from mhr_toolkit import load_mhr, ensure_assets, plan_for

print(plan_for(lods=[1, 4]).summary())   # "…, 33.4 MB download -> 764.0 MB on disk"
ensure_assets([1, 4])                    # fetch (skips anything already present)
model = load_mhr(lod=1)                  # also calls ensure_assets internally
```

---

## 4. Quick start

```python
from mhr_toolkit import load_mhr, PoseBuilder, Renderer, Camera, save_image

model = load_mhr(lod=1)                       # downloads ~27 MB on first run
print(model.describe())

pose = PoseBuilder(model).preset("t_pose").set(l_elbow_bend=0.8).build()
out = model.forward(pose=pose)

out.save_mesh("tpose.ply")
image = Renderer(width=512, height=768).render(
    out.vertices_np(), out.faces, Camera.auto_frame(out.vertices_np(), azimuth=20)
)
save_image(image, "tpose.png")
```

Then work through the demos in order:

| Demo | Subject |
|---|---|
| `demos/01_hello_mhr.py` | load, evaluate, export, render |
| `demos/02_identity_explorer.py` | the 45-D shape space, shape vs. skeleton |
| `demos/03_pose_and_expression.py` | rig channels, presets, sign conventions, faces |
| `demos/04_image_reconstruction.py` | **images**: SAM 3D Body → MHR → overlay |
| `demos/05_video_animation.py` | **video**: walk cycle, keyframes, turntable |
| `demos/06_video_from_sam3d.py` | **video**: per-frame predictions → clean clip |
| `demos/07_torchscript_backend.py` | the dependency-free backend, benchmarks |
| `demos/08_lod_and_correctives.py` | LOD resampling, corrective ablation |

And `examples/examples.py` holds 25 short recipes:

```bash
python examples/examples.py --list
python examples/examples.py 8 19        # by number
python examples/examples.py sam3d       # by name
```

---

## 5. The parameter spaces

### Identity — `[45]`

Zero-mean, roughly unit-variance shape coefficients, split into three blocks:

| Block | Indices | Controls |
|---|---|---|
| body | 0–19 | torso, limb and overall body shape |
| head | 20–39 | facial structure and skull shape |
| hands | 40–44 | hand shape |

`±1` is an ordinary body, `±2.5` is distinctive, beyond `±4` gets strange.

> **The decoupling that catches everyone out.** Identity coefficients reshape
> the **surface** only. They do *not* change bone lengths, so no identity
> coefficient makes the body taller. **Stature lives in the `scale_*` channels
> of the pose vector.** Measured on LOD 1: pushing identity coefficient 0 to ±2
> changes torso depth by ~6 cm and total height by under a millimetre, whereas
> `scale_uplegs = 0.4` adds ~4 cm of height and changes depth not at all. This
> is ATLAS's decoupled parameterisation and it is a feature — you can retarget a
> body shape onto a different skeleton — but it surprises people who sample
> random identities and find every body is 172.7 cm.

```python
from mhr_toolkit import identity_vector
from mhr_toolkit.params import IDENTITY_BLOCKS

body_only = identity_vector(coefficients={0: 1.5, 3: -0.8})

vector = identity_vector()
vector[IDENTITY_BLOCKS["head"]] = rng.normal(size=20)   # new face, same body
```

### Pose — `[204]`

Not a flat list of joint rotations. The rig is *decomposed* into anatomically
meaningful channels, which is what makes it hand-authorable. The layout,
in vector order:

| Indices | Group | Examples |
|---|---|---|
| 0–5 | root transform | `root_tx…tz` (**decimetres**, 1 unit = 10 cm), `root_rx…rz` (rad) |
| 6–29 | spine, neck, head | `spine_twist0`, `spine_lean1`, `spine_bend0`, `neck_bend`, `head_twist` |
| 30–49 | arms | `l_clavicle_rz`, `r_uparm_ry`, `l_elbow_bend`, `r_wrist_rz` |
| 50–67 | legs and feet | `l_upleg_rz`, `r_knee_bend`, `l_foot_bend`, `r_ball_bend` |
| 68–123 | fingers | `l_index1_rz`, `r_thumb2_rz`, … (25 per hand) |
| 124–131 | ankle detail | `l_talocrural_rx_flexible`, `r_subtalar_rz_flexible` |
| 132–203 | scales | `scale_uplegs`, `scale_shoulder_width`, `scale_r_index1_length` |

All angles are radians. Naming conventions:

- `_bend` / `_twist` / `_lean` — decomposed single-axis rotations
- `_r{x,y,z}` — a raw Euler channel
- `_flexible` — a secondary/soft channel the rig blends over the primary one
- `scale_*` — bone lengths and proportions

### Expression — `[72]`

Facial blendshape weights, roughly `[-1, 1]`. The released assets ship **no
semantic names** for them, so `mhr_toolkit.params.EXPRESSION_PRESETS` contains
index picks found by sweeping coefficients — illustrative, not a FACS mapping.
Peak displacement for a single coefficient at weight 1.0 is around 0.5–1.0 cm,
confined to the face.

---

## 6. Posing the rig

### Named access

```python
from mhr_toolkit import ParamIndex, PoseBuilder

index = ParamIndex(model.param_names)
index["l_elbow_bend"]          # 46
index.find("knee")             # ['r_knee_bend', 'l_knee_bend']
index.find("^l_thumb")         # regex works too
index.group("left_arm")        # a semantic group
index.describe(pose)           # the largest-magnitude entries of a pose vector
```

Groups: `root`, `spine`, `neck`, `head`, `shoulder`, `arms`, `left_arm`,
`right_arm`, `hands`, `left_hand`, `right_hand`, `legs`, `left_leg`,
`right_leg`, `feet`, `scales`.

### Building a pose

`PoseBuilder` is fluent; every method returns `self`.

```python
pose = (
    PoseBuilder(model)
    .preset("a_pose")
    .set(l_elbow_bend=1.2, r_elbow_bend=0.4)
    .add(neck_twist=0.1)
    .set_group("hands", 0.0)
    .translate(y=0.3)            # decimetres: 3 cm up, unlike vertices which are cm
    .rotate(y=0.3)               # radians
    .build()                     # -> np.ndarray [204]
)
```

### Sign conventions

**Left and right channels are already mirrored inside the rig.** Setting
`l_uparm_rz` and `r_uparm_rz` to the *same* value gives a symmetric pose.
Most rigs work the other way round, so this is the single most common source
of accidentally-asymmetric poses.

Measured on the released LOD-1 rig (axes: +X = model's left, +Y = up,
+Z = the way it faces):

| Channel | Positive value does |
|---|---|
| `*_uparm_rz` | raises the arm forwards and up (shoulder flexion) |
| `*_uparm_ry` | abducts the arm out to the side and back; negative crosses the chest |
| `*_uparm_twist` | internally rotates the upper arm |
| `*_elbow_bend` | flexes the elbow |
| `*_knee_bend` | flexes the knee (heel back) |
| `*_upleg_rz` | **extends** the hip; negative flexes it (knee forwards) |
| `*_upleg_ry` | adducts; negative abducts (leg out sideways) |
| `*_foot_bend` | plantarflexes (toes down) |
| `spine_bend*` | leans the torso forwards |
| `spine_lean*` | leans to the model's right |
| `spine_twist*` | rotates the torso to the model's left |
| finger `*_rz` | curls the digit into the palm |

Do not take that table on faith — **measure**. Perturb a channel and read off
how a downstream joint moved:

```python
joints = {name: i for i, name in enumerate(model.joint_names)}
rest = model.forward().joint_positions[0].numpy()
moved = model.forward(
    pose=PoseBuilder(model).set(l_elbow_bend=1.0).build()
).joint_positions[0].numpy()
print(moved[joints["l_wrist"]] - rest[joints["l_wrist"]])   # -> [-9.3 +25.7 +13.8]
```

`examples/examples.py 8` and `demos/03_pose_and_expression.py` do this
systematically.

### Presets

`rest`, `a_pose`, `t_pose`, `arms_up`, `hands_on_chest`, `sitting`, `squat`,
`walk_contact`, `fists`, `twist`.

```python
from mhr_toolkit import POSE_PRESETS, make_pose

pose = make_pose(model, "squat", spine_bend0=0.4)   # preset plus overrides
print(sorted(POSE_PRESETS))
```

These were hand-tuned by sweeping channels and looking at the result — they are
illustrative starting points, not motion-capture ground truth. Note that
`t_pose` is not perfectly horizontal: the shoulder channels alone cannot reach a
true T without clavicle help, and forcing them there collapses the shoulder.

---

## 7. Rendering

`mhr_toolkit.render` is a **z-buffered software rasteriser in numpy**. No GL
context, no OSMesa, no EGL — it works headless, in containers, in CI, and on
macOS, where the usual offscreen-GL workarounds do not apply.

```python
from mhr_toolkit import Renderer, Camera, save_image, tile

renderer = Renderer(
    width=512, height=768,
    palette="clay",        # clay | slate | bone | mint | amber | paper
    supersample=2,         # render 2x and box-downsample; 4x the cost, clean edges
    smooth=True,           # per-vertex normals (False = faceted)
    ambient=0.26, fill=0.22, rim=0.28,
)
camera = Camera.auto_frame(vertices, azimuth=20.0, elevation=8.0)
image = renderer.render(vertices, faces, camera)             # HxWx3 uint8
```

### Cameras

```python
Camera.auto_frame(vertices, azimuth=0)      # head on; azimuth turns to the model's left
Camera.auto_frame(vertices, azimuth=90)     # profile
Camera(eye=(30, 160, 110), target=(0, 158, 0), fov_y_deg=18)   # manual
orbit_cameras(vertices, frames=60)          # a full turntable
```

`auto_frame` derives the distance from the mesh's bounding sphere, so framing is
correct for any body size or LOD. Frame against *all* frames of an animation
(`np.concatenate(all_vertices)`) so the shot does not pop mid-clip.

### Modes and extras

```python
renderer.render(v, f, cam, mode="depth")             # normalised depth
renderer.render(v, f, cam, mode="normal")            # camera-space normals as RGB
renderer.render(v, f, cam, vertex_colors=heatmap)    # per-vertex albedo
renderer.render(v, f, cam, return_alpha=True)        # HxWx4, for compositing
```

`vertex_colors` **replaces** the palette albedo rather than tinting it, so
heatmaps keep their contrast. Pair it with a high `ambient` and low `rim` so
shading does not fight the colour scale.

Helpers: `tile(images, columns)` for contact sheets, `merge_meshes([...])` to
draw several bodies with correct mutual occlusion, `overlay(photo, rgba)` to
composite onto a photograph, `draw_points` / `draw_box` for 2D annotations.

### Cost

Roughly 0.3 s per 384×576 frame at `supersample=1` on one CPU core, ~0.8 s at
`supersample=2`. The loop runs once per *visible* triangle with everything
inside vectorised over that triangle's bounding box, so cost scales with
triangle count rather than pixel count. Use `supersample=2` for stills and 1
for video, where motion hides the aliasing.

The renderer does **not** do textures, shadows or transparency. If you need
those, export `.glb` and render in Blender.

---

## 8. Animation and video

An animation is a `[T, 204]` array. Build the trajectory, evaluate it as one
batch, render, encode.

```python
from mhr_toolkit import interpolate, lerp_poses, write_video
from mhr_toolkit.params import oscillate, sweep, smooth_sequence, resample_sequence

# keyframes, with smoothstep easing and a seamless loop
poses = interpolate(
    [PoseBuilder(model).preset(p).build() for p in ("rest", "squat", "arms_up")],
    frames_per_segment=24, ease=True, loop=True,
)

# procedural: sine waves on named channels, right limbs in antiphase
poses = oscillate(
    base_pose, {"l_upleg_rz": -0.5, "r_upleg_rz": -0.5}, index,
    frames=48, phase={"r_upleg_rz": np.pi},
)

out = model.forward_chunked(pose=poses, chunk_size=8)   # bounded memory
frames = renderer.render_sequence(
    [out.vertices_np(i) for i in range(len(out))], out.faces, camera
)
write_video(frames, "walk.mp4", fps=30)
write_video(frames, "walk.gif", fps=15)
```

Notes:

- `interpolate` operates componentwise on Euler channels. That is right for the
  small-to-moderate rotations these channels carry; for large **root**
  rotations you would want quaternion slerp instead.
- `forward_chunked` chunks over whichever input carries the frame axis —
  identity for a morph, pose for an animation, expression for a talking head.
  The corrective MLP materialises a dense `[B, V*3]` intermediate, so a
  thousand-frame LOD-1 batch would want tens of gigabytes; chunking keeps memory
  flat and the output identical.
- `write_video` tries imageio + bundled ffmpeg, then a system `ffmpeg`, then
  Pillow for GIF. Frames are cropped to a multiple of 16 for H.264 so encoders
  do not silently rescale them.

---

## 9. Images and video in: SAM 3D Body

This is the honest answer to "how do I use MHR on an image?". You run
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body), which predicts
MHR parameters, and MHR turns those back into geometry.

```
photo/video ──▶ SAM 3D Body ──▶ .npz per person per frame ──▶ MHR ──▶ mesh
                                        │
                                        └── shape_params[45], mhr_model_params[204],
                                            expr_params[72], pred_cam_t[3], keypoints…
```

`data/sam3d_body_outputs/` contains four example predictions from the upstream
repo — four people detected in one photograph — so the demos run with no extra
downloads.

### Reading predictions

```python
from mhr_toolkit import load_mhr, load_predictions

model = load_mhr(lod=1)              # SAM 3D Body predicts LOD-1 meshes
people = load_predictions("data/sam3d_body_outputs")

person = people[0]
out = model.forward(person.identity, person.pose, person.expression)
print(person.reconstruction_error(out.vertices_np()))    # ~7e-06 cm
```

That last check matters: every prediction stores both its parameters *and* the
mesh they produced, so re-running MHR on the parameters must reproduce the mesh.
A large error means the parameters and the model are out of sync (usually a LOD
mismatch), and nothing downstream will line up.

### Coordinate conventions

Two frames to keep straight:

- **MHR** — centimetres, +X the subject's left, +Y up, +Z the way it faces.
- **SAM 3D Body's camera frame** — metres, Y-down, Z-forward (the standard
  computer-vision convention).

Related by a 180° rotation about X plus a factor of 100:

```
vertices_camera_metres = mhr_vertices_cm * (1, -1, -1) / 100 + pred_cam_t
```

`to_camera_space()` and `to_render_space()` do the conversion in each direction.

### Recovering the camera

Intrinsics are not stored, but they are exactly recoverable: `pred_keypoints_2d`
is the perspective projection of `pred_keypoints_3d + pred_cam_t`, so a
two-parameter least-squares fit gets them to floating-point precision.

```python
from mhr_toolkit.sam3d import any_intrinsics, camera_for_image, to_render_space

intrinsics = any_intrinsics(people)
width, height = intrinsics.image_size          # (1920, 890) for the bundled data
print(intrinsics.residual)                     # 1e-4 px — exact, as expected

camera = camera_for_image(intrinsics, width, height)
placed = [to_render_space(out.vertices_np(i), p.cam_t) for i, p in enumerate(people)]
```

### Overlaying on the photograph

```python
from mhr_toolkit.render import merge_meshes, overlay

scene_v, scene_f = merge_meshes([(v, out.faces) for v in placed])   # shared z-buffer
rgba = Renderer(width=width, height=height).render(
    scene_v, scene_f, camera, return_alpha=True
)
save_image(overlay(photo, rgba, alpha=0.8), "overlay.png")
```

`demos/04_image_reconstruction.py` runs this end to end and draws the detection
boxes and keypoints on top, so you can verify the alignment by eye. Pass
`--image your_photo.jpg` to composite over the real frame.

### Video

Per-frame predictions replayed naively look bad for two reasons that are not the
model's fault:

1. **Identity flicker** — identity is re-estimated every frame, so the body
   subtly changes shape. Identity is constant for one person, so average it.
2. **Pose jitter** — per-frame estimates are independent, so high-frequency
   noise shimmers. A short centred moving average removes most of it.

```python
from mhr_toolkit.sam3d import mean_identity
from mhr_toolkit.params import smooth_sequence, resample_sequence

frames = load_predictions("my_video_outputs")           # zero-pad your filenames!
identity = mean_identity(frames)                        # kills the flicker
poses = smooth_sequence(np.stack([f.pose for f in frames]), window=5)
poses = resample_sequence(poses, 240)                   # retime to 10 s at 24 fps
out = model.forward_chunked(identity, poses, chunk_size=8)
```

On the bundled data this cut mean frame-to-frame pose change from 0.13 to
0.03 rad. `demos/06_video_from_sam3d.py` runs the whole pipeline and offers
`--camera-space` to keep the subject's original translation through the frame
rather than re-centring it.

### Converting to SMPL / SMPL-X

Out of scope for this toolkit, but upstream ships it:
`upstream_MHR/tools/mhr_smpl_conversion/`. It needs `smplx` plus the official
SMPL/SMPL-X model files (registration required) and offers PyMomentum
(Gauss-Newton, CPU, temporally aware) and PyTorch (GPU, per-frame) backends.

> **Licensing.** The conversion code is Apache 2.0, but the SMPL/SMPL-X model
> files it needs are **not** — they are non-commercial research licences by
> default, and commercial use requires a separate agreement with Max Planck.
> This is the only path in this guide that leaves permissively-licensed
> territory, and it is entirely optional. See [LICENSES.md](LICENSES.md).

> **SAM 3D Body** itself is not open source either: it ships under Meta's custom
> SAM License. Commercial use *is* permitted, but with acceptable-use
> restrictions (no military/weapons applications, trade-control compliance,
> research attribution). MHR and its weights are plain Apache 2.0 and carry
> none of this.

---

## 10. Levels of detail

Seven rigs ship. Measured vertex counts:

| LOD | Vertices | Rig download | Correctives (download / unpacked) |
|---:|---:|---:|---:|
| 0 | 73,639 | 28.7 MB | 84.1 MB / 2651 MB |
| 1 | 18,439 | 7.4 MB | 19.0 MB / 664 MB |
| 2 | 10,661 | 3.8 MB | 14.1 MB / 384 MB |
| 3 | 4,899 | 1.7 MB | 6.8 MB / 176 MB |
| 4 | 2,461 | 0.8 MB | 3.4 MB / 89 MB |
| 5 | 971 | 0.5 MB | 1.0 MB / 35 MB |
| 6 | 595 | 0.3 MB | 0.9 MB / 21 MB |

LOD 1 is the standard research resolution and what SAM 3D Body predicts.

### Resampling instead of downloading

Upstream also ships barycentric mapping files, so a mesh evaluated at LOD 1 can
be resampled onto any other LOD with one matrix multiply:

```python
from mhr_toolkit import LODConverter

converter = LODConverter()            # reads data/lod_mappings/
coarse = converter.convert(out.vertices_np(), target_lod=5, source_faces=model.faces)
faces  = converter.faces_for(5)       # the target LOD's topology (small .fbx download)
```

Resampling is not the same as evaluating: the target LOD has its own fitted
correctives, which an interpolated LOD-1 surface cannot reproduce. Measured
LOD 1 → LOD 6 against a native LOD-6 evaluation of the same pose: median error
0.017 cm, mean 0.13 cm, max 6.7 cm. **Resample for display, evaluate natively
for accuracy.**

---

## 11. The TorchScript backend

`assets/mhr_model.pt` is a traced graph of the whole forward pass. It needs
nothing but PyTorch.

```python
from mhr_toolkit import load_torchscript_mhr

scripted = load_torchscript_mhr()               # downloads ~26 MB
out = scripted.forward(identity, pose, expression)
```

| | Full model | TorchScript |
|---|---|---|
| Dependencies | `pymomentum` + assets | `torch` only |
| Levels of detail | 0–6 | 1 only |
| Parameter / joint names | yes | no |
| Rest mesh, skinning weights | yes | no |
| `apply_correctives=False` | yes | no (always on) |
| Mesh topology | yes | not in the export |

Measured parity on random inputs: mean |Δ| 7.0e-06 cm, max 7.6e-05 cm — the same
weights in float32, differing only through operator fusion in the trace.

Two gotchas:

- **The export carries no face indices.** `load_torchscript_mhr` borrows them
  from `lod1.fbx` if one is present; otherwise `scripted.faces` is `None` and
  you are limited to vertices, joint positions and point clouds. Download just
  the rig (`--lods 1 --no-correctives`, ~8 MB) to get topology.
- **It cannot broadcast a single identity across a batch of poses.** The trace
  builds its parameter padding with `zeros_like(identity)`, so all three inputs
  must share one batch size. `TorchScriptMHR.forward` expands them for you.

Parameter *names* live in the full model, but the vectors are interchangeable —
build a pose with `PoseBuilder(full_model)` and evaluate it on TorchScript.

---

## 12. Common use cases

### Generate a synthetic-data set

```python
rng = np.random.default_rng(0)
identities = 0.9 * rng.normal(size=(64, 45)).astype(np.float32)
poses = np.stack([PoseBuilder(model).preset("a_pose").randomize("arms", 0.3, rng).build()
                  for _ in range(64)])
out = model.forward_chunked(identities, poses, chunk_size=8)
for i in range(len(out)):
    save_image(renderer.render(out.vertices_np(i), out.faces), f"synth/{i:04d}.png")
    out.save_mesh(f"synth/{i:04d}.ply", i)
```

Remember to vary `scale_*` too if you want height variation — see
[section 5](#5-the-parameter-spaces).

### Take body measurements

Everything is in centimetres, so measurements come straight off the geometry.
Joint positions give cleaner skeletal measurements than the surface:

```python
joints = {n: i for i, n in enumerate(model.joint_names)}
v, j = out.vertices_np(), out.joint_positions[0].numpy()
height    = v[:, 1].max() - v[:, 1].min()
depth     = np.ptp(v[:, 2])
shoulders = abs(j[joints["l_uparm"]][0] - j[joints["r_uparm"]][0])
```

### Export to Blender / Maya / Unreal

```python
out.save_mesh("body.glb")                        # ply / obj / glb, via trimesh
out.save_meshes("sequence/", stem="walk")        # numbered mesh sequence
```

### Fit MHR to your own keypoints

Not built in, but the model is differentiable end to end, so the standard
approach works: optimise `pose` (and optionally `identity`) with Adam against a
2D reprojection loss on the joint positions, using the intrinsics recovered as
in [section 9](#9-images-and-video-in-sam-3d-body). Add a small L2 prior on
`identity` to keep bodies plausible. For a production-grade solver, use
`pymomentum`'s Gauss-Newton IK directly on `model.character`.

### Retarget a captured body onto a different build

Because shape and skeleton are decoupled, keep the captured `identity` and edit
only the `scale_*` channels — or vice versa, to put a new body shape on a
captured skeleton.

### Run headless in CI

```bash
pip install torch numpy pillow
python -m mhr_toolkit --lods 1 --torchscript
python -m pytest tests/
```

No GL, no display, no `pymomentum`.

---

## 13. Performance

Measured on an Apple M-series CPU, LOD 1, single core.

| Operation | Cost |
|---|---|
| `load_mhr(lod=1)` | ~1.5 s |
| `forward`, batch 1 | ~21 ms |
| `forward`, batch 8 | ~7 ms/frame |
| `forward`, batch 32 | ~3 ms/frame |
| `render`, 384×576, `supersample=1` | ~0.35 s |
| `render`, 384×576, `supersample=2` | ~0.8 s |
| LOD resample, any target | < 10 ms |

Practical advice:

- **Batch.** Per-frame cost falls ~7× from batch 1 to batch 32; the corrective
  MLP is one big matmul and batching amortises it.
- **Rendering dominates**, not evaluation. A 48-frame clip is ~0.3 s of model
  and ~20 s of rasterising. Drop `supersample` to 1 and the resolution to taste.
- **Chunk long sequences** with `forward_chunked` to bound memory.
- **CUDA** helps the model, not the renderer (which is numpy). MPS is *not*
  auto-selected: `pymomentum` builds its skeleton state on CPU tensors and
  several corrective ops fall back to CPU anyway, so MPS is usually slower here.
  Pass `device="mps"` explicitly to try it.
- **Drop the correctives** (`pose_correctives=False`) for a big speed and
  download win when deformation quality does not matter — at the cost of the
  feature that distinguishes MHR.

---

## 14. Troubleshooting

**`ImportError: No module named pymomentum`**
`pymomentum` publishes cp312/cp313 wheels only. Check with
`python -c "import sys; print(sys.version)"`. Either use Python 3.12/3.13, or
switch to `load_torchscript_mhr()`, which needs only torch.

**`FileNotFoundError: Missing MHR assets: lod1.fbx`**
Run `python -m mhr_toolkit --lods 1`. If you are behind a proxy that blocks
`Range` requests, download the full archive with upstream's
`mhr-download-assets` and unzip it into `assets/`.

**Reconstruction error is huge for a SAM 3D Body prediction**
Almost always a LOD mismatch — SAM 3D Body predicts LOD-1 meshes, so load the
model with `lod=1`.

**My symmetric pose came out asymmetric**
Left/right channels are already mirrored inside the rig. Use the *same* value
on both sides, not opposite ones. See [section 6](#6-sign-conventions).

**Random identities all come out the same height**
Expected — identity does not change bone lengths. Drive `scale_*` as well. See
[section 5](#5-the-parameter-spaces).

**The overlay is offset from the people in the photo**
Check `intrinsics.residual` (should be ≪ 1 px) and that the photograph's
dimensions match `intrinsics.image_size`. `camera_for_image` warns if the
principal point is more than 2 px off centre, which the renderer cannot model.

**`RuntimeWarning: 'mhr_toolkit.assets' found in sys.modules`**
Harmless. Use `python -m mhr_toolkit` instead of `python -m mhr_toolkit.assets`.

**Renders are black**
The camera is probably inside or behind the mesh. Use
`Camera.auto_frame(vertices)` rather than a hand-placed camera, and remember
units are centimetres — a camera at `z=3` is 3 cm from the origin.

**`translate(y=10)` launched the body into orbit**
`root_t*` is in decimetres, not centimetres — that was a 1-metre move. One unit
is 10 cm. Vertex positions themselves are in centimetres; only this channel is
scaled.

**Video encoding warns about resizing**
Should not happen — `write_video` crops to a multiple of 16 first. If you see
it, you are calling `imageio` directly.

---

## 15. API reference

### `mhr_toolkit.model`

| Symbol | Purpose |
|---|---|
| `load_mhr(lod=1, device=None, folder=None, pose_correctives=True, download=True)` | Load the full model |
| `load_torchscript_mhr(device=None, folder=None, download=True)` | Load the traced export |
| `MHRModel.forward(identity, pose, expression, apply_correctives=True)` | Evaluate |
| `MHRModel.forward_chunked(..., chunk_size=16, progress=False)` | Evaluate in slices |
| `MHRModel.zeros(batch_size=1)` | Neutral parameter tensors |
| `MHRModel.random_identity(batch_size=1, scale=0.8)` | Sample plausible bodies |
| `MHRModel.param_names` / `.joint_names` / `.faces` / `.describe()` | Introspection |
| `MHROutput.vertices` / `.skeleton_state` / `.joint_positions` / `.joint_quaternions` | Results |
| `MHROutput.vertices_np(i)` / `.trimesh(i)` / `.save_mesh(path, i)` / `.save_meshes(dir)` | Export |
| `LOD_VERTEX_COUNTS`, `NUM_IDENTITY`, `NUM_MODEL_PARAMS`, `NUM_EXPRESSION`, `NUM_JOINTS` | Constants |

### `mhr_toolkit.params`

`ParamIndex`, `PoseBuilder`, `POSE_PRESETS`, `EXPRESSION_PRESETS`,
`IDENTITY_BLOCKS`, `make_pose`, `identity_vector`, `expression_vector`,
`lerp_poses`, `interpolate`, `oscillate`, `sweep`, `smooth_sequence`,
`resample_sequence`, `smoothstep`, `stack_identities`.

### `mhr_toolkit.render`

`Renderer`, `Camera`, `orbit_cameras`, `PALETTES`, `save_image`, `tile`,
`merge_meshes`, `overlay`, `draw_points`, `draw_box`.

### `mhr_toolkit.video`

`write_video`, `write_gif`, `write_frames`, `available_backends`.

### `mhr_toolkit.sam3d`

`load_prediction`, `load_predictions`, `SAM3DPrediction`, `Intrinsics`,
`infer_intrinsics`, `any_intrinsics`, `camera_for_image`, `to_camera_space`,
`to_render_space`, `mean_identity`, `stack_parameters`, `describe_all`.

### `mhr_toolkit.assets`

`ensure_assets`, `plan_for`, `list_archive`, `required_members`,
`asset_folder`, `AssetPlan`, `DEFAULT_ARCHIVE_URL`.

### `mhr_toolkit.lod`

`LODConverter` (`convert`, `convert_all`, `available_lods`, `faces_for`),
`mapping_search_paths`.

---

## 16. Project layout

```
mhr_env/
├── README.md                    project overview and quick start
├── setup.sh                     one-command environment + asset setup
├── requirements.txt             dependencies, with notes on what needs what
├── docs/GUIDE.md                this document
├── docs/LICENSES.md             licensing and commercial-use audit
├── mhr_toolkit/                 the library
│   ├── assets.py                ranged downloader (stdlib only)
│   ├── model.py                 load_mhr, MHRModel, TorchScriptMHR, MHROutput
│   ├── params.py                named parameters, presets, interpolation
│   ├── render.py                numpy software rasteriser
│   ├── video.py                 MP4 / GIF encoding
│   ├── sam3d.py                 SAM 3D Body predictions
│   └── lod.py                   level-of-detail resampling
├── demos/                       eight end-to-end walkthroughs
├── examples/examples.py         25 short recipes
├── tests/test_toolkit.py        pytest suite
├── data/
│   ├── sam3d_body_outputs/      four example predictions (from upstream)
│   └── lod_mappings/            barycentric LOD mappings (from upstream)
├── assets/                      downloaded model weights (gitignored)
├── outputs/                     demo output (gitignored)
└── upstream_MHR/                reference clone of facebookresearch/MHR
```

---

## 17. Licensing

Short version: **MHR's code and its model weights are both Apache 2.0**, which
is unusual for a parametric body model and means commercial use is fine. Every
Python dependency is MIT/BSD/Apache/PSF.

The one genuine restriction is the `ffmpeg` binary bundled by `imageio-ffmpeg`
for MP4 output: it is built with `--enable-gpl --enable-libx264` and is
therefore GPL-2.0-or-later. Running it locally is unencumbered; redistributing
it in a closed-source product is not. MP4 is optional — GIF and PNG output need
no ffmpeg at all.

Two referenced-but-not-installed items differ: **SAM 3D Body** (Meta SAM
License — commercial use permitted, acceptable-use restrictions apply) and
**SMPL/SMPL-X** (non-commercial research by default).

Full audit, including how to reproduce it: **[LICENSES.md](LICENSES.md)**.

---

## 18. Citation

```bibtex
@misc{MHR:2025,
      title={MHR: Momentum Human Rig},
      author={Aaron Ferguson and Ahmed A. A. Osman and Berta Bescos and others},
      year={2025},
      eprint={2511.15586},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2511.15586},
}
```

MHR is licensed under Apache 2.0. The model assets carry their own licence —
see `assets/LICENSE.txt`, which the downloader always fetches.
