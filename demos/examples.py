#!/usr/bin/env python3
"""Twelve short, self-contained examples of using MHR.

Each ``example_*`` function is independent and prints what it found, so you can
read one, run one, and copy it into your own code. The demos in ``demos/`` are the
same ideas at full length, with rendered output.

Run::

    python examples.py                 # list the examples
    python examples.py 4               # run one
    python examples.py 4 5 6           # run several
    python examples.py --all           # run all of them (~15 s once assets are cached)

Examples 3 and 12 download extra assets on first use (about 5 MB and 26 MB).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mhr_kit import render
from mhr_kit.assets import ASSET_DIR, TORCHSCRIPT_MEMBER, download_assets
from mhr_kit.model import (
    faces,
    joint_names,
    joint_positions,
    joint_rotations,
    load_mhr,
    model_summary,
    save_mesh,
)
from mhr_kit.params import (
    IDENTITY_GROUPS,
    POSES,
    interpolate_keyframes,
    model_parameters,
    parameter_index,
    parameter_names,
    pose,
    random_identity,
)
from mhr_kit.sam3d import fit_camera, load_predictions, mhr_inputs, to_camera_space

OUTPUT_DIR = Path("outputs/examples")

# Loading a model costs a second and a few hundred MB, so examples share one.
_CACHE: dict[int, object] = {}


def get_model(lod: int = 1):
    """Load (and remember) the MHR model at a given level of detail."""
    if lod not in _CACHE:
        _CACHE[lod] = load_mhr(lod=lod)
    return _CACHE[lod]


# --------------------------------------------------------------------------- #
# 1. The smallest complete program
# --------------------------------------------------------------------------- #
def example_01_minimal() -> None:
    """Build one mesh from the three parameter blocks."""
    model = get_model()

    identity = torch.zeros(1, 45)  # the average body
    parameters = torch.zeros(1, 204)  # the rest pose (a T-pose)
    expression = torch.zeros(1, 72)  # a neutral face

    with torch.no_grad():
        vertices, skeleton_state = model(identity, parameters, expression)

    print(f"vertices       {tuple(vertices.shape)}  in cm, +y up")
    print(f"skeleton_state {tuple(skeleton_state.shape)}  per joint: tx ty tz qx qy qz qw scale")
    print(f"stature        {vertices[0, :, 1].max() - vertices[0, :, 1].min():.1f} cm")
    print(model_summary(model, lod=1))


# --------------------------------------------------------------------------- #
# 2. Many poses at once
# --------------------------------------------------------------------------- #
def example_02_batching() -> None:
    """Skin a batch of poses in one call -- far cheaper than looping."""
    model = get_model()
    identity = torch.zeros(1, 45)  # broadcast across the batch

    for batch_size in (1, 8, 64):
        parameters = 0.2 * (torch.rand(batch_size, 204) - 0.5)
        with torch.no_grad():
            model(identity, parameters, None)  # warm up
            started = time.perf_counter()
            vertices, _ = model(identity, parameters, None)
            elapsed = time.perf_counter() - started
        print(f"batch {batch_size:3d}: {elapsed * 1000:6.1f} ms total, {elapsed / batch_size * 1000:5.2f} ms/mesh"
              f"  -> {tuple(vertices.shape)}")


# --------------------------------------------------------------------------- #
# 3. Levels of detail
# --------------------------------------------------------------------------- #
def example_03_levels_of_detail() -> None:
    """Compare mesh resolutions. Coarser LODs load faster and use much less RAM."""
    for lod in (1, 4, 6):
        download_assets(lod=lod)  # a no-op once the files are there
        started = time.perf_counter()
        model = get_model(lod)
        load_time = time.perf_counter() - started
        with torch.no_grad():
            vertices, _ = model(torch.zeros(1, 45), pose(model, "relaxed"), None)
        print(f"LOD {lod}: {vertices.shape[1]:6,} vertices, {faces(model).shape[0]:6,} triangles,"
              f" loaded in {load_time:4.1f} s")
    print("\nAll LODs share the same parameters, so a pose fitted at LOD 4 renders at LOD 1 unchanged.")


# --------------------------------------------------------------------------- #
# 4. Posing by name
# --------------------------------------------------------------------------- #
def example_04_named_parameters() -> None:
    """Find rig controls by name and set them, instead of indexing blindly."""
    model = get_model()
    names = parameter_names(model)
    index = parameter_index(model)

    print(f"{len(names)} model parameters; the first six are the rigid transform:")
    print("  ", names[:6])
    for keyword in ("elbow", "knee", "spine", "wrist"):
        matches = [name for name in names if keyword in name]
        print(f"  {keyword:6s} -> {len(matches):2d} parameters, e.g. {matches[:4]}")

    parameters = model_parameters(model, {"l_elbow_bend": 1.4, "r_elbow_bend": 0.7, "spine_twist0": 0.3})
    print(f"\nl_elbow_bend is column {index['l_elbow_bend']}, set to {parameters[0, index['l_elbow_bend']]:.1f} rad")
    print(f"Named poses available: {', '.join(POSES)}")


# --------------------------------------------------------------------------- #
# 5. Reading the skeleton
# --------------------------------------------------------------------------- #
def example_05_skeleton() -> None:
    """Use the skeleton state for measurements: joint positions, bone lengths."""
    model = get_model()
    names = joint_names(model)
    with torch.no_grad():
        _, skeleton_state = model(torch.zeros(1, 45), pose(model, "relaxed"), None)

    positions = joint_positions(skeleton_state)[0]
    rotations = joint_rotations(skeleton_state)[0]
    print(f"{len(names)} joints, e.g. {names[:5]}")
    print(f"quaternions are unit length: {rotations.norm(dim=-1).min():.4f} .. {rotations.norm(dim=-1).max():.4f}")

    for parent, child, label in (
        ("l_uparm", "l_lowarm", "upper arm"),
        ("l_lowarm", "l_wrist", "forearm"),
        ("l_upleg", "l_lowleg", "thigh"),
        ("l_lowleg", "l_foot", "shin"),
    ):
        length = (positions[names.index(parent)] - positions[names.index(child)]).norm()
        print(f"  {label:10s} {length:5.1f} cm")


# --------------------------------------------------------------------------- #
# 6. Exporting meshes
# --------------------------------------------------------------------------- #
def example_06_export_mesh() -> None:
    """Write meshes to disk for Blender, MeshLab or a renderer of your choice."""
    model = get_model()
    with torch.no_grad():
        vertices, _ = model(random_identity(scale=1.0, seed=42), pose(model, "wave"), None)

    for suffix in (".ply", ".obj"):
        path = save_mesh(vertices, model, OUTPUT_DIR / f"wave{suffix}")
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print("Vertex order and triangle indices are fixed per LOD, so meshes stay comparable across poses.")


# --------------------------------------------------------------------------- #
# 7. What the pose correctives contribute
# --------------------------------------------------------------------------- #
def example_07_pose_correctives() -> None:
    """Measure MHR's non-linear pose correctives by switching them off."""
    model = get_model()
    identity = torch.zeros(1, 45)

    for name in ("t_pose", "relaxed", "walk_stride", "squat"):
        parameters = pose(model, name)
        with torch.no_grad():
            with_correctives, _ = model(identity, parameters, None, apply_correctives=True)
            without, _ = model(identity, parameters, None, apply_correctives=False)
        offsets = (with_correctives - without)[0].norm(dim=-1) * 10.0  # cm -> mm
        print(f"{name:12s} mean {offsets.mean():5.2f} mm, max {offsets.max():6.2f} mm")
    print("\nThe rest pose is (almost) unaffected by design; the further a joint bends, the more work")
    print("the correctives do -- that is what keeps elbows, knees and shoulders from collapsing.")


# --------------------------------------------------------------------------- #
# 8. Identity: bodies, heads, hands
# --------------------------------------------------------------------------- #
def example_08_identity_groups() -> None:
    """Randomise one identity group at a time and measure the effect."""
    import trimesh

    model = get_model()
    triangles = faces(model)
    parameters = pose(model, "a_pose")

    with torch.no_grad():
        mean_body, _ = model(torch.zeros(1, 45), parameters, None)
    reference = mean_body[0].numpy()
    print(f"mean body: {trimesh.Trimesh(reference, triangles, process=False).volume / 1000:.1f} litres")

    for group in IDENTITY_GROUPS:
        with torch.no_grad():
            vertices, _ = model(random_identity(scale=1.5, groups=(group,), seed=5), parameters, None)
        mesh = vertices[0].numpy()
        moved = np.linalg.norm(mesh - reference, axis=1)
        volume = trimesh.Trimesh(mesh, triangles, process=False).volume / 1000
        print(f"  randomising {group:6s}: {volume:6.1f} litres, {(moved > 0.5).sum():6,} vertices moved >5 mm")


# --------------------------------------------------------------------------- #
# 9. Fitting parameters with autograd
# --------------------------------------------------------------------------- #
def example_09_fit_with_autograd() -> None:
    """MHR is differentiable: recover parameters from a target mesh by gradient descent.

    This is the core of any optimisation-based fitting pipeline (to keypoints, to
    scans, to another body model). Here the target is a mesh MHR produced itself,
    so we know the answer and can report how close the fit gets.

    It runs at LOD 4: the parameters are shared across levels of detail, so fitting
    on the coarse mesh (2.5k vertices instead of 18k) is ~5x cheaper and the result
    can be evaluated at LOD 1 afterwards.
    """
    model = get_model(4)
    identity = torch.zeros(1, 45)

    truth = model_parameters(model, {"l_elbow_bend": 1.1, "r_uparm_rz": 0.8, "spine_twist0": 0.25, "l_knee_bend": 0.6})
    with torch.no_grad():
        target, _ = model(identity, truth, None)

    estimate = torch.zeros(1, 204, requires_grad=True)
    optimiser = torch.optim.Adam([estimate], lr=0.1)
    for iteration in range(300):
        optimiser.zero_grad()
        vertices, _ = model(identity, estimate, None)
        loss = (vertices - target).pow(2).mean()
        loss.backward()
        optimiser.step()
        if iteration % 60 == 0:
            print(f"  iter {iteration:3d}: loss {loss.sqrt().item() * 10:7.3f} mm (root mean square coordinate error)")

    with torch.no_grad():
        final, _ = model(identity, estimate, None)
        error = (final - target).norm(dim=-1)
    print(f"final: mean {error.mean() * 10:.3f} mm, max {error.max() * 10:.3f} mm")
    print("recovered parameters (non-zero in the target):")
    index = parameter_index(model)
    for name in ("l_elbow_bend", "r_uparm_rz", "spine_twist0", "l_knee_bend"):
        print(f"  {name:15s} true {truth[0, index[name]]:+.3f}  fitted {estimate[0, index[name]].item():+.3f}")
    print("\nThe rig is deliberately redundant (204 controls for 127 joints), so several parameter")
    print("combinations produce nearly the same surface: judge a fit by mesh error, not by parameters.")


# --------------------------------------------------------------------------- #
# 10. From an image: SAM 3D Body predictions
# --------------------------------------------------------------------------- #
def example_10_from_image() -> None:
    """Turn SAM 3D Body output (estimated from a photograph) into MHR meshes."""
    predictions = load_predictions()  # data/*.npz, downloaded by mhr_kit.assets
    model = get_model()

    identity, parameters, expression = mhr_inputs(predictions)
    with torch.no_grad():
        vertices, _ = model(identity, parameters, expression)

    camera, residual = fit_camera(predictions)
    print(f"{len(predictions)} people, camera focal {camera.focal:.0f} px, reprojection error {residual:.1e} px")
    for index, prediction in enumerate(predictions):
        in_camera = to_camera_space(vertices[index].numpy(), prediction)
        uv, depth = camera.project(in_camera)
        print(f"  {prediction.name}: {depth.mean() / 100:.2f} m away, projects into"
              f" x {uv[:, 0].min():.0f}..{uv[:, 0].max():.0f}, y {uv[:, 1].min():.0f}..{uv[:, 1].max():.0f} px"
              f" (detection box {np.round(prediction.bbox).astype(int).tolist()})")


# --------------------------------------------------------------------------- #
# 11. Rendering: stills and video
# --------------------------------------------------------------------------- #
def example_11_render() -> None:
    """Render a still and a short clip without a GPU, display or OpenGL context."""
    model = get_model()
    triangles = faces(model)

    with torch.no_grad():
        vertices, _ = model(torch.zeros(1, 45), pose(model, "wave"), None)
    mesh = vertices[0].numpy()

    camera = render.orbit_camera(mesh, azimuth=25.0, elevation=5.0, size=(420, 520))
    print(f"wrote {render.save_image(render.render(mesh, triangles, camera), OUTPUT_DIR / 'still.png')}")

    # A clip: two keyframes, interpolated, skinned in one batch, then rendered.
    animation = interpolate_keyframes([pose(model, "relaxed"), pose(model, "arms_up")], 12, loop=True)
    with torch.no_grad():
        frames_vertices, _ = model(torch.zeros(1, 45), animation, None)
    meshes = frames_vertices.numpy()
    camera = render.orbit_camera(meshes, azimuth=25.0, size=(320, 400))
    frames = [render.render(mesh, triangles, camera) for mesh in meshes]
    print(f"wrote {render.save_video(frames, OUTPUT_DIR / 'clip.mp4', fps=12)} ({len(frames)} frames)")


# --------------------------------------------------------------------------- #
# 12. TorchScript: MHR with PyTorch only
# --------------------------------------------------------------------------- #
def example_12_torchscript() -> None:
    """Run the traced model, which needs neither pymomentum nor the rig files."""
    script_path = ASSET_DIR / Path(TORCHSCRIPT_MEMBER).name
    if not script_path.exists():
        print(f"downloading {script_path.name} (26 MB compressed, 696 MB on disk)")
        download_assets(lod=1, torchscript=True)

    scripted = torch.jit.load(script_path)
    scripted.eval()

    batch = 4
    identity = torch.zeros(batch, 45)  # the traced graph needs one identity row per pose
    parameters = 0.2 * (torch.rand(batch, 204) - 0.5)
    with torch.no_grad():
        vertices, skeleton_state = scripted(identity, parameters, torch.zeros(batch, 72))
    print(f"TorchScript output: {tuple(vertices.shape)} vertices, {tuple(skeleton_state.shape)} skeleton state")
    print("LOD 1 only, and no triangle indices or joint names -- store those separately if you need them.")


EXAMPLES = [value for name, value in sorted(globals().items()) if name.startswith("example_")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("numbers", nargs="*", type=int, help="which examples to run (1-based)")
    parser.add_argument("--all", action="store_true", help="run every example")
    args = parser.parse_args()

    if not args.numbers and not args.all:
        print(__doc__)
        print("Available examples:")
        for number, function in enumerate(EXAMPLES, start=1):
            print(f"  {number:2d}. {function.__doc__.splitlines()[0]}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chosen = range(1, len(EXAMPLES) + 1) if args.all else args.numbers
    for number in chosen:
        if not 1 <= number <= len(EXAMPLES):
            print(f"no example {number}; there are {len(EXAMPLES)}")
            continue
        function = EXAMPLES[number - 1]
        print(f"\n{'=' * 78}\n{number}. {function.__doc__.splitlines()[0]}\n{'=' * 78}")
        function()


if __name__ == "__main__":
    sys.exit(main())
