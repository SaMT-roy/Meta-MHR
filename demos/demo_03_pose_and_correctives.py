#!/usr/bin/env python3
"""Demo 3 -- posing MHR, and what the non-linear pose correctives actually do.

Two things are shown:

1. **Named parameters.** MHR's 204 model parameters are rig controls with names
   (``l_elbow_bend``, ``r_upleg_rz``, ``spine_twist0``, ...), not an opaque vector,
   so a pose can be written down and read back. The demo renders the pose library
   from :mod:`mhr_kit.params` and a manual pose built from names.

2. **Pose correctives.** This is MHR's headline feature: a small network predicts
   per-vertex offsets from the joint rotations *before* skinning, which is what
   keeps elbows, shoulders and knees from collapsing the way plain linear blend
   skinning does. ``model(..., apply_correctives=False)`` turns them off, so the
   contribution can be measured and drawn directly.

Run::

    python demos/demo_03_pose_and_correctives.py
    python demos/demo_03_pose_and_correctives.py --pose squat
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np
import torch
from mhr_kit import render
from mhr_kit.model import faces, load_mhr
from mhr_kit.params import POSES, model_parameters, pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lod", type=int, default=1, choices=range(7), help="level of detail (default: 1)")
    parser.add_argument("--pose", default="squat", help=f"pose to analyse: {', '.join(POSES)}")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/03_pose"), help="where to write results")
    args = parser.parse_args()

    model = load_mhr(lod=args.lod)
    triangles = faces(model)
    identity = torch.zeros(1, 45)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- pose library
    names = list(POSES)
    batch = torch.cat([pose(model, name) for name in names])
    with torch.no_grad():
        vertices, _ = model(identity, batch, None)
    meshes = vertices.numpy()

    tiles = []
    for azimuth in (25.0, 85.0):  # three-quarter view and profile
        camera = render.orbit_camera(meshes, azimuth=azimuth, size=(220, 300))
        tiles += [render.render(mesh, triangles, camera) for mesh in meshes]
    library_path = render.save_image(render.image_grid(tiles, columns=len(names)), args.outdir / "pose_library.png")
    print("Pose library:", ", ".join(names))

    # ------------------------------------------------------ a pose, by name
    # Anything not named stays at its rest value, so a readable dict is enough.
    handmade = model_parameters(
        model,
        {
            "root_ry": 0.35,  # turn the whole body 20 degrees to its left
            "spine_bend0": 0.20,  # lean forward from the lower spine
            "neck_bend": -0.25,  # look up
            "r_uparm_ry": 1.00,  # right arm out and up
            "r_uparm_rz": 1.20,
            "r_elbow_bend": 0.60,
            "l_uparm_ry": -0.65,  # left arm hanging
            "l_uparm_rz": -0.85,
            "l_elbow_bend": 0.30,
            "r_upleg_rz": -0.30,  # right leg a step forward
            "l_knee_bend": 0.20,
        },
    )
    with torch.no_grad():
        handmade_vertices, _ = model(identity, handmade, None)
    handmade_path = render.save_image(
        render.render(
            handmade_vertices[0].numpy(),
            triangles,
            render.orbit_camera(handmade_vertices[0].numpy(), azimuth=20.0, size=(420, 520)),
        ),
        args.outdir / "handmade_pose.png",
    )

    # -------------------------------------------------- correctives on / off
    parameters = pose(model, args.pose)
    with torch.no_grad():
        started = time.perf_counter()
        with_correctives, _ = model(identity, parameters, None, apply_correctives=True)
        time_on = time.perf_counter() - started

        started = time.perf_counter()
        without_correctives, _ = model(identity, parameters, None, apply_correctives=False)
        time_off = time.perf_counter() - started

    offsets = (with_correctives - without_correctives)[0].numpy()  # cm, per vertex
    magnitude = np.linalg.norm(offsets, axis=1)
    print(f"\nPose correctives on the '{args.pose}' pose:")
    print(f"  mean offset {magnitude.mean() * 10:.2f} mm, max {magnitude.max() * 10:.2f} mm")
    print(f"  vertices moved more than 5 mm: {(magnitude > 0.5).sum():,} of {len(magnitude):,}")
    print(f"  forward pass: {time_on * 1000:.0f} ms with correctives, {time_off * 1000:.0f} ms without")

    # Colour every triangle by how far the correctives moved it. The hot spots are
    # exactly the places linear blend skinning gets wrong: elbows, knees, hips,
    # shoulders and the armpits.
    face_magnitude = magnitude[triangles].mean(axis=1)
    normalised = np.clip(face_magnitude / max(face_magnitude.max(), 1e-6), 0.0, 1.0)
    face_colors = matplotlib.colormaps["inferno"](normalised)[:, :3]

    mesh = with_correctives[0].numpy()
    tiles = []
    for azimuth in (25.0, 155.0):
        camera = render.orbit_camera(mesh, azimuth=azimuth, size=(360, 460))
        tiles.append(render.render(mesh, triangles, camera))  # plain shading
        tiles.append(render.render(mesh, triangles, camera, color=face_colors, ambient=1.0))  # heat map
    heatmap_path = render.save_image(render.image_grid(tiles, columns=4), args.outdir / f"correctives_{args.pose}.png")

    # Side by side, so the shape difference itself is visible.
    camera = render.orbit_camera(mesh, azimuth=70.0, size=(360, 460))
    comparison = render.image_grid(
        [
            render.render(without_correctives[0].numpy(), triangles, camera),
            render.render(mesh, triangles, camera),
        ],
        columns=2,
    )
    comparison_path = render.save_image(comparison, args.outdir / f"correctives_{args.pose}_side_by_side.png")

    for path in (library_path, handmade_path, heatmap_path, comparison_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
