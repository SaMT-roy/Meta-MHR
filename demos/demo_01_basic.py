#!/usr/bin/env python3
"""Demo 1 -- the shortest useful MHR program.

What it shows:

* loading the model (and downloading only the assets that LOD needs);
* the three parameter blocks and what changing each one does;
* reading the two outputs: mesh vertices and the 127-joint skeleton state;
* saving a mesh you can open in Blender/MeshLab, and a rendered turntable sheet.

Run::

    python demos/demo_01_basic.py                # LOD 1, writes into outputs/
    python demos/demo_01_basic.py --lod 4        # coarser mesh, 8x less memory
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Demos are scripts, not a package: put the project root on the import path so
# `mhr_kit` resolves no matter where they are run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mhr_kit import render
from mhr_kit.model import faces, joint_names, joint_positions, load_mhr, model_summary, save_mesh
from mhr_kit.params import model_parameters, pose, random_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lod", type=int, default=1, choices=range(7), help="level of detail (default: 1)")
    parser.add_argument("--device", default="cpu", help="torch device (pymomentum is CPU-only on macOS)")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/01_basic"), help="where to write results")
    args = parser.parse_args()

    # ------------------------------------------------------------------ load
    started = time.perf_counter()
    model = load_mhr(lod=args.lod, device=args.device)
    print(f"Loaded MHR LOD {args.lod} in {time.perf_counter() - started:.1f} s")
    print(model_summary(model, lod=args.lod))
    triangles = faces(model)

    # ------------------------------------------------------- one forward pass
    # The average body: every identity coefficient at its mean of zero.
    identity = torch.zeros(1, 45, device=args.device)
    # `None` for the expression block is shorthand for "neutral face".
    parameters = pose(model, "relaxed", device=args.device)

    with torch.no_grad():
        vertices, skeleton_state = model(identity, parameters, None)
    print(f"\nvertices       {tuple(vertices.shape)}  (batch, vertex, xyz in cm)")
    print(f"skeleton_state {tuple(skeleton_state.shape)}  (batch, joint, [tx ty tz qx qy qz qw scale])")

    # The skeleton state is world space, so joint positions are a plain slice.
    positions = joint_positions(skeleton_state)[0]
    names = joint_names(model)
    height = float(vertices[0, :, 1].max() - vertices[0, :, 1].min())
    print(f"\nstature        {height:.1f} cm")
    for joint in ("root", "c_head", "l_wrist", "r_wrist"):
        index = names.index(joint)
        print(f"  {joint:10s} at {positions[index].tolist()} cm")

    # --------------------------------------------------- vary each parameter
    # One batch, four rows: the same body posed and shaped four different ways.
    # Note that MHR broadcasts a single identity row over the whole batch, which
    # is why `identity` can stay (1, 45) while `parameters` has four rows.
    labels = ["rest (T-pose)", "relaxed", "walk stride", "sitting"]
    batch = torch.cat(
        [
            model_parameters(model, device=args.device),  # all zeros = rest pose
            pose(model, "relaxed", device=args.device),
            pose(model, "walk_stride", device=args.device),
            pose(model, "sitting", device=args.device),
        ]
    )
    started = time.perf_counter()
    with torch.no_grad():
        posed, _ = model(identity, batch, None)
    elapsed = time.perf_counter() - started
    print(f"\nSkinned {len(labels)} poses in {elapsed * 1000:.0f} ms ({elapsed / len(labels) * 1000:.0f} ms/mesh)")

    # A different body in the same pose: identity coefficients, not parameters,
    # are what make someone tall, short, broad or slight.
    with torch.no_grad():
        other, _ = model(random_identity(scale=1.2, seed=7, device=args.device), parameters, None)
    other_height = float(other[0, :, 1].max() - other[0, :, 1].min())
    print(f"Random identity (seed 7) is {other_height:.1f} cm tall, {other_height - height:+.1f} cm vs the mean body")

    # --------------------------------------------------------------- outputs
    args.outdir.mkdir(parents=True, exist_ok=True)
    mesh_path = save_mesh(posed[1], model, args.outdir / "relaxed.ply")
    print(f"\nWrote {mesh_path}")

    # One camera framed on every pose keeps the figure size consistent tile to tile.
    all_vertices = posed.numpy()
    tiles = []
    for index, label in enumerate(labels):
        camera = render.orbit_camera(all_vertices, azimuth=25.0, size=(280, 360))
        tiles.append(render.render(all_vertices[index], triangles, camera))
        print(f"  rendered {label}")
    sheet = render.save_image(render.image_grid(tiles, columns=4), args.outdir / "poses.png")

    # Turntable: same mesh, four camera azimuths.
    views = [
        render.render(all_vertices[1], triangles, render.orbit_camera(all_vertices[1], azimuth=angle, size=(280, 360)))
        for angle in (0, 90, 180, 270)
    ]
    turntable = render.save_image(render.image_grid(views, columns=4), args.outdir / "turntable.png")
    print(f"Wrote {sheet}\nWrote {turntable}")


if __name__ == "__main__":
    main()
