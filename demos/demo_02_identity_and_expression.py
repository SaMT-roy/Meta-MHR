#!/usr/bin/env python3
"""Demo 2 -- who the person is: identity coefficients and facial expressions.

MHR separates *identity* (45 coefficients: 20 body, 20 head, 5 hands) from *pose*
(the 204 model parameters), so the same body can be re-posed and the same pose can
be re-bodied. This demo makes both halves visible:

* a sweep of a single identity coefficient from -2.5 to +2.5;
* random bodies with the head held fixed, and random heads on a fixed body;
* single facial expression blendshapes, rendered as a face close-up.

Run::

    python demos/demo_02_identity_and_expression.py
    python demos/demo_02_identity_and_expression.py --coefficient 3 --lod 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import trimesh
from mhr_kit import render
from mhr_kit.model import faces, joint_names, joint_positions, load_mhr
from mhr_kit.params import IDENTITY_GROUPS, pose, random_expression, random_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lod", type=int, default=1, choices=range(7), help="level of detail (default: 1)")
    parser.add_argument("--coefficient", type=int, default=0, help="which identity coefficient to sweep")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/02_identity"), help="where to write results")
    args = parser.parse_args()

    model = load_mhr(lod=args.lod)
    triangles = faces(model)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Everything is shown in the same relaxed pose, so only identity changes.
    parameters = pose(model, "relaxed")
    print("Identity coefficient groups:", {name: (span.start, span.stop) for name, span in IDENTITY_GROUPS.items()})

    # ------------------------------------------------- sweep one coefficient
    amounts = torch.linspace(-2.5, 2.5, 7)
    tiles, statures, volumes = [], [], []
    for amount in amounts:
        identity = torch.zeros(1, 45)
        identity[0, args.coefficient] = amount
        with torch.no_grad():
            vertices, _ = model(identity, parameters, None)
        mesh = vertices[0].numpy()
        statures.append(float(mesh[:, 1].max() - mesh[:, 1].min()))
        volumes.append(trimesh.Trimesh(mesh, triangles, process=False).volume / 1000.0)  # cm^3 -> litres
        # One camera for the whole sweep, so size differences stay comparable.
        camera = render.orbit_camera(mesh, azimuth=20.0, size=(190, 340), center=[0.0, 86.0, 0.0], radius=100.0)
        tiles.append(render.render(mesh, triangles, camera))
    print(f"\nSweeping identity coefficient {args.coefficient} from {amounts[0]:+.1f} to {amounts[-1]:+.1f}:")
    print("  stature (cm):", " ".join(f"{value:5.0f}" for value in statures))
    print("  volume  (l): ", " ".join(f"{value:5.0f}" for value in volumes))
    # Identity mostly reshapes the *mesh*: MHR keeps the skeleton separate, and
    # bone lengths are driven by the `scale_*` model parameters instead. So a
    # coefficient can change build a lot while barely moving stature.
    sweep_path = render.save_image(render.image_grid(tiles, columns=7), args.outdir / f"sweep_coeff{args.coefficient}.png")

    # ----------------------------------------- random bodies vs random heads
    # `groups` keeps the other coefficients at their mean, isolating the change.
    tiles = []
    for seed in range(4):
        identity = random_identity(scale=1.4, groups=("body",), seed=seed)
        with torch.no_grad():
            vertices, _ = model(identity, parameters, None)
        mesh = vertices[0].numpy()
        camera = render.orbit_camera(mesh, azimuth=20.0, size=(190, 340), center=[0.0, 86.0, 0.0], radius=100.0)
        tiles.append(render.render(mesh, triangles, camera))
    bodies_path = render.save_image(render.image_grid(tiles, columns=4), args.outdir / "random_bodies.png")

    head_index = joint_names(model).index("c_head")
    tiles = []
    for seed in range(4):
        identity = random_identity(scale=1.6, groups=("head",), seed=100 + seed)
        with torch.no_grad():
            vertices, skeleton = model(identity, parameters, None)
        mesh = vertices[0].numpy()
        head = joint_positions(skeleton)[0, head_index].numpy()
        # Zoom in: a 14 cm radius around the head joint fills the frame with a face.
        camera = render.orbit_camera(mesh, azimuth=12.0, size=(240, 260), center=head, radius=14.0)
        tiles.append(render.render(mesh, triangles, camera))
    heads_path = render.save_image(render.image_grid(tiles, columns=4), args.outdir / "random_heads.png")

    # ------------------------------------------------------ expression basis
    # Each of the 72 coefficients is one facial blendshape; driving them one at a
    # time shows what the basis contains (jaw, brows, eyelids, mouth shapes, ...).
    identity = torch.zeros(1, 45)
    tiles = []
    shown = [0, 1, 2, 3, 8, 12, 16, 24, 32, 40, 48, 60]
    for coefficient in shown:
        expression = torch.zeros(1, 72)
        expression[0, coefficient] = 1.0
        with torch.no_grad():
            vertices, skeleton = model(identity, parameters, expression)
        mesh = vertices[0].numpy()
        head = joint_positions(skeleton)[0, head_index].numpy()
        camera = render.orbit_camera(mesh, azimuth=8.0, size=(200, 220), center=head, radius=13.0)
        tiles.append(render.render(mesh, triangles, camera))
    print(f"Rendered expression blendshapes {shown}")
    expressions_path = render.save_image(render.image_grid(tiles, columns=6), args.outdir / "expression_basis.png")

    # A random blend of all 72, which is what an estimator would predict.
    with torch.no_grad():
        vertices, skeleton = model(identity, parameters, random_expression(scale=0.6, seed=3))
    mesh = vertices[0].numpy()
    head = joint_positions(skeleton)[0, head_index].numpy()
    camera = render.orbit_camera(mesh, azimuth=8.0, size=(320, 360), center=head, radius=13.0)
    random_face_path = render.save_image(render.render(mesh, triangles, camera), args.outdir / "random_expression.png")

    for path in (sweep_path, bodies_path, heads_path, expressions_path, random_face_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
