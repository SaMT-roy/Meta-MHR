#!/usr/bin/env python3
"""Demo 7 -- the TorchScript model: MHR without pymomentum or the FBX rig.

MHR also ships as a single traced TorchScript file, ``mhr_model.pt``. It needs
nothing but PyTorch -- no ``pymomentum``, no rig, no corrective ``.npz`` -- which
makes it the easy way to embed MHR in an existing pipeline or ship it somewhere
awkward. The trade-offs: LOD 1 only, 696 MB on disk, and none of the model
metadata (no triangle indices, joint names or parameter names).

The demo loads both models, checks they agree, and times them.

Run::

    python demos/demo_07_torchscript.py          # downloads mhr_model.pt (26 MB) on first use
    python demos/demo_07_torchscript.py --batch 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mhr_kit import render
from mhr_kit.assets import ASSET_DIR, TORCHSCRIPT_MEMBER, download_assets
from mhr_kit.model import faces, load_mhr
from mhr_kit.params import pose, random_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=8, help="batch size for the timing comparison")
    parser.add_argument("--assets", type=Path, default=ASSET_DIR, help="asset folder")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/07_torchscript"), help="where to write results")
    args = parser.parse_args()

    script_path = args.assets / Path(TORCHSCRIPT_MEMBER).name
    if not script_path.exists():
        print(f"{script_path.name} is missing; downloading it (26 MB compressed, 696 MB on disk)")
        download_assets(lod=1, torchscript=True, dest=args.assets)

    # ------------------------------------------------------------ load both
    started = time.perf_counter()
    scripted = torch.jit.load(script_path)
    scripted.eval()
    print(f"Loaded {script_path.name} in {time.perf_counter() - started:.1f} s (no pymomentum involved)")

    model = load_mhr(lod=1, assets=args.assets)

    # ------------------------------------------------------------- identical?
    # Same inputs, both models: the traced graph should reproduce the eager one.
    torch.manual_seed(0)
    identity = random_identity(scale=0.8, seed=0)
    parameters = torch.cat([pose(model, name) for name in ("relaxed", "wave", "sitting", "walk_stride")])
    expression = torch.zeros(len(parameters), 72)

    # Difference worth knowing about: the eager model broadcasts a single identity
    # row over the batch, the traced graph does not -- it needs one row per pose.
    identity_rows = identity.repeat(len(parameters), 1)
    with torch.no_grad():
        reference, _ = model(identity, parameters, expression)
        traced, _ = scripted(identity_rows, parameters, expression)
    difference = (reference - traced).abs()
    print(f"\nAgreement over {len(parameters)} poses: mean {difference.mean() * 10:.2e} mm,"
          f" max {difference.max() * 10:.2e} mm")

    # ------------------------------------------------------------- how fast?
    batch_parameters = parameters[:1].repeat(args.batch, 1)
    batch_expression = expression[:1].repeat(args.batch, 1)
    batch_identity = identity.repeat(args.batch, 1)
    timings = {}
    for name, runnable in (("mhr.MHR", model), ("TorchScript", scripted)):
        with torch.no_grad():
            runnable(batch_identity, batch_parameters, batch_expression)  # warm up
            started = time.perf_counter()
            runnable(batch_identity, batch_parameters, batch_expression)
            timings[name] = time.perf_counter() - started
    print(f"\nBatch of {args.batch}:")
    for name, elapsed in timings.items():
        print(f"  {name:12s} {elapsed * 1000:6.0f} ms  ({elapsed / args.batch * 1000:.1f} ms/mesh)")

    # The TorchScript module has no mesh metadata, so the triangle indices have to
    # come from somewhere else -- here, from the full model. In a deployment you
    # would save them once (they never change for a given LOD).
    args.outdir.mkdir(parents=True, exist_ok=True)
    triangles = faces(model)
    mesh = traced[1].numpy()
    image = render.render(mesh, triangles, render.orbit_camera(mesh, azimuth=20.0, size=(420, 520)))
    print(f"\nWrote {render.save_image(image, args.outdir / 'torchscript_wave.png')}")


if __name__ == "__main__":
    main()
