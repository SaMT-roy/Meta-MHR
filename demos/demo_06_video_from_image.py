#!/usr/bin/env python3
"""Demo 6 -- video from image-derived MHR parameters.

Two videos, both starting from SAM 3D Body predictions (see demo 4):

``--mode orbit`` (default)
    Reconstruct one person from a photograph and orbit the camera 360 degrees
    around them: a "bullet time" turn from a single still. The parameters never
    change, only the camera -- so this shows off what a *3D* reconstruction buys
    you over a 2D pose estimate.

``--mode sequence``
    Treat a folder of ``.npz`` files as consecutive frames -- what per-frame
    tracking of a video produces -- and render them as a clip in the image frame.
    ``--interpolate`` inserts intermediate frames by blending parameters, which
    both smooths jitter and slows the clip down.

    The four files that ship with MHR are four people in *one* photograph, not a
    time sequence, so the default clip morphs between them. Point ``--data`` at
    your own per-frame predictions for the real thing.

Run::

    python demos/demo_06_video_from_image.py
    python demos/demo_06_video_from_image.py --mode sequence --interpolate 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from mhr_kit import render
from mhr_kit.assets import DATA_DIR, download_sam3d_examples
from mhr_kit.model import faces, load_mhr
from mhr_kit.params import interpolate_keyframes
from mhr_kit.sam3d import CM_PER_M, MHR_TO_CAMERA, fit_camera, load_predictions, mhr_inputs


def orbit_clip(model, predictions, args) -> list[np.ndarray]:
    """Render one reconstructed person from ``args.frames`` viewpoints around them."""
    prediction = predictions[min(args.person, len(predictions) - 1)]
    identity, parameters, expression = mhr_inputs([prediction])
    with torch.no_grad():
        vertices, _ = model(identity, parameters, expression)
    mesh = vertices[0].numpy()
    print(f"Orbiting {prediction.name} ({args.frames} frames)")

    # Keep the framing fixed: one look-at point and radius for every frame, so
    # only the viewing angle changes.
    look_at = 0.5 * (mesh.min(0) + mesh.max(0))
    radius = float(np.linalg.norm(mesh - look_at, axis=1).max())
    triangles = faces(model)

    frames = []
    for azimuth in np.linspace(0.0, 360.0, args.frames, endpoint=False):
        camera = render.orbit_camera(
            mesh,
            azimuth=float(azimuth),
            elevation=8.0,
            size=(args.width, args.height),
            center=look_at,
            radius=radius,
        )
        frames.append(render.render(mesh, triangles, camera))
    return frames


def sequence_clip(model, predictions, args) -> list[np.ndarray]:
    """Render a folder of predictions as consecutive frames, in the image frame."""
    identity, parameters, expression = mhr_inputs(predictions)
    camera, residual = fit_camera(predictions)
    print(f"Sequence of {len(predictions)} frames, camera focal {camera.focal:.0f} px"
          f" (reprojection error {residual:.1e} px)")

    # Camera translation is per frame too, so interpolate it alongside the pose.
    translations = np.stack([prediction.camera_translation for prediction in predictions])
    if args.interpolate > 1:
        parameters = interpolate_keyframes(list(parameters), frames_per_segment=args.interpolate, smooth=True)
        expression = interpolate_keyframes(list(expression), frames_per_segment=args.interpolate, smooth=True)
        translations = interpolate_keyframes(
            [torch.from_numpy(row) for row in translations], frames_per_segment=args.interpolate, smooth=True
        ).numpy()
        identity = identity[:1]  # one body for the whole clip
        print(f"  interpolated to {len(parameters)} frames")

    with torch.no_grad():
        vertices, _ = model(identity, parameters, expression)

    triangles = faces(model)
    frames = []
    for index in range(len(vertices)):
        # The same conversion `mhr_kit.sam3d.to_camera_space` does, but with the
        # interpolated translation rather than the prediction's own.
        mesh = vertices[index].numpy() @ MHR_TO_CAMERA.T + CM_PER_M * translations[index]
        frames.append(render.render(mesh, triangles, camera))
        if (index + 1) % 20 == 0:
            print(f"  rendered {index + 1}/{len(vertices)} frames")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("orbit", "sequence"), default="orbit", help="which clip to make")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="folder of SAM 3D Body .npz predictions")
    parser.add_argument("--person", type=int, default=2, help="which prediction to orbit (orbit mode)")
    parser.add_argument("--interpolate", type=int, default=1, help="frames to insert between predictions (sequence mode)")
    parser.add_argument("--frames", type=int, default=72, help="number of viewpoints (orbit mode)")
    parser.add_argument("--fps", type=int, default=24, help="frames per second of the output video")
    parser.add_argument("--width", type=int, default=480, help="output width in pixels (orbit mode)")
    parser.add_argument("--height", type=int, default=600, help="output height in pixels (orbit mode)")
    parser.add_argument("--lod", type=int, default=1, choices=range(7), help="level of detail (default: 1)")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/06_video_from_image"), help="output folder")
    args = parser.parse_args()

    if not any(args.data.glob("*.npz")):
        download_sam3d_examples(dest=args.data)
    predictions = load_predictions(args.data)
    model = load_mhr(lod=args.lod)
    args.outdir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    frames = orbit_clip(model, predictions, args) if args.mode == "orbit" else sequence_clip(model, predictions, args)
    print(f"Rendered {len(frames)} frames in {time.perf_counter() - started:.1f} s")

    video_path = render.save_video(frames, args.outdir / f"{args.mode}.mp4", fps=args.fps)
    sheet_path = render.save_image(render.image_grid(frames[:: max(len(frames) // 8, 1)], columns=8),
                                  args.outdir / f"{args.mode}_frames.png")
    print(f"Wrote {video_path}\nWrote {sheet_path}")


if __name__ == "__main__":
    main()
