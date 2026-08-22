#!/usr/bin/env python3
"""Demo 5 -- animating MHR and writing a video.

Because the 204 model parameters are joint *angles*, animation needs no special
machinery: interpolate between keyframes, stack the frames into one batch, and let
a single forward pass skin the whole clip. That is both the simplest and the
fastest way to use the model -- batching amortises the pose-corrective network,
which is the expensive part.

The clip is a loop through the pose library, rendered while the camera drifts, and
written as MP4 (or GIF if no encoder is available).

Run::

    python demos/demo_05_video_animation.py                       # 5 s clip, 480x600
    python demos/demo_05_video_animation.py --fps 30 --frames-per-pose 24
    python demos/demo_05_video_animation.py --lod 4 --width 320 --height 400
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
from mhr_kit.model import faces, load_mhr
from mhr_kit.params import interpolate_keyframes, pose, random_identity

# The poses the clip visits, in order; it loops back to the first one at the end.
CLIP = ("relaxed", "a_pose", "arms_up", "wave", "arms_up", "walk_stride", "squat")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lod", type=int, default=1, choices=range(7), help="level of detail (default: 1)")
    parser.add_argument("--fps", type=int, default=24, help="frames per second of the output video")
    parser.add_argument("--frames-per-pose", type=int, default=18, help="frames between consecutive keyframes")
    parser.add_argument("--width", type=int, default=480, help="output width in pixels")
    parser.add_argument("--height", type=int, default=600, help="output height in pixels")
    parser.add_argument("--identity-seed", type=int, help="randomise the body (default: the mean body)")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/05_video"), help="where to write results")
    args = parser.parse_args()

    model = load_mhr(lod=args.lod)
    triangles = faces(model)
    args.outdir.mkdir(parents=True, exist_ok=True)

    identity = torch.zeros(1, 45) if args.identity_seed is None else random_identity(1.1, seed=args.identity_seed)

    # ------------------------------------------------------------- the motion
    keyframes = [pose(model, name) for name in CLIP]
    animation = interpolate_keyframes(keyframes, frames_per_segment=args.frames_per_pose, loop=True, smooth=True)
    print(f"Clip: {' -> '.join(CLIP)} -> {CLIP[0]}")
    print(f"{len(animation)} frames at {args.fps} fps = {len(animation) / args.fps:.1f} s")

    # --------------------------------------------------- skin every frame at once
    started = time.perf_counter()
    with torch.no_grad():
        vertices, _ = model(identity, animation, None)
    elapsed = time.perf_counter() - started
    print(f"Skinned the whole clip in {elapsed:.2f} s ({elapsed / len(animation) * 1000:.1f} ms/frame)")
    meshes = vertices.numpy()

    # ------------------------------------------------------------- render it
    # Frame the camera on every frame at once so the subject never jumps, then
    # rebuild it per frame only to move the azimuth (a slow drift adds parallax).
    look_at = 0.5 * (meshes.reshape(-1, 3).min(0) + meshes.reshape(-1, 3).max(0))
    radius = float(np.linalg.norm(meshes.reshape(-1, 3) - look_at, axis=1).max())
    azimuths = 25.0 + 20.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(meshes)))

    started = time.perf_counter()
    frames = []
    for index, (mesh, azimuth) in enumerate(zip(meshes, azimuths)):
        camera = render.orbit_camera(
            mesh, azimuth=float(azimuth), size=(args.width, args.height), center=look_at, radius=radius
        )
        frames.append(render.render(mesh, triangles, camera))
        if (index + 1) % 20 == 0:
            print(f"  rendered {index + 1}/{len(meshes)} frames")
    print(f"Rendered {len(frames)} frames in {time.perf_counter() - started:.1f} s")

    video_path = render.save_video(frames, args.outdir / "animation.mp4", fps=args.fps)
    print(f"Wrote {video_path}")

    # A contact sheet of every 6th frame, useful when a video player is not handy.
    sheet = render.image_grid(frames[::6], columns=7)
    print(f"Wrote {render.save_image(sheet, args.outdir / 'animation_frames.png')}")


if __name__ == "__main__":
    main()
