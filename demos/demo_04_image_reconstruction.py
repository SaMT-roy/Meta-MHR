#!/usr/bin/env python3
"""Demo 4 -- from a photograph to an MHR mesh (the "image" demo).

MHR itself never looks at pixels: it maps parameters to a mesh. Going the other
way -- image to parameters -- is the job of an estimator, and the one built on MHR
is `SAM 3D Body <https://github.com/facebookresearch/sam-3d-body>`_. It writes one
``.npz`` per detected person containing MHR identity, pose and expression plus the
camera translation that places them in the image.

This demo consumes those files (four ship with MHR, downloaded into ``data/``) and:

* rebuilds each person's mesh from ``mhr_model_params`` and checks it against the
  ``pred_vertices`` the estimator stored -- they agree to ~1e-3 mm, which proves
  the parameter and coordinate conventions in :mod:`mhr_kit.sam3d` are right;
* recovers the camera intrinsics exactly from the stored 2D/3D keypoint pairs;
* renders everybody back into the image frame, plus per-person crops from the
  detection boxes;
* writes the meshes out as ``.ply``.

Run::

    python demos/demo_04_image_reconstruction.py
    python demos/demo_04_image_reconstruction.py --image photo.jpg   # overlay on the source photo

To run it on *your* photographs, install SAM 3D Body, run its estimator, and save
each person's prediction dict with ``np.savez`` -- the keys are listed in
``mhr_kit/sam3d.py``. Nothing else here changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from mhr_kit import render
from mhr_kit.assets import DATA_DIR, download_sam3d_examples
from mhr_kit.model import faces, load_mhr, save_mesh
from mhr_kit.sam3d import CM_PER_M, MHR_TO_CAMERA, fit_camera, load_predictions, mhr_inputs, to_camera_space


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="folder of SAM 3D Body .npz predictions")
    parser.add_argument("--image", type=Path, help="optional source photograph to composite the meshes over")
    parser.add_argument("--lod", type=int, default=1, choices=range(7), help="level of detail (default: 1)")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/04_image"), help="where to write results")
    args = parser.parse_args()

    if not any(args.data.glob("*.npz")):
        download_sam3d_examples(dest=args.data)
    predictions = load_predictions(args.data)
    print(f"Loaded {len(predictions)} prediction(s) from {args.data}")

    model = load_mhr(lod=args.lod)
    triangles = faces(model)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------- parameters -> meshes, in batch
    # Every person is a row: one forward pass reconstructs the whole photograph.
    identity, parameters, expression = mhr_inputs(predictions)
    with torch.no_grad():
        vertices, _ = model(identity, parameters, expression)
    print(f"Rebuilt {tuple(vertices.shape)} vertices from ({identity.shape[1]}, {parameters.shape[1]},"
          f" {expression.shape[1]}) parameters per person")

    # ------------------------------------------------------------ sanity check
    # `pred_vertices` is what SAM 3D Body itself produced, in metres and camera
    # axes. Our reconstruction should match it to floating-point noise.
    print("\nReconstruction vs. the estimator's own vertices:")
    for index, prediction in enumerate(predictions):
        ours = vertices[index].numpy() @ MHR_TO_CAMERA.T / CM_PER_M  # cm, y-up -> m, camera axes
        error_mm = np.abs(ours - prediction.vertices).max() * 1000.0
        distance = np.linalg.norm(prediction.camera_translation)
        print(f"  {prediction.name}: max error {error_mm:.6f} mm, {distance:.2f} m from the camera")

    # ------------------------------------------------------------- the camera
    camera, residual = fit_camera(predictions)
    print(f"\nRecovered camera: focal {camera.focal:.1f} px, principal point"
          f" ({camera.center[0]:.1f}, {camera.center[1]:.1f}), image {camera.size[0]}x{camera.size[1]} px")
    print(f"  worst keypoint reprojection error: {residual:.2e} px")

    background: np.ndarray | tuple[float, float, float] = render.DEFAULT_BACKGROUND
    if args.image is not None:
        import imageio.v3 as iio

        background = iio.imread(args.image)[..., :3]
        height, width = background.shape[:2]
        if (width, height) != tuple(camera.size):
            print(f"  note: {args.image.name} is {width}x{height}, not {camera.size[0]}x{camera.size[1]};"
                  " using the image size (the fitted principal point is kept)")
            camera = render.Camera(camera.focal, camera.center, (width, height), camera.rotation, camera.translation)

    # --------------------------------------------- render the whole photograph
    # Each person carries their own camera translation, so convert first, then
    # merge into one mesh: the depth sort then resolves who occludes whom.
    in_camera = [to_camera_space(vertices[index].numpy(), prediction) for index, prediction in enumerate(predictions)]
    merged_vertices, merged_faces = render.merge_meshes(in_camera, triangles)
    scene = render.render(merged_vertices, merged_faces, camera, background=background, alpha=0.85)
    scene_path = render.save_image(scene, args.outdir / "reconstruction.png")

    # ------------------------------------------------------- per-person crops
    # `bbox` is (x, y, width, height) in the same pixel frame we just rendered.
    crops = []
    for prediction in predictions:
        x, y, width, height = prediction.bbox
        pad = 0.08 * max(width, height)
        left, top = int(max(x - pad, 0)), int(max(y - pad, 0))
        right = int(min(x + width + pad, scene.shape[1]))
        bottom = int(min(y + height + pad, scene.shape[0]))
        crops.append(scene[top:bottom, left:right])

    # Centre each crop on a common canvas so they can be tiled side by side.
    tallest = max(crop.shape[0] for crop in crops)
    widest = max(crop.shape[1] for crop in crops)
    padded = []
    for crop in crops:
        canvas = np.full((tallest, widest, 3), 20, dtype=np.uint8)
        top = (tallest - crop.shape[0]) // 2
        left = (widest - crop.shape[1]) // 2
        canvas[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
        padded.append(canvas)
    crops_path = render.save_image(render.image_grid(padded, columns=len(padded)), args.outdir / "person_crops.png")

    # --------------------------------------------------------- export meshes
    for index, prediction in enumerate(predictions):
        save_mesh(vertices[index], model, args.outdir / f"{prediction.name}.ply")
    print(f"\nWrote {len(predictions)} meshes to {args.outdir}")
    print(f"Wrote {scene_path}\nWrote {crops_path}")


if __name__ == "__main__":
    main()
