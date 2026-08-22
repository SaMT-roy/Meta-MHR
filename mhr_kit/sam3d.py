# Copyright (c) 2026 -- helper utilities for the MHR body model. Apache-2.0, like MHR itself.
"""Reading SAM 3D Body predictions -- the "image -> MHR" direction.

MHR is a body *model*: it turns parameters into a mesh. It does not look at
pixels. To get MHR parameters out of a photograph you run an estimator, and the
one MHR is built for is `SAM 3D Body <https://github.com/facebookresearch/sam-3d-body>`_,
which regresses MHR identity, pose and expression per detected person.

This module reads the ``.npz`` files that estimator writes, so the demos work on
genuinely image-derived parameters. Four examples ship with the MHR repository
(``python -m mhr_kit.assets`` downloads them into ``data/``); they are four people
detected in a single 1920x890 photograph.

``.npz`` contents (per person)
------------------------------

=========================  ==============  =================================================
key                        shape           meaning
=========================  ==============  =================================================
``shape_params``           ``(45,)``       MHR ``identity_coeffs``
``mhr_model_params``       ``(204,)``      MHR ``model_parameters`` (no global translation)
``expr_params``            ``(72,)``       MHR ``face_expr_coeffs``
``pred_vertices``          ``(V, 3)``      posed mesh in **metres**, camera axes, no translation
``pred_cam_t``             ``(3,)``        camera-space translation of this person, in **metres**
``pred_keypoints_3d``      ``(70, 3)``     keypoints in metres, same frame as ``pred_vertices``
``pred_keypoints_2d``      ``(70, 2)``     the same keypoints in **image pixels**
``bbox``                   ``(4,)``        detection box ``(x, y, width, height)`` in pixels
``lhand_bbox``/``rhand_bbox``  ``(4,)``    hand boxes in pixels
``pred_joint_coords``      ``(127, 3)``    skeleton joint positions, metres
``pred_global_rots``       ``(127, 3, 3)`` skeleton joint rotations
``scale_params``           ``(28,)``       bone scales (already folded into ``mhr_model_params``)
``pred_pose_raw``          ``(266,)``      the network's raw pose head output
=========================  ==============  =================================================

Two conventions matter, and both are verified by the demos:

1. **Units.** MHR outputs centimetres; SAM 3D Body works in metres.
2. **Axes.** MHR is y-up with the body facing +z; the estimator reports camera
   coordinates (y down, z into the scene). Converting is a sign flip on y and z::

       vertices_cm = mhr_vertices_cm * (1, -1, -1) + 100 * pred_cam_t

   Doing that reproduces ``pred_vertices`` from ``mhr_model_params`` to within
   ~1e-6 m, which :mod:`demos.04_image_reconstruction` checks explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .assets import DATA_DIR
from .render import Camera

# MHR world axes (y up, facing +z) -> camera axes (y down, z into the scene).
MHR_TO_CAMERA = np.diag([1.0, -1.0, -1.0])
CM_PER_M = 100.0


@dataclass
class Sam3dPrediction:
    """One person predicted from one image."""

    name: str
    identity_coeffs: np.ndarray  # (45,)
    model_parameters: np.ndarray  # (204,)
    expression_coeffs: np.ndarray  # (72,)
    camera_translation: np.ndarray  # (3,) metres
    vertices: np.ndarray  # (V, 3) metres, camera axes, translation not applied
    keypoints_3d: np.ndarray  # (70, 3) metres, same frame as `vertices`
    keypoints_2d: np.ndarray  # (70, 2) pixels
    bbox: np.ndarray  # (4,) pixels: x, y, width, height


def load_prediction(path: str | Path) -> Sam3dPrediction:
    """Load one SAM 3D Body ``.npz`` file."""
    path = Path(path)
    with np.load(path) as data:
        return Sam3dPrediction(
            name=path.stem,
            identity_coeffs=data["shape_params"].astype(np.float32),
            model_parameters=data["mhr_model_params"].astype(np.float32),
            expression_coeffs=data["expr_params"].astype(np.float32),
            camera_translation=data["pred_cam_t"].astype(np.float64),
            vertices=data["pred_vertices"].astype(np.float64),
            keypoints_3d=data["pred_keypoints_3d"].astype(np.float64),
            keypoints_2d=data["pred_keypoints_2d"].astype(np.float64),
            bbox=data["bbox"].astype(np.float64),
        )


def load_predictions(source: str | Path = DATA_DIR, pattern: str = "*.npz") -> list[Sam3dPrediction]:
    """Load every prediction in a folder (sorted by filename), or a single file."""
    source = Path(source)
    paths = sorted(source.glob(pattern)) if source.is_dir() else [source]
    if not paths:
        raise FileNotFoundError(
            f"no {pattern} files in {source}. Run: python -m mhr_kit.assets  (downloads the examples into data/)"
        )
    return [load_prediction(path) for path in paths]


def mhr_inputs(
    predictions: list[Sam3dPrediction], device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack predictions into the three tensors ``MHR.forward`` expects.

    Returns:
        ``(identity_coeffs, model_parameters, face_expr_coeffs)`` with shapes
        ``(B, 45)``, ``(B, 204)`` and ``(B, 72)``.

    Note that MHR's ``forward`` broadcasts a single identity across the batch, so
    when every frame is the same person you can pass ``identity[:1]`` instead and
    save the (small) blendshape evaluation.
    """
    stack = lambda arrays: torch.from_numpy(np.stack(arrays)).float().to(device)  # noqa: E731
    return (
        stack([prediction.identity_coeffs for prediction in predictions]),
        stack([prediction.model_parameters for prediction in predictions]),
        stack([prediction.expression_coeffs for prediction in predictions]),
    )


def to_camera_space(vertices_cm: np.ndarray, prediction: Sam3dPrediction) -> np.ndarray:
    """Place MHR output in the estimator's camera frame.

    Args:
        vertices_cm: ``(V, 3)`` vertices straight out of ``MHR.forward``, in cm.
        prediction: The prediction the parameters came from (for its camera translation).

    Returns:
        ``(V, 3)`` vertices in camera space, still in cm, ready for
        :func:`mhr_kit.render.render` with a camera from :func:`fit_camera`.
    """
    return vertices_cm @ MHR_TO_CAMERA.T + CM_PER_M * prediction.camera_translation


def fit_camera(
    predictions: list[Sam3dPrediction],
    image_size: tuple[int, int] | None = None,
) -> tuple[Camera, float]:
    """Recover the pinhole camera that produced the 2D keypoints.

    SAM 3D Body stores 3D keypoints in camera space and their 2D projections in
    image pixels, but not the intrinsics it used. Since projection is linear in
    the focal length and principal point once the 3D points are known --
    ``u * z = f * x + cx * z`` -- they can be recovered exactly by least squares
    from the correspondences. All people detected in one image share one camera,
    so passing several predictions makes the fit more robust.

    Args:
        predictions: Predictions from the *same* image.
        image_size: ``(width, height)`` of that image. If omitted it is inferred
            from the principal point, assuming it sits at the image centre.

    Returns:
        ``(camera, residual)``. The camera has identity rotation and zero
        translation: feed it vertices that are already in camera space
        (:func:`to_camera_space`). ``residual`` is the largest reprojection error
        in pixels -- expect ~1e-4 on genuine SAM 3D Body output, and treat a large
        value as a sign that the predictions come from different images.
    """
    rows, targets = [], []
    for prediction in predictions:
        points = prediction.keypoints_3d + prediction.camera_translation
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        u, v = prediction.keypoints_2d[:, 0], prediction.keypoints_2d[:, 1]
        zeros = np.zeros_like(z)
        # Unknowns: [focal, cx, cy].
        rows.append(np.stack([x, z, zeros], axis=1))
        targets.append(u * z)
        rows.append(np.stack([y, zeros, z], axis=1))
        targets.append(v * z)

    solution, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(targets), rcond=None)
    focal, cx, cy = (float(value) for value in solution)

    residual = 0.0
    for prediction in predictions:
        points = prediction.keypoints_3d + prediction.camera_translation
        projected = points[:, :2] / points[:, 2:3] * focal + np.array([cx, cy])
        residual = max(residual, float(np.abs(projected - prediction.keypoints_2d).max()))

    if image_size is None:
        image_size = (int(round(2 * cx)), int(round(2 * cy)))
    camera = Camera(
        focal=focal,
        center=(cx, cy),
        size=image_size,
        rotation=np.eye(3),
        translation=np.zeros(3),
    )
    return camera, residual
