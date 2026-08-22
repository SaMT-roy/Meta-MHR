# Copyright (c) 2026 -- helper utilities for the MHR body model. Apache-2.0, like MHR itself.
"""Building MHR parameter vectors by name, plus a small library of poses.

MHR takes three parameter blocks. This module is about making them readable:

* ``identity_coeffs`` -- 45 numbers, roughly zero-mean/unit-variance. The first 20
  describe the body, the next 20 the head, the last 5 the hands
  (:data:`IDENTITY_GROUPS`).
* ``model_parameters`` -- 204 numbers. Unlike SMPL's flat axis-angle vector these
  are *named* rig controls: ``root_tx``..``root_rz`` for the 6 rigid degrees of
  freedom, then joint angles such as ``l_elbow_bend`` or ``r_upleg_rz``, then
  bone-length and ``scale_*`` controls. :func:`model_parameters` lets you set them
  by name instead of remembering indices.
* ``face_expr_coeffs`` -- 72 facial expression blendshape weights, usually in
  ``[-1, 1]``.

Conventions, measured against the LOD 1 rig (see ``README.md`` for how):

* Rotations are in **radians**; the rig is mirrored, so the same sign means the
  same anatomical motion on both sides (``l_uparm_rz`` and ``r_uparm_rz`` both
  raise their arm when positive).
* ``root_tx/ty/tz`` are in **units of 10 cm** -- ``root_ty = -4.5`` lowers the
  body by 45 cm -- while the mesh itself comes out in cm.
* The rest pose (all parameters zero) is a T-pose standing on ``y = 0`` and
  facing ``+z``.
"""

from __future__ import annotations

import numpy as np
import torch
from mhr.mhr import MHR

from .model import NUM_EXPRESSION_COEFFS, NUM_IDENTITY_COEFFS, NUM_MODEL_PARAMETERS, parameter_names

# Which identity coefficients affect which part of the body.
IDENTITY_GROUPS = {"body": slice(0, 20), "head": slice(20, 40), "hands": slice(40, 45)}

# A few poses, expressed as {parameter name: radians}. Everything not mentioned
# stays at 0, i.e. at its rest value.
POSES: dict[str, dict[str, float]] = {
    # The rest pose itself: arms straight out to the sides.
    "t_pose": {},
    # Arms lowered to ~30 degrees below horizontal, the usual authoring pose.
    "a_pose": {
        "l_uparm_ry": 0.10,
        "r_uparm_ry": 0.10,
        "l_uparm_rz": -0.40,
        "r_uparm_rz": -0.40,
        "l_elbow_bend": 0.10,
        "r_elbow_bend": 0.10,
    },
    # Arms hanging by the hips, elbows slightly bent: someone standing still.
    # `uparm_rz` alone would swing the arms backwards, so `uparm_ry` brings them
    # level with the body again (the wrists end up within a cm of z = 0).
    "relaxed": {
        "l_uparm_ry": -0.65,
        "r_uparm_ry": -0.65,
        "l_uparm_rz": -0.85,
        "r_uparm_rz": -0.85,
        "l_elbow_bend": 0.28,
        "r_elbow_bend": 0.28,
        "l_lowarm_twist": 0.35,
        "r_lowarm_twist": 0.35,
        "spine_bend0": 0.04,
    },
    # Both arms overhead in a V. `uparm_rz` alone swings the arm forwards, so a
    # true overhead raise needs `uparm_ry` (up and back) as well.
    "arms_up": {
        "l_uparm_ry": 1.00,
        "r_uparm_ry": 1.00,
        "l_uparm_rz": 1.50,
        "r_uparm_rz": 1.50,
        "l_elbow_bend": 0.15,
        "r_elbow_bend": 0.15,
    },
    # Right arm raised and bent so the hand is beside the head, left arm resting.
    "wave": {
        "r_uparm_ry": 1.05,
        "r_uparm_rz": 0.75,
        "r_elbow_bend": 1.15,
        "r_lowarm_twist": -0.20,
        "l_uparm_ry": -0.65,
        "l_uparm_rz": -0.85,
        "l_elbow_bend": 0.30,
    },
    # Hips and knees flexed, pelvis dropped 45 cm: sitting on a chair.
    "sitting": {
        "l_upleg_rz": -1.45,
        "r_upleg_rz": -1.45,
        "l_knee_bend": 1.50,
        "r_knee_bend": 1.50,
        "root_ty": -4.5,
        "l_uparm_ry": -0.55,
        "r_uparm_ry": -0.55,
        "l_uparm_rz": -0.70,
        "r_uparm_rz": -0.70,
        "l_elbow_bend": 0.55,
        "r_elbow_bend": 0.55,
        "spine_bend0": 0.10,
    },
    # Mid-stride: left leg forward, right leg back, with the opposite arm swing.
    # Around the hanging rest values, `uparm_ry` more negative swings an arm
    # forwards and less negative swings it back.
    "walk_stride": {
        "l_upleg_rz": -0.55,
        "l_knee_bend": 0.25,
        "r_upleg_rz": 0.35,
        "r_knee_bend": 0.55,
        "r_uparm_ry": -1.05,
        "r_uparm_rz": -0.85,
        "r_elbow_bend": 0.55,
        "l_uparm_ry": -0.25,
        "l_uparm_rz": -0.85,
        "l_elbow_bend": 0.35,
        "spine_twist0": -0.10,
    },
    # Knees and hips deeply flexed, torso leaning forward, arms out for balance.
    "squat": {
        "l_upleg_rz": -1.55,
        "r_upleg_rz": -1.55,
        "l_knee_bend": 1.85,
        "r_knee_bend": 1.85,
        "l_foot_bend": 0.45,
        "r_foot_bend": 0.45,
        "root_ty": -4.6,
        "spine_bend0": 0.40,
        "l_uparm_rz": 1.05,
        "r_uparm_rz": 1.05,
        "l_elbow_bend": 0.45,
        "r_elbow_bend": 0.45,
    },
}


def parameter_index(model: MHR) -> dict[str, int]:
    """Return ``{parameter name: column}`` for the 204 model parameters."""
    return {name: index for index, name in enumerate(parameter_names(model))}


def model_parameters(
    model: MHR,
    values: dict[str, float] | None = None,
    batch: int = 1,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a ``(batch, 204)`` model-parameter tensor from named values.

    Args:
        model: A loaded MHR model (only its parameter names are used).
        values: ``{parameter name: value}``; unset parameters stay at 0. Rotations
            are radians, root translations are in units of 10 cm.
        batch: Number of identical rows to produce.
        device: Device for the result.

    Returns:
        ``(batch, 204)`` float32 tensor, ready to pass to ``model(...)``.

    Raises:
        KeyError: If a name is not a model parameter, with near-miss suggestions.
    """
    parameters = torch.zeros(batch, NUM_MODEL_PARAMETERS, device=device)
    if not values:
        return parameters

    index = parameter_index(model)
    for name, value in values.items():
        if name not in index:
            similar = [candidate for candidate in index if name.split("_")[-1] in candidate][:6]
            raise KeyError(f"unknown model parameter {name!r}" + (f"; did you mean one of {similar}?" if similar else ""))
        parameters[:, index[name]] = value
    return parameters


def pose(model: MHR, name: str, batch: int = 1, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return the ``(batch, 204)`` parameters of a named pose from :data:`POSES`."""
    if name not in POSES:
        raise KeyError(f"unknown pose {name!r}; available: {', '.join(sorted(POSES))}")
    return model_parameters(model, POSES[name], batch=batch, device=device)


def random_identity(
    scale: float = 1.0,
    groups: tuple[str, ...] = ("body", "head", "hands"),
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample a (1, 45) identity vector directly on the target device."""
    
    # 1. Initialize an empty tensor directly on the target device
    identity = torch.zeros(1, NUM_IDENTITY_COEFFS, device=device)
    
    # 2. Create the random number generator directly on the target device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
        
    # 3. Generate random noise for the entire vector at once
    noise = torch.randn(1, NUM_IDENTITY_COEFFS, generator=generator, device=device) * scale
    
    # 4. Mask out the groups we don't want to randomize
    for group in IDENTITY_GROUPS:
        if group not in groups:
            span = IDENTITY_GROUPS[group]
            noise[:, span] = 0.0  # Keep these at the average (0)
            
    # 5. Add the noise to the base
    identity = identity + noise
    return identity


def random_expression(scale: float = 0.35, seed: int | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
    """Sample a ``(1, 72)`` facial expression vector (blendshape weights)."""
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    return (scale * torch.randn(1, NUM_EXPRESSION_COEFFS, generator=generator)).to(device)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Ease-in/ease-out curve on ``[0, 1]``, so keyframes are not hit at full speed."""
    return t * t * (3.0 - 2.0 * t)


def interpolate_keyframes(
    keyframes: list[torch.Tensor],
    frames_per_segment: int = 20,
    loop: bool = False,
    smooth: bool = True,
) -> torch.Tensor:
    """Interpolate between parameter keyframes to make an animation.

    Linear interpolation of MHR parameters is well behaved because they are joint
    *angles*, not matrices -- there is nothing to re-orthonormalise. The result is
    one big batch, which is also the fastest way to evaluate it: a single call to
    ``model(...)`` skins every frame at once.

    Args:
        keyframes: Poses to visit, each ``(1, 204)`` or ``(204,)``.
        frames_per_segment: Frames generated between consecutive keyframes.
        loop: Append a segment back to the first keyframe, for a seamless cycle.
        smooth: Ease in and out of each keyframe instead of moving linearly.

    Returns:
        ``(frames, 204)`` tensor of per-frame model parameters.
    """
    if len(keyframes) < 2:
        raise ValueError("need at least two keyframes")

    stacked = torch.stack([key.reshape(-1) for key in keyframes])
    if loop:
        stacked = torch.cat([stacked, stacked[:1]], dim=0)

    weights = np.linspace(0.0, 1.0, frames_per_segment, endpoint=False)
    if smooth:
        weights = _smoothstep(weights)

    segments = []
    for start, end in zip(stacked[:-1], stacked[1:]):
        blend = torch.from_numpy(weights).to(start.dtype)[:, None]
        segments.append(start[None, :] * (1.0 - blend) + end[None, :] * blend)
    segments.append(stacked[-1:])  # land exactly on the final keyframe
    return torch.cat(segments, dim=0)
