# Copyright (c) 2026 -- helper utilities for the MHR body model. Apache-2.0, like MHR itself.
"""Loading MHR and reading its outputs.

:class:`mhr.mhr.MHR` is a plain ``torch.nn.Module``. Its call signature is::

    vertices, skeleton_state = model(identity_coeffs, model_parameters, face_expr_coeffs)

with

===================  ================  =========================================
argument             shape             meaning
===================  ================  =========================================
``identity_coeffs``  ``(1 or B, 45)``  who the person is (body/head/hand shape)
``model_parameters`` ``(B, 204)``      how they are posed and proportioned
``face_expr_coeffs`` ``(B, 72)``       facial expression, or ``None`` for neutral
===================  ================  =========================================

and outputs

=====================  =================  =======================================
output                 shape              meaning
=====================  =================  =======================================
``vertices``           ``(B, V, 3)``      skinned mesh in cm, +y up
``skeleton_state``     ``(B, 127, 8)``    per joint ``(tx,ty,tz, qx,qy,qz,qw, s)``
=====================  =================  =======================================

``V`` depends on the level of detail (18 439 at LOD 1). The skeleton state is in
world space, so :func:`joint_positions` is just a slice.

This module adds the conveniences the upstream package leaves to the caller:
asset-aware loading, a printable summary, and small readers for the outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mhr.mhr import MHR

from .assets import ASSET_DIR, LOD_BLENDSHAPE_MB, download_assets, members_for_lod

NUM_IDENTITY_COEFFS = 45
NUM_MODEL_PARAMETERS = 204
NUM_EXPRESSION_COEFFS = 72


def load_mhr(
    lod: int = 1,
    device: str | torch.device = "cpu",
    assets: Path = ASSET_DIR,
    correctives: bool = True,
    download: bool = True,
) -> MHR:
    """Load an MHR model, downloading the assets it needs if they are missing.

    Args:
        lod: Level of detail, 0 (densest, ~2.6 GB of correctives) to 6 (coarsest).
            LOD 1 is the model everything else in MHR is defined against.
        device: ``"cpu"``, ``"cuda"``, or a :class:`torch.device`. Apple Silicon
            users should stay on CPU: ``pymomentum`` has no MPS backend.
        assets: Folder holding ``lod{n}.fbx``, ``compact_v6_1.model`` and the
            corrective ``.npz`` files.
        correctives: Load the non-linear pose correctives. Turning them off saves
            most of the memory and some time, at the cost of the pose-dependent
            deformations that distinguish MHR from a plain linear blend-skinned rig.
        download: Fetch any missing assets automatically (see :mod:`mhr_kit.assets`).

    Returns:
        The loaded :class:`mhr.mhr.MHR` module, in eval mode.
    """
    assets = Path(assets)
    missing = [name for name in members_for_lod(lod) if not (assets / Path(name).name).exists()]
    if missing:
        if not download:
            raise FileNotFoundError(
                f"missing assets for LOD {lod} in {assets}: {', '.join(Path(m).name for m in missing)}\n"
                f"Run: python -m mhr_kit.assets --lod {lod}"
            )
        download_assets(lod=lod, dest=assets)

    device = torch.device(device)
    model = MHR.from_files(folder=assets, device=device, lod=lod, wants_pose_correctives=correctives)
    model.eval()
    return model


def faces(model: MHR) -> np.ndarray:
    """Return the ``(F, 3)`` triangle indices of the model's mesh."""
    return np.asarray(model.character.mesh.faces)


def parameter_names(model: MHR) -> list[str]:
    """Return the names of the 204 model parameters, in order.

    The names are self-describing and worth reading once: ``root_tx``..``root_rz``
    are the 6 rigid degrees of freedom, then ~124 joint rotations such as
    ``l_elbow_bend`` or ``r_uparm_twist``, then bone-length and ``scale_*``
    parameters that stretch individual body parts.
    """
    return list(model.character.parameter_transform.names)[:NUM_MODEL_PARAMETERS]


def joint_names(model: MHR) -> list[str]:
    """Return the 127 skeleton joint names, in the order used by the skeleton state."""
    return list(model.character.skeleton.joint_names)


def joint_positions(skeleton_state: torch.Tensor) -> torch.Tensor:
    """Extract world-space joint positions ``(..., J, 3)`` in cm from a skeleton state."""
    return skeleton_state[..., :3]


def joint_rotations(skeleton_state: torch.Tensor) -> torch.Tensor:
    """Extract world-space joint rotations ``(..., J, 4)`` as ``xyzw`` quaternions."""
    return skeleton_state[..., 3:7]


def model_summary(model: MHR, lod: int | None = None) -> str:
    """Return a printable description of a loaded model."""
    vertex_count = int(np.asarray(model.character.mesh.vertices).shape[0])
    lines = [
        f"vertices                {vertex_count:,}",
        f"triangles               {faces(model).shape[0]:,}",
        f"skeleton joints         {len(joint_names(model))}",
        f"identity coefficients   {NUM_IDENTITY_COEFFS} (20 body + 20 head + 5 hands)",
        f"model parameters        {NUM_MODEL_PARAMETERS} (6 rigid + joint angles + scales)",
        f"expression coefficients {NUM_EXPRESSION_COEFFS}",
        f"pose correctives        {'on' if model.pose_correctives_model is not None else 'off'}",
    ]
    if lod is not None:
        lines.insert(0, f"level of detail         {lod} (correctives ~{LOD_BLENDSHAPE_MB[lod]} MB in RAM)")
    return "\n".join("  " + line for line in lines)


def save_mesh(vertices: torch.Tensor | np.ndarray, model: MHR, path: str | Path) -> Path:
    """Write one posed mesh to any format ``trimesh`` supports (``.ply``, ``.obj``, ...).

    Args:
        vertices: ``(V, 3)`` vertices in cm; a leading batch dimension of 1 is fine.
        model: The model the vertices came from (used for the triangle indices).
        path: Output path; the extension picks the format.
    """
    import trimesh

    array = vertices.detach().cpu().numpy() if isinstance(vertices, torch.Tensor) else np.asarray(vertices)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"save_mesh expects a single mesh, got a batch of {array.shape[0]}")
        array = array[0]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # process=False keeps MHR's vertex order intact, so vertex indices stay comparable.
    trimesh.Trimesh(vertices=array, faces=faces(model), process=False).export(path)
    return path
