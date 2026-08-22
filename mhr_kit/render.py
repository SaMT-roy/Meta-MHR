# Copyright (c) 2026 -- helper utilities for the MHR body model. Apache-2.0, like MHR itself.
"""A small dependency-light renderer for MHR meshes.

MHR ships no renderer: :meth:`mhr.mhr.MHR.forward` returns vertices and a skeleton
state, and it is up to the caller to turn those into pictures. This module fills
that gap with a painter's-algorithm rasteriser built on Matplotlib's Agg backend,
which needs no GPU, no OpenGL context and no display -- so the demos also run over
SSH or in CI.

The pieces:

* :class:`Camera` -- a pinhole camera (OpenCV convention: +x right, +y down,
  +z into the scene) that maps MHR world coordinates to pixels.
* :func:`orbit_camera` -- builds a camera that automatically frames a mesh from a
  given azimuth/elevation, for "studio turntable" renders.
* :func:`render` -- flat-shaded render of one mesh, optionally composited over a
  background photograph.
* :func:`save_image`, :func:`save_video`, :func:`image_grid` -- output helpers.

Units: MHR works in **centimetres** with +y up. Everything here does too, and the
conversion to the metre-based SAM 3D Body camera happens in :mod:`mhr_kit.sam3d`.

Accuracy note: triangles are sorted back-to-front and drawn as flat polygons
instead of being depth-tested per pixel. For a closed body mesh with backfaces
culled that is visually indistinguishable from a z-buffer; for meshes that
self-intersect a lot you may see occasional bleed-through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

# Warm skin-like grey used when no colour is requested.
DEFAULT_COLOR = (0.82, 0.74, 0.68)
DEFAULT_BACKGROUND = (0.11, 0.12, 0.14)
# Direction the light travels, in camera space (from the upper left, towards the scene).
DEFAULT_LIGHT = (0.35, -0.45, 0.82)


@dataclass(frozen=True)
class Camera:
    """A pinhole camera with square pixels.

    Attributes:
        focal: Focal length in pixels (shared by x and y).
        center: Principal point ``(cx, cy)`` in pixels.
        size: Image ``(width, height)`` in pixels.
        rotation: ``(3, 3)`` rotation mapping world axes to camera axes.
        translation: ``(3,)`` camera-space position of the world origin, in cm.

    A world point ``p`` lands at ``project(p)``; the camera looks down its own
    ``+z`` axis, ``+x`` is right and ``+y`` is down, matching OpenCV and the
    convention SAM 3D Body uses.
    """

    focal: float
    center: tuple[float, float]
    size: tuple[int, int]
    rotation: np.ndarray
    translation: np.ndarray

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        """Transform world points ``(..., 3)`` in cm into camera space, in cm."""
        return points @ np.asarray(self.rotation, dtype=np.float64).T + np.asarray(self.translation, dtype=np.float64)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixels.

        Returns:
            ``(uv, depth)`` where ``uv`` is ``(..., 2)`` in pixels and ``depth`` is
            ``(...)`` camera-space z in cm (positive = in front of the camera).
        """
        camera_points = self.to_camera(points)
        depth = camera_points[..., 2]
        # Guard against division by ~0 for points level with the camera plane.
        safe_depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        uv = camera_points[..., :2] / safe_depth[..., None] * self.focal + np.asarray(self.center)
        return uv, depth


def _rotation_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """World-to-camera rotation for a camera orbiting a y-up scene.

    ``azimuth_deg = 0, elevation_deg = 0`` looks at the model's front from the
    ``+z`` side, with world +y appearing up in the image.
    """
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    # Flip y and z: world y-up/z-towards-viewer becomes camera y-down/z-into-scene.
    flip = np.diag([1.0, -1.0, -1.0])
    spin = np.array(
        [
            [np.cos(azimuth), 0.0, np.sin(azimuth)],
            [0.0, 1.0, 0.0],
            [-np.sin(azimuth), 0.0, np.cos(azimuth)],
        ]
    )
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elevation), -np.sin(elevation)],
            [0.0, np.sin(elevation), np.cos(elevation)],
        ]
    )
    return tilt @ flip @ spin


def orbit_camera(
    vertices: np.ndarray,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    size: tuple[int, int] = (512, 640),
    focal_scale: float = 1.6,
    margin: float = 1.12,
    center: np.ndarray | None = None,
    radius: float | None = None,
) -> Camera:
    """Build a camera that frames ``vertices`` from a given viewing direction.

    Args:
        vertices: ``(V, 3)`` or ``(B, V, 3)`` mesh vertices in cm. All of them are
            used to pick the distance, so a whole animation can be framed once and
            stay stable across frames.
        azimuth: Degrees around the vertical axis (0 = front, 90 = the model's left).
        elevation: Degrees above the horizon.
        size: Output ``(width, height)`` in pixels.
        focal_scale: Focal length as a multiple of the image width. Larger values
            are more telephoto (less perspective distortion).
        margin: Extra room around the subject; 1.0 means "exactly fits".
        center: Optional look-at point in cm; defaults to the mesh bounding-box centre.
        radius: Optional size of the region to frame, in cm. Defaults to enclosing
            every vertex; set it small (with ``center``) to zoom in on the face.

    Returns:
        A :class:`Camera` looking at the subject.
    """
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    look_at = np.asarray(center, dtype=np.float64) if center is not None else 0.5 * (points.min(0) + points.max(0))
    if radius is None:
        radius = float(np.linalg.norm(points - look_at, axis=1).max())

    width, height = size
    focal = focal_scale * width
    # Half-angle of the smaller image dimension, then the distance at which a
    # sphere of `radius` fits inside it.
    half_angle = np.arctan(0.5 * min(width, height) / focal)
    distance = radius / np.sin(half_angle) * margin

    rotation = _rotation_from_angles(azimuth, elevation)
    # Place the subject `distance` in front of the camera: camera-space origin
    # translation = -R @ look_at + [0, 0, distance].
    translation = -rotation @ look_at + np.array([0.0, 0.0, distance])
    return Camera(focal=focal, center=(width / 2.0, height / 2.0), size=size, rotation=rotation, translation=translation)


def render(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: Camera,
    color: tuple[float, float, float] | np.ndarray = DEFAULT_COLOR,
    background: np.ndarray | tuple[float, float, float] = DEFAULT_BACKGROUND,
    light: tuple[float, float, float] = DEFAULT_LIGHT,
    ambient: float = 0.28,
    alpha: float = 1.0,
) -> np.ndarray:
    """Flat-shade one mesh into an RGB image.

    Args:
        vertices: ``(V, 3)`` vertex positions in cm, in MHR world space.
        faces: ``(F, 3)`` triangle vertex indices.
        camera: Camera to render through; its ``size`` sets the output resolution.
        color: Base colour, either one RGB triple or ``(F, 3)`` per-triangle colours.
        background: RGB triple, or an ``(H, W, 3)`` image matching ``camera.size``
            (e.g. the photograph the pose was estimated from).
        light: Direction the light travels, in camera space.
        ambient: Fraction of the base colour visible in shadow, in ``[0, 1]``.
        alpha: Mesh opacity; below 1.0 the background shows through.

    Returns:
        ``(H, W, 3)`` ``uint8`` image.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    width, height = camera.size

    triangles = camera.to_camera(vertices)[faces]  # (F, 3, 3) in camera space
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    # Keep only triangles in front of the camera whose normal points towards it.
    depth = triangles[..., 2].mean(axis=1)
    visible = (normals[:, 2] < 0.0) & (triangles[..., 2].min(axis=1) > 1e-3)
    order = np.argsort(-depth)  # far to near: the painter's algorithm
    order = order[visible[order]]

    # Lambertian term against the view-space light, plus a constant ambient floor.
    lambert = np.clip(-(normals[order] @ np.asarray(light, dtype=np.float64)), 0.0, 1.0)
    shade = ambient + (1.0 - ambient) * lambert
    base = np.asarray(color, dtype=np.float64)
    face_colors = np.clip(shade[:, None] * (base[order] if base.ndim == 2 else base[None, :]), 0.0, 1.0)

    uv, _ = camera.project(vertices)

    figure = Figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_axis_off()
    axes.set_xlim(0.0, width)
    axes.set_ylim(height, 0.0)  # image row 0 at the top

    if isinstance(background, np.ndarray):
        axes.imshow(background, extent=(0.0, float(width), float(height), 0.0), interpolation="nearest")
    else:
        figure.patch.set_facecolor(background)
        axes.set_facecolor(background)

    axes.add_collection(
        PolyCollection(uv[faces][order], facecolors=face_colors, edgecolors="none", alpha=alpha, antialiaseds=True)
    )
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def merge_meshes(meshes: list[np.ndarray], faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine several meshes that share one topology into a single mesh.

    Handy for rendering everyone detected in a photograph at once: the merged mesh
    goes through :func:`render` in one pass, so the depth sort resolves occlusions
    between people as well as within each body.

    Args:
        meshes: List of ``(V, 3)`` vertex arrays, all using ``faces``.
        faces: ``(F, 3)`` triangle indices.

    Returns:
        ``(vertices, faces)`` for the combined mesh.
    """
    vertex_count = meshes[0].shape[0]
    vertices = np.concatenate(meshes, axis=0)
    combined_faces = np.concatenate([np.asarray(faces) + index * vertex_count for index in range(len(meshes))], axis=0)
    return vertices, combined_faces


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Write an ``(H, W, 3)`` ``uint8`` image to ``path`` (PNG/JPEG by extension)."""
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image)
    return path


def save_video(frames: list[np.ndarray], path: str | Path, fps: int = 30) -> Path:
    """Write frames to a video, falling back to an animated GIF.

    MP4 needs ``imageio-ffmpeg``; if that is unavailable (or the encoder fails)
    the frames are written as a GIF next to the requested path instead.

    Returns:
        The path actually written.
    """
    import imageio.v2 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # macro_block_size=1 keeps our exact resolution instead of padding to 16.
        with iio.get_writer(path, fps=fps, macro_block_size=1, quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)
        return path
    except Exception as error:  # noqa: BLE001 - any encoder problem falls back to GIF
        gif_path = path.with_suffix(".gif")
        print(f"  MP4 encoding unavailable ({error.__class__.__name__}), writing {gif_path.name} instead")
        iio.mimsave(gif_path, frames, duration=1.0 / fps, loop=0)
        return gif_path


def image_grid(images: list[np.ndarray], columns: int = 4, pad: int = 4, fill: int = 20) -> np.ndarray:
    """Tile equally sized images into a contact sheet, padded with ``fill`` grey."""
    if not images:
        raise ValueError("no images to tile")
    height, width = images[0].shape[:2]
    rows = -(-len(images) // columns)  # ceiling division
    sheet = np.full((rows * (height + pad) + pad, columns * (width + pad) + pad, 3), fill, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        top, left = pad + row * (height + pad), pad + column * (width + pad)
        sheet[top : top + height, left : left + width] = image
    return sheet
