"""CameraView2D and CameraView3D: camera state extracted from pygfx cameras."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygfx as gfx


@dataclass(frozen=True)
class CameraView2D:
    """Camera state for a 2D orthographic view.

    Parameters
    ----------
    bounds : tuple of float
        World-space extents of the visible area as ``(xmin, xmax, ymin, ymax)``.
    viewport_size_px : tuple of int
        Width and height of the viewport in pixels as ``(width, height)``.
    world_per_pixel : float
        Number of world-space units per pixel (world width divided by viewport
        width).
    """

    bounds: tuple[float, float, float, float]
    viewport_size_px: tuple[int, int]
    world_per_pixel: float


@dataclass(frozen=True)
class CameraView3D:
    """Camera state for a 3D perspective view.

    Parameters
    ----------
    frustum_corners : numpy.ndarray
        Shape ``(8, 3)`` array of the eight frustum corners in world
        coordinates. The first four rows are the near-plane corners and the
        last four rows are the far-plane corners.
    camera_position : numpy.ndarray
        Shape ``(3,)`` array giving the camera position in world coordinates.
    viewport_size_px : tuple of int
        Width and height of the viewport in pixels as ``(width, height)``.
    fov_y_rad : float
        Vertical field-of-view in radians.
    """

    frustum_corners: np.ndarray
    camera_position: np.ndarray
    viewport_size_px: tuple[int, int]
    fov_y_rad: float


def camera_view_from_gfx_3d(
    camera: gfx.PerspectiveCamera, canvas
) -> CameraView3D:
    """Build a CameraView3D from a pygfx perspective camera.

    Parameters
    ----------
    camera : gfx.PerspectiveCamera
        The pygfx perspective camera whose state will be captured.
    canvas : pygfx-compatible canvas
        The canvas associated with the camera, used to query the current
        viewport size via ``get_logical_size()``.

    Returns
    -------
    CameraView3D
        A frozen snapshot of the camera's frustum corners, position,
        viewport size, and vertical field-of-view.
    """
    width_px, height_px = canvas.get_logical_size()
    return CameraView3D(
        frustum_corners=np.asarray(camera.frustum, dtype=np.float64).copy(),
        camera_position=np.array(camera.world.position, dtype=np.float64),
        viewport_size_px=(int(width_px), int(height_px)),
        fov_y_rad=math.radians(camera.fov),
    )


def camera_view_from_gfx_2d(
    camera: gfx.OrthographicCamera, canvas
) -> CameraView2D:
    """Build a CameraView2D from a pygfx orthographic camera.

    The pygfx ``OrthographicCamera`` width and height must be scaled by the
    canvas aspect ratio to obtain the true world-space extents; this function
    performs that correction automatically.

    Parameters
    ----------
    camera : gfx.OrthographicCamera
        The pygfx orthographic camera whose state will be captured.
    canvas : pygfx-compatible canvas
        The canvas associated with the camera, used to query the current
        viewport size via ``get_logical_size()``.

    Returns
    -------
    CameraView2D
        A frozen snapshot of the camera's world-space bounds, viewport size,
        and world-units-per-pixel scale.
    """
    # Get the size of the canvas
    width_px, height_px = canvas.get_logical_size()
    canvas_aspect = width_px / height_px

    # Get the center point of the camera
    cx, cy = float(camera.world.position[0]), float(camera.world.position[1])

    # Get the height and width of the camera in world space.
    # The OrthographicCamera width and height need to be scaled
    # by the canvas size. See pygfx OrthographicCamera docstring.
    cam_width, cam_height = float(camera.width), float(camera.height)
    cam_aspect = cam_width / cam_height
    if canvas_aspect >= cam_aspect:
        world_height = cam_height
        world_width = cam_height * canvas_aspect
    else:
        world_width = cam_width
        world_height = cam_width / canvas_aspect

    # Calculate the world extents
    xmin, xmax = cx - world_width / 2, cx + world_width / 2
    ymin, ymax = cy - world_height / 2, cy + world_height / 2
    return CameraView2D(
        bounds=(xmin, xmax, ymin, ymax),
        viewport_size_px=(int(width_px), int(height_px)),
        world_per_pixel=cam_width / max(width_px, 1),
    )
