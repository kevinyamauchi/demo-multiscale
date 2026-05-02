"""CameraView2D and CameraView3D — camera state extracted from pygfx cameras."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygfx as gfx


@dataclass(frozen=True)
class CameraView2D:
    bounds: tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax) world coords
    viewport_size_px: tuple[int, int]           # (width, height)
    world_per_pixel: float                      # world_width / viewport_width


@dataclass(frozen=True)
class CameraView3D:
    frustum_corners: np.ndarray   # shape (8, 3): 4 near + 4 far corners, world coords
    camera_position: np.ndarray   # shape (3,), world coords
    viewport_size_px: tuple[int, int]
    fov_y_rad: float


def camera_view_from_gfx_3d(
    camera: gfx.PerspectiveCamera, canvas
) -> CameraView3D:
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
