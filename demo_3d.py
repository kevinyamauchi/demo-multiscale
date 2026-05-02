"""demo_3d.py — 3-D multiscale brick-cache viewer (perspective + orbit).

Usage::

    uv run demo_3d.py --zarr-path /tmp/example.ome.zarr

Controls:
    Orbit:  left-drag
    Zoom:   scroll
    Pan:    right-drag / middle-drag
"""

from __future__ import annotations

import argparse
import math
import sys
from uuid import uuid4

import numpy as np
import pygfx as gfx
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from rendercanvas.qt import QRenderWidget

from demo_multiscale.data_store import OMEZarrImageDataStore
from demo_multiscale.render_visual import (
    GFXMultiscaleImageVisual,
    VolumeGeometry,
)
from demo_multiscale.slicer import AsyncSlicer
from demo_multiscale.state import AxisAlignedSelectionState, DimsState

# ---------------------------------------------------------------------------
# Tunable constants (adjust per dataset / hardware)
# ---------------------------------------------------------------------------

CLIM_LOW: float = 0.0
CLIM_HIGH: float = 1000.0
ISO_THRESHOLD: float = 0.2
LOD_BIAS: float = 1.0
BLOCK_SIZE: int = 32
GPU_BUDGET_3D_BYTES: int = 4096 * 1024**2


# ---------------------------------------------------------------------------
# Reslice helpers
# ---------------------------------------------------------------------------


def reslice_3d(
    visual: GFXMultiscaleImageVisual,
    data_store: OMEZarrImageDataStore,
    camera: gfx.PerspectiveCamera,
    canvas,
    slicer: AsyncSlicer,
    lod_bias: float = LOD_BIAS,
) -> None:
    visual.cancel_pending()

    camera_pos = np.array(camera.world.position, dtype=np.float64)
    frustum_corners = np.asarray(camera.frustum, dtype=np.float64).copy()
    _width_px, height_px = canvas.get_logical_size()
    fov_y_rad = math.radians(camera.fov)

    axis_labels = tuple(data_store.axis_names)
    n_spatial = min(3, len(axis_labels))
    spatial_axes = tuple(range(len(axis_labels) - n_spatial, len(axis_labels)))
    dims_state = DimsState(
        axis_labels=axis_labels,
        selection=AxisAlignedSelectionState(
            displayed_axes=spatial_axes,
            slice_indices={},
        ),
    )

    requests = visual.build_slice_request(
        camera_pos_world=camera_pos,
        frustum_corners_world=frustum_corners,
        fov_y_rad=fov_y_rad,
        screen_height_px=height_px,
        lod_bias=lod_bias,
        dims_state=dims_state,
    )

    def on_batch(batch):
        visual.on_data_ready(batch)

    slicer.submit(requests, fetch_fn=data_store.get_data, callback=on_batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_visual(
    data_store: OMEZarrImageDataStore,
    voxel_scales: list[float],
) -> GFXMultiscaleImageVisual:
    # Assume last 3 axes are spatial (z, y, x).
    n = data_store.n_levels
    shapes = data_store.level_shapes  # list of (z, y, x) tuples
    n_spatial = min(3, len(shapes[0]))
    spatial_idx = tuple(range(len(shapes[0]) - n_spatial, len(shapes[0])))

    shapes_3d = [tuple(s[ax] for ax in spatial_idx) for s in shapes]

    # Level transforms: scale factors between levels (level-k → level-0).
    level_transforms = data_store.level_transforms
    transforms_3d = [t.set_slice(spatial_idx) for t in level_transforms]

    volume_geometry = VolumeGeometry(
        level_shapes=shapes_3d,
        level_transforms=transforms_3d,
        block_size=BLOCK_SIZE,
    )

    visual = GFXMultiscaleImageVisual(
        visual_model_id=uuid4(),
        volume_geometry=volume_geometry,
        image_geometry_2d=None,
        render_modes={"3d"},
        displayed_axes=spatial_idx,
        colormap=gfx.cm.viridis,
        clim=(CLIM_LOW, CLIM_HIGH),
        threshold=ISO_THRESHOLD,
        voxel_scales=voxel_scales,
        full_level_shapes=list(shapes),
        full_level_transforms=list(level_transforms),
        gpu_budget_bytes_3d=GPU_BUDGET_3D_BYTES,
    )
    return visual


def main() -> None:
    parser = argparse.ArgumentParser(description="3-D multiscale brick-cache viewer")
    parser.add_argument("--zarr-path", required=True, help="Path or URI to OME-Zarr store")
    args = parser.parse_args()

    # ── Load data store ──────────────────────────────────────────────────
    print(f"Opening: {args.zarr_path}")
    data_store = OMEZarrImageDataStore.from_path(args.zarr_path)
    print(f"  levels  : {data_store.n_levels}")
    print(f"  shapes  : {data_store.level_shapes}")
    print(f"  axes    : {data_store.axis_names}")
    print(f"  dtype   : {data_store.dtype}")
    print(f"  scales  : {data_store.voxel_sizes}")

    # ── Depth range from world extents ───────────────────────────────────
    vox_shape = np.array(data_store.level_shapes[0], dtype=np.float64)
    vs = np.array(data_store.voxel_sizes, dtype=np.float64)
    world_extents = vox_shape * vs
    max_extent = float(world_extents.max())
    near = max(1.0, max_extent * 0.0001)
    far = max_extent * 10.0

    # ── Build visual ─────────────────────────────────────────────────────
    vs_3d = list(vs[-3:])  # last 3 axes = spatial ZYX
    visual = build_visual(data_store, vs_3d)

    # ── pygfx setup ──────────────────────────────────────────────────────
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    canvas = QRenderWidget(update_mode="continuous")
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    scene.add(gfx.AmbientLight())
    scene.add(visual.node_3d)

    camera = gfx.PerspectiveCamera(70, depth_range=(near, far))
    camera.show_object(scene, up=(0, 0, 1))
    controller = gfx.OrbitController(camera, register_events=renderer)

    slicer_obj = AsyncSlicer()

    # ── Camera-change detection (compare state each frame) ───────────────
    _last_cam_pos = [None]

    def _camera_changed() -> bool:
        pos = tuple(camera.world.position)
        rot = tuple(camera.world.rotation)
        s = (pos, rot)
        if s != _last_cam_pos[0]:
            _last_cam_pos[0] = s
            return True
        return False

    # ── Draw function (called each frame by QRenderWidget) ───────────────
    def draw_frame():
        if _camera_changed():
            reslice_3d(visual, data_store, camera, canvas, slicer_obj)
        renderer.render(scene, camera)

    canvas.request_draw(draw_frame)
    # Seed the last-known camera state so the very first frame fires a reslice.
    _last_cam_pos[0] = None  # already None; explicit for clarity

    # ── Qt side panel ────────────────────────────────────────────────────
    panel = _build_panel(visual, data_store, camera, canvas, slicer_obj)

    win = QtWidgets.QWidget()
    win.setWindowTitle("demo_3d")
    layout = QtWidgets.QHBoxLayout(win)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(canvas, stretch=1)
    layout.addWidget(panel)
    win.resize(1100, 700)
    win.show()

    app.exec()


def _build_panel(
    visual,
    data_store,
    camera,
    canvas,
    slicer_obj,
) -> QtWidgets.QWidget:
    panel = QtWidgets.QWidget()
    panel.setFixedWidth(220)
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setAlignment(Qt.AlignTop)

    # Info label
    shape_str = " × ".join(str(s) for s in data_store.level_shapes[0])
    info = QtWidgets.QLabel(
        f"Shape: {shape_str}\n"
        f"Levels: {data_store.n_levels}\n"
        f"Axes: {' '.join(data_store.axis_names)}\n"
        f"Dtype: {data_store.dtype}\n"
        f"Scales: {data_store.voxel_sizes}\n"
        f"clim: [{CLIM_LOW}, {CLIM_HIGH}]"
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    # Render mode
    render_group = QtWidgets.QGroupBox("Render mode")
    rg_layout = QtWidgets.QVBoxLayout(render_group)
    mode_combo = QtWidgets.QComboBox()
    mode_combo.addItems(["iso", "mip"])
    rg_layout.addWidget(mode_combo)

    def on_mode_change(value):
        if visual.material_3d is not None:
            visual.material_3d.render_mode = value
        reslice_3d(visual, data_store, camera, canvas, slicer_obj)

    mode_combo.currentTextChanged.connect(on_mode_change)
    layout.addWidget(render_group)

    layout.addStretch()

    return panel




if __name__ == "__main__":
    main()
