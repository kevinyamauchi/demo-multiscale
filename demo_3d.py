"""demo_3d.py — 3-D multiscale brick-cache viewer (perspective + orbit).

Usage:

    uv run demo_3d.py --zarr-path ./example.ome.zarr

Controls:
    Orbit:  left-drag
    Zoom:   scroll
    Pan:    right-drag / middle-drag
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from uuid import uuid4

import numpy as np
import pygfx as gfx
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QTimer
from rendercanvas.qt import QRenderWidget
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider

from demo_multiscale.data_store import OMEZarrImageDataStore
from demo_multiscale.render_visual import (
    GFXMultiscaleImageVisual,
    VolumeGeometry,
)
from demo_multiscale.slicer import AsyncSlicer
from demo_multiscale.state import AxisAlignedSelectionState, DimsState


# Constants for rendering settings
INITIAL_CLIM_LOW: float = 0.0
INITIAL_CLIM_HIGH: float = 1000.0
INITIAL_ISO_THRESHOLD: float = 0.2
LOD_BIAS: float = 1.0
BLOCK_SIZE: int = 32
GPU_BUDGET_3D_BYTES: int = 4096 * 1024**2
CHUNKS_PER_FRAME: int = 32  # max GPU uploads per draw frame


def _dtype_max(dtype: np.dtype) -> float:
    """Return the maximum representable value for a numpy dtype."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return float(np.finfo(dtype).max)


def reslice_3d(
    visual: GFXMultiscaleImageVisual,
    data_store: OMEZarrImageDataStore,
    camera: gfx.PerspectiveCamera,
    canvas,
    slicer: AsyncSlicer,
    upload_queue: collections.deque,
    lod_bias: float = LOD_BIAS,
) -> None:
    visual.cancel_pending()
    upload_queue.clear()  # drop any items queued for the previous reslice

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

    slice_id = requests[0].slice_request_id if requests else None
    print(f"[RESLICE] {len(requests)} requests, slice_id={str(slice_id)[:8] if slice_id else 'none'}")

    def on_batch(batch):
        # Drop batches that were already in the Qt signal queue when a newer
        # reslice superseded this one. current_slice_id is updated on the main
        # thread by submit(), so this check is always safe here.
        if slicer.current_slice_id != slice_id:
            print(f"[CALLBACK] dropping stale batch, "
                  f"batch={str(slice_id)[:8]}, current={str(slicer.current_slice_id)[:8]}")
            return
        upload_queue.extend(batch)

    slicer.submit(requests, fetch_fn=data_store.get_data, callback=on_batch)


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
        clim=(INITIAL_CLIM_LOW, INITIAL_CLIM_HIGH),
        threshold=INITIAL_ISO_THRESHOLD,
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
    upload_queue: collections.deque = collections.deque()

    # ── Settle timer (debounce 3-D reslice until camera is still) ────────
    settle_ms = [150]

    settle_timer = QTimer()
    settle_timer.setSingleShot(True)
    settle_timer.timeout.connect(
        lambda: reslice_3d(visual, data_store, camera, canvas, slicer_obj, upload_queue)
    )

    # ── View-change detection (compare state each frame) ─────────────────
    _last_view_state: list[tuple[object, ...] | None] = [None]

    def _active_view_state() -> tuple[object, ...]:
        return (
            tuple(camera.world.position),
            tuple(camera.world.rotation),
            float(camera.fov),
            tuple(canvas.get_logical_size()),
        )

    def _camera_changed() -> bool:
        s = _active_view_state()
        if s != _last_view_state[0]:
            _last_view_state[0] = s
            return True
        return False

    # ── Draw function (called each frame by QRenderWidget) ───────────────
    def draw_frame():
        if _camera_changed():
            settle_timer.start(settle_ms[0])
        if upload_queue:
            drain = [upload_queue.popleft() for _ in range(min(CHUNKS_PER_FRAME, len(upload_queue)))]
            visual.on_data_ready(drain)
        renderer.render(scene, camera)

    canvas.request_draw(draw_frame)
    # Seed the last-known view state so the very first frame fires a reslice.
    _last_view_state[0] = None  # already None; explicit for clarity

    # ── Qt side panel ────────────────────────────────────────────────────
    panel = _build_panel(visual, data_store, camera, canvas, slicer_obj, settle_ms, upload_queue)

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
    settle_ms: list[int],
    upload_queue: collections.deque,
) -> QtWidgets.QWidget:
    panel = QtWidgets.QWidget()
    panel.setFixedWidth(220)
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setAlignment(Qt.AlignTop)

    dtype_max = _dtype_max(np.dtype(data_store.dtype))

    # Info label
    shape_str = " × ".join(str(s) for s in data_store.level_shapes[0])
    info = QtWidgets.QLabel(
        f"Shape: {shape_str}\n"
        f"Levels: {data_store.n_levels}\n"
        f"Axes: {' '.join(data_store.axis_names)}\n"
        f"Dtype: {data_store.dtype}\n"
        f"Scales: {data_store.voxel_sizes}"
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    # Contrast limits
    clim_group = QtWidgets.QGroupBox("Contrast limits")
    clim_layout = QtWidgets.QVBoxLayout(clim_group)
    clim_slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
    clim_slider.setRange(0.0, dtype_max)
    clim_slider.setValue((INITIAL_CLIM_LOW, INITIAL_CLIM_HIGH))

    def on_clim_change(values: tuple[float, float]) -> None:
        if visual.material_3d is not None:
            visual.material_3d.clim = values

    clim_slider.valueChanged.connect(on_clim_change)
    clim_layout.addWidget(clim_slider)
    layout.addWidget(clim_group)

    # ISO threshold
    threshold_group = QtWidgets.QGroupBox("ISO threshold")
    threshold_layout = QtWidgets.QVBoxLayout(threshold_group)
    threshold_slider = QLabeledDoubleSlider(Qt.Horizontal)
    threshold_slider.setRange(0.0, dtype_max)
    threshold_slider.setValue(INITIAL_ISO_THRESHOLD)

    def on_threshold_change(value: float) -> None:
        if visual.material_3d is not None:
            visual.material_3d.threshold = value

    threshold_slider.valueChanged.connect(on_threshold_change)
    threshold_layout.addWidget(threshold_slider)
    layout.addWidget(threshold_group)

    # Render mode
    render_group = QtWidgets.QGroupBox("Render mode")
    rg_layout = QtWidgets.QVBoxLayout(render_group)
    mode_combo = QtWidgets.QComboBox()
    mode_combo.addItems(["iso", "mip"])
    rg_layout.addWidget(mode_combo)

    def on_mode_change(value):
        if visual.material_3d is not None:
            visual.material_3d.render_mode = value
        reslice_3d(visual, data_store, camera, canvas, slicer_obj, upload_queue)

    mode_combo.currentTextChanged.connect(on_mode_change)
    layout.addWidget(render_group)

    # Settle time
    settle_group = QtWidgets.QGroupBox("Camera settle (ms)")
    settle_layout = QtWidgets.QVBoxLayout(settle_group)
    settle_spin = QtWidgets.QSpinBox()
    settle_spin.setRange(0, 2000)
    settle_spin.setValue(settle_ms[0])

    def on_settle_change(v: int) -> None:
        settle_ms[0] = v

    settle_spin.valueChanged.connect(on_settle_change)
    settle_layout.addWidget(settle_spin)
    layout.addWidget(settle_group)

    layout.addStretch()

    return panel



if __name__ == "__main__":
    main()
