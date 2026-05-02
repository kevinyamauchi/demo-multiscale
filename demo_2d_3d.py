"""demo_2d_3d.py — toggle between 3-D brick renderer and 2-D tile renderer.

Usage::

    uv run demo_2d_3d.py --zarr-path /tmp/example.ome.zarr

Controls:
    3-D mode:  orbit (left-drag), zoom (scroll), pan (right-drag)
    2-D mode:  pan (left-drag), zoom (scroll)
    Toggle:    "Toggle 2-D / 3-D" button in side panel
    Z-slice:   spin box (2-D mode only)
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
    ImageGeometry2D,
    VolumeGeometry,
)
from demo_multiscale.slicer import AsyncSlicer
from demo_multiscale.state import AxisAlignedSelectionState, DimsState

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

INITIAL_CLIM_LOW: float = 0.0
INITIAL_CLIM_HIGH: float = 1000.0
INITIAL_ISO_THRESHOLD: float = 0.2
LOD_BIAS: float = 1.0
BLOCK_SIZE: int = 32
GPU_BUDGET_3D_BYTES: int = 4096 * 1024**2
GPU_BUDGET_2D_BYTES: int = 64 * 1024**2
CHUNKS_PER_FRAME_3D: int = 32   # max GPU uploads per draw frame (3-D bricks)
CHUNKS_PER_FRAME_2D: int = 64   # max GPU uploads per draw frame (2-D tiles, smaller)


def _dtype_max(dtype: np.dtype) -> float:
    """Return the maximum representable value for a numpy dtype."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return float(np.finfo(dtype).max)


# ---------------------------------------------------------------------------
# Reslice helpers
# ---------------------------------------------------------------------------


def reslice_3d(
    visual: GFXMultiscaleImageVisual,
    data_store: OMEZarrImageDataStore,
    camera: gfx.PerspectiveCamera,
    canvas,
    slicer: AsyncSlicer,
    axis_labels: tuple[str, ...],
    spatial_axes: tuple[int, ...],
    upload_queue: collections.deque,
    lod_bias: float = LOD_BIAS,
) -> None:
    visual.cancel_pending()
    upload_queue.clear()  # drop any items queued for the previous reslice

    camera_pos = np.array(camera.world.position, dtype=np.float64)
    frustum_corners = np.asarray(camera.frustum, dtype=np.float64).copy()
    _width_px, height_px = canvas.get_logical_size()
    fov_y_rad = math.radians(camera.fov)

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

    def on_batch(batch):
        # Drop batches that were already in the Qt signal queue when a newer
        # reslice superseded this one.
        if slicer.current_slice_id != slice_id:
            return
        upload_queue.extend(batch)

    slicer.submit(requests, fetch_fn=data_store.get_data, callback=on_batch)


def reslice_2d(
    visual: GFXMultiscaleImageVisual,
    data_store: OMEZarrImageDataStore,
    camera: gfx.OrthographicCamera,
    canvas,
    slicer: AsyncSlicer,
    axis_labels: tuple[str, ...],
    yx_axes: tuple[int, int],
    z_axis: int,
    z_slice: int,
    upload_queue: collections.deque,
    lod_bias: float = LOD_BIAS,
) -> None:
    visual.cancel_pending_2d()
    visual.invalidate_2d_cache()
    upload_queue.clear()  # drop any items queued for the previous reslice

    camera_pos = np.array(camera.world.position, dtype=np.float64)
    width_px, _height_px = canvas.get_logical_size()
    world_width = float(camera.width)

    dims_state = DimsState(
        axis_labels=axis_labels,
        selection=AxisAlignedSelectionState(
            displayed_axes=yx_axes,
            slice_indices={z_axis: z_slice},
        ),
    )

    requests = visual.build_slice_request_2d(
        camera_pos_world=camera_pos,
        viewport_width_px=float(width_px),
        world_width=world_width,
        view_min_world=None,
        view_max_world=None,
        dims_state=dims_state,
        lod_bias=lod_bias,
    )

    slice_id = requests[0].slice_request_id if requests else None

    def on_batch(batch):
        # Drop batches that were already in the Qt signal queue when a newer
        # reslice superseded this one.
        if slicer.current_slice_id != slice_id:
            return
        upload_queue.extend(batch)

    slicer.submit(requests, fetch_fn=data_store.get_data, callback=on_batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_visual(
    data_store: OMEZarrImageDataStore,
    voxel_scales: list[float],
) -> tuple[GFXMultiscaleImageVisual, tuple[int, ...], tuple[int, int], int]:
    """Build visual and return (visual, spatial_axes_3d, yx_axes, z_axis)."""
    shapes = data_store.level_shapes
    n_axes = len(shapes[0])

    # Last 3 axes: spatial ZYX
    spatial_axes = tuple(range(n_axes - 3, n_axes))
    z_axis, y_axis, x_axis = spatial_axes
    yx_axes = (y_axis, x_axis)

    shapes_3d = [tuple(s[ax] for ax in spatial_axes) for s in shapes]
    shapes_2d = [(s[y_axis], s[x_axis]) for s in shapes]

    level_transforms = data_store.level_transforms
    transforms_3d = [t.set_slice(spatial_axes) for t in level_transforms]
    transforms_2d = [t.set_slice(yx_axes) for t in level_transforms]

    volume_geometry = VolumeGeometry(
        level_shapes=shapes_3d,
        level_transforms=transforms_3d,
        block_size=BLOCK_SIZE,
    )

    image_geometry_2d = ImageGeometry2D(
        level_shapes=shapes_2d,
        block_size=BLOCK_SIZE,
        n_levels=data_store.n_levels,
        level_transforms=transforms_2d,
    )

    visual = GFXMultiscaleImageVisual(
        visual_model_id=uuid4(),
        volume_geometry=volume_geometry,
        image_geometry_2d=image_geometry_2d,
        render_modes={"2d", "3d"},
        displayed_axes=spatial_axes,
        colormap=gfx.cm.viridis,
        clim=(INITIAL_CLIM_LOW, INITIAL_CLIM_HIGH),
        threshold=INITIAL_ISO_THRESHOLD,
        voxel_scales=voxel_scales,
        full_level_shapes=list(shapes),
        full_level_transforms=list(level_transforms),
        gpu_budget_bytes_3d=GPU_BUDGET_3D_BYTES,
        gpu_budget_bytes_2d=GPU_BUDGET_2D_BYTES,
    )

    return visual, spatial_axes, yx_axes, z_axis


def main() -> None:
    parser = argparse.ArgumentParser(description="2-D / 3-D multiscale viewer")
    parser.add_argument("--zarr-path", required=True)
    args = parser.parse_args()

    # ── Load data store ──────────────────────────────────────────────────
    print(f"Opening: {args.zarr_path}")
    data_store = OMEZarrImageDataStore.from_path(args.zarr_path)
    print(f"  levels : {data_store.n_levels}")
    print(f"  shapes : {data_store.level_shapes}")
    print(f"  axes   : {data_store.axis_names}")
    print(f"  scales : {data_store.voxel_sizes}")

    vox_shape = np.array(data_store.level_shapes[0], dtype=np.float64)
    vs_full = np.array(data_store.voxel_sizes, dtype=np.float64)
    voxel_scales_3d = list(vs_full[-3:])  # ZYX

    world_extents = vox_shape * vs_full
    max_extent = float(world_extents.max())
    near = max(1.0, max_extent * 0.0001)
    far = max_extent * 10.0

    # ── Build visual ─────────────────────────────────────────────────────
    visual, spatial_axes, yx_axes, z_axis = build_visual(data_store, voxel_scales_3d)
    axis_labels = tuple(data_store.axis_names)

    z_depth = data_store.level_shapes[0][z_axis]

    # ── pygfx setup ──────────────────────────────────────────────────────
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    canvas = QRenderWidget(update_mode="continuous")
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    scene.add(gfx.AmbientLight())
    scene.add(visual.node_3d)

    # 3-D camera
    camera_3d = gfx.PerspectiveCamera(70, depth_range=(near, far))
    camera_3d.show_object(scene, up=(0, 0, 1))
    controller_3d = gfx.OrbitController(camera_3d, register_events=renderer)

    # 2-D camera (initially disabled)
    h0, w0 = data_store.level_shapes[0][yx_axes[0]], data_store.level_shapes[0][yx_axes[1]]
    world_w = float(w0) * float(voxel_scales_3d[2])
    world_h = float(h0) * float(voxel_scales_3d[1])
    camera_2d = gfx.OrthographicCamera(width=world_w, height=world_h)
    camera_2d.local.position = (world_w / 2.0, world_h / 2.0, 0.0)
    controller_2d = gfx.PanZoomController(camera_2d, register_events=renderer)
    controller_2d.enabled = False

    slicer_obj = AsyncSlicer()
    upload_queue_3d: collections.deque = collections.deque()
    upload_queue_2d: collections.deque = collections.deque()

    # ── Shared mutable state ─────────────────────────────────────────────
    state = {"mode": "3d", "z_slice": z_depth // 2}

    # ── Reslice dispatcher ───────────────────────────────────────────────
    def reslice():
        if state["mode"] == "3d":
            reslice_3d(
                visual, data_store, camera_3d, canvas, slicer_obj,
                axis_labels, spatial_axes, upload_queue_3d,
            )
        else:
            reslice_2d(
                visual, data_store, camera_2d, canvas, slicer_obj,
                axis_labels, yx_axes, z_axis, state["z_slice"], upload_queue_2d,
            )

    # ── Settle timer (debounce reslice until camera is still) ────────────
    settle_ms = [150]

    settle_timer = QTimer()
    settle_timer.setSingleShot(True)
    settle_timer.timeout.connect(reslice)

    # ── View-change detection (compare state each frame) ─────────────────
    _last_view_state: list[tuple[object, ...] | None] = [None]

    def _active_cam_state() -> tuple:
        logical_size = tuple(canvas.get_logical_size())
        if state["mode"] == "3d":
            return (
                "3d",
                tuple(camera_3d.world.position),
                tuple(camera_3d.world.rotation),
                float(camera_3d.fov),
                logical_size,
            )
        return (
            "2d",
            tuple(camera_2d.world.position),
            tuple(camera_2d.world.rotation),
            float(camera_2d.width),
            float(camera_2d.height),
            logical_size,
        )

    def _camera_changed() -> bool:
        s = _active_cam_state()
        if s != _last_view_state[0]:
            _last_view_state[0] = s
            return True
        return False

    # ── Draw function (called each frame by QRenderWidget) ───────────────
    def draw_frame():
        if _camera_changed():
            settle_timer.start(settle_ms[0])
        if upload_queue_3d:
            drain = [upload_queue_3d.popleft() for _ in range(min(CHUNKS_PER_FRAME_3D, len(upload_queue_3d)))]
            visual.on_data_ready(drain)
        if upload_queue_2d:
            drain = [upload_queue_2d.popleft() for _ in range(min(CHUNKS_PER_FRAME_2D, len(upload_queue_2d)))]
            visual.on_data_ready_2d(drain)
        active_cam = camera_3d if state["mode"] == "3d" else camera_2d
        renderer.render(scene, active_cam)

    canvas.request_draw(draw_frame)

    # ── Qt side panel ────────────────────────────────────────────────────
    panel = QtWidgets.QWidget()
    panel.setFixedWidth(240)
    p_layout = QtWidgets.QVBoxLayout(panel)
    p_layout.setAlignment(Qt.AlignTop)

    # Info
    shape_str = " × ".join(str(s) for s in data_store.level_shapes[0])
    info_label = QtWidgets.QLabel(
        f"Shape: {shape_str}\n"
        f"Levels: {data_store.n_levels}\n"
        f"Axes: {' '.join(data_store.axis_names)}\n"
        f"Scales: {data_store.voxel_sizes}"
    )
    info_label.setWordWrap(True)
    p_layout.addWidget(info_label)

    dtype_max = _dtype_max(np.dtype(data_store.dtype))

    # Contrast limits
    clim_group = QtWidgets.QGroupBox("Contrast limits")
    clim_layout = QtWidgets.QVBoxLayout(clim_group)
    clim_slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
    clim_slider.setRange(0.0, dtype_max)
    clim_slider.setValue((INITIAL_CLIM_LOW, INITIAL_CLIM_HIGH))

    def on_clim_change(values: tuple[float, float]) -> None:
        if visual.material_3d is not None:
            visual.material_3d.clim = values
        if visual.material_2d is not None:
            visual.material_2d.clim = values

    clim_slider.valueChanged.connect(on_clim_change)
    clim_layout.addWidget(clim_slider)
    p_layout.addWidget(clim_group)

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
    p_layout.addWidget(threshold_group)

    # Mode label
    mode_label = QtWidgets.QLabel("Mode: 3-D")
    p_layout.addWidget(mode_label)

    # Toggle button
    toggle_btn = QtWidgets.QPushButton("Toggle 2-D / 3-D")
    p_layout.addWidget(toggle_btn)

    # Render mode (3D only)
    render_group = QtWidgets.QGroupBox("Render mode")
    rg_layout = QtWidgets.QVBoxLayout(render_group)
    mode_combo = QtWidgets.QComboBox()
    mode_combo.addItems(["iso", "mip"])
    rg_layout.addWidget(mode_combo)
    p_layout.addWidget(render_group)

    def on_render_mode_change(value):
        if visual.material_3d is not None:
            visual.material_3d.render_mode = value
        reslice()

    mode_combo.currentTextChanged.connect(on_render_mode_change)

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
    p_layout.addWidget(settle_group)

    # Z-slice spinner (2D only, initially hidden)
    z_group = QtWidgets.QGroupBox("Z slice")
    z_layout = QtWidgets.QVBoxLayout(z_group)
    z_spin = QtWidgets.QSpinBox()
    z_spin.setRange(0, z_depth - 1)
    z_spin.setValue(state["z_slice"])
    z_layout.addWidget(z_spin)
    z_group.setVisible(False)
    p_layout.addWidget(z_group)

    z_spin.valueChanged.connect(
        lambda v: (state.update({"z_slice": v}), reslice())
    )

    # Toggle handler
    def on_toggle():
        visual.cancel_pending()
        visual.cancel_pending_2d()
        settle_timer.stop()
        _last_view_state[0] = None

        if state["mode"] == "3d":
            state["mode"] = "2d"
            scene.remove(visual.node_3d)
            scene.add(visual.node_2d)
            controller_3d.enabled = False
            controller_2d.enabled = True
            render_group.setVisible(False)
            z_group.setVisible(True)
            mode_label.setText("Mode: 2-D")
            # Fit 2-D camera to physical image footprint
            camera_2d.local.position = (world_w / 2.0, world_h / 2.0, 0.0)
            camera_2d.width = world_w
        else:
            state["mode"] = "3d"
            scene.remove(visual.node_2d)
            scene.add(visual.node_3d)
            controller_3d.enabled = True
            controller_2d.enabled = False
            render_group.setVisible(True)
            z_group.setVisible(False)
            mode_label.setText("Mode: 3-D")

        reslice()

    toggle_btn.clicked.connect(on_toggle)

    # ── Main window ──────────────────────────────────────────────────────
    win = QtWidgets.QWidget()
    win.setWindowTitle("demo_2d_3d")
    w_layout = QtWidgets.QHBoxLayout(win)
    w_layout.setContentsMargins(0, 0, 0, 0)
    w_layout.addWidget(canvas, stretch=1)
    w_layout.addWidget(panel)
    win.resize(1140, 700)
    win.show()

    app.exec()


if __name__ == "__main__":
    main()
