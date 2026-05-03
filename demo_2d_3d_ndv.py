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
import sys
from uuid import uuid4

import numpy as np
import pygfx as gfx
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QTimer
from rendercanvas.qt import QRenderWidget
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider

from ndv.models._array_display_model import ArrayDisplayModel
from ndv.models._resolve import ResolvedDisplayState, resolve

from demo_multiscale_ndv._camera_view import camera_view_from_gfx_2d, camera_view_from_gfx_3d
from demo_multiscale_ndv._ome_zarr_wrapper import OMEZarrDataWrapper
from demo_multiscale_ndv._plan_slice import plan_slice_2d, plan_slice_3d
from demo_multiscale_ndv.render_visual import (
    GFXMultiscaleImageVisual,
    ImageGeometry2D,
    VolumeGeometry,
)
from demo_multiscale_ndv.slicer import AsyncSlicer

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
OVERLAP_3D: int = 3
OVERLAP_2D: int = 1
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
    wrapper: OMEZarrDataWrapper,
    camera: gfx.PerspectiveCamera,
    canvas,
    slicer: AsyncSlicer,
    resolved: ResolvedDisplayState,
    upload_queue: collections.deque,
    lod_bias: float = LOD_BIAS,
) -> None:
    """Plan and submit 3-D brick requests.

    The 3-D planner emits core brick indices and delegates overlap expansion to
    ``volume_handle.expand_fetch_index``. Result batches are filtered by
    ``slice_id`` before appending into ``upload_queue``.
    """
    visual.cancel_pending()
    upload_queue.clear()

    camera_view = camera_view_from_gfx_3d(camera, canvas)
    visual.begin_frame_3d(resolved.visible_axes)
    volume_handle = visual.volume_handle

    requests, _ = plan_slice_3d(
        camera_view=camera_view,
        cache_query=volume_handle.cache_query(),
        geo=visual.volume_geometry,
        voxel_scales=visual.voxel_scales,
        full_level_shapes=visual.full_level_shapes,
        world_to_level_transforms=visual.world_to_level_transforms,
        expand_fetch_index=volume_handle.expand_fetch_index,
        resolved=resolved,
        lod_bias=lod_bias,
    )

    slice_id = requests[0].slice_request_id if requests else None

    def on_batch(batch):
        if slicer.current_slice_id != slice_id:
            return
        upload_queue.extend(batch)

    slicer.submit(
        requests,
        fetch_fn=lambda req: wrapper.isel(req.index, level=req.level),
        callback=on_batch,
    )


def reslice_2d(
    visual: GFXMultiscaleImageVisual,
    wrapper: OMEZarrDataWrapper,
    camera: gfx.OrthographicCamera,
    canvas,
    slicer: AsyncSlicer,
    resolved: ResolvedDisplayState,
    upload_queue: collections.deque,
    lod_bias: float = LOD_BIAS,
) -> tuple[tuple[int, int], ...]:
    """Plan and submit 2-D tile requests.

    Returns the slice_coord used for this reslice so the caller can pass it to
    ``visual.on_data_ready_2d`` when draining the upload queue.
    """
    visual.cancel_pending_2d()
    visual.invalidate_2d_cache()
    upload_queue.clear()

    camera_view = camera_view_from_gfx_2d(camera, canvas)

    visible = set(resolved.visible_axes)
    slice_coord: tuple[tuple[int, int], ...] = tuple(sorted(
        (ax, v) for ax, v in resolved.current_index.items()
        if isinstance(v, int) and ax not in visible
    ))

    visual.begin_frame_2d(resolved.visible_axes)
    image_handle = visual.image_handle

    requests, _, target_level = plan_slice_2d(
        camera_view=camera_view,
        cache_query=image_handle.cache_query(),
        geo=visual.image_geometry_2d,
        voxel_scales=visual.voxel_scales,
        full_level_shapes=visual.full_level_shapes,
        world_to_level_transforms=visual.world_to_level_transforms,
        resolved=resolved,
        current_slice_coord=slice_coord,
        expand_fetch_index=image_handle.expand_fetch_index,
        lod_bias=lod_bias,
        use_culling=True,
    )

    n_evicted = image_handle.evict_finer_than(target_level)
    if n_evicted > 0 and not requests:
        image_handle.rebuild_lut(slice_coord)

    slice_id = requests[0].slice_request_id if requests else None

    def on_batch(batch):
        if slicer.current_slice_id != slice_id:
            return
        upload_queue.extend(batch)

    slicer.submit(
        requests,
        fetch_fn=lambda req: wrapper.isel(req.index, level=req.level),
        callback=on_batch,
    )

    return slice_coord


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_visual(
    wrapper: OMEZarrDataWrapper,
    voxel_scales: list[float],
) -> tuple[GFXMultiscaleImageVisual, tuple[int, ...], tuple[int, int], int]:
    """Build visual and return (visual, spatial_axes_3d, yx_axes, z_axis)."""
    shapes = wrapper.level_shapes
    n_axes = len(shapes[0])

    # Last 3 axes: spatial ZYX
    spatial_axes = tuple(range(n_axes - 3, n_axes))
    z_axis, y_axis, x_axis = spatial_axes
    yx_axes = (y_axis, x_axis)

    shapes_3d = [tuple(s[ax] for ax in spatial_axes) for s in shapes]
    shapes_2d = [(s[y_axis], s[x_axis]) for s in shapes]

    level_transforms = wrapper.level_transforms
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
        n_levels=wrapper.n_levels,
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
        overlap_3d=OVERLAP_3D,
        overlap_2d=OVERLAP_2D,
    )

    return visual, spatial_axes, yx_axes, z_axis


class Viewer(QtWidgets.QWidget):
    """Interactive 2-D/3-D multiscale viewer widget."""

    def __init__(self, wrapper: OMEZarrDataWrapper, parent=None):
        super().__init__(parent=parent)
        self.wrapper = wrapper

        self._init_models()
        self._init_rendering()
        self._init_ui()
        self._connect_signals()

    def _init_models(self) -> None:
        vox_shape = np.array(self.wrapper.level_shapes[0], dtype=np.float64)
        vs_full = np.array(self.wrapper.voxel_sizes, dtype=np.float64)
        self.voxel_scales_3d = list(vs_full[-3:])  # ZYX

        world_extents = vox_shape * vs_full
        max_extent = float(world_extents.max())
        self.near = max(1.0, max_extent * 0.0001)
        self.far = max_extent * 10.0

        self.visual, self.spatial_axes, self.yx_axes, self.z_axis = build_visual(
            self.wrapper, self.voxel_scales_3d
        )
        self.z_depth = self.wrapper.level_shapes[0][self.z_axis]

        self.display_model_3d = ArrayDisplayModel(visible_axes=self.spatial_axes)
        self.display_model_2d = ArrayDisplayModel(visible_axes=self.yx_axes)

        self.mode = "3d"
        self.z_slice = self.z_depth // 2
        self.slice_coord_2d: tuple[tuple[int, int], ...] = ()
        self.settle_ms = 150
        self._last_view_state: tuple[object, ...] | None = None

    def _init_rendering(self) -> None:
        self.canvas = QRenderWidget(update_mode="continuous")
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight())
        self.scene.add(self.visual.node_3d)

        self.camera_3d = gfx.PerspectiveCamera(70, depth_range=(self.near, self.far))
        self.camera_3d.show_object(self.scene, up=(0, 0, 1))
        self.controller_3d = gfx.OrbitController(self.camera_3d, register_events=self.renderer)

        h0 = self.wrapper.level_shapes[0][self.yx_axes[0]]
        w0 = self.wrapper.level_shapes[0][self.yx_axes[1]]
        self.world_w = float(w0) * float(self.voxel_scales_3d[2])
        self.world_h = float(h0) * float(self.voxel_scales_3d[1])

        self.camera_2d = gfx.OrthographicCamera(width=self.world_w, height=self.world_h)
        self.camera_2d.local.position = (self.world_w / 2.0, self.world_h / 2.0, 0.0)
        self.controller_2d = gfx.PanZoomController(self.camera_2d, register_events=self.renderer)
        self.controller_2d.enabled = False

        self.slicer_obj = AsyncSlicer()
        self.upload_queue_3d: collections.deque = collections.deque()
        self.upload_queue_2d: collections.deque = collections.deque()

        self.settle_timer = QTimer()
        self.settle_timer.setSingleShot(True)
        self.settle_timer.timeout.connect(self._reslice)

        self.canvas.request_draw(self._draw_frame)

    def _init_ui(self) -> None:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(240)
        p_layout = QtWidgets.QVBoxLayout(panel)
        p_layout.setAlignment(Qt.AlignTop)

        shape_str = " x ".join(str(s) for s in self.wrapper.level_shapes[0])
        info_label = QtWidgets.QLabel(
            f"Shape: {shape_str}\n"
            f"Levels: {self.wrapper.n_levels}\n"
            f"Axes: {' '.join(self.wrapper.axis_names)}\n"
            f"Scales: {self.wrapper.voxel_sizes}"
        )
        info_label.setWordWrap(True)
        p_layout.addWidget(info_label)

        dtype_max = _dtype_max(np.dtype(self.wrapper.dtype))

        clim_group = QtWidgets.QGroupBox("Contrast limits")
        clim_layout = QtWidgets.QVBoxLayout(clim_group)
        self.clim_slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
        self.clim_slider.setRange(0.0, dtype_max)
        self.clim_slider.setValue((INITIAL_CLIM_LOW, INITIAL_CLIM_HIGH))
        clim_layout.addWidget(self.clim_slider)
        p_layout.addWidget(clim_group)

        threshold_group = QtWidgets.QGroupBox("ISO threshold")
        threshold_layout = QtWidgets.QVBoxLayout(threshold_group)
        self.threshold_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0.0, dtype_max)
        self.threshold_slider.setValue(INITIAL_ISO_THRESHOLD)
        threshold_layout.addWidget(self.threshold_slider)
        p_layout.addWidget(threshold_group)

        self.mode_label = QtWidgets.QLabel("Mode: 3-D")
        p_layout.addWidget(self.mode_label)

        self.toggle_btn = QtWidgets.QPushButton("Toggle 2-D / 3-D")
        p_layout.addWidget(self.toggle_btn)

        self.render_group = QtWidgets.QGroupBox("Render mode")
        rg_layout = QtWidgets.QVBoxLayout(self.render_group)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["iso", "mip"])
        rg_layout.addWidget(self.mode_combo)
        p_layout.addWidget(self.render_group)

        settle_group = QtWidgets.QGroupBox("Camera settle (ms)")
        settle_layout = QtWidgets.QVBoxLayout(settle_group)
        self.settle_spin = QtWidgets.QSpinBox()
        self.settle_spin.setRange(0, 2000)
        self.settle_spin.setValue(self.settle_ms)
        settle_layout.addWidget(self.settle_spin)
        p_layout.addWidget(settle_group)

        self.z_group = QtWidgets.QGroupBox("Z slice")
        z_layout = QtWidgets.QVBoxLayout(self.z_group)
        self.z_spin = QtWidgets.QSpinBox()
        self.z_spin.setRange(0, self.z_depth - 1)
        self.z_spin.setValue(self.z_slice)
        z_layout.addWidget(self.z_spin)
        self.z_group.setVisible(False)
        p_layout.addWidget(self.z_group)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(panel)

    def _connect_signals(self) -> None:
        self.clim_slider.valueChanged.connect(self._on_clim_change)
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        self.mode_combo.currentTextChanged.connect(self._on_render_mode_change)
        self.settle_spin.valueChanged.connect(self._on_settle_change)
        self.z_spin.valueChanged.connect(self._on_z_slice_change)
        self.toggle_btn.clicked.connect(self._on_toggle)

    def _reslice(self) -> None:
        if self.mode == "3d":
            resolved = resolve(self.display_model_3d, self.wrapper)
            reslice_3d(
                self.visual,
                self.wrapper,
                self.camera_3d,
                self.canvas,
                self.slicer_obj,
                resolved,
                self.upload_queue_3d,
            )
            return

        self.display_model_2d.current_index[self.z_axis] = self.z_slice
        resolved = resolve(self.display_model_2d, self.wrapper)
        self.slice_coord_2d = reslice_2d(
            self.visual,
            self.wrapper,
            self.camera_2d,
            self.canvas,
            self.slicer_obj,
            resolved,
            self.upload_queue_2d,
        )

    def _active_cam_state(self) -> tuple[object, ...]:
        logical_size = tuple(self.canvas.get_logical_size())
        if self.mode == "3d":
            return (
                "3d",
                tuple(self.camera_3d.world.position),
                tuple(self.camera_3d.world.rotation),
                float(self.camera_3d.fov),
                logical_size,
            )
        return (
            "2d",
            tuple(self.camera_2d.world.position),
            tuple(self.camera_2d.world.rotation),
            float(self.camera_2d.width),
            float(self.camera_2d.height),
            logical_size,
        )

    def _camera_changed(self) -> bool:
        state = self._active_cam_state()
        if state != self._last_view_state:
            self._last_view_state = state
            return True
        return False

    def _draw_frame(self) -> None:
        if self._camera_changed():
            self.settle_timer.start(self.settle_ms)

        if self.upload_queue_3d:
            n_3d = min(CHUNKS_PER_FRAME_3D, len(self.upload_queue_3d))
            drain = [self.upload_queue_3d.popleft() for _ in range(n_3d)]
            self.visual.on_data_ready(drain)

        if self.upload_queue_2d:
            n_2d = min(CHUNKS_PER_FRAME_2D, len(self.upload_queue_2d))
            drain = [self.upload_queue_2d.popleft() for _ in range(n_2d)]
            self.visual.on_data_ready_2d(drain, self.slice_coord_2d)

        active_cam = self.camera_3d if self.mode == "3d" else self.camera_2d
        self.renderer.render(self.scene, active_cam)

    def _on_clim_change(self, values: tuple[float, float]) -> None:
        if self.visual.material_3d is not None:
            self.visual.material_3d.clim = values
        if self.visual.material_2d is not None:
            self.visual.material_2d.clim = values

    def _on_threshold_change(self, value: float) -> None:
        if self.visual.material_3d is not None:
            self.visual.material_3d.threshold = value

    def _on_render_mode_change(self, value: str) -> None:
        if self.visual.material_3d is not None:
            self.visual.material_3d.render_mode = value
        self._reslice()

    def _on_settle_change(self, value: int) -> None:
        self.settle_ms = value

    def _on_z_slice_change(self, value: int) -> None:
        self.z_slice = value
        self._reslice()

    def _on_toggle(self) -> None:
        self.visual.cancel_pending()
        self.visual.cancel_pending_2d()
        self.settle_timer.stop()
        self._last_view_state = None

        if self.mode == "3d":
            self.mode = "2d"
            self.scene.remove(self.visual.node_3d)
            self.scene.add(self.visual.node_2d)
            self.controller_3d.enabled = False
            self.controller_2d.enabled = True
            self.render_group.setVisible(False)
            self.z_group.setVisible(True)
            self.mode_label.setText("Mode: 2-D")
            self.camera_2d.local.position = (self.world_w / 2.0, self.world_h / 2.0, 0.0)
            self.camera_2d.width = self.world_w
        else:
            self.mode = "3d"
            self.scene.remove(self.visual.node_2d)
            self.scene.add(self.visual.node_3d)
            self.controller_3d.enabled = True
            self.controller_2d.enabled = False
            self.render_group.setVisible(True)
            self.z_group.setVisible(False)
            self.mode_label.setText("Mode: 3-D")

        self._reslice()


def main() -> None:
    parser = argparse.ArgumentParser(description="2-D / 3-D multiscale viewer")
    parser.add_argument("--zarr-path", required=True)
    args = parser.parse_args()

    print(f"Opening: {args.zarr_path}")
    wrapper = OMEZarrDataWrapper.from_path(args.zarr_path)
    print(f"  levels : {wrapper.n_levels}")
    print(f"  shapes : {wrapper.level_shapes}")
    print(f"  axes   : {wrapper.axis_names}")
    print(f"  scales : {wrapper.voxel_sizes}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer = Viewer(wrapper)
    viewer.setWindowTitle("demo_2d_3d")
    viewer.resize(1140, 700)
    viewer.show()
    app.exec()


if __name__ == "__main__":
    main()
