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

from demo_multiscale_ndv._camera_view import CameraView2D, CameraView3D, camera_view_from_gfx_2d, camera_view_from_gfx_3d
from demo_multiscale_ndv._handle import MultiscaleImageHandle, MultiscaleVolumeHandle
from demo_multiscale_ndv._ome_zarr_wrapper import OMEZarrDataWrapper
from demo_multiscale_ndv._plan_slice import (
    build_fetch_requests_2d,
    build_fetch_requests_3d,
    select_visible_bricks_2d,
    select_visible_bricks_3d,
)
from demo_multiscale_ndv.render_visual import (
    GFXMultiscaleImageVisual,
    MultiscaleBrickLayout2D,
    MultiscaleBrickLayout3D,
)
from demo_multiscale_ndv.slicer import AsyncSlicer
from demo_multiscale_ndv.transform import AffineTransform


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


def reslice_3d(
    volume_handle: MultiscaleVolumeHandle,
    wrapper: OMEZarrDataWrapper,
    camera_view: CameraView3D,
    slicer: AsyncSlicer,
    resolved: ResolvedDisplayState,
    upload_queue: collections.deque,
    lod_bias: float = LOD_BIAS,
    prev_slice_coord: tuple[tuple[int, int], ...] = (),
) -> tuple[tuple[int, int], ...]:
    """Plan and submit asynchronous 3-D brick requests for the current view.

    Cancels any pending work, selects visible bricks for the current camera
    view, allocates GPU cache slots for cache misses, and submits the resulting
    fetch requests to the asynchronous slicer. Completed batches are appended
    to ``upload_queue`` for later GPU upload by the draw loop.

    Parameters
    ----------
    volume_handle : MultiscaleVolumeHandle
        Handle that owns the 3-D GPU cache and accepts brick writes.
    wrapper : OMEZarrDataWrapper
        Multiscale data source used to load each requested chunk.
    camera_view : CameraView3D
        Frozen snapshot of the perspective camera (position, frustum, FOV).
    slicer : AsyncSlicer
        Background loader that executes chunk reads and posts completed
        batches back to the Qt thread.
    resolved : ResolvedDisplayState
        Resolved ndv display state describing the visible axes and any fixed
        indices for non-visible axes.
    upload_queue : collections.deque
        Receives completed ``(request, data)`` pairs until the draw loop
        drains and uploads them to the GPU.
    lod_bias : float, optional
        Scale applied to LOD thresholds. Values > 1 favour coarser levels;
        values < 1 favour finer levels.
    """
    # Discard any in-flight requests and queued uploads from the previous frame.
    volume_handle.invalidate_pending()
    upload_queue.clear()

    # Advance the LRU frame counter so cache residency checks this frame use
    # a consistent timestamp.
    volume_handle.advance_frame()

    # Derive the cache key for the current non-displayed axis position.
    visible = set(resolved.visible_axes)
    slice_coord: tuple[tuple[int, int], ...] = tuple(sorted(
        (ax, v)
        for ax, v in resolved.current_index.items()
        if isinstance(v, int) and ax not in visible
    ))

    # Evict bricks from stale slice coordinates when the axis position changes.
    # Keeps one previous coord as fallback for the two-phase LUT rebuild.
    if slice_coord != prev_slice_coord:
        n_evicted = volume_handle.evict_stale_slice_coords(slice_coord)
    else:
        n_evicted = 0

    # Scales for the displayed axes in display order (used by LOD and culling).
    voxel_scales = np.asarray(wrapper.voxel_sizes, dtype=np.float64)[list(resolved.visible_axes)]

    # Pure selection: choose and prioritize visible bricks based on camera
    # position, frustum, and LOD thresholds. No cache interaction here.
    sorted_block_keys = select_visible_bricks_3d(
        camera_view,
        volume_handle.brick_layout,
        voxel_scales,
        lod_bias,
        slice_coord=slice_coord,
    )

    # Build inverse level transforms (world → level-k data coordinates) needed
    # to map non-displayed axis indices into each resolution level.
    world_to_level = [AffineTransform(matrix=t.inverse_matrix) for t in wrapper.level_transforms]

    # Cache-aware pass: skip resident bricks, allocate slots for misses, and
    # assemble MultiscaleChunkRequest objects ready for the slicer.
    requests = build_fetch_requests_3d(
        sorted_block_keys,
        volume_handle.brick_layout.block_size,
        volume_handle.cache_query(),
        resolved,
        list(wrapper.level_shapes),
        world_to_level,
        volume_handle.expand_fetch_index,
        slice_coord=slice_coord,
    )

    # If evictions changed the cache but no new fetches are needed, rebuild
    # the LUT immediately so the display reflects the evictions right away.
    if n_evicted > 0 and not requests:
        volume_handle.commit(current_slice_coord=slice_coord)

    # Capture the slice ID so the callback can drop stale batches if a newer
    # reslice supersedes this one before results arrive.
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


def reslice_2d(
    image_handle: MultiscaleImageHandle,
    wrapper: OMEZarrDataWrapper,
    camera_view: CameraView2D,
    slicer: AsyncSlicer,
    resolved: ResolvedDisplayState,
    upload_queue: collections.deque,
    lod_bias: float = LOD_BIAS,
) -> tuple[tuple[int, int], ...]:
    """Plan and submit asynchronous 2-D tile requests for the current slice.

    Cancels any pending work, selects visible tiles for the current camera
    view and slice position, evicts tiles from finer LOD levels that are no
    longer needed, allocates GPU cache slots for cache misses, and submits the
    resulting fetch requests to the asynchronous slicer. Completed batches are
    appended to ``upload_queue`` for later GPU upload by the draw loop.

    Parameters
    ----------
    image_handle : MultiscaleImageHandle
        Handle that owns the 2-D GPU cache and accepts tile writes.
    wrapper : OMEZarrDataWrapper
        Multiscale data source used to load each requested chunk.
    camera_view : CameraView2D
        Frozen snapshot of the orthographic camera (bounds, viewport size).
    slicer : AsyncSlicer
        Background loader that executes chunk reads and posts completed
        batches back to the Qt thread.
    resolved : ResolvedDisplayState
        Resolved ndv display state describing the visible axes and any fixed
        indices for non-visible axes.
    upload_queue : collections.deque
        Receives completed ``(request, data)`` pairs until the draw loop
        drains and uploads them to the GPU.
    lod_bias : float, optional
        Scale applied to LOD thresholds. Values > 1 favour coarser levels;
        values < 1 favour finer levels.

    Returns
    -------
    tuple of tuple of int
        Sorted ``(axis, index)`` pairs encoding the current non-visible axis
        positions. The caller passes this to ``on_data_ready_2d`` so uploaded
        tiles are committed into the correct 2-D slice cache entry.
    """
    # Discard any in-flight requests and queued uploads from the previous frame.
    image_handle.invalidate_pending()
    upload_queue.clear()

    # Derive a stable cache key for the current slice position: sorted
    # (axis, index) pairs for all non-displayed integer axes.
    visible = set(resolved.visible_axes)
    slice_coord: tuple[tuple[int, int], ...] = tuple(sorted(
        (ax, v) for ax, v in resolved.current_index.items()
        if isinstance(v, int) and ax not in visible
    ))

    # Advance the LRU frame counter so cache residency checks this frame use
    # a consistent timestamp.
    image_handle.advance_frame()

    # Scales for the displayed axes in display order (used by LOD and culling).
    voxel_scales = np.asarray(wrapper.voxel_sizes, dtype=np.float64)[list(resolved.visible_axes)]

    # View selection: choose and prioritize visible tiles based on camera
    # bounds, LOD thresholds, and viewport culling. No cache interaction here.
    required_block_keys, target_level = select_visible_bricks_2d(
        camera_view,
        image_handle.brick_layout,
        voxel_scales,
        slice_coord,
        lod_bias,
    )

    # Remove tiles at finer resolution than the current target level; they
    # would be immediately superseded and waste cache slots.
    n_evicted = image_handle.evict_finer_than(target_level)

    # Build inverse level transforms (world → level-k data coordinates) needed
    # to map non-displayed axis indices into each resolution level.
    world_to_level = [AffineTransform(matrix=t.inverse_matrix) for t in wrapper.level_transforms]

    # Cache-aware pass: skip resident tiles, allocate slots for misses, and
    # assemble MultiscaleChunkRequest objects ready for the slicer.
    requests = build_fetch_requests_2d(
        required_block_keys,
        slice_coord,
        image_handle.cache_query(),
        resolved,
        list(wrapper.level_shapes),
        world_to_level,
        image_handle.expand_fetch_index,
        image_handle.brick_layout.block_size,
    )

    # If evictions changed the cache but no new fetches are needed, rebuild
    # the LUT immediately so the display reflects the evictions right away.
    if n_evicted > 0 and not requests:
        image_handle.rebuild_lut(slice_coord)

    # Capture the slice ID so the callback can drop stale batches if a newer
    # reslice supersedes this one before results arrive.
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


def _find_spatial_axes(
    wrapper: OMEZarrDataWrapper,
) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Return (spatial_axes_3d, yx_axes) derived from OME-Zarr axis-type metadata.

    Selects the last three axes whose type is "space" for the 3-D view and the
    last two of those for the 2-D view.  Falls back to the trailing three data
    axes if fewer than three spatial axes are recorded in the metadata.
    """
    axis_types = wrapper.axis_types
    n = len(axis_types)
    spatial = [i for i, t in enumerate(axis_types) if t == "space"]
    if len(spatial) >= 3:
        spatial_axes: tuple[int, ...] = tuple(spatial[-3:])
    else:
        spatial_axes = tuple(range(n - 3, n))
    yx_axes: tuple[int, int] = (spatial_axes[-2], spatial_axes[-1])
    return spatial_axes, yx_axes


def build_visual(
    wrapper: OMEZarrDataWrapper,
    spatial_axes: tuple[int, ...],
    yx_axes: tuple[int, int],
) -> GFXMultiscaleImageVisual:
    """Build and return the GFXMultiscaleImageVisual for the given displayed axes."""
    shapes = wrapper.level_shapes
    vs = np.asarray(wrapper.voxel_sizes, dtype=np.float64)
    voxel_scales_3d = list(vs[list(spatial_axes)])

    shapes_3d = [tuple(s[ax] for ax in spatial_axes) for s in shapes]
    shapes_2d = [(s[yx_axes[0]], s[yx_axes[1]]) for s in shapes]

    level_transforms = wrapper.level_transforms
    transforms_3d = [t.set_slice(spatial_axes) for t in level_transforms]
    transforms_2d = [t.set_slice(yx_axes) for t in level_transforms]

    volume_geometry = MultiscaleBrickLayout3D(
        level_shapes=shapes_3d,
        level_transforms=transforms_3d,
        block_size=BLOCK_SIZE,
    )

    image_geometry_2d = MultiscaleBrickLayout2D(
        level_shapes=shapes_2d,
        block_size=BLOCK_SIZE,
        n_levels=wrapper.n_levels,
        level_transforms=transforms_2d,
    )

    return GFXMultiscaleImageVisual(
        visual_model_id=uuid4(),
        volume_geometry=volume_geometry,
        image_geometry_2d=image_geometry_2d,
        render_modes={"2d", "3d"},
        displayed_axes=spatial_axes,
        colormap=gfx.cm.viridis,
        clim=(INITIAL_CLIM_LOW, INITIAL_CLIM_HIGH),
        threshold=INITIAL_ISO_THRESHOLD,
        voxel_scales=voxel_scales_3d,
        full_level_shapes=list(shapes),
        gpu_budget_bytes_3d=GPU_BUDGET_3D_BYTES,
        gpu_budget_bytes_2d=GPU_BUDGET_2D_BYTES,
        overlap_3d=OVERLAP_3D,
        overlap_2d=OVERLAP_2D,
    )


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

        world_extents = vox_shape * vs_full
        max_extent = float(world_extents.max())
        self.near = max(1.0, max_extent * 0.0001)
        self.far = max_extent * 10.0

        self.spatial_axes, self.yx_axes = _find_spatial_axes(self.wrapper)
        self._spatial_axes_set = set(self.spatial_axes)

        self.visual = build_visual(self.wrapper, self.spatial_axes, self.yx_axes)

        self.display_model_3d = ArrayDisplayModel(visible_axes=self.spatial_axes)
        self.display_model_2d = ArrayDisplayModel(visible_axes=self.yx_axes)

        self.mode = "3d"
        self.slice_coord_2d: tuple[tuple[int, int], ...] = ()
        self.slice_coord_3d: tuple[tuple[int, int], ...] = ()
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
        vs = self.wrapper.voxel_sizes
        self.world_w = float(w0) * float(vs[self.yx_axes[1]])
        self.world_h = float(h0) * float(vs[self.yx_axes[0]])

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

        self._slice_spins: dict[int, QtWidgets.QSpinBox] = {}
        self._slice_group = QtWidgets.QGroupBox("Slice axes")
        slice_layout = QtWidgets.QFormLayout(self._slice_group)
        n_axes = len(self.wrapper.level_shapes[0])
        axis_names = self.wrapper.axis_names
        for ax in range(n_axes):
            if ax in set(self.yx_axes):
                continue
            depth = self.wrapper.level_shapes[0][ax]
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, depth - 1)
            spin.setValue(depth // 2)
            label = axis_names[ax] if ax < len(axis_names) else str(ax)
            slice_layout.addRow(label, spin)
            self._slice_spins[ax] = spin
        p_layout.addWidget(self._slice_group)
        self._update_slice_spin_visibility()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(panel)

    def _connect_signals(self) -> None:
        self.clim_slider.valueChanged.connect(self._on_clim_change)
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        self.mode_combo.currentTextChanged.connect(self._on_render_mode_change)
        self.settle_spin.valueChanged.connect(self._on_settle_change)
        for spin in self._slice_spins.values():
            spin.valueChanged.connect(self._reslice)
        self.toggle_btn.clicked.connect(self._on_toggle)

    def _update_slice_spin_visibility(self) -> None:
        # Use intended visibility (mode + axis role), not Qt's propagated
        # runtime visibility state, to avoid parent/child circular hiding.
        visible_axes = {
            ax for ax in self._slice_spins
            if self.mode == "2d" or ax not in self._spatial_axes_set
        }
        self._slice_group.setVisible(bool(visible_axes))
        for ax, spin in self._slice_spins.items():
            spin.setVisible(ax in visible_axes)

    def _reslice(self) -> None:
        if self.mode == "3d":
            for ax, spin in self._slice_spins.items():
                if ax not in self._spatial_axes_set:
                    self.display_model_3d.current_index[ax] = spin.value()
            resolved = resolve(self.display_model_3d, self.wrapper)
            self.visual.update_display_axes(resolved.visible_axes)
            camera_view = camera_view_from_gfx_3d(self.camera_3d, self.canvas)
            self.slice_coord_3d = reslice_3d(
                volume_handle=self.visual.volume_handle,
                wrapper=self.wrapper,
                camera_view=camera_view,
                slicer=self.slicer_obj,
                resolved=resolved,
                upload_queue=self.upload_queue_3d,
                prev_slice_coord=self.slice_coord_3d,
            )
            return

        for ax, spin in self._slice_spins.items():
            self.display_model_2d.current_index[ax] = spin.value()
        resolved = resolve(self.display_model_2d, self.wrapper)
        self.visual.update_display_axes(resolved.visible_axes)
        camera_view = camera_view_from_gfx_2d(self.camera_2d, self.canvas)
        self.slice_coord_2d = reslice_2d(
            image_handle=self.visual.image_handle,
            wrapper=self.wrapper,
            camera_view=camera_view,
            slicer=self.slicer_obj,
            resolved=resolved,
            upload_queue=self.upload_queue_2d,
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
            self.visual.on_data_ready(drain, self.slice_coord_3d)

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

    def _on_toggle(self) -> None:
        if self.visual.volume_handle is not None:
            self.visual.volume_handle.invalidate_pending()
        if self.visual.image_handle is not None:
            self.visual.image_handle.invalidate_pending()
        self.settle_timer.stop()
        self._last_view_state = None

        if self.mode == "3d":
            self.mode = "2d"
            self.scene.remove(self.visual.node_3d)
            self.scene.add(self.visual.node_2d)
            self.controller_3d.enabled = False
            self.controller_2d.enabled = True
            self.render_group.setVisible(False)
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
            self.mode_label.setText("Mode: 3-D")
        self._update_slice_spin_visibility()

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
