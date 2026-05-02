"""GFXMultiscaleImageVisual — stripped demo render visual (no painting, no EventBus)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx

from demo_multiscale.block_cache import (
    BlockCache3D,
    BlockKey3D,
    TileSlot,
    compute_block_cache_parameters_3d,
)
from demo_multiscale.block_cache._block_cache_2d import BlockCache2D
from demo_multiscale.block_cache._cache_parameters_2d import (
    compute_block_cache_parameters_2d,
)
from demo_multiscale.data_store import ChunkRequest
from demo_multiscale.logging import _GPU_LOGGER, _PERF_LOGGER
from demo_multiscale.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from demo_multiscale.lut_indirection._layout_2d import BlockLayout2D
from demo_multiscale.lut_indirection._lut_buffers_2d import (
    build_block_scales_buffer_2d,
    build_lut_params_buffer_2d,
)
from demo_multiscale.lut_indirection._lut_indirection_manager_2d import (
    LutIndirectionManager2D,
)
from demo_multiscale.shaders._block_image import ImageBlockMaterial
from demo_multiscale.shaders._multiscale_volume_brick import (
    MultiscaleVolumeBrickMaterial,
    build_brick_scales_buffer,
    build_vol_params_buffer,
    compose_world_transform,
    compute_normalized_size,
)
from demo_multiscale.transform import AffineTransform
from demo_multiscale._frustum import (
    bricks_in_frustum_arr,
    frustum_planes_from_corners,
)
from demo_multiscale._level_of_detail import (
    arr_to_brick_keys,
    build_level_grids,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from demo_multiscale._level_of_detail_2d import (
    arr_to_block_keys_2d,
    build_tile_grids_2d,
    select_lod_2d,
    sort_tiles_by_distance_2d,
    viewport_cull_2d,
)

if TYPE_CHECKING:
    from pygfx.resources import Buffer

    from demo_multiscale.block_cache._tile_manager_2d import BlockKey2D
    from demo_multiscale.block_cache._tile_manager_2d import TileSlot as TileSlot2D
    from demo_multiscale.state import AxisAlignedSelectionState, DimsState

# Register shader classes with pygfx via @register_wgpu_render_function.
import demo_multiscale.shaders._block_image as _image_reg  # noqa: F401
import demo_multiscale.shaders._multiscale_volume_brick as _brick_reg  # noqa: F401


# ---------------------------------------------------------------------------
# Inlined wireframe / matrix helpers
# ---------------------------------------------------------------------------


def _box_wireframe_positions(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    x0, y0, z0 = float(box_min[0]), float(box_min[1]), float(box_min[2])
    x1, y1, z1 = float(box_max[0]), float(box_max[1]), float(box_max[2])
    return np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y0, z0], [x1, y1, z0],
            [x1, y1, z0], [x0, y1, z0], [x0, y1, z0], [x0, y0, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y0, z1], [x1, y1, z1],
            [x1, y1, z1], [x0, y1, z1], [x0, y1, z1], [x0, y0, z1],
            [x0, y0, z0], [x0, y0, z1], [x1, y0, z0], [x1, y0, z1],
            [x1, y1, z0], [x1, y1, z1], [x0, y1, z0], [x0, y1, z1],
        ],
        dtype=np.float32,
    )


def _rect_wireframe_positions(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    x0, y0 = float(box_min[0]), float(box_min[1])
    x1, y1 = float(box_max[0]), float(box_max[1])
    return np.array(
        [
            [x0, y0, 0.0], [x1, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0],
            [x1, y1, 0.0], [x0, y1, 0.0], [x0, y1, 0.0], [x0, y0, 0.0],
        ],
        dtype=np.float32,
    )


def _make_aabb_line(
    positions: np.ndarray, color: str, line_width: float = 2.0
) -> gfx.Line:
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineSegmentMaterial(color=color, thickness=line_width),
    )
    line.visible = False
    return line


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def _extract_scale_and_translation(
    level_transforms: list[AffineTransform],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    scale_vecs: list[np.ndarray] = []
    translation_vecs: list[np.ndarray] = []
    for t in level_transforms:
        nd = t.ndim
        scale_vecs.append(np.diag(t.matrix[:nd, :nd]).copy())
        translation_vecs.append(t.matrix[:nd, nd].copy())
    return scale_vecs, translation_vecs


# ---------------------------------------------------------------------------
# VolumeGeometry
# ---------------------------------------------------------------------------


class VolumeGeometry:
    """Pre-built metadata cache for a multiscale 3-D volume."""

    def __init__(
        self,
        level_shapes: list[tuple[int, ...]],
        level_transforms: list[AffineTransform],
        block_size: int,
    ) -> None:
        self.level_transforms = list(level_transforms)
        self.block_size = block_size
        self.n_levels = len(level_shapes)

        ndim = level_transforms[0].ndim
        assert np.allclose(
            level_transforms[0].matrix, np.eye(ndim + 1)
        ), "level_transforms[0] must be the identity"

        sv_data, tv_data = _extract_scale_and_translation(level_transforms)
        self._scale_vecs_data = sv_data
        self._translation_vecs_data = tv_data
        self._level_scale_factors = [
            float(np.prod(sv) ** (1.0 / len(sv))) for sv in sv_data
        ]

        self._scale_vecs_shader = [sv[[2, 1, 0]] for sv in sv_data]
        self._translation_vecs_shader = [tv[[2, 1, 0]] for tv in tv_data]

        self._scale_arr_shader = np.stack(self._scale_vecs_shader, axis=0)
        self._translation_arr_shader = np.stack(self._translation_vecs_shader, axis=0)

        self._rebuild(level_shapes)

    def _rebuild(self, level_shapes: list[tuple[int, ...]]) -> None:
        self.level_shapes = list(level_shapes)
        self.layouts = [
            BlockLayout3D(volume_shape=shape, block_size=self.block_size)
            for shape in level_shapes
        ]
        self.base_layout = self.layouts[0]
        self._level_grids = build_level_grids(
            self.base_layout,
            self.n_levels,
            level_shapes=self.level_shapes,
            scale_vecs_shader=self._scale_vecs_shader,
            translation_vecs_shader=self._translation_vecs_shader,
        )

    def update(self, level_shapes: list[tuple[int, ...]]) -> None:
        self._rebuild(level_shapes)


# ---------------------------------------------------------------------------
# ImageGeometry2D
# ---------------------------------------------------------------------------


class ImageGeometry2D:
    """Pre-built metadata cache for a multiscale 2-D image."""

    def __init__(
        self,
        level_shapes: list[tuple[int, int]],
        block_size: int,
        n_levels: int,
        level_transforms: list[AffineTransform] | None = None,
    ) -> None:
        self.block_size = block_size
        self.n_levels = n_levels
        self.level_shapes = list(level_shapes)

        if level_transforms is None:
            level_transforms = [
                AffineTransform.identity(ndim=2) for _ in range(n_levels)
            ]
        self.level_transforms = list(level_transforms)

        sv_data, tv_data = _extract_scale_and_translation(self.level_transforms)
        self._scale_vecs_data = sv_data
        self._translation_vecs_data = tv_data

        self._scale_vecs_shader = [sv[[1, 0]] for sv in sv_data]
        self._translation_vecs_shader = [tv[[1, 0]] for tv in tv_data]

        self._scale_arr_shader = np.stack(self._scale_vecs_shader, axis=0)
        self._translation_arr_shader = np.stack(self._translation_vecs_shader, axis=0)

        self._level_scale_factors = [float(np.sqrt(np.prod(sv))) for sv in sv_data]

        self.base_layout = BlockLayout2D.from_shape(
            shape=(level_shapes[0][0], level_shapes[0][1]),
            block_size=block_size,
        )
        self._level_grids = build_tile_grids_2d(
            self.base_layout,
            n_levels,
            level_shapes=self.level_shapes,
            scale_vecs_shader=self._scale_vecs_shader,
            translation_vecs_shader=self._translation_vecs_shader,
        )

    def update(self, level_shapes: list[tuple[int, int]]) -> None:
        self.level_shapes = list(level_shapes)
        self.base_layout = BlockLayout2D.from_shape(
            shape=(level_shapes[0][0], level_shapes[0][1]),
            block_size=self.block_size,
        )
        self._level_grids = build_tile_grids_2d(
            self.base_layout,
            self.n_levels,
            level_shapes=self.level_shapes,
            scale_vecs_shader=self._scale_vecs_shader,
            translation_vecs_shader=self._translation_vecs_shader,
        )


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _brick_key_to_padded_coords(
    key: BlockKey3D,
    block_size: int,
    overlap: int,
) -> tuple[int, int, int, int, int, int]:
    padded = block_size + 2 * overlap
    z0 = key.gz * block_size - overlap
    y0 = key.gy * block_size - overlap
    x0 = key.gx * block_size - overlap
    return z0, y0, x0, z0 + padded, y0 + padded, x0 + padded


def _block_key_2d_to_padded_coords(
    key: BlockKey2D,
    block_size: int,
    overlap: int,
) -> tuple[int, int, int, int]:
    padded = block_size + 2 * overlap
    y0 = key.gy * block_size - overlap
    x0 = key.gx * block_size - overlap
    return y0, x0, y0 + padded, x0 + padded


def _build_axis_selections(
    sel: AxisAlignedSelectionState,
    ndim: int,
    display_coords: list[tuple[int, int]],
    level_shape: tuple[int, ...],
    world_to_level_k: AffineTransform,
) -> tuple[int | tuple[int, int], ...]:
    display_pos = {ax: i for i, ax in enumerate(sel.displayed_axes)}

    world_pt = np.zeros(ndim, dtype=np.float64)
    for ax, idx in sel.slice_indices.items():
        world_pt[ax] = float(idx)

    level_k_pt = world_to_level_k.map_coordinates(world_pt.reshape(1, -1)).flatten()

    result: list[int | tuple[int, int]] = []
    for data_axis in range(ndim):
        if data_axis in display_pos:
            result.append(display_coords[display_pos[data_axis]])
        else:
            raw = float(level_k_pt[data_axis])
            clamped = int(round(raw))
            clamped = max(0, min(clamped, level_shape[data_axis] - 1))
            result.append(clamped)
    return tuple(result)


# ---------------------------------------------------------------------------
# GFXMultiscaleImageVisual
# ---------------------------------------------------------------------------


class GFXMultiscaleImageVisual:
    """Demo render-layer visual for one multiscale image (no painting, no EventBus).

    Parameters
    ----------
    visual_model_id : UUID
        Opaque identifier for this visual.
    volume_geometry : VolumeGeometry or None
        Pre-built 3-D metadata cache.
    image_geometry_2d : ImageGeometry2D or None
        Pre-built 2-D metadata cache.
    render_modes : set[str]
        Which nodes to build: ``{"3d"}``, ``{"2d"}``, or ``{"2d", "3d"}``.
    voxel_scales : list[float] or None
        Physical voxel size in data-axis order (z, y, x).  Defaults to 1.0
        for each axis if not provided.
    full_level_shapes : list[tuple[int, ...]] or None
        Full nD shapes for each level (needed for non-displayed axis mapping).
    full_level_transforms : list[AffineTransform] or None
        Full nD level transforms (level-k → level-0 in data order).
    """

    cancellable: bool = True

    def __init__(
        self,
        visual_model_id: UUID,
        volume_geometry: VolumeGeometry | None,
        image_geometry_2d: ImageGeometry2D | None,
        render_modes: set[str],
        displayed_axes: tuple[int, ...] | None = None,
        colormap: gfx.TextureMap | None = None,
        clim: tuple[float, float] = (0.0, 1.0),
        threshold: float = 0.5,
        interpolation: str = "nearest",
        gpu_budget_bytes_3d: int = 1 * 1024**3,
        gpu_budget_bytes_2d: int = 64 * 1024**2,
        voxel_scales: list[float] | None = None,
        full_level_shapes: list[tuple[int, ...]] | None = None,
        full_level_transforms: list[AffineTransform] | None = None,
        aabb_enabled: bool = False,
        aabb_color: str = "#ffffff",
        aabb_line_width: float = 2.0,
        render_order: int = 0,
        pick_write: bool = True,
    ) -> None:
        self.visual_model_id = visual_model_id
        self.render_modes = render_modes

        # ndim from geometry or shapes.
        if full_level_shapes is not None:
            self._ndim = len(full_level_shapes[0])
        elif volume_geometry is not None:
            self._ndim = len(volume_geometry.level_shapes[0])
        elif image_geometry_2d is not None:
            self._ndim = len(image_geometry_2d.level_shapes[0])
        else:
            self._ndim = 3

        # Physical voxel scales in data-axis order (z, y, x) for 3D.
        if voxel_scales is None:
            self._voxel_scales = np.ones(self._ndim, dtype=np.float64)
        else:
            self._voxel_scales = np.asarray(voxel_scales, dtype=np.float64)

        self._volume_geometry = volume_geometry
        self._image_geometry_2d = image_geometry_2d

        # Full-ndim level transforms for world→level-k mapping.
        if full_level_transforms is not None:
            self._level_transforms = list(full_level_transforms)
        elif volume_geometry is not None:
            self._level_transforms = volume_geometry.level_transforms
        elif image_geometry_2d is not None:
            self._level_transforms = image_geometry_2d.level_transforms
        else:
            self._level_transforms = []

        # Full-ndim level shapes (for non-displayed axis clamping).
        if full_level_shapes is not None:
            self._full_level_shapes = list(full_level_shapes)
        elif volume_geometry is not None:
            self._full_level_shapes = list(volume_geometry.level_shapes)
        elif image_geometry_2d is not None:
            self._full_level_shapes = list(image_geometry_2d.level_shapes)
        else:
            self._full_level_shapes = []

        # Precompute world→level-k transforms (just inv_level; world == level-0 voxel).
        self._world_to_level_transforms = self._build_world_to_level_transforms()

        self._last_displayed_axes: tuple[int, ...] | None = displayed_axes
        self._gpu_budget_bytes = gpu_budget_bytes_3d
        self._frame_number = 0
        self._pending_slot_map: dict[UUID, tuple[BlockKey3D, TileSlot]] = {}
        self._pending_slot_map_2d: dict[UUID, tuple[BlockKey2D, TileSlot2D]] = {}
        self._last_plan_stats: dict = {}

        self._data_ready_3d: bool = False
        self._data_ready_2d: bool = False

        self._aabb_enabled: bool = aabb_enabled
        self._aabb_color: str = aabb_color
        self._aabb_line_width: float = aabb_line_width

        self._current_slice_coord: tuple[tuple[int, int], ...] | None = None

        # ── 3D GPU resources ────────────────────────────────────────────────
        self._block_cache_3d: BlockCache3D | None = None
        self._lut_manager_3d: LutIndirectionManager3D | None = None
        if volume_geometry is not None:
            cache_parameters_3d = compute_block_cache_parameters_3d(
                block_size=volume_geometry.block_size,
                gpu_budget_bytes=gpu_budget_bytes_3d,
                overlap=3,
            )
            self._block_cache_3d = BlockCache3D(cache_parameters=cache_parameters_3d)
            self._lut_manager_3d = LutIndirectionManager3D(
                base_layout=volume_geometry.base_layout,
                n_levels=volume_geometry.n_levels,
                level_scale_vecs_data=volume_geometry._scale_vecs_data,
            )

        # ── 2D GPU resources ────────────────────────────────────────────────
        self._block_cache_2d: BlockCache2D | None = None
        self._lut_manager_2d: LutIndirectionManager2D | None = None
        self._lut_params_buffer_2d = None
        self._block_scales_buffer_2d = None
        if image_geometry_2d is not None:
            cache_parameters_2d = compute_block_cache_parameters_2d(
                gpu_budget_bytes=gpu_budget_bytes_2d,
                block_size=image_geometry_2d.block_size,
            )
            self._block_cache_2d = BlockCache2D(cache_parameters=cache_parameters_2d)
            self._lut_manager_2d = LutIndirectionManager2D(
                base_layout=image_geometry_2d.base_layout,
                n_levels=image_geometry_2d.n_levels,
                scale_vecs_data=image_geometry_2d._scale_vecs_data,
            )
            self._lut_params_buffer_2d = build_lut_params_buffer_2d(
                image_geometry_2d.base_layout, cache_parameters_2d
            )
            self._block_scales_buffer_2d = build_block_scales_buffer_2d(
                level_scale_vecs_data=image_geometry_2d._scale_vecs_data,
            )

        # ── 3D brick shader buffers ─────────────────────────────────────────
        self._vol_params_buffer: Buffer | None = None
        self._brick_scales_buffer: Buffer | None = None
        self._norm_size: np.ndarray | None = None
        self._dataset_size: np.ndarray | None = None
        if volume_geometry is not None:
            ds = volume_geometry.level_shapes[0]  # (D, H, W)
            self._dataset_size = np.array(
                [float(ds[2]), float(ds[1]), float(ds[0])], dtype=np.float64
            )
            # Physical size per axis in shader order (x, y, z).
            scales_xyz = self._voxel_scales[::-1]  # (sx, sy, sz)
            phys_size = self._dataset_size * scales_xyz
            self._norm_size = phys_size / phys_size.max()
            self._vol_params_buffer = build_vol_params_buffer(
                norm_size=self._norm_size,
                dataset_size=self._dataset_size,
                base_layout=volume_geometry.base_layout,
                cache_info=self._block_cache_3d.info,
            )
            self._brick_scales_buffer = build_brick_scales_buffer(
                volume_geometry._scale_vecs_data
            )

        if colormap is None:
            colormap = gfx.cm.viridis

        # ── 3D node ─────────────────────────────────────────────────────────
        self.node_3d: gfx.Group | None = None
        self._inner_node_3d: gfx.Volume | None = None
        self.material_3d: MultiscaleVolumeBrickMaterial | None = None
        self._proxy_tex_3d: gfx.Texture | None = None
        self._aabb_line_3d: gfx.Line | None = None
        if "3d" in render_modes and volume_geometry is not None:
            inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                pick_write=pick_write,
            )
            self._inner_node_3d = inner
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
            self._aabb_line_3d = self._build_aabb_line_3d()
            self.node_3d.add(self._aabb_line_3d)

        # ── 2D node ─────────────────────────────────────────────────────────
        self.node_2d: gfx.Group | None = None
        self._inner_node_2d: gfx.Image | None = None
        self.material_2d: ImageBlockMaterial | None = None
        self._proxy_tex_2d: gfx.Texture | None = None
        self._aabb_line_2d: gfx.Line | None = None
        if "2d" in render_modes and image_geometry_2d is not None:
            inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap,
                clim=clim,
                interpolation=interpolation,
                pick_write=pick_write,
            )
            self._inner_node_2d = inner
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            self._aabb_line_2d = self._build_aabb_line_2d()
            self.node_2d.add(self._aabb_line_2d)

        if self.node_3d is not None:
            self.node_3d.render_order = render_order
        if self.node_2d is not None:
            self.node_2d.render_order = render_order

        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def n_levels(self) -> int:
        if self._volume_geometry is not None:
            return self._volume_geometry.n_levels
        if self._image_geometry_2d is not None:
            return self._image_geometry_2d.n_levels
        raise RuntimeError("No geometry available")

    # ── Node selection ──────────────────────────────────────────────────────

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        _old_node, new_node = self.rebuild_geometry(
            self._full_level_shapes, displayed_axes
        )
        return new_node

    # ── Geometry rebuild ────────────────────────────────────────────────────

    def rebuild_geometry(
        self,
        level_shapes: list[tuple[int, ...]],
        displayed_axes: tuple[int, ...],
    ) -> tuple[gfx.WorldObject | None, gfx.WorldObject | None]:
        self._full_level_shapes = list(level_shapes)
        self._last_displayed_axes = displayed_axes

        old_node: gfx.WorldObject | None = None
        new_node: gfx.WorldObject | None = None

        if "3d" in self.render_modes and len(displayed_axes) == 3:
            old_node = self.node_3d
            if self._volume_geometry is not None:
                shapes_3d = [
                    tuple(s[ax] for ax in displayed_axes) for s in level_shapes
                ]
                if shapes_3d != self._volume_geometry.level_shapes:
                    self._volume_geometry.update(shapes_3d)
                    self._rebuild_3d_resources()
                new_node = self.node_3d

        if "2d" in self.render_modes and len(displayed_axes) == 2:
            old_node = self.node_2d
            if self._image_geometry_2d is not None:
                shapes_2d_full = [
                    tuple(s[ax] for ax in displayed_axes) for s in level_shapes
                ]
                shapes_2d = [(s[0], s[1]) for s in shapes_2d_full]
                if shapes_2d != self._image_geometry_2d.level_shapes:
                    self._image_geometry_2d.update(shapes_2d)
                    self._rebuild_2d_resources()
                new_node = self.node_2d

        return old_node, new_node

    def _rebuild_3d_resources(self) -> None:
        geo = self._volume_geometry
        self._block_cache_3d.tile_manager.release_all_in_flight()
        self._lut_manager_3d = LutIndirectionManager3D(
            base_layout=geo.base_layout,
            n_levels=geo.n_levels,
            level_scale_vecs_data=geo._scale_vecs_data,
        )
        if self.node_3d is not None:
            colormap = self.material_3d.map
            clim = self.material_3d.clim
            threshold = self.material_3d.threshold
            inner, self.material_3d, self._proxy_tex_3d = self._build_3d_node(
                colormap=colormap,
                clim=clim,
                threshold=threshold,
            )
            self._inner_node_3d = inner
            self._aabb_line_3d = self._build_aabb_line_3d()
            self.node_3d = gfx.Group()
            self.node_3d.add(inner)
            self.node_3d.add(self._aabb_line_3d)
            if self._last_displayed_axes is not None:
                self._update_node_matrix(self._last_displayed_axes)
        self._pending_slot_map = {}

    def _rebuild_2d_resources(self) -> None:
        geo2d = self._image_geometry_2d
        self._block_cache_2d.tile_manager.release_all_in_flight()
        self._lut_manager_2d = LutIndirectionManager2D(
            base_layout=geo2d.base_layout,
            n_levels=geo2d.n_levels,
            scale_vecs_data=geo2d._scale_vecs_data,
        )
        self._lut_params_buffer_2d = build_lut_params_buffer_2d(
            geo2d.base_layout, self._block_cache_2d.info
        )
        self._block_scales_buffer_2d = build_block_scales_buffer_2d(
            level_scale_vecs_data=geo2d._scale_vecs_data,
        )
        if self.node_2d is not None:
            colormap = self.material_2d.map
            clim = self.material_2d.clim
            interpolation = getattr(self.material_2d, "interpolation", "linear")
            inner, self.material_2d, self._proxy_tex_2d = self._build_2d_node(
                colormap=colormap, clim=clim, interpolation=interpolation
            )
            self._inner_node_2d = inner
            self._aabb_line_2d = self._build_aabb_line_2d()
            self.node_2d = gfx.Group()
            self.node_2d.add(inner)
            self.node_2d.add(self._aabb_line_2d)
            if self._last_displayed_axes is not None:
                self._update_node_matrix(self._last_displayed_axes)
        self._pending_slot_map_2d = {}

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        self._last_displayed_axes = displayed_axes
        # Physical scale in pygfx (x, y, z) order = data (z, y, x) reversed.
        scales_xyz = self._voxel_scales[::-1]  # (sx, sy, sz)

        if self.node_3d is not None:
            data_to_world = np.diag(
                [scales_xyz[0], scales_xyz[1], scales_xyz[2], 1.0]
            ).astype(np.float64)
            m = compose_world_transform(data_to_world, self._dataset_size, self._norm_size)
            self.node_3d.local.matrix = m

        if self.node_2d is not None:
            m = np.eye(4, dtype=np.float32)
            m[0, 0] = float(self._voxel_scales[2])  # x (column) scale
            m[1, 1] = float(self._voxel_scales[1])  # y (row) scale
            self.node_2d.local.matrix = m

    def _build_world_to_level_transforms(self) -> list[AffineTransform]:
        """Return list of transforms: world (= level-0 voxel) → level-k voxel."""
        result: list[AffineTransform] = []
        for lt in self._level_transforms:
            inv_level = AffineTransform(matrix=lt.inverse_matrix)
            result.append(inv_level)
        return result

    # ── 3D SliceCoordinator interface ───────────────────────────────────────

    def build_slice_request(
        self,
        camera_pos_world: np.ndarray,
        frustum_corners_world: np.ndarray | None,
        fov_y_rad: float,
        screen_height_px: float,
        lod_bias: float = 1.0,
        dims_state: DimsState | None = None,
        force_level: int | None = None,
    ) -> list[ChunkRequest]:
        t_plan_start = time.perf_counter()
        self._frame_number += 1
        geo = self._volume_geometry

        if dims_state is not None:
            displayed = dims_state.selection.displayed_axes
            if displayed != self._last_displayed_axes:
                self._update_node_matrix(displayed)

        # Map camera and frustum from world (physical) to level-0 voxel space.
        # world = voxel * voxel_scales, so voxel = world / voxel_scales.
        # voxel_scales is in data order (sz, sy, sx); pygfx world is (x, y, z).
        scales_xyz = self._voxel_scales[::-1].astype(np.float32)  # (sx, sy, sz)
        camera_pos_data = camera_pos_world / scales_xyz

        if frustum_corners_world is not None:
            corners_data = frustum_corners_world.reshape(-1, 3) / scales_xyz
            corners_data = corners_data.reshape(frustum_corners_world.shape)
            frustum_planes = frustum_planes_from_corners(corners_data)
        else:
            frustum_planes = None

        _PERF_LOGGER.debug(
            "[frame %d]  camera_pos_world_xyz=%s  camera_pos_voxel_xyz=%s",
            self._frame_number,
            np.round(camera_pos_world, 1).tolist(),
            np.round(camera_pos_data, 1).tolist(),
        )

        # 1. LOD selection
        if force_level is None and fov_y_rad > 0:
            focal_half_height_world = (screen_height_px / 2.0) / np.tan(fov_y_rad / 2.0)
            thresholds: list[float] | None = [
                geo._level_scale_factors[k - 1] * focal_half_height_world * lod_bias
                for k in range(1, geo.n_levels)
            ]
        else:
            thresholds = None

        t0 = time.perf_counter()
        if force_level is not None:
            brick_arr = select_levels_arr_forced(
                geo.base_layout, force_level, geo._level_grids
            )
        else:
            brick_arr = select_levels_from_cache(
                geo._level_grids,
                geo.n_levels,
                camera_pos_data,
                thresholds=thresholds,
                base_layout=geo.base_layout,
            )
        lod_select_ms = (time.perf_counter() - t0) * 1000

        # 2. Distance sort
        t0 = time.perf_counter()
        brick_arr = sort_arr_by_distance(
            brick_arr,
            camera_pos_data,
            geo.block_size,
            scale_vecs_shader=geo._scale_arr_shader,
            translation_vecs_shader=geo._translation_arr_shader,
        )
        distance_sort_ms = (time.perf_counter() - t0) * 1000
        n_total = len(brick_arr)

        # 3. Frustum cull
        cull_timings: dict = {}
        n_culled = 0
        frustum_cull_ms = 0.0
        if frustum_planes is not None:
            t0 = time.perf_counter()
            brick_arr, cull_timings = bricks_in_frustum_arr(
                brick_arr,
                geo.block_size,
                frustum_planes,
                level_scale_arr_shader=geo._scale_arr_shader,
                level_translation_arr_shader=geo._translation_arr_shader,
            )
            frustum_cull_ms = (time.perf_counter() - t0) * 1000
            n_culled = n_total - len(brick_arr)

        # 4. Budget truncation
        n_needed = len(brick_arr)
        n_budget = self._block_cache_3d.info.n_slots - 1
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            brick_arr = brick_arr[:n_budget]

        # 5. Stage
        t0 = time.perf_counter()
        sorted_required = arr_to_brick_keys(brick_arr)
        fill_plan = self._block_cache_3d.tile_manager.stage(
            sorted_required, self._frame_number
        )
        stage_ms = (time.perf_counter() - t0) * 1000

        # 6. Build ChunkRequests
        slice_id = uuid4()
        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map = {}

        for brick_key, slot in fill_plan:
            chunk_id = uuid4()
            z0, y0, x0, z1, y1, x1 = _brick_key_to_padded_coords(
                brick_key, geo.block_size, self._block_cache_3d.info.overlap
            )
            level_index = brick_key.level - 1
            display_coords = [(z0, z1), (y0, y1), (x0, x1)]
            if dims_state is not None:
                ndim = len(dims_state.axis_labels)
                axis_selections = _build_axis_selections(
                    dims_state.selection,
                    ndim,
                    display_coords,
                    level_shape=self._full_level_shapes[level_index],
                    world_to_level_k=self._world_to_level_transforms[level_index],
                )
            else:
                axis_selections = tuple(display_coords)
            req = ChunkRequest(
                chunk_request_id=chunk_id,
                slice_request_id=slice_id,
                scale_index=brick_key.level - 1,
                axis_selections=axis_selections,
            )
            chunk_requests.append(req)
            self._pending_slot_map[chunk_id] = (brick_key, slot)

        plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

        self._last_plan_stats = stats = {
            "hits": len(sorted_required) - len(fill_plan),
            "misses": len(fill_plan),
            "total_required": n_total,
            "n_culled": n_culled,
            "n_needed": n_needed,
            "n_budget": n_budget,
            "n_dropped": n_dropped,
            "lod_select_ms": lod_select_ms,
            "distance_sort_ms": distance_sort_ms,
            "frustum_cull_ms": frustum_cull_ms,
            "stage_ms": stage_ms,
            "plan_total_ms": plan_total_ms,
        }

        _PERF_LOGGER.info(
            "[frame %d]  lod=%.1fms  sort=%.1fms  cull=%.1fms  stage=%.1fms"
            "  |  req=%d  culled=%d  hits=%d  misses=%d",
            self._frame_number,
            stats["lod_select_ms"],
            stats["distance_sort_ms"],
            stats["frustum_cull_ms"],
            stats["stage_ms"],
            stats["total_required"],
            stats["n_culled"],
            stats["hits"],
            stats["misses"],
        )

        return chunk_requests

    def on_data_ready(
        self,
        batch: list[tuple[ChunkRequest, np.ndarray]],
    ) -> None:
        for req, data in batch:
            entry = self._pending_slot_map.get(req.chunk_request_id)
            if entry is None:
                continue
            brick_key, slot = entry
            slot.brick_max = float(data.max())
            self._block_cache_3d.write_brick(slot, data, key=brick_key)
            self._block_cache_3d.tile_manager.commit(brick_key, slot)

        _GPU_LOGGER.info(
            "gpu_flush  bricks_in_batch=%d  resident=%d",
            len(batch),
            self._block_cache_3d.n_resident,
        )

        self._lut_manager_3d.rebuild(self._block_cache_3d.tile_manager)

        if not self._data_ready_3d and self._aabb_line_3d is not None:
            self._data_ready_3d = True
            self._aabb_line_3d.visible = self._aabb_enabled

    def cancel_pending(self) -> None:
        if self._block_cache_3d is None:
            return
        self._block_cache_3d.tile_manager.release_all_in_flight()
        self._pending_slot_map = {}

    # ── 2D SliceCoordinator interface ───────────────────────────────────────

    def build_slice_request_2d(
        self,
        camera_pos_world: np.ndarray,
        viewport_width_px: float,
        world_width: float,
        view_min_world: np.ndarray | None,
        view_max_world: np.ndarray | None,
        dims_state: DimsState,
        lod_bias: float = 1.0,
        force_level: int | None = None,
        use_culling: bool = True,
    ) -> list[ChunkRequest]:
        t_plan_start = time.perf_counter()
        self._frame_number += 1

        self._current_slice_coord = tuple(
            sorted(dims_state.selection.slice_indices.items())
        )

        geo2d = self._image_geometry_2d
        block_size = geo2d.block_size
        n_levels = geo2d.n_levels

        displayed = dims_state.selection.displayed_axes
        if displayed != self._last_displayed_axes:
            self._update_node_matrix(displayed)

        # Map camera from world (physical) to level-0 tile-grid voxel space.
        # pygfx 2D world: X = second displayed axis (columns), Y = first (rows).
        scale_x = float(self._voxel_scales[2])  # column axis (x data axis)
        scale_y = float(self._voxel_scales[1])  # row axis (y data axis)
        voxel_width = world_width / (scale_x * scale_y) ** 0.5
        camera_pos = np.array(
            [
                float(camera_pos_world[0]) / scale_x,
                float(camera_pos_world[1]) / scale_y,
                0.0,
            ],
            dtype=np.float32,
        )

        if use_culling and view_min_world is not None and view_max_world is not None:
            view_min = np.array(
                [float(view_min_world[0]) / scale_x, float(view_min_world[1]) / scale_y],
                dtype=np.float32,
            )
            view_max = np.array(
                [float(view_max_world[0]) / scale_x, float(view_max_world[1]) / scale_y],
                dtype=np.float32,
            )
        else:
            view_min = None
            view_max = None

        # 1. LOD selection
        t0 = time.perf_counter()
        tile_arr = select_lod_2d(
            geo2d._level_grids,
            n_levels,
            viewport_width_px=viewport_width_px,
            voxel_width=voxel_width,
            lod_bias=lod_bias,
            force_level=force_level,
            level_scale_factors=geo2d._level_scale_factors,
        )
        lod_select_ms = (time.perf_counter() - t0) * 1000

        # 2. Distance sort
        t0 = time.perf_counter()
        tile_arr = sort_tiles_by_distance_2d(
            tile_arr,
            camera_pos,
            block_size,
            level_scale_arr_shader=geo2d._scale_arr_shader,
            level_translation_arr_shader=geo2d._translation_arr_shader,
        )
        distance_sort_ms = (time.perf_counter() - t0) * 1000

        required = arr_to_block_keys_2d(tile_arr, slice_coord=self._current_slice_coord)
        n_total = len(required)

        # 3. Viewport culling
        n_culled = 0
        cull_ms = 0.0
        if use_culling and view_min is not None and view_max is not None:
            t0 = time.perf_counter()
            required, n_culled = viewport_cull_2d(
                required,
                block_size,
                view_min,
                view_max,
                level_scale_arr_shader=geo2d._scale_arr_shader,
                level_translation_arr_shader=geo2d._translation_arr_shader,
            )
            cull_ms = (time.perf_counter() - t0) * 1000

        # 4. Budget truncation
        n_needed = len(required)
        n_budget = self._block_cache_2d.info.n_slots - 1
        n_dropped = max(0, n_needed - n_budget)
        if n_dropped:
            keys_to_keep = list(required.keys())[:n_budget]
            required = {k: required[k] for k in keys_to_keep}

        # 5. Evict finer-than-target tiles, then stage
        target_level = int(tile_arr[0, 0]) if len(tile_arr) > 0 else 1
        n_evicted = self._block_cache_2d.tile_manager.evict_finer_than(target_level)

        t0 = time.perf_counter()
        fill_plan = self._block_cache_2d.tile_manager.stage(
            required, self._frame_number
        )
        stage_ms = (time.perf_counter() - t0) * 1000

        if n_evicted > 0 and not fill_plan:
            self._lut_manager_2d.rebuild(
                self._block_cache_2d.tile_manager,
                current_slice_coord=self._current_slice_coord,
            )

        # 6. Build ChunkRequests
        slice_id = uuid4()
        chunk_requests: list[ChunkRequest] = []
        self._pending_slot_map_2d = {}

        overlap = self._block_cache_2d.info.overlap
        ndim = len(dims_state.axis_labels)
        sel = dims_state.selection

        for tile_key, slot in fill_plan:
            chunk_id = uuid4()
            y0, x0, y1, x1 = _block_key_2d_to_padded_coords(tile_key, block_size, overlap)
            level_index = tile_key.level - 1
            display_coords = [(y0, y1), (x0, x1)]
            axis_selections = _build_axis_selections(
                sel,
                ndim,
                display_coords,
                level_shape=self._full_level_shapes[level_index],
                world_to_level_k=self._world_to_level_transforms[level_index],
            )
            req = ChunkRequest(
                chunk_request_id=chunk_id,
                slice_request_id=slice_id,
                scale_index=tile_key.level - 1,
                axis_selections=axis_selections,
            )
            chunk_requests.append(req)
            self._pending_slot_map_2d[chunk_id] = (tile_key, slot)

        plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

        self._last_plan_stats = stats = {
            "hits": len(required) - len(fill_plan),
            "misses": len(fill_plan),
            "total_required": n_total,
            "n_culled": n_culled,
            "n_needed": n_needed,
            "n_budget": n_budget,
            "n_dropped": n_dropped,
            "lod_select_ms": lod_select_ms,
            "distance_sort_ms": distance_sort_ms,
            "cull_ms": cull_ms,
            "stage_ms": stage_ms,
            "plan_total_ms": plan_total_ms,
        }

        _PERF_LOGGER.info(
            "[frame %d]  lod=%.1fms  sort=%.1fms  cull=%.1fms  stage=%.1fms"
            "  |  req=%d  culled=%d  hits=%d  misses=%d",
            self._frame_number,
            stats["lod_select_ms"],
            stats["distance_sort_ms"],
            stats["cull_ms"],
            stats["stage_ms"],
            stats["total_required"],
            stats["n_culled"],
            stats["hits"],
            stats["misses"],
        )

        return chunk_requests

    def on_data_ready_2d(
        self,
        batch: list[tuple[ChunkRequest, np.ndarray]],
    ) -> None:
        for req, data in batch:
            entry = self._pending_slot_map_2d.get(req.chunk_request_id)
            if entry is None:
                continue
            tile_key, slot = entry
            self._block_cache_2d.write_tile(slot, data, key=tile_key)
            self._block_cache_2d.tile_manager.commit(tile_key, slot)

        n_resident = len(self._block_cache_2d.tile_manager.tilemap)
        _GPU_LOGGER.info(
            "gpu_flush  tiles_in_batch=%d  resident=%d", len(batch), n_resident
        )

        self._lut_manager_2d.rebuild(
            self._block_cache_2d.tile_manager,
            current_slice_coord=self._current_slice_coord,
        )

        if not self._data_ready_2d and self._aabb_line_2d is not None:
            self._data_ready_2d = True
            self._aabb_line_2d.visible = self._aabb_enabled

    def cancel_pending_2d(self) -> None:
        if self._block_cache_2d is None:
            return
        self._block_cache_2d.tile_manager.release_all_in_flight()
        self._pending_slot_map_2d = {}

    def invalidate_2d_cache(self) -> None:
        if self._block_cache_2d is None or self._lut_manager_2d is None:
            return
        self.cancel_pending_2d()

    # ── Private helpers ─────────────────────────────────────────────────────

    def _build_aabb_line_3d(self) -> gfx.Line:
        if self._norm_size is not None:
            half = self._norm_size / 2.0
            positions = _box_wireframe_positions(-half, half)
        else:
            positions = _box_wireframe_positions(np.zeros(3), np.ones(3))
        line = _make_aabb_line(positions, self._aabb_color, self._aabb_line_width)
        line.visible = self._aabb_enabled
        return line

    def _build_aabb_line_2d(self) -> gfx.Line:
        if self._image_geometry_2d is not None:
            h, w = self._image_geometry_2d.level_shapes[0]
            positions = _rect_wireframe_positions(
                np.array([0.0, 0.0]),
                np.array([float(w), float(h)]),
            )
        else:
            positions = _rect_wireframe_positions(np.zeros(2), np.ones(2))
        line = _make_aabb_line(positions, self._aabb_color, self._aabb_line_width)
        line.visible = self._aabb_enabled
        return line

    def _build_3d_node(
        self,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        threshold: float,
        pick_write: bool = True,
    ) -> tuple[gfx.Volume, MultiscaleVolumeBrickMaterial, gfx.Texture]:
        proxy_data = np.zeros((2, 2, 2), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=3)

        material = MultiscaleVolumeBrickMaterial(
            cache_texture=self._block_cache_3d.cache_tex,
            lut_texture=self._lut_manager_3d.lut_tex,
            brick_max_texture=self._lut_manager_3d.brick_max_tex,
            vol_params_buffer=self._vol_params_buffer,
            block_scales_buffer=self._brick_scales_buffer,
            clim=clim,
            map=colormap,
            threshold=threshold,
            pick_write=pick_write,
        )

        geometry = gfx.Geometry(grid=proxy_tex)
        vol = gfx.Volume(geometry, material)
        return vol, material, proxy_tex

    def _build_2d_node(
        self,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        interpolation: str,
        pick_write: bool = True,
    ) -> tuple[gfx.Image, ImageBlockMaterial, gfx.Texture]:
        gh, gw = self._image_geometry_2d.base_layout.grid_dims

        proxy_data = np.zeros((gh, gw), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=2)

        material = ImageBlockMaterial(
            cache_texture=self._block_cache_2d.cache_tex,
            lut_texture=self._lut_manager_2d.lut_tex,
            lut_params_buffer=self._lut_params_buffer_2d,
            block_scales_buffer=self._block_scales_buffer_2d,
            paint_cache_texture=None,
            paint_lut_texture=None,
            clim=clim,
            map=colormap,
            pick_write=pick_write,
        )

        geometry = gfx.Geometry(grid=proxy_tex)
        image = gfx.Image(geometry, material)

        bs = self._image_geometry_2d.block_size
        h, w = self._image_geometry_2d.level_shapes[0]
        sx = float(w) / float(gw * bs)
        sy = float(h) / float(gh * bs)
        scale_x = float(bs) * sx
        scale_y = float(bs) * sy
        image.local.scale = (scale_x, scale_y, 1.0)
        image.local.position = (scale_x * 0.5, scale_y * 0.5, 0.0)

        return image, material, proxy_tex
