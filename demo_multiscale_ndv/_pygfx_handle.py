"""GFXMultiscaleVolumeHandle and GFXMultiscaleImageHandle: pygfx implementations."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx as gfx

from demo_multiscale_ndv._cache_query import CacheQuery, CacheQuery2D, SlotId
from demo_multiscale_ndv._data_wrapper import BrickKey
from demo_multiscale_ndv._handle import MultiscaleImageHandle, MultiscaleVolumeHandle, SliceCoord
from demo_multiscale_ndv.block_cache import (
    BlockCache3D,
    BlockKey3D,
    TileSlot,
    compute_block_cache_parameters_3d,
)
from demo_multiscale_ndv.block_cache._block_cache_2d import BlockCache2D
from demo_multiscale_ndv.block_cache._cache_parameters_2d import (
    compute_block_cache_parameters_2d,
)
from demo_multiscale_ndv.block_cache._tile_manager_2d import BlockKey2D
from demo_multiscale_ndv.block_cache._tile_manager_2d import TileSlot as TileSlot2D
from demo_multiscale_ndv.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from demo_multiscale_ndv.lut_indirection._layout_2d import BlockLayout2D
from demo_multiscale_ndv.lut_indirection._lut_buffers_2d import (
    build_block_scales_buffer_2d,
    build_lut_params_buffer_2d,
)
from demo_multiscale_ndv.lut_indirection._lut_indirection_manager_2d import (
    LutIndirectionManager2D,
)
from demo_multiscale_ndv.shaders._block_image import ImageBlockMaterial
from demo_multiscale_ndv.shaders._multiscale_volume_brick import (
    MultiscaleVolumeBrickMaterial,
    build_brick_scales_buffer,
    build_vol_params_buffer,
    compose_world_transform,
)

if TYPE_CHECKING:
    import cmap as _cmap

    from demo_multiscale_ndv.render_visual import ImageGeometry2D, VolumeGeometry

# Register shader render functions with pygfx.
import demo_multiscale_ndv.shaders._block_image as _image_reg  # noqa: F401
import demo_multiscale_ndv.shaders._multiscale_volume_brick as _brick_reg  # noqa: F401


# ---------------------------------------------------------------------------
# Key conversion helpers
# ---------------------------------------------------------------------------


def _bk_to_block_key_3d(key: BrickKey) -> BlockKey3D:
    """BrickKey (0-indexed level) → BlockKey3D (1-indexed level)."""
    return BlockKey3D(
        level=key.level + 1,
        gz=key.brick_coords[0],
        gy=key.brick_coords[1],
        gx=key.brick_coords[2],
    )


def _bk_to_block_key_2d(
    key: BrickKey, slice_coord: tuple[tuple[int, int], ...]
) -> BlockKey2D:
    """BrickKey + slice_coord → BlockKey2D (1-indexed level)."""
    return BlockKey2D(
        level=key.level + 1,
        gy=key.brick_coords[0],
        gx=key.brick_coords[1],
        slice_coord=slice_coord,
    )


# ---------------------------------------------------------------------------
# 3-D cache-query adapter
# ---------------------------------------------------------------------------


class _CacheQueryAdapter3D:
    """Implements CacheQuery backed by a GFXMultiscaleVolumeHandle."""

    def __init__(self, handle: GFXMultiscaleVolumeHandle) -> None:
        self._handle = handle

    @property
    def capacity(self) -> int:
        return self._handle._block_cache.info.n_slots - 1

    def is_resident(self, key: BrickKey) -> bool:
        """Return True and refresh LRU timestamp if brick is resident."""
        block_key = _bk_to_block_key_3d(key)
        tm = self._handle._block_cache.tile_manager
        if block_key not in tm.tilemap:
            return False
        slot = tm.tilemap[block_key]
        fn = self._handle.frame_number
        slot.timestamp = fn
        heapq.heappush(tm._lru_heap, (fn, slot.index))
        return True

    def allocate_slot(self, key: BrickKey) -> SlotId:
        """Reserve a cache slot for *key*, evicting LRU if necessary."""
        block_key = _bk_to_block_key_3d(key)
        tm = self._handle._block_cache.tile_manager
        fn = self._handle.frame_number
        if tm.free_slots:
            slot_idx = tm.free_slots.pop()
        else:
            slot_idx = tm._evict_lru()
        grid_pos = tm._slot_grid_pos(slot_idx)
        slot = TileSlot(index=slot_idx, grid_pos=grid_pos, timestamp=fn)
        tm._in_flight[slot_idx] = block_key
        self._handle._pending_writes[slot_idx] = (block_key, slot)
        return slot_idx


# ---------------------------------------------------------------------------
# 2-D cache-query adapter
# ---------------------------------------------------------------------------


class _CacheQueryAdapter2D:
    """Implements CacheQuery2D backed by a GFXMultiscaleImageHandle."""

    def __init__(self, handle: GFXMultiscaleImageHandle) -> None:
        self._handle = handle

    @property
    def capacity(self) -> int:
        return self._handle._block_cache.info.n_slots - 1

    def is_resident(
        self,
        key: BrickKey,
        slice_coord: tuple[tuple[int, int], ...],
    ) -> bool:
        """Return True and refresh LRU timestamp if tile is resident."""
        block_key = _bk_to_block_key_2d(key, slice_coord)
        tm = self._handle._block_cache.tile_manager
        if block_key not in tm.tilemap:
            return False
        slot = tm.tilemap[block_key]
        fn = self._handle.frame_number
        slot.timestamp = fn
        heapq.heappush(tm._lru_heap, (fn, slot.index))
        return True

    def allocate_slot(
        self,
        key: BrickKey,
        slice_coord: tuple[tuple[int, int], ...],
    ) -> SlotId:
        """Reserve a cache slot for *(key, slice_coord)*, evicting LRU if needed."""
        block_key = _bk_to_block_key_2d(key, slice_coord)
        tm = self._handle._block_cache.tile_manager
        fn = self._handle.frame_number
        if tm.free_slots:
            slot_idx = tm.free_slots.pop()
        else:
            slot_idx = tm._evict_lru()
        grid_pos = tm._slot_grid_pos(slot_idx)
        slot = TileSlot2D(index=slot_idx, grid_pos=grid_pos, timestamp=fn)
        tm._in_flight[slot_idx] = block_key
        self._handle._pending_writes[slot_idx] = (block_key, slot)
        return slot_idx


# ---------------------------------------------------------------------------
# 3-D handle
# ---------------------------------------------------------------------------


class GFXMultiscaleVolumeHandle(MultiscaleVolumeHandle):
    """pygfx 3-D multiscale handle.

    Owns the block cache, LUT manager, GPU buffers, and pygfx scene node
    for one 3-D multiscale image.

    Parameters
    ----------
    volume_geometry :
        Pre-built 3-D metadata (level shapes, transforms, block size).
    voxel_scales :
        Physical voxel sizes in data-axis order (z, y, x).
    gpu_budget_bytes :
        Maximum GPU memory for the brick cache.
    colormap :
        Initial pygfx colormap (TextureMap).
    clim :
        Initial contrast limits ``(low, high)``.
    threshold :
        Initial ISO threshold.
    pick_write :
        Whether the pygfx Volume node participates in picking.
    """

    def __init__(
        self,
        volume_geometry: VolumeGeometry,
        voxel_scales: np.ndarray,
        gpu_budget_bytes: int = 1 * 1024**3,
        colormap: gfx.TextureMap | None = None,
        clim: tuple[float, float] = (0.0, 1.0),
        threshold: float = 0.5,
        pick_write: bool = True,
    ) -> None:
        self._voxel_scales = np.asarray(voxel_scales, dtype=np.float64)
        self.frame_number: int = 0

        if colormap is None:
            colormap = gfx.cm.viridis

        cache_params = compute_block_cache_parameters_3d(
            block_size=volume_geometry.block_size,
            gpu_budget_bytes=gpu_budget_bytes,
            overlap=3,
        )
        self._block_cache = BlockCache3D(cache_params)
        self._cache_query_adapter = _CacheQueryAdapter3D(self)

        self._pending_writes: dict[SlotId, tuple[BlockKey3D, TileSlot]] = {}
        self._pending_commits: list[tuple[BlockKey3D, TileSlot]] = []

        self._build_from_geometry(volume_geometry, colormap, clim, threshold, pick_write)

    # ── Geometry-dependent initialisation ──────────────────────────────

    def _build_from_geometry(
        self,
        geometry: VolumeGeometry,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        threshold: float,
        pick_write: bool = True,
    ) -> None:
        ds = geometry.level_shapes[0]  # (D, H, W)
        self._dataset_size = np.array(
            [float(ds[2]), float(ds[1]), float(ds[0])], dtype=np.float64
        )
        scales_xyz = self._voxel_scales[::-1]
        phys_size = self._dataset_size * scales_xyz
        self._norm_size = phys_size / phys_size.max()

        self._vol_params_buffer = build_vol_params_buffer(
            norm_size=self._norm_size,
            dataset_size=self._dataset_size,
            base_layout=geometry.base_layout,
            cache_info=self._block_cache.info,
        )
        self._brick_scales_buffer = build_brick_scales_buffer(
            geometry._scale_vecs_data
        )
        self._lut_manager = LutIndirectionManager3D(
            base_layout=geometry.base_layout,
            n_levels=geometry.n_levels,
            level_scale_vecs_data=geometry._scale_vecs_data,
        )
        inner, self.material, self._proxy_tex = self._make_node(
            colormap, clim, threshold, pick_write
        )
        self._inner_node = inner
        self.node = gfx.Group()
        self.node.add(inner)

    def _make_node(
        self,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        threshold: float,
        pick_write: bool = True,
    ) -> tuple[gfx.Volume, MultiscaleVolumeBrickMaterial, gfx.Texture]:
        proxy_data = np.zeros((2, 2, 2), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=3)
        material = MultiscaleVolumeBrickMaterial(
            cache_texture=self._block_cache.cache_tex,
            lut_texture=self._lut_manager.lut_tex,
            brick_max_texture=self._lut_manager.brick_max_tex,
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

    # ── rebuild for geometry changes ────────────────────────────────────

    def rebuild(self, geometry: VolumeGeometry) -> None:
        """Reinitialize LUT, buffers, and pygfx node for new level shapes."""
        self._block_cache.tile_manager.release_all_in_flight()
        self._pending_writes.clear()
        self._pending_commits.clear()
        colormap = self.material.map
        clim = self.material.clim
        threshold = self.material.threshold

        self._vol_params_buffer = build_vol_params_buffer(
            norm_size=self._norm_size,
            dataset_size=self._dataset_size,
            base_layout=geometry.base_layout,
            cache_info=self._block_cache.info,
        )
        self._brick_scales_buffer = build_brick_scales_buffer(geometry._scale_vecs_data)
        self._lut_manager = LutIndirectionManager3D(
            base_layout=geometry.base_layout,
            n_levels=geometry.n_levels,
            level_scale_vecs_data=geometry._scale_vecs_data,
        )
        inner, self.material, self._proxy_tex = self._make_node(colormap, clim, threshold)
        self._inner_node = inner
        self.node = gfx.Group()
        self.node.add(inner)

    def apply_world_transform(self) -> None:
        """Apply physical voxel scaling to the scene node."""
        scales_xyz = self._voxel_scales[::-1]
        data_to_world = np.diag(
            [scales_xyz[0], scales_xyz[1], scales_xyz[2], 1.0]
        ).astype(np.float64)
        m = compose_world_transform(data_to_world, self._dataset_size, self._norm_size)
        self.node.local.matrix = m

    # ── MultiscaleVolumeHandle abstract methods ─────────────────────────

    def set_brick(self, slot: SlotId, data: np.ndarray) -> None:
        entry = self._pending_writes.get(slot)
        if entry is None:
            return
        block_key, tile_slot = entry
        tile_slot.brick_max = float(data.max())
        self._block_cache.write_brick(tile_slot, data, key=block_key)
        self._pending_commits.append((block_key, tile_slot))

    def commit(self) -> None:
        for block_key, slot in self._pending_commits:
            self._block_cache.tile_manager.commit(block_key, slot)
        self._pending_commits.clear()
        self._lut_manager.rebuild(self._block_cache.tile_manager)

    def invalidate_pending(self) -> None:
        self._block_cache.tile_manager.release_all_in_flight()
        self._pending_writes.clear()
        self._pending_commits.clear()

    def cache_query(self) -> CacheQuery:
        return self._cache_query_adapter

    @property
    def overlap(self) -> int:
        return self._block_cache.info.overlap

    # ── LUTView stub implementations ────────────────────────────────────

    def set_clims(self, clims: tuple[float, float]) -> None:
        self.material.clim = clims

    def set_colormap(self, colormap: _cmap.Colormap) -> None:
        pass  # cmap.Colormap → gfx.TextureMap conversion not wired yet

    # ── CanvasElement stubs ─────────────────────────────────────────────

    def visible(self) -> bool:
        return bool(self.node.visible)

    def set_visible(self, visible: bool) -> None:
        self.node.visible = visible


# ---------------------------------------------------------------------------
# 2-D handle
# ---------------------------------------------------------------------------


class GFXMultiscaleImageHandle(MultiscaleImageHandle):
    """pygfx 2-D multiscale handle.

    Owns the tile cache, LUT manager, GPU buffers, and pygfx scene node
    for one 2-D multiscale image.

    Parameters
    ----------
    image_geometry_2d :
        Pre-built 2-D metadata (level shapes, transforms, block size).
    voxel_scales :
        Physical voxel sizes in data-axis order (z, y, x).
    gpu_budget_bytes :
        Maximum GPU memory for the tile cache.
    colormap :
        Initial pygfx colormap (TextureMap).
    clim :
        Initial contrast limits ``(low, high)``.
    interpolation :
        Texture interpolation mode (``"nearest"`` or ``"linear"``).
    pick_write :
        Whether the pygfx Image node participates in picking.
    """

    def __init__(
        self,
        image_geometry_2d: ImageGeometry2D,
        voxel_scales: np.ndarray,
        gpu_budget_bytes: int = 64 * 1024**2,
        colormap: gfx.TextureMap | None = None,
        clim: tuple[float, float] = (0.0, 1.0),
        interpolation: str = "nearest",
        pick_write: bool = True,
    ) -> None:
        self._voxel_scales = np.asarray(voxel_scales, dtype=np.float64)
        self.frame_number: int = 0

        if colormap is None:
            colormap = gfx.cm.viridis

        cache_params = compute_block_cache_parameters_2d(
            gpu_budget_bytes=gpu_budget_bytes,
            block_size=image_geometry_2d.block_size,
        )
        self._block_cache = BlockCache2D(cache_params)
        self._cache_query_adapter = _CacheQueryAdapter2D(self)

        self._pending_writes: dict[SlotId, tuple[BlockKey2D, TileSlot2D]] = {}
        self._pending_commits: list[tuple[BlockKey2D, TileSlot2D]] = []

        self._build_from_geometry(
            image_geometry_2d, colormap, clim, interpolation, pick_write
        )

    # ── Geometry-dependent initialisation ──────────────────────────────

    def _build_from_geometry(
        self,
        geometry: ImageGeometry2D,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        interpolation: str,
        pick_write: bool = True,
    ) -> None:
        self._lut_manager = LutIndirectionManager2D(
            base_layout=geometry.base_layout,
            n_levels=geometry.n_levels,
            scale_vecs_data=geometry._scale_vecs_data,
        )
        self._lut_params_buffer = build_lut_params_buffer_2d(
            geometry.base_layout, self._block_cache.info
        )
        self._block_scales_buffer = build_block_scales_buffer_2d(
            level_scale_vecs_data=geometry._scale_vecs_data,
        )
        inner, self.material, self._proxy_tex = self._make_node(
            geometry, colormap, clim, interpolation, pick_write
        )
        self._inner_node = inner
        self.node = gfx.Group()
        self.node.add(inner)

    def _make_node(
        self,
        geometry: ImageGeometry2D,
        colormap: gfx.TextureMap,
        clim: tuple[float, float],
        interpolation: str,
        pick_write: bool = True,
    ) -> tuple[gfx.Image, ImageBlockMaterial, gfx.Texture]:
        gh, gw = geometry.base_layout.grid_dims
        proxy_data = np.zeros((gh, gw), dtype=np.float32)
        proxy_tex = gfx.Texture(proxy_data, dim=2)
        material = ImageBlockMaterial(
            cache_texture=self._block_cache.cache_tex,
            lut_texture=self._lut_manager.lut_tex,
            lut_params_buffer=self._lut_params_buffer,
            block_scales_buffer=self._block_scales_buffer,
            clim=clim,
            map=colormap,
            pick_write=pick_write,
        )
        geom = gfx.Geometry(grid=proxy_tex)
        image = gfx.Image(geom, material)
        bs = geometry.block_size
        h, w = geometry.level_shapes[0]
        sx = float(w) / float(gw * bs)
        sy = float(h) / float(gh * bs)
        image.local.scale = (float(bs) * sx, float(bs) * sy, 1.0)
        image.local.position = (float(bs) * sx * 0.5, float(bs) * sy * 0.5, 0.0)
        return image, material, proxy_tex

    # ── rebuild for geometry changes ────────────────────────────────────

    def rebuild(self, geometry: ImageGeometry2D) -> None:
        """Reinitialize LUT, buffers, and pygfx node for new level shapes."""
        self._block_cache.tile_manager.release_all_in_flight()
        self._pending_writes.clear()
        self._pending_commits.clear()
        colormap = self.material.map
        clim = self.material.clim
        interpolation = getattr(self.material, "interpolation", "linear")

        self._lut_manager = LutIndirectionManager2D(
            base_layout=geometry.base_layout,
            n_levels=geometry.n_levels,
            scale_vecs_data=geometry._scale_vecs_data,
        )
        self._lut_params_buffer = build_lut_params_buffer_2d(
            geometry.base_layout, self._block_cache.info
        )
        self._block_scales_buffer = build_block_scales_buffer_2d(
            level_scale_vecs_data=geometry._scale_vecs_data,
        )
        inner, self.material, self._proxy_tex = self._make_node(
            geometry, colormap, clim, interpolation
        )
        self._inner_node = inner
        self.node = gfx.Group()
        self.node.add(inner)

    def apply_world_transform(self) -> None:
        """Apply physical voxel scaling to the scene node."""
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = float(self._voxel_scales[2])  # x (column)
        m[1, 1] = float(self._voxel_scales[1])  # y (row)
        self.node.local.matrix = m

    # ── MultiscaleImageHandle abstract methods ──────────────────────────

    def set_brick(self, slot: SlotId, data: np.ndarray) -> None:
        entry = self._pending_writes.get(slot)
        if entry is None:
            return
        tile_key, tile_slot = entry
        self._block_cache.write_tile(tile_slot, data, key=tile_key)
        self._pending_commits.append((tile_key, tile_slot))

    def commit(self, slice_coord: SliceCoord) -> None:
        for tile_key, slot in self._pending_commits:
            self._block_cache.tile_manager.commit(tile_key, slot)
        self._pending_commits.clear()
        self._lut_manager.rebuild(
            self._block_cache.tile_manager,
            current_slice_coord=slice_coord,
        )

    def invalidate_pending(self) -> None:
        self._block_cache.tile_manager.release_all_in_flight()
        self._pending_writes.clear()
        self._pending_commits.clear()

    def cache_query(self) -> CacheQuery2D:
        return self._cache_query_adapter

    @property
    def overlap(self) -> int:
        return self._block_cache.info.overlap

    def evict_finer_than(self, target_level: int) -> int:
        return self._block_cache.tile_manager.evict_finer_than(target_level)

    def rebuild_lut(self, slice_coord: SliceCoord) -> None:
        self._lut_manager.rebuild(
            self._block_cache.tile_manager,
            current_slice_coord=slice_coord,
        )

    # ── LUTView stub implementations ────────────────────────────────────

    def set_clims(self, clims: tuple[float, float]) -> None:
        self.material.clim = clims

    def set_colormap(self, colormap: _cmap.Colormap) -> None:
        pass  # cmap.Colormap → gfx.TextureMap conversion not wired yet

    # ── CanvasElement stubs ─────────────────────────────────────────────

    def visible(self) -> bool:
        return bool(self.node.visible)

    def set_visible(self, visible: bool) -> None:
        self.node.visible = visible
