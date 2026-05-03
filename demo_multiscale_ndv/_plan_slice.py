"""Pure planning functions: plan_slice_3d, plan_slice_2d.

These functions contain all brick/tile selection, LOD, culling, and cache-query
logic. They are free of GPU side-effects and do not touch the scene graph.

Both planners compute *core* fetch indices (no overlap/halo) and delegate
backend-specific fetch expansion to ``expand_fetch_index`` callbacks.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from demo_multiscale_ndv._cache_query import CacheQuery, CacheQuery2D
from demo_multiscale_ndv._camera_view import CameraView2D, CameraView3D
from demo_multiscale_ndv._chunk_request import MultiscaleChunkRequest
from demo_multiscale_ndv._data_wrapper import BrickKey
from demo_multiscale_ndv._frustum import (
    bricks_in_frustum_arr,
    frustum_planes_from_corners,
)
from demo_multiscale_ndv._indexing import ExpandFetchIndex
from demo_multiscale_ndv._level_of_detail_2d import (
    arr_to_block_keys_2d,
    select_lod_2d,
    sort_tiles_by_distance_2d,
    viewport_cull_2d,
)
from demo_multiscale_ndv._level_of_detail_3d import (
    arr_to_brick_keys,
    select_levels_arr_forced,
    select_levels_from_cache,
    sort_arr_by_distance,
)
from demo_multiscale_ndv.transform import AffineTransform
from ndv.models._resolve import ResolvedDisplayState

if TYPE_CHECKING:
    from demo_multiscale_ndv.render_visual import ImageGeometry2D, VolumeGeometry


# ---------------------------------------------------------------------------
# Coordinate helpers (moved from render_visual)
# ---------------------------------------------------------------------------


def _brick_key_to_core_coords(
    key,
    block_size: int,
) -> tuple[int, int, int, int, int, int]:
    """Return core (non-overlapped) zyx bounds for one brick key."""
    z0 = key.gz * block_size
    y0 = key.gy * block_size
    x0 = key.gx * block_size
    return z0, y0, x0, z0 + block_size, y0 + block_size, x0 + block_size


def _block_key_2d_to_core_coords(
    key,
    block_size: int,
) -> tuple[int, int, int, int]:
    """Return core (non-overlapped) yx bounds for one tile key."""
    y0 = key.gy * block_size
    x0 = key.gx * block_size
    return y0, x0, y0 + block_size, x0 + block_size


def _build_axis_selections(
    visible_axes: tuple[int, ...],
    slice_indices: dict[int, int],
    ndim: int,
    display_coords: list[tuple[int, int]],
    level_shape: tuple[int, ...],
    world_to_level_k: AffineTransform,
) -> tuple[int | tuple[int, int], ...]:
    display_pos = {ax: i for i, ax in enumerate(visible_axes)}

    world_pt = np.zeros(ndim, dtype=np.float64)
    for ax, idx in slice_indices.items():
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
# 3-D planner
# ---------------------------------------------------------------------------


def plan_slice_3d(
    camera_view: CameraView3D,
    cache_query: CacheQuery,
    geo: VolumeGeometry,
    voxel_scales: np.ndarray,
    full_level_shapes: list[tuple[int, ...]],
    world_to_level_transforms: list[AffineTransform],
    expand_fetch_index: ExpandFetchIndex,
    resolved: ResolvedDisplayState | None = None,
    lod_bias: float = 1.0,
    force_level: int | None = None,
) -> tuple[list[MultiscaleChunkRequest], dict[str, Any]]:
    """Select bricks, assign cache slots, and build fetch-ready requests.

    The planner computes core data-space brick indices (no overlap/halo).
    Backend-specific fetch expansion (halo, borders, filter footprint) is
    delegated to ``expand_fetch_index``.

    Parameters
    ----------
    camera_view :
        Frozen snapshot of the 3-D perspective camera.
    cache_query :
        Read/allocate surface for the GPU brick cache.
    geo :
        Pre-built 3-D level metadata (grids, scale arrays, block size).
    voxel_scales :
        Physical voxel sizes in data-axis order ``(z, y, x)``.
    full_level_shapes :
        Full nD shapes for each resolution level.
    world_to_level_transforms :
        Inverse of each level's affine transform (world → level-k).
    expand_fetch_index :
        Callback that converts core brick indices into backend-specific
        fetch indices (e.g. overlap/halo expansion).
    resolved :
        ndv resolved display state; supplies visible axes and slice indices.
        If ``None``, all data axes are treated as display axes.
    lod_bias :
        Scale applied to LOD thresholds (> 1 → coarser, < 1 → finer).
    force_level :
        If set, override LOD selection and use this level for all bricks.

    Returns
    -------
    requests :
        List of :class:`MultiscaleChunkRequest` ready to submit.
    stats :
        Timing and count diagnostics for the planning step.
    """
    t_plan_start = time.perf_counter()

    camera_pos_world = camera_view.camera_position
    frustum_corners_world = camera_view.frustum_corners
    fov_y_rad = camera_view.fov_y_rad
    screen_height_px = camera_view.viewport_size_px[1]

    scales_xyz = voxel_scales[::-1].astype(np.float32)
    camera_pos_data = camera_pos_world / scales_xyz

    if frustum_corners_world is not None:
        corners_data = frustum_corners_world.reshape(-1, 3) / scales_xyz
        corners_data = corners_data.reshape(frustum_corners_world.shape)
        frustum_planes = frustum_planes_from_corners(corners_data)
    else:
        frustum_planes = None

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

    t0 = time.perf_counter()
    sorted_block_keys = arr_to_brick_keys(brick_arr)
    n_budget = cache_query.capacity
    n_needed = len(sorted_block_keys)
    n_dropped = max(0, n_needed - n_budget)
    if n_dropped:
        sorted_block_keys = sorted_block_keys[:n_budget]

    slice_id = uuid4()
    chunk_requests: list[MultiscaleChunkRequest] = []
    n_hits = 0
    n_misses = 0

    if resolved is not None:
        ndim = len(resolved.data_coords)
        visible = set(resolved.visible_axes)
        slice_indices: dict[int, int] = {
            ax: v for ax, v in resolved.current_index.items()
            if isinstance(v, int) and ax not in visible
        }

    for block_key in sorted_block_keys:
        bk = BrickKey(
            level=block_key.level - 1,
            brick_coords=(block_key.gz, block_key.gy, block_key.gx),
        )
        if cache_query.is_resident(bk):
            n_hits += 1
            continue
        slot_id = cache_query.allocate_slot(bk)
        n_misses += 1
        chunk_id = uuid4()
        z0, y0, x0, z1, y1, x1 = _brick_key_to_core_coords(block_key, geo.block_size)
        level_index = block_key.level - 1
        display_coords = [(z0, z1), (y0, y1), (x0, x1)]
        if resolved is not None:
            axis_selections = _build_axis_selections(
                resolved.visible_axes,
                slice_indices,
                ndim,
                display_coords,
                level_shape=full_level_shapes[level_index],
                world_to_level_k=world_to_level_transforms[level_index],
            )
        else:
            axis_selections = tuple(display_coords)
        core_index = {
            ax_i: sel if isinstance(sel, int) else slice(sel[0], sel[1])
            for ax_i, sel in enumerate(axis_selections)
        }
        index = expand_fetch_index(
            level_index,
            core_index,
            full_level_shapes[level_index],
            (),
        )
        chunk_requests.append(MultiscaleChunkRequest(
            chunk_request_id=chunk_id,
            slice_request_id=slice_id,
            level=level_index,
            index=index,
            slot_id=slot_id,
        ))

    stage_ms = (time.perf_counter() - t0) * 1000
    plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

    stats: dict[str, Any] = {
        "hits": n_hits,
        "misses": n_misses,
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

    return chunk_requests, stats


# ---------------------------------------------------------------------------
# 2-D planner
# ---------------------------------------------------------------------------


def plan_slice_2d(
    camera_view: CameraView2D,
    cache_query: CacheQuery2D,
    geo: ImageGeometry2D,
    voxel_scales: np.ndarray,
    full_level_shapes: list[tuple[int, ...]],
    world_to_level_transforms: list[AffineTransform],
    resolved: ResolvedDisplayState,
    current_slice_coord: tuple[tuple[int, int], ...],
    expand_fetch_index: ExpandFetchIndex,
    lod_bias: float = 1.0,
    force_level: int | None = None,
    use_culling: bool = True,
) -> tuple[list[MultiscaleChunkRequest], dict[str, Any], int]:
    """Select tiles, assign cache slots, and build fetch-ready requests.

    The planner computes core data-space tile indices (no overlap/halo).
    Backend-specific fetch expansion (halo, borders, filter footprint) is
    delegated to ``expand_fetch_index``.

    Parameters
    ----------
    camera_view :
        Frozen snapshot of the 2-D orthographic camera.
    cache_query :
        Read/allocate surface for the GPU tile cache.
    geo :
        Pre-built 2-D level metadata (grids, scale arrays, block size).
    voxel_scales :
        Physical voxel sizes in data-axis order ``(z, y, x)``.
    full_level_shapes :
        Full nD shapes for each resolution level.
    world_to_level_transforms :
        Inverse of each level's affine transform (world → level-k).
    resolved :
        ndv resolved display state; supplies visible axes and slice indices.
    current_slice_coord :
        Sorted ``((axis, value), ...)`` tuple encoding the current slice
        position (used as part of the tile cache key).
    expand_fetch_index :
        Callback that converts core tile indices into backend-specific
        fetch indices (e.g. overlap/halo expansion).
    lod_bias :
        Scale applied to LOD thresholds (> 1 → coarser, < 1 → finer).
    force_level :
        If set, override LOD selection and use this level for all tiles.
    use_culling :
        Whether to cull tiles outside the current viewport.

    Returns
    -------
    requests :
        List of :class:`MultiscaleChunkRequest` ready to submit.
    stats :
        Timing and count diagnostics for the planning step.
    target_level :
        1-indexed LOD level of the closest tile; used by the caller to evict
        finer-resolution tiles that are no longer needed.
    """
    t_plan_start = time.perf_counter()

    xmin, xmax, ymin, ymax = camera_view.bounds
    viewport_width_px = camera_view.viewport_size_px[0]
    world_width = camera_view.world_per_pixel * viewport_width_px

    scale_x = float(voxel_scales[2])
    scale_y = float(voxel_scales[1])
    voxel_width = world_width / (scale_x * scale_y) ** 0.5
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    camera_pos = np.array([cx / scale_x, cy / scale_y, 0.0], dtype=np.float32)

    if use_culling:
        view_min = np.array([xmin / scale_x, ymin / scale_y], dtype=np.float32)
        view_max = np.array([xmax / scale_x, ymax / scale_y], dtype=np.float32)
    else:
        view_min = None
        view_max = None

    print(
        f"[slice2d] world bounds x=[{xmin:.1f}, {xmax:.1f}]  y=[{ymin:.1f}, {ymax:.1f}]  "
        f"world_size axis0(y)={ymax - ymin:.1f}  axis1(x)={xmax - xmin:.1f}  "
        f"voxel_width={voxel_width:.1f}  scale_x={scale_x:.3f}  scale_y={scale_y:.3f}  "
        f"view_min={view_min}  view_max={view_max}  use_culling={use_culling}"
    )

    t0 = time.perf_counter()
    tile_arr = select_lod_2d(
        geo._level_grids,
        geo.n_levels,
        viewport_width_px=viewport_width_px,
        voxel_width=voxel_width,
        lod_bias=lod_bias,
        force_level=force_level,
        level_scale_factors=geo._level_scale_factors,
    )
    lod_select_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    tile_arr = sort_tiles_by_distance_2d(
        tile_arr,
        camera_pos,
        geo.block_size,
        level_scale_arr_shader=geo._scale_arr_shader,
        level_translation_arr_shader=geo._translation_arr_shader,
    )
    distance_sort_ms = (time.perf_counter() - t0) * 1000

    required_block_keys = arr_to_block_keys_2d(
        tile_arr, slice_coord=current_slice_coord
    )
    n_total = len(required_block_keys)

    n_culled = 0
    cull_ms = 0.0
    if use_culling and view_min is not None and view_max is not None:
        t0 = time.perf_counter()
        required_block_keys, n_culled = viewport_cull_2d(
            required_block_keys,
            geo.block_size,
            view_min,
            view_max,
            level_scale_arr_shader=geo._scale_arr_shader,
            level_translation_arr_shader=geo._translation_arr_shader,
        )
        cull_ms = (time.perf_counter() - t0) * 1000

    print(
        f"[cull2d] tiles before={n_total}  after_cull={len(required_block_keys)}  "
        f"n_culled={n_culled}  use_culling={use_culling}  "
        f"view_min={view_min}  view_max={view_max}"
    )

    n_budget = cache_query.capacity
    n_needed = len(required_block_keys)
    n_dropped = max(0, n_needed - n_budget)
    if n_dropped:
        keys_to_keep = list(required_block_keys.keys())[:n_budget]
        required_block_keys = {k: required_block_keys[k] for k in keys_to_keep}

    target_level = int(tile_arr[0, 0]) if len(tile_arr) > 0 else 1

    t0 = time.perf_counter()
    ndim = len(resolved.data_coords)
    visible = set(resolved.visible_axes)
    slice_indices: dict[int, int] = {
        ax: v for ax, v in resolved.current_index.items()
        if isinstance(v, int) and ax not in visible
    }
    slice_id = uuid4()
    chunk_requests: list[MultiscaleChunkRequest] = []
    n_hits = 0
    n_misses = 0

    for tile_key in required_block_keys:
        bk = BrickKey(
            level=tile_key.level - 1,
            brick_coords=(tile_key.gy, tile_key.gx),
        )
        if cache_query.is_resident(bk, current_slice_coord):
            n_hits += 1
            continue
        slot_id = cache_query.allocate_slot(bk, current_slice_coord)
        n_misses += 1
        chunk_id = uuid4()
        y0, x0, y1, x1 = _block_key_2d_to_core_coords(tile_key, geo.block_size)
        level_index = tile_key.level - 1
        display_coords = [(y0, y1), (x0, x1)]
        axis_selections = _build_axis_selections(
            resolved.visible_axes,
            slice_indices,
            ndim,
            display_coords,
            level_shape=full_level_shapes[level_index],
            world_to_level_k=world_to_level_transforms[level_index],
        )
        core_index = {
            ax_i: sel if isinstance(sel, int) else slice(sel[0], sel[1])
            for ax_i, sel in enumerate(axis_selections)
        }
        index = expand_fetch_index(
            level_index,
            core_index,
            full_level_shapes[level_index],
            current_slice_coord,
        )
        chunk_requests.append(MultiscaleChunkRequest(
            chunk_request_id=chunk_id,
            slice_request_id=slice_id,
            level=level_index,
            index=index,
            slot_id=slot_id,
        ))

    stage_ms = (time.perf_counter() - t0) * 1000
    plan_total_ms = (time.perf_counter() - t_plan_start) * 1000

    stats: dict[str, Any] = {
        "hits": n_hits,
        "misses": n_misses,
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

    return chunk_requests, stats, target_level
