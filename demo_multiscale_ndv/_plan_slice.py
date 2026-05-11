"""Brick/tile selection and fetch-request building for 2-D and 3-D views.

Each dimensionality has two functions:
  select_visible_bricks_*  — select bricks based on what is in view
  build_fetch_requests_*   — cache-aware brick subselection, allocates slots, builds requests
"""

from __future__ import annotations

from typing import TYPE_CHECKING
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
    from demo_multiscale_ndv._handle import SliceCoord
    from demo_multiscale_ndv.render_visual import MultiscaleBrickLayout2D, MultiscaleBrickLayout3D


def _brick_key_to_core_coords(
    key,
    block_size: int,
) -> tuple[int, int, int, int, int, int]:
    """Return core (non-overlapped) zyx voxel bounds for one 3-D brick key.

    Parameters
    ----------
    key : BlockKey3D
        Grid-space brick identifier carrying ``gz``, ``gy``, and ``gx``
        coarse-grid indices (all 1-indexed in the caller's convention).
    block_size : int
        Side length of each brick in voxels at the brick's LOD level.

    Returns
    -------
    z0, y0, x0, z1, y1, x1 : int
        Half-open voxel range ``[z0, z1) × [y0, y1) × [x0, x1)`` in
        level-k data order (no overlap padding).
    """
    # Multiply grid indices by block_size to get the voxel origin of the brick.
    z0 = key.gz * block_size
    y0 = key.gy * block_size
    x0 = key.gx * block_size
    return z0, y0, x0, z0 + block_size, y0 + block_size, x0 + block_size


def _block_key_2d_to_core_coords(
    key,
    block_size: int,
) -> tuple[int, int, int, int]:
    """Return core (non-overlapped) yx voxel bounds for one 2-D tile key.

    Parameters
    ----------
    key : BlockKey2D
        Grid-space tile identifier carrying ``gy`` and ``gx`` coarse-grid
        indices (1-indexed).
    block_size : int
        Side length of each tile in voxels at the tile's LOD level.

    Returns
    -------
    y0, x0, y1, x1 : int
        Half-open voxel range ``[y0, y1) × [x0, x1)`` in level-k data
        order (no overlap padding).
    """
    # Multiply grid indices by block_size to get the voxel origin of the tile.
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
    """Build a per-axis index/range selection for one brick or tile fetch.

    For each data axis the result is either:

    * a ``(start, stop)`` range — for axes that are being *displayed*
      (i.e. the brick or tile spans that axis in the viewport), or
    * a single ``int`` — for axes that are *sliced* (collapsed to one
      plane), mapped from world space to level-k voxel coordinates and
      clamped to the level's extent.

    Parameters
    ----------
    visible_axes : tuple[int, ...]
        Data-axis indices that are mapped to screen dimensions (order
        matches ``display_coords``).
    slice_indices : dict[int, int]
        World-space index for each non-displayed (sliced) axis.
    ndim : int
        Total number of data dimensions.
    display_coords : list[tuple[int, int]]
        ``(start, stop)`` voxel ranges for each visible axis in level-k
        space, in the same order as ``visible_axes``.
    level_shape : tuple[int, ...]
        Full voxel shape of the array at this LOD level; used to clamp
        mapped slice indices so they never fall outside the array bounds.
    world_to_level_k : AffineTransform
        Affine mapping from world (viewer) coordinates to level-k voxel
        coordinates.

    Returns
    -------
    selections : tuple[int | tuple[int, int], ...]
        Length-``ndim`` tuple; element ``i`` is a ``(start, stop)`` pair
        for a display axis or an ``int`` for a sliced axis.
    """
    # Map each visible axis to its position within display_coords so we can
    # look it up in O(1) inside the loop below.
    display_pos = {ax: i for i, ax in enumerate(visible_axes)}

    # Build a world-space point that carries the current slice index on every
    # sliced axis (display axes are left at 0 — their value is irrelevant
    # because we use display_coords for those axes, not this mapped point).
    world_pt = np.zeros(ndim, dtype=np.float64)
    for ax, idx in slice_indices.items():
        world_pt[ax] = float(idx)

    # Project the world point into level-k voxel space so we can compute the
    # correct voxel index for each sliced axis at this LOD level.
    level_k_pt = world_to_level_k.map_coordinates(world_pt.reshape(1, -1)).flatten()

    result: list[int | tuple[int, int]] = []
    for data_axis in range(ndim):
        if data_axis in display_pos:
            # Display axis: pass the brick/tile range through unchanged.
            result.append(display_coords[display_pos[data_axis]])
        else:
            # Sliced axis: round the mapped coordinate to the nearest voxel
            # and clamp to [0, size-1] to handle floating-point overshoot at
            # the array boundaries.
            raw = float(level_k_pt[data_axis])
            clamped = int(round(raw))
            clamped = max(0, min(clamped, level_shape[data_axis] - 1))
            result.append(clamped)
    return tuple(result)


def select_visible_bricks_3d(
    camera_view: CameraView3D,
    brick_layout: MultiscaleBrickLayout3D,
    voxel_scales: np.ndarray,
    lod_bias: float = 1.0,
    force_level: int | None = None,
    slice_coord: tuple[tuple[int, int], ...] = (),
) -> list:
    """Select and prioritize visible bricks for the current 3-D camera view.

    The pipeline is:

    1. Convert world-space camera position and frustum corners to data
       (voxel) space using ``voxel_scales``.
    2. Compute per-level distance thresholds from the focal length so
       each level is shown only where its resolution is appropriate.
    3. Select one LOD level per brick from the precomputed grid cache.
    4. Sort bricks nearest-first so the most important bricks are fetched
       first when the budget is tight.
    5. Cull bricks that lie entirely outside the view frustum.

    Parameters
    ----------
    camera_view : CameraView3D
        Current 3-D camera state (position, frustum corners, FOV,
        viewport size).
    brick_layout : MultiscaleBrickLayout3D
        Data describing how the shader brick grid textures are setup.
    voxel_scales : np.ndarray
        Physical size of one voxel along each data axis in zyx order;
        used to convert world distances to data-space distances.
    lod_bias : float, optional
        Multiplier applied to the LOD distance thresholds.  Values > 1
        favor coarser levels; values < 1 favor finer levels.
    force_level : int or None, optional
        When set, skip the automatic LOD selection and use this 1-indexed
        level for every brick.

    Returns
    -------
    sorted_block_keys : list
        Brick keys sorted nearest-first and frustum-culled.
    """
    camera_pos_world = camera_view.camera_position
    frustum_corners_world = camera_view.frustum_corners
    fov_y_rad = camera_view.fov_y_rad
    screen_height_px = camera_view.viewport_size_px[1]

    # voxel_scales is in zyx order; reverse to xyz to match the shader convention
    # used by the level grid arrays, then divide to convert world coords to data coords.
    scales_xyz = voxel_scales[::-1].astype(np.float32)
    camera_pos_data = camera_pos_world / scales_xyz

    # Convert frustum corners from world to data space so the frustum culling
    # step operates in the same coordinate system as the brick centres.
    if frustum_corners_world is not None:
        corners_data = frustum_corners_world.reshape(-1, 3) / scales_xyz
        corners_data = corners_data.reshape(frustum_corners_world.shape)
        frustum_planes = frustum_planes_from_corners(corners_data)
    else:
        frustum_planes = None

    # Derive per-level distance thresholds from the pinhole camera geometry.
    # focal_half_height_world is the world-space half-height of the image
    # plane at unit distance; multiplying by lod_bias shifts the LOD bands.
    if force_level is None and fov_y_rad > 0:
        focal_half_height_world = (screen_height_px / 2.0) / np.tan(fov_y_rad / 2.0)
        thresholds: list[float] | None = [
            brick_layout._level_scale_factors[k - 1] * focal_half_height_world / lod_bias
            for k in range(1, brick_layout.n_levels)
        ]
    else:
        thresholds = None

    # Select which LOD level each brick should use.
    if force_level is not None:
        brick_arr = select_levels_arr_forced(
            brick_layout.base_layout, force_level, brick_layout._level_grids
        )
    else:
        brick_arr = select_levels_from_cache(
            brick_layout._level_grids,
            brick_layout.n_levels,
            camera_pos_data,
            thresholds=thresholds,
            base_layout=brick_layout.base_layout,
        )

    # Sort nearest-first so the fetch budget is spent on the most visible bricks.
    brick_arr = sort_arr_by_distance(
        brick_arr,
        camera_pos_data,
        brick_layout.block_size,
        scale_vecs_shader=brick_layout._scale_arr_shader,
        translation_vecs_shader=brick_layout._translation_arr_shader,
    )

    # Remove bricks whose AABBs lie entirely outside the view frustum.
    if frustum_planes is not None:
        brick_arr, _ = bricks_in_frustum_arr(
            brick_arr,
            brick_layout.block_size,
            frustum_planes,
            level_scale_arr_shader=brick_layout._scale_arr_shader,
            level_translation_arr_shader=brick_layout._translation_arr_shader,
        )

    return arr_to_brick_keys(brick_arr, slice_coord=slice_coord)


def build_fetch_requests_3d(
    sorted_block_keys: list,
    block_size: int,
    cache_query: CacheQuery,
    resolved: ResolvedDisplayState | None,
    full_level_shapes: list[tuple[int, ...]],
    world_to_level_transforms: list[AffineTransform],
    expand_fetch_index: ExpandFetchIndex,
    slice_coord: tuple[tuple[int, int], ...] = (),
) -> list[MultiscaleChunkRequest]:
    """Check cache residency, allocate slots, and build fetch requests for 3-D bricks.

    Iterates through ``sorted_block_keys`` in priority order (nearest-first),
    skips bricks that are already resident in the GPU cache, allocates a slot
    for each missing brick, and constructs a :class:`MultiscaleChunkRequest`
    that describes exactly which voxel range to fetch.

    Parameters
    ----------
    sorted_block_keys : list
        Brick keys in fetch priority order, as returned by
        :func:`select_visible_bricks_3d`.
    block_size : int
        Side length of each brick in voxels at its LOD level.
    cache_query : CacheQuery
        Interface to the GPU brick cache for residency checks and slot
        allocation.
    resolved : ResolvedDisplayState or None
        Current viewer state (visible axes, current index) used to build
        correct multi-dimensional index expressions.  When ``None``, a
        raw 3-D index is used.
    full_level_shapes : list[tuple[int, ...]]
        Voxel shape of the full array at each LOD level (0-indexed).
    world_to_level_transforms : list[AffineTransform]
        Affine transforms from world to level-k data coordinates, one per
        level (0-indexed).
    expand_fetch_index : ExpandFetchIndex
        Callable that expands the indices of the chunk to fetch based on
        the needs of the sampler (e.g., cache overlap). This method
        is implemented on the MultiscaleImageHandle.

    Returns
    -------
    chunk_requests : list[MultiscaleChunkRequest]
        One request per brick that is not yet resident in the cache,
        capped at ``cache_query.capacity`` bricks total.
    """
    n_budget = cache_query.capacity
    n_needed = len(sorted_block_keys)
    # Trim to budget before iterating to avoid unnecessary cache queries.
    if n_needed > n_budget:
        sorted_block_keys = list(sorted_block_keys)[:n_budget]

    # Extract slice (non-display) axis indices from the resolved display state.
    if resolved is not None:
        ndim = len(resolved.data_coords)
        visible = set(resolved.visible_axes)
        slice_indices: dict[int, int] = {
            ax: v for ax, v in resolved.current_index.items()
            if isinstance(v, int) and ax not in visible
        }
    else:
        ndim = 0
        slice_indices = {}

    # All requests produced in this call share one slice_id so the renderer
    # can atomically swap a full set of bricks belonging to the same frame.
    slice_id = uuid4()
    chunk_requests: list[MultiscaleChunkRequest] = []

    for block_key in sorted_block_keys:
        # Convert the 1-indexed level from the block key to the 0-indexed level
        # used everywhere else in the codebase.
        bk = BrickKey(
            level=block_key.level - 1,
            brick_coords=(block_key.gz, block_key.gy, block_key.gx),
        )
        # Skip bricks already loaded in the GPU cache; they need no re-fetch.
        if cache_query.is_resident(bk, slice_coord):
            continue
        slot_id = cache_query.allocate_slot(bk, slice_coord)
        chunk_id = uuid4()
        z0, y0, x0, z1, y1, x1 = _brick_key_to_core_coords(block_key, block_size)
        level_index = block_key.level - 1
        # The three spatial display ranges, ordered zyx to match the data axes.
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
        # Convert (start, stop) tuples to slice objects; leave integer
        # selections as-is (they index a single plane on a sliced axis).
        core_index = {
            ax_i: sel if isinstance(sel, int) else slice(sel[0], sel[1])
            for ax_i, sel in enumerate(axis_selections)
        }
        # Expand the indices of the chunk to fetch based on
        # the needs of the sampler (e.g., cache overlap).
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

    return chunk_requests


def select_visible_bricks_2d(
    camera_view: CameraView2D,
    brick_layout: MultiscaleBrickLayout2D,
    voxel_scales: np.ndarray,
    current_slice_coord: SliceCoord,
    lod_bias: float = 1.0,
    force_level: int | None = None,
    use_culling: bool = True,
) -> tuple[dict, int]:
    """Select and prioritize visible tiles for the current 2-D camera view.

    The pipeline mirrors :func:`select_visible_bricks_3d` but operates in 2-D:

    1. Derive an effective "voxel width" of the viewport to drive LOD selection.
    2. Select the appropriate LOD level for each tile.
    3. Sort tiles nearest-first.
    4. Optionally cull tiles outside the viewport bounds.

    Parameters
    ----------
    camera_view : CameraView2D
        Current 2-D camera state (bounds, viewport size, world-per-pixel).
    brick_layout : MultiscaleBrickLayout2D
        Data describing how the shader brick grid textures are setup.
    voxel_scales : np.ndarray
        Physical size of one voxel along each data axis in zyx order.
    current_slice_coord : SliceCoord
        Current position along the axes that are not displayed (e.g. the
        z-plane for an xy view); baked into the tile keys so that cache
        entries are keyed per-slice.
    lod_bias : float, optional
        Multiplier on the LOD selection threshold.  Values > 1 favor
        coarser levels; values < 1 favor finer levels.
    force_level : int or None, optional
        When set, bypass automatic LOD selection and use this 1-indexed
        level for every tile.
    use_culling : bool, optional
        When ``True`` (default), tiles that lie entirely outside the
        viewport bounds are removed before returning.

    Returns
    -------
    required_block_keys : dict
        Ordered mapping of BlockKey2D → ... for tiles that should be
        displayed, sorted nearest-first and culled to the viewport.
    target_level : int
        1-indexed LOD level of the nearest tile; used by the caller to
        evict finer-resolution tiles that are no longer needed.
    """
    xmin, xmax, ymin, ymax = camera_view.bounds
    viewport_width_px = camera_view.viewport_size_px[0]
    world_width = camera_view.world_per_pixel * viewport_width_px

    # Convert the world-space viewport width to an effective voxel width using
    # the geometric mean of the x and y scales so that non-square pixels are
    # handled without introducing axis bias.
    scale_x = float(voxel_scales[1])
    scale_y = float(voxel_scales[0])
    voxel_width = world_width / (scale_x * scale_y) ** 0.5
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    # Camera position in data (voxel) space; z=0 because this is a 2-D view.
    camera_pos = np.array([cx / scale_x, cy / scale_y, 0.0], dtype=np.float32)

    # Convert viewport bounds from world to data space for the culling step.
    if use_culling:
        view_min = np.array([xmin / scale_x, ymin / scale_y], dtype=np.float32)
        view_max = np.array([xmax / scale_x, ymax / scale_y], dtype=np.float32)
    else:
        view_min = None
        view_max = None

    # Select one LOD level per tile position based on the viewport resolution.
    tile_arr = select_lod_2d(
        brick_layout._level_grids,
        brick_layout.n_levels,
        viewport_width_px=viewport_width_px,
        voxel_width=voxel_width,
        lod_bias=lod_bias,
        force_level=force_level,
        level_scale_factors=brick_layout._level_scale_factors,
    )

    # Sort tiles so the nearest (most important) ones come first.
    tile_arr = sort_tiles_by_distance_2d(
        tile_arr,
        camera_pos,
        brick_layout.block_size,
        level_scale_arr_shader=brick_layout._scale_arr_shader,
        level_translation_arr_shader=brick_layout._translation_arr_shader,
    )

    # Convert the compact array representation to a dict of BlockKey2D objects,
    # embedding the current slice coordinate into each key.
    required_block_keys = arr_to_block_keys_2d(
        tile_arr, slice_coord=current_slice_coord
    )

    # Remove tiles that fall entirely outside the visible viewport rectangle.
    if use_culling and view_min is not None and view_max is not None:
        required_block_keys, _ = viewport_cull_2d(
            required_block_keys,
            brick_layout.block_size,
            view_min,
            view_max,
            level_scale_arr_shader=brick_layout._scale_arr_shader,
            level_translation_arr_shader=brick_layout._translation_arr_shader,
        )

    # Report the level of the first (nearest) tile; fall back to level 1 if
    # no tiles survived culling.
    target_level = int(tile_arr[0, 0]) if len(tile_arr) > 0 else 1

    return required_block_keys, target_level


def build_fetch_requests_2d(
    required_block_keys: dict,
    current_slice_coord: SliceCoord,
    cache_query: CacheQuery2D,
    resolved: ResolvedDisplayState,
    full_level_shapes: list[tuple[int, ...]],
    world_to_level_transforms: list[AffineTransform],
    expand_fetch_index: ExpandFetchIndex,
    block_size: int,
) -> list[MultiscaleChunkRequest]:
    """Check cache residency, allocate slots, and build fetch requests for 2-D tiles.

    Mirrors :func:`build_fetch_requests_3d` for the 2-D case.  The key
    difference is that cache residency is keyed on both the tile identity
    *and* ``current_slice_coord``, because the same tile at the same LOD
    level must be re-fetched when the user moves along a sliced axis.

    Parameters
    ----------
    required_block_keys : dict
        Ordered mapping of BlockKey2D → ... in priority order, as returned
        by :func:`select_visible_bricks_2d`.
    current_slice_coord : SliceCoord
        Coordinates along the non-displayed axes; embedded in the cache key
        so that stale tiles from a previous slice position are not reused.
    cache_query : CacheQuery2D
        Interface to the GPU tile cache for residency checks and slot
        allocation.
    resolved : ResolvedDisplayState
        Current viewer state (visible axes, current index).
    full_level_shapes : list[tuple[int, ...]]
        Voxel shape of the full array at each LOD level (0-indexed).
    world_to_level_transforms : list[AffineTransform]
        Affine transforms from world to level-k data coordinates, one per
        level (0-indexed).
    expand_fetch_index : ExpandFetchIndex
        Callable that expands the indices of the chunk to fetch based on
        the needs of the sampler (e.g., cache overlap). This method
        is implemented on the MultiscaleImageHandle.
    block_size : int
        Side length of each tile in voxels at its LOD level.

    Returns
    -------
    chunk_requests : list[MultiscaleChunkRequest]
        One request per tile that is not yet resident in the cache for the
        current slice position, capped at ``cache_query.capacity`` tiles.
    """
    n_budget = cache_query.capacity
    n_needed = len(required_block_keys)
    # Trim to budget before iterating to avoid unnecessary cache queries.
    if n_needed > n_budget:
        keys_to_keep = list(required_block_keys.keys())[:n_budget]
        required_block_keys = {k: required_block_keys[k] for k in keys_to_keep}

    # Extract slice (non-display) axis indices from the resolved display state.
    ndim = len(resolved.data_coords)
    visible = set(resolved.visible_axes)
    slice_indices: dict[int, int] = {
        ax: v for ax, v in resolved.current_index.items()
        if isinstance(v, int) and ax not in visible
    }
    # All requests share one slice_id so the renderer can swap a complete
    # set of tiles atomically when a frame is ready.
    slice_id = uuid4()
    chunk_requests: list[MultiscaleChunkRequest] = []

    for tile_key in required_block_keys:
        # Convert the 1-indexed level to the 0-indexed level used internally.
        bk = BrickKey(
            level=tile_key.level - 1,
            brick_coords=(tile_key.gy, tile_key.gx),
        )
        # The 2-D cache includes slice_coord in the lookup so that a tile
        # loaded for a different z-plane is not incorrectly considered resident.
        if cache_query.is_resident(bk, current_slice_coord):
            continue
        slot_id = cache_query.allocate_slot(bk, current_slice_coord)
        chunk_id = uuid4()
        y0, x0, y1, x1 = _block_key_2d_to_core_coords(tile_key, block_size)
        level_index = tile_key.level - 1
        # Two spatial display ranges for the 2-D case (yx order).
        display_coords = [(y0, y1), (x0, x1)]
        axis_selections = _build_axis_selections(
            resolved.visible_axes,
            slice_indices,
            ndim,
            display_coords,
            level_shape=full_level_shapes[level_index],
            world_to_level_k=world_to_level_transforms[level_index],
        )
        # Convert (start, stop) tuples to slice objects for non-sliced axes.
        core_index = {
            ax_i: sel if isinstance(sel, int) else slice(sel[0], sel[1])
            for ax_i, sel in enumerate(axis_selections)
        }
        # Expand the indices of the chunk to fetch based on
        # the needs of the sampler (e.g., cache overlap).
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

    return chunk_requests
