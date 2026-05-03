"""GFXMultiscaleImageVisual — stripped demo render visual (no painting, no EventBus)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import pygfx as gfx

from demo_multiscale_ndv._chunk_request import MultiscaleChunkRequest
from demo_multiscale_ndv._pygfx_handle import GFXMultiscaleImageHandle, GFXMultiscaleVolumeHandle
from demo_multiscale_ndv.lut_indirection import BlockLayout3D
from demo_multiscale_ndv.lut_indirection._layout_2d import BlockLayout2D
from demo_multiscale_ndv.transform import AffineTransform
from demo_multiscale_ndv._level_of_detail_3d import build_level_grids
from demo_multiscale_ndv._level_of_detail_2d import build_tile_grids_2d

from ndv.models._resolve import ResolvedDisplayState

from demo_multiscale_ndv._camera_view import CameraView2D, CameraView3D

if TYPE_CHECKING:
    from pygfx.resources import Buffer

    from demo_multiscale_ndv.block_cache._tile_manager_2d import BlockKey2D
    from demo_multiscale_ndv.block_cache._tile_manager_2d import TileSlot as TileSlot2D
    from demo_multiscale_ndv.shaders._block_image import ImageBlockMaterial
    from demo_multiscale_ndv.shaders._multiscale_volume_brick import MultiscaleVolumeBrickMaterial


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
        render_order: int = 0,
        pick_write: bool = True,
    ) -> None:
        self.visual_model_id = visual_model_id
        self.render_modes = render_modes

        if full_level_shapes is not None:
            self._ndim = len(full_level_shapes[0])
        elif volume_geometry is not None:
            self._ndim = len(volume_geometry.level_shapes[0])
        elif image_geometry_2d is not None:
            self._ndim = len(image_geometry_2d.level_shapes[0])
        else:
            self._ndim = 3

        if voxel_scales is None:
            self._voxel_scales = np.ones(self._ndim, dtype=np.float64)
        else:
            self._voxel_scales = np.asarray(voxel_scales, dtype=np.float64)

        self._volume_geometry = volume_geometry
        self._image_geometry_2d = image_geometry_2d

        if full_level_transforms is not None:
            self._level_transforms = list(full_level_transforms)
        elif volume_geometry is not None:
            self._level_transforms = volume_geometry.level_transforms
        elif image_geometry_2d is not None:
            self._level_transforms = image_geometry_2d.level_transforms
        else:
            self._level_transforms = []

        if full_level_shapes is not None:
            self._full_level_shapes = list(full_level_shapes)
        elif volume_geometry is not None:
            self._full_level_shapes = list(volume_geometry.level_shapes)
        elif image_geometry_2d is not None:
            self._full_level_shapes = list(image_geometry_2d.level_shapes)
        else:
            self._full_level_shapes = []

        self._world_to_level_transforms = self._build_world_to_level_transforms()

        self._last_displayed_axes: tuple[int, ...] | None = displayed_axes
        self._frame_number = 0
        self._last_plan_stats: dict = {}

        if colormap is None:
            colormap = gfx.cm.viridis

        # ── 3-D handle ──────────────────────────────────────────────────
        self._volume_handle: GFXMultiscaleVolumeHandle | None = None
        if "3d" in render_modes and volume_geometry is not None:
            self._volume_handle = GFXMultiscaleVolumeHandle(
                volume_geometry=volume_geometry,
                voxel_scales=self._voxel_scales,
                gpu_budget_bytes=gpu_budget_bytes_3d,
                colormap=colormap,
                clim=clim,
                threshold=threshold,
                pick_write=pick_write,
            )
            self._volume_handle.node.render_order = render_order

        # ── 2-D handle ──────────────────────────────────────────────────
        self._image_handle: GFXMultiscaleImageHandle | None = None
        if "2d" in render_modes and image_geometry_2d is not None:
            self._image_handle = GFXMultiscaleImageHandle(
                image_geometry_2d=image_geometry_2d,
                voxel_scales=self._voxel_scales,
                gpu_budget_bytes=gpu_budget_bytes_2d,
                colormap=colormap,
                clim=clim,
                interpolation=interpolation,
                pick_write=pick_write,
            )
            self._image_handle.node.render_order = render_order

        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    # ── Public scene-node / material properties ─────────────────────────

    @property
    def node_3d(self) -> gfx.Group | None:
        return self._volume_handle.node if self._volume_handle is not None else None

    @property
    def node_2d(self) -> gfx.Group | None:
        return self._image_handle.node if self._image_handle is not None else None

    @property
    def material_3d(self) -> MultiscaleVolumeBrickMaterial | None:
        return self._volume_handle.material if self._volume_handle is not None else None

    @property
    def material_2d(self) -> ImageBlockMaterial | None:
        return self._image_handle.material if self._image_handle is not None else None

    @property
    def n_levels(self) -> int:
        if self._volume_geometry is not None:
            return self._volume_geometry.n_levels
        if self._image_geometry_2d is not None:
            return self._image_geometry_2d.n_levels
        raise RuntimeError("No geometry available")

    # ── Public 3-D planning surface ─────────────────────────────────────

    @property
    def volume_handle(self) -> GFXMultiscaleVolumeHandle | None:
        return self._volume_handle

    @property
    def volume_geometry(self) -> VolumeGeometry | None:
        return self._volume_geometry

    def begin_frame_3d(self, visible_axes: tuple[int, ...]) -> None:
        """Advance the LRU frame counter and sync scene-node axes if needed.

        Must be called once per reslice, before ``volume_handle.cache_query()``
        is used, so the LRU timestamps are consistent within a single frame.
        """
        self._frame_number += 1
        if self._volume_handle is not None:
            self._volume_handle.frame_number = self._frame_number
        if visible_axes != self._last_displayed_axes:
            self._update_node_matrix(visible_axes)

    # ── Public 2-D planning surface ─────────────────────────────────────

    @property
    def image_handle(self) -> GFXMultiscaleImageHandle | None:
        return self._image_handle

    @property
    def image_geometry_2d(self) -> ImageGeometry2D | None:
        return self._image_geometry_2d

    @property
    def voxel_scales(self) -> np.ndarray:
        return self._voxel_scales

    @property
    def full_level_shapes(self) -> list[tuple[int, ...]]:
        return self._full_level_shapes

    @property
    def world_to_level_transforms(self) -> list[AffineTransform]:
        return self._world_to_level_transforms

    def begin_frame_2d(self, visible_axes: tuple[int, ...]) -> None:
        """Advance the LRU frame counter and sync scene-node axes if needed.

        Must be called once per reslice, before ``image_handle.cache_query()``
        is used, so the LRU timestamps are consistent within a single frame.
        """
        self._frame_number += 1
        if self._image_handle is not None:
            self._image_handle.frame_number = self._frame_number
        if visible_axes != self._last_displayed_axes:
            self._update_node_matrix(visible_axes)

    # ── Geometry rebuild ────────────────────────────────────────────────

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]) -> gfx.Group | None:
        _old_node, new_node = self.rebuild_geometry(
            self._full_level_shapes, displayed_axes
        )
        return new_node

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
            if self._volume_geometry is not None and self._volume_handle is not None:
                shapes_3d = [
                    tuple(s[ax] for ax in displayed_axes) for s in level_shapes
                ]
                if shapes_3d != self._volume_geometry.level_shapes:
                    self._volume_geometry.update(shapes_3d)
                    self._rebuild_3d_resources()
                new_node = self.node_3d

        if "2d" in self.render_modes and len(displayed_axes) == 2:
            old_node = self.node_2d
            if self._image_geometry_2d is not None and self._image_handle is not None:
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
        self._volume_handle.rebuild(self._volume_geometry)
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def _rebuild_2d_resources(self) -> None:
        self._image_handle.rebuild(self._image_geometry_2d)
        if self._last_displayed_axes is not None:
            self._update_node_matrix(self._last_displayed_axes)

    def _update_node_matrix(self, displayed_axes: tuple[int, ...]) -> None:
        self._last_displayed_axes = displayed_axes
        if self._volume_handle is not None:
            self._volume_handle.apply_world_transform()
        if self._image_handle is not None:
            self._image_handle.apply_world_transform()

    def _build_world_to_level_transforms(self) -> list[AffineTransform]:
        result: list[AffineTransform] = []
        for lt in self._level_transforms:
            result.append(AffineTransform(matrix=lt.inverse_matrix))
        return result

    # ── 3-D upload ──────────────────────────────────────────────────────

    def on_data_ready(
        self,
        batch: list[tuple[MultiscaleChunkRequest, np.ndarray]],
    ) -> None:
        for req, data in batch:
            if req.slot_id is None:
                continue
            self._volume_handle.set_brick(req.slot_id, data)
        self._volume_handle.commit()

    def cancel_pending(self) -> None:
        if self._volume_handle is None:
            return
        self._volume_handle.invalidate_pending()

    # ── 2-D upload ──────────────────────────────────────────────────────

    def on_data_ready_2d(
        self,
        batch: list[tuple[MultiscaleChunkRequest, np.ndarray]],
        slice_coord: tuple[tuple[int, int], ...],
    ) -> None:
        for req, data in batch:
            if req.slot_id is None:
                continue
            self._image_handle.set_brick(req.slot_id, data)
        self._image_handle.commit(slice_coord)

    def cancel_pending_2d(self) -> None:
        if self._image_handle is None:
            return
        self._image_handle.invalidate_pending()

    def invalidate_2d_cache(self) -> None:
        if self._image_handle is None:
            return
        self.cancel_pending_2d()
