"""OMEZarrDataWrapper — MultiscaleDataWrapper backed by OMEZarrImageDataStore."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from demo_multiscale_ndv._data_wrapper import BrickInfo, BrickKey, MultiscaleDataWrapper
from demo_multiscale_ndv.data_store import OMEZarrImageDataStore
from demo_multiscale_ndv.transform import AffineTransform


class OMEZarrDataWrapper(MultiscaleDataWrapper):
    """Full MultiscaleDataWrapper implementation backed by OMEZarrImageDataStore.

    Open via ``OMEZarrDataWrapper.from_path(zarr_path)``.  After Phase 1,
    nothing outside this class references ``OMEZarrImageDataStore`` directly.
    """

    PRIORITY: ClassVar[int] = 10

    # _data (inherited from DataWrapper) holds the OMEZarrImageDataStore.

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        series_index: int = 0,
        anonymous: bool = False,
        name: str = "ome zarr image data store",
    ) -> OMEZarrDataWrapper:
        store = OMEZarrImageDataStore.from_path(
            zarr_path,
            multiscale_index=multiscale_index,
            series_index=series_index,
            anonymous=anonymous,
            name=name,
        )
        return cls(store)

    @classmethod
    def supports(cls, obj: Any) -> bool:
        return isinstance(obj, OMEZarrImageDataStore)

    # ── DataWrapper interface ────────────────────────────────────────────

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return tuple(self._data.axis_names)

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        shape = self._data.level_shapes[0]
        return {name: range(size) for name, size in zip(self._data.axis_names, shape)}

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    # ── MultiscaleDataWrapper interface ─────────────────────────────────

    @property
    def n_levels(self) -> int:
        return self._data.n_levels

    def level_shape(self, level: int) -> tuple[int, ...]:
        return self._data.level_shapes[level]

    def level_transform(self, level: int) -> AffineTransform:
        return self._data.level_transforms[level]

    def brick_grid(
        self, level: int, spatial_axes: tuple[int, ...]
    ) -> Iterable[BrickInfo]:
        # Brick partitioning belongs to the visual layer (Phase 4 promotes this).
        raise NotImplementedError("brick_grid is not used until Phase 4")

    def isel(
        self, index: Mapping[int, int | slice], level: int = 0
    ) -> np.ndarray:
        """Read a padded brick from ``level``, zero-padding out-of-bounds regions."""
        store = self._data._ts_stores[level]
        store_shape = tuple(int(d) for d in store.domain.shape)
        ndim = len(store_shape)

        out_shape: list[int] = []
        for ax in range(ndim):
            sel = index.get(ax)
            if isinstance(sel, slice):
                start = sel.start if sel.start is not None else 0
                stop = sel.stop if sel.stop is not None else store_shape[ax]
                out_shape.append(stop - start)

        out = np.zeros(out_shape, dtype=np.float32)

        store_idx: list[int | slice] = []
        dest_starts: list[int] = []
        valid = True

        for ax in range(ndim):
            size = store_shape[ax]
            sel = index.get(ax)
            if isinstance(sel, int):
                store_idx.append(max(0, min(sel, size - 1)))
            elif isinstance(sel, slice):
                start = sel.start if sel.start is not None else 0
                stop = sel.stop if sel.stop is not None else size
                c_start = max(start, 0)
                c_stop = min(stop, size)
                if c_stop <= c_start:
                    valid = False
                    break
                store_idx.append(slice(c_start, c_stop))
                dest_starts.append(c_start - start)
            else:
                store_idx.append(slice(None))
                dest_starts.append(0)

        if valid:
            region = np.asarray(
                store[tuple(store_idx)].read().result(),
                dtype=np.float32,
            )
            dest_idx = tuple(
                slice(d, d + s) for d, s in zip(dest_starts, region.shape)
            )
            out[dest_idx] = region

        return out

    # ── Convenience properties for the demo ─────────────────────────────

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        return self._data.level_shapes

    @property
    def level_transforms(self) -> list[AffineTransform]:
        return self._data.level_transforms

    @property
    def voxel_sizes(self) -> list[float]:
        return self._data.voxel_sizes

    @property
    def axis_names(self) -> list[str]:
        return self._data.axis_names

    @property
    def axis_types(self) -> list[str]:
        return self._data.axis_types
