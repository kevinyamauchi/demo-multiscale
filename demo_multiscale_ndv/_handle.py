"""MultiscaleVolumeHandle and MultiscaleImageHandle ABCs."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from ndv.views.bases._graphics._canvas_elements import CanvasElement
from ndv.views.bases._lut_view import LUTView

from demo_multiscale_ndv._cache_query import CacheQuery, CacheQuery2D, SlotId
from demo_multiscale_ndv._indexing import IndexSpec

if TYPE_CHECKING:
    import cmap

    from ndv.models._lut_model import ClimPolicy

    from demo_multiscale_ndv.render_visual import MultiscaleBrickLayout2D, MultiscaleBrickLayout3D

# Sorted tuple of (axis_index, slice_value) pairs encoding the sliced axes.
SliceCoord = tuple[tuple[int, int], ...]


class MultiscaleVolumeHandle(CanvasElement, LUTView):
    """3-D multiscale handle: owns GPU resources, accepts brick writes.

    Separates *rendering* (GPU textures, LUT rebuild) from *planning*
    (brick selection, LOD). Planner functions emit core data-space fetch
    indices; the handle owns backend-specific expansion through
    :meth:`expand_fetch_index`.
    """

    # ── Core brick surface ──────────────────────────────────────────────

    @abstractmethod
    def set_brick(self, slot: SlotId, data: np.ndarray) -> None:
        """Write *data* into the GPU cache at *slot* and queue for commit."""

    @abstractmethod
    def commit(self) -> None:
        """Move all queued writes to the tilemap and rebuild the LUT."""

    @abstractmethod
    def invalidate_pending(self) -> None:
        """Discard all in-flight (staged but not yet committed) slots."""

    @abstractmethod
    def cache_query(self) -> CacheQuery:
        """Return the cache-query adapter for the current frame."""

    @property
    @abstractmethod
    def brick_layout(self) -> MultiscaleBrickLayout3D: ...

    @abstractmethod
    def advance_frame(self) -> None:
        """Increment the LRU frame counter for this frame's cache queries."""

    @abstractmethod
    def expand_fetch_index(
        self,
        level: int,
        core_index: IndexSpec,
        level_shape: tuple[int, ...],
        current_slice_coord: SliceCoord,
    ) -> IndexSpec:
        """Expand planner-produced core index to a backend fetch index.

        Parameters
        ----------
        level :
            Zero-indexed multiscale level for the fetch request.
        core_index :
            Planner-generated index without overlap/halo.
        level_shape :
            Full nD shape of ``level``; useful for backend-specific bounds logic.
        current_slice_coord :
            Current non-displayed slice position. Included for signature
            compatibility with 2-D/3-D planners; 3-D implementations may ignore it.
        """

    # ── LUTView abstract methods ────────────────────────────────────────

    @abstractmethod
    def set_clims(self, clims: tuple[float, float]) -> None: ...

    @abstractmethod
    def set_colormap(self, cmap: cmap.Colormap) -> None: ...

    # ── Stubs for CanvasElement / LUTView methods not needed here ───────

    def visible(self) -> bool:
        return True

    def set_visible(self, visible: bool) -> None:
        pass

    def can_select(self) -> bool:
        return False

    def selected(self) -> bool:
        return False

    def set_selected(self, selected: bool) -> None:
        pass

    def frontend_widget(self) -> Any:
        return None

    def close(self) -> None:
        pass

    def set_channel_name(self, name: str) -> None:
        pass

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        pass

    def set_channel_visible(self, visible: bool) -> None:
        pass

    def set_gamma(self, gamma: float) -> None:
        pass


class MultiscaleImageHandle(CanvasElement, LUTView):
    """2-D multiscale handle: owns GPU resources, accepts tile writes.

    Like ``MultiscaleVolumeHandle`` but 2-D: every cache lookup also
    takes ``slice_coord`` because the tile key includes the current
    slice position.
    """

    # ── Core tile surface ───────────────────────────────────────────────

    @abstractmethod
    def set_brick(self, slot: SlotId, data: np.ndarray) -> None:
        """Write *data* into the GPU cache at *slot* and queue for commit."""

    @abstractmethod
    def commit(self, slice_coord: SliceCoord) -> None:
        """Move all queued writes to the tilemap and rebuild the LUT."""

    @abstractmethod
    def invalidate_pending(self) -> None:
        """Discard all in-flight (staged but not yet committed) slots."""

    @abstractmethod
    def cache_query(self) -> CacheQuery2D:
        """Return the cache-query adapter for the current frame."""

    @property
    @abstractmethod
    def brick_layout(self) -> MultiscaleBrickLayout2D: ...

    @abstractmethod
    def advance_frame(self) -> None:
        """Increment the LRU frame counter for this frame's cache queries."""

    @abstractmethod
    def expand_fetch_index(
        self,
        level: int,
        core_index: IndexSpec,
        level_shape: tuple[int, ...],
        current_slice_coord: SliceCoord,
    ) -> IndexSpec:
        """Expand planner-produced core index to a backend fetch index.

        ``current_slice_coord`` is part of the 2-D cache key and can influence
        backend expansion policy.
        """

    @abstractmethod
    def evict_finer_than(self, target_level: int) -> int:
        """Evict all tiles coarser than *target_level*; return eviction count."""

    @abstractmethod
    def rebuild_lut(self, slice_coord: SliceCoord) -> None:
        """Rebuild the indirection LUT without a preceding brick write."""

    # ── LUTView abstract methods ────────────────────────────────────────

    @abstractmethod
    def set_clims(self, clims: tuple[float, float]) -> None: ...

    @abstractmethod
    def set_colormap(self, cmap: cmap.Colormap) -> None: ...

    # ── Stubs for CanvasElement / LUTView methods not needed here ───────

    def visible(self) -> bool:
        return True

    def set_visible(self, visible: bool) -> None:
        pass

    def can_select(self) -> bool:
        return False

    def selected(self) -> bool:
        return False

    def set_selected(self, selected: bool) -> None:
        pass

    def frontend_widget(self) -> Any:
        return None

    def close(self) -> None:
        pass

    def set_channel_name(self, name: str) -> None:
        pass

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        pass

    def set_channel_visible(self, visible: bool) -> None:
        pass

    def set_gamma(self, gamma: float) -> None:
        pass
