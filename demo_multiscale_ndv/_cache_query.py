"""CacheQuery protocols and SlotId type alias for Phase 3+."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from demo_multiscale_ndv._data_wrapper import BrickKey

SlotId = int  # opaque cache-slot identifier; equals TileSlot.index


class CacheQuery(Protocol):
    """Read/allocate surface for the 3-D brick cache."""

    @property
    def capacity(self) -> int:
        """Number of allocatable slots (excludes reserved slot 0)."""
        ...

    def is_resident(
        self,
        key: BrickKey,
        slice_coord: tuple[tuple[int, int], ...] = (),
    ) -> bool:
        """Return True and refresh LRU timestamp if the brick is in the cache."""
        ...

    def allocate_slot(
        self,
        key: BrickKey,
        slice_coord: tuple[tuple[int, int], ...] = (),
    ) -> SlotId:
        """Reserve a cache slot for *key* (evicting LRU if necessary).

        The slot is in-flight until the handle's ``commit()`` is called.
        """
        ...


class CacheQuery2D(Protocol):
    """Read/allocate surface for the 2-D tile cache.

    Identical to ``CacheQuery`` except every method also takes
    ``slice_coord`` because the 2-D cache key includes the current
    slice position.
    """

    @property
    def capacity(self) -> int:
        """Number of allocatable slots (excludes reserved slot 0)."""
        ...

    def is_resident(
        self,
        key: BrickKey,
        slice_coord: tuple[tuple[int, int], ...],
    ) -> bool:
        """Return True and refresh LRU timestamp if the tile is in the cache."""
        ...

    def allocate_slot(
        self,
        key: BrickKey,
        slice_coord: tuple[tuple[int, int], ...],
    ) -> SlotId:
        """Reserve a cache slot for *(key, slice_coord)* (evicting LRU if needed).

        The slot is in-flight until the handle's ``commit()`` is called.
        """
        ...
