"""MultiscaleChunkRequest — replaces ChunkRequest from data_store.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import UUID

from demo_multiscale_ndv._cache_query import SlotId  # noqa: F401  single definition


@dataclass(frozen=True)
class MultiscaleChunkRequest:
    """A request for one padded brick/tile at a specific resolution level."""

    chunk_request_id: UUID
    slice_request_id: UUID
    level: int                        # resolution level (replaces scale_index)
    index: Mapping[int, int | slice]  # level-k coordinates (replaces axis_selections)
    slot_id: SlotId | None = None     # always None in Phases 1–2; assigned in Phase 3
