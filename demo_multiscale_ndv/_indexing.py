from __future__ import annotations

from typing import Callable, TypeAlias

IndexSpec: TypeAlias = dict[int, int | slice]

# Function for expanding the indices of the chunk to fetch
# based on the needs of the shader.
# Args:
#   level, core_index, level_shape, current_slice_coord
ExpandFetchIndex: TypeAlias = Callable[
    [int, IndexSpec, tuple[int, ...], tuple[tuple[int, int], ...]],
    IndexSpec,
]
