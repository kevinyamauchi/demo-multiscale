"""MultiscaleDataWrapper ABC — extends ndv's DataWrapper with multiscale surface."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, TypeVar

import numpy as np
from ndv.models._data_wrapper import DataWrapper

from demo_multiscale_ndv.transform import AffineTransform

ArrayT = TypeVar("ArrayT")


class BrickKey(NamedTuple):
    level: int
    brick_coords: tuple[int, ...]  # one int per spatial axis


@dataclass(frozen=True)
class BrickInfo:
    key: BrickKey
    level_index: tuple[slice, ...]             # per-axis slice in level-k coordinates
    world_aabb: tuple[np.ndarray, np.ndarray]  # (min_xyz, max_xyz)


class MultiscaleDataWrapper(DataWrapper[Any]):
    """DataWrapper extended with multiscale metadata and fetch surface."""

    @property
    @abstractmethod
    def n_levels(self) -> int: ...

    @abstractmethod
    def level_shape(self, level: int) -> tuple[int, ...]: ...

    @abstractmethod
    def level_transform(self, level: int) -> AffineTransform: ...

    @abstractmethod
    def brick_grid(
        self, level: int, spatial_axes: tuple[int, ...]
    ) -> Iterable[BrickInfo]: ...

    @abstractmethod
    def isel(
        self, index: Mapping[int, int | slice], level: int = 0
    ) -> np.ndarray: ...
