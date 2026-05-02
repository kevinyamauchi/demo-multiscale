"""Affine coordinate transforms (BaseTransform + AffineTransform merged)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing_extensions import Self


class BaseTransform(BaseModel, ABC):
    """Base class for coordinate transforms."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @abstractmethod
    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def map_normal_vector(self, normal_vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def imap_normal_vector(self, normal_vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _to_homogeneous(coordinates: np.ndarray, ndim: int) -> np.ndarray:
    coordinates = np.atleast_2d(coordinates)
    n_components = coordinates.shape[1]
    if n_components == ndim:
        return np.pad(coordinates, pad_width=((0, 0), (0, 1)), constant_values=1)
    elif n_components == ndim + 1:
        return coordinates
    else:
        raise ValueError(
            f"coordinates must have {ndim} or {ndim + 1} components per point, "
            f"got {n_components}"
        )


class AffineTransform(BaseTransform):
    """N-dimensional affine transformation using a homogeneous matrix."""

    matrix: np.ndarray

    @field_validator("matrix", mode="before")
    @classmethod
    def _coerce_to_ndarray_float32(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
            raise ValueError(
                f"matrix must be square with shape (N, N) where N >= 2, got {arr.shape}"
            )
        return arr

    @field_serializer("matrix")
    def _serialize_matrix(self, v: np.ndarray) -> list:
        return v.tolist()

    @cached_property
    def ndim(self) -> int:
        return self.matrix.shape[0] - 1

    @cached_property
    def inverse_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.matrix).astype(np.float32)

    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        return np.dot(_to_homogeneous(coordinates, self.ndim), self.matrix.T)[
            :, : self.ndim
        ]

    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        return np.dot(_to_homogeneous(coordinates, self.ndim), self.inverse_matrix.T)[
            :, : self.ndim
        ]

    def map_normal_vector(self, normal_vector: np.ndarray) -> np.ndarray:
        nd = self.ndim
        transformed = np.matmul(
            _to_homogeneous(normal_vector, nd), self.inverse_matrix
        )[:, :nd]
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        return transformed / norms

    def imap_normal_vector(self, normal_vector: np.ndarray) -> np.ndarray:
        nd = self.ndim
        transformed = np.matmul(_to_homogeneous(normal_vector, nd), self.matrix)[:, :nd]
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        return transformed / norms

    def set_slice(self, axes: tuple[int, ...]) -> AffineTransform:
        k = len(axes)
        m = np.eye(k + 1, dtype=np.float32)
        tc = self.ndim
        for out_i, src_i in enumerate(axes):
            for out_j, src_j in enumerate(axes):
                m[out_i, out_j] = self.matrix[src_i, src_j]
            m[out_i, k] = self.matrix[src_i, tc]
        return AffineTransform(matrix=m)

    def expand_dims(self, target_ndim: int) -> AffineTransform:
        if target_ndim < self.ndim:
            raise ValueError(
                f"target_ndim ({target_ndim}) must be >= self.ndim ({self.ndim})"
            )
        if target_ndim == self.ndim:
            return self
        m = np.eye(target_ndim + 1, dtype=np.float32)
        offset = target_ndim - self.ndim
        nd = self.ndim
        m[offset:target_ndim, offset:target_ndim] = self.matrix[:nd, :nd]
        m[offset:target_ndim, target_ndim] = self.matrix[:nd, nd]
        return AffineTransform(matrix=m)

    @classmethod
    def identity(cls, ndim: int = 3) -> Self:
        return cls(matrix=np.eye(ndim + 1, dtype=np.float32))

    @classmethod
    def from_scale(cls, scale: tuple[float, ...]) -> Self:
        return cls.from_scale_and_translation(scale, tuple(0.0 for _ in scale))

    @classmethod
    def from_scale_and_translation(
        cls,
        scale: tuple[float, ...],
        translation: tuple[float, ...] | None = None,
    ) -> Self:
        ndim = len(scale)
        if translation is None:
            translation = tuple(0.0 for _ in range(ndim))
        if len(translation) != ndim:
            raise ValueError(
                f"scale and translation must have the same length, "
                f"got {len(scale)} and {len(translation)}"
            )
        matrix = np.eye(ndim + 1, dtype=np.float32)
        for i in range(ndim):
            matrix[i, i] = scale[i]
            matrix[i, ndim] = translation[i]
        return cls(matrix=matrix)

    @classmethod
    def from_translation(cls, translation: tuple[float, ...]) -> Self:
        ndim = len(translation)
        matrix = np.eye(ndim + 1, dtype=np.float32)
        for i in range(ndim):
            matrix[i, ndim] = translation[i]
        return cls(matrix=matrix)

    def compose(self, other: AffineTransform) -> AffineTransform:
        if self.ndim != other.ndim:
            raise ValueError(
                f"Cannot compose transforms with different ndim: "
                f"{self.ndim} vs {other.ndim}"
            )
        return AffineTransform(matrix=other.matrix @ self.matrix)

    def __matmul__(self, other: AffineTransform) -> AffineTransform:
        return self.compose(other)
