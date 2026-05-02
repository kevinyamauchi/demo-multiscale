"""Data store classes: ChunkRequest, AxisInfo, BaseDataStore, OMEZarrImageDataStore."""

from __future__ import annotations

import dataclasses
import pathlib
import uuid
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import tensorstore as ts
from pydantic import UUID4, AfterValidator, BaseModel, ConfigDict, Field, PrivateAttr

from demo_multiscale.transform import AffineTransform

if TYPE_CHECKING:
    from yaozarrs import v05


# ---------------------------------------------------------------------------
# ChunkRequest
# ---------------------------------------------------------------------------


class ChunkRequest(NamedTuple):
    """A request for one padded brick / chunk of data."""

    chunk_request_id: uuid.UUID
    slice_request_id: uuid.UUID
    scale_index: int
    axis_selections: tuple[int | tuple[int, int], ...]


# ---------------------------------------------------------------------------
# AxisInfo
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AxisInfo:
    """Descriptor for one axis of an OME-Zarr array."""

    name: str
    unit: str | None
    type: str
    array_dim: int


# ---------------------------------------------------------------------------
# BaseDataStore
# ---------------------------------------------------------------------------


class BaseDataStore(BaseModel):
    """Base class for all DataStores."""

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    name: str = "data store"


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

_SUPPORTED_SCHEMES = frozenset({"file", "s3", "gs", "gcs", "https", "http"})


def _is_windows_drive_prefix(text: str) -> bool:
    return len(text) >= 2 and text[1] == ":" and text[0].isalpha()


def _file_uri_to_local_path(uri: str) -> pathlib.Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Expected file:// URI, got {uri!r}")

    netloc = parsed.netloc.replace("\\", "/")
    path = parsed.path.replace("\\", "/")

    if len(netloc) == 0 and len(path) == 0:
        raise ValueError(f"URI couldn't be parsed: {parsed}")

    if (
        len(netloc) == 0
        and len(path) >= 3
        and path[0] == "/"
        and _is_windows_drive_prefix(path[1:])
    ):
        return pathlib.Path(path[1:])

    if len(netloc) > 0:
        if _is_windows_drive_prefix(netloc):
            return pathlib.Path(netloc.rstrip("/") + path)
        return pathlib.Path(f"//{netloc}{path}")

    return pathlib.Path(path)


def _join_uri_path(uri: str, child_path: str) -> str:
    child = child_path.replace("\\", "/").lstrip("/")
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return (_file_uri_to_local_path(uri) / pathlib.PurePosixPath(child)).as_uri()
    return f"{uri.rstrip('/')}/{child}"


def _validate_uri_scheme(uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme not in _SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported URI scheme {parsed.scheme!r} in {uri!r}. "
            f"Supported schemes: {', '.join(sorted(_SUPPORTED_SCHEMES))}."
        )


def _build_kvstore_spec(uri: str, array_path: str) -> dict:
    parsed = urlparse(uri)
    scheme = parsed.scheme

    if scheme == "file":
        root = _file_uri_to_local_path(uri)
        return {
            "driver": "file",
            "path": str(root / pathlib.PurePosixPath(array_path)),
        }
    elif scheme in ("s3",):
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        full = f"{prefix}/{array_path}" if prefix else array_path
        return {"driver": "s3", "bucket": bucket, "path": full}
    elif scheme in ("gs", "gcs"):
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        full = f"{prefix}/{array_path}" if prefix else array_path
        return {"driver": "gcs", "bucket": bucket, "path": full}
    elif scheme in ("http", "https"):
        base = uri.rstrip("/")
        return {"driver": "http", "base_url": f"{base}/{array_path}"}
    else:
        raise ValueError(f"Unsupported URI scheme: {scheme!r}")


def _detect_zarr_driver(uri: str, array_path: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        level_path = _file_uri_to_local_path(uri) / pathlib.PurePosixPath(array_path)

        if (level_path / ".zarray").exists():
            return "zarr"
        if (level_path / "zarr.json").exists():
            return "zarr3"
        raise FileNotFoundError(
            f"Cannot determine zarr format for '{level_path}': "
            f"neither '.zarray' (zarr v2) nor 'zarr.json' (zarr v3) found."
        )
    return "zarr3"


def _open_ome_ts_stores(
    zarr_path: str,
    scale_names: list[str],
    anonymous: bool = False,
) -> list[ts.TensorStore]:
    stores: list[ts.TensorStore] = []
    scheme = urlparse(zarr_path).scheme
    for name in scale_names:
        driver = _detect_zarr_driver(zarr_path, name)
        spec: dict[str, Any] = {
            "driver": driver,
            "kvstore": _build_kvstore_spec(zarr_path, name),
        }
        if anonymous and scheme in ("s3", "gs", "gcs"):
            if scheme == "s3":
                spec.setdefault("context", {})["aws_credentials"] = {
                    "anonymous": True,
                }
            else:
                spec.setdefault("context", {})["gcs_user_project"] = ""
        store = ts.open(spec).result()
        shape = tuple(int(d) for d in store.domain.shape)
        print(f"  {name}: opened as {driver}  shape={shape}")
        stores.append(store)
    return stores


# ---------------------------------------------------------------------------
# OME metadata helpers
# ---------------------------------------------------------------------------


def _extract_global_transform(
    ms: v05.Multiscale,
) -> tuple[list[float], list[float]]:
    from yaozarrs.v05 import ScaleTransformation, TranslationTransformation

    n = len(ms.axes)
    global_scale = [1.0] * n
    global_translation = [0.0] * n

    if ms.coordinateTransformations is not None:
        for ct in ms.coordinateTransformations:
            if isinstance(ct, ScaleTransformation):
                global_scale = list(ct.scale)
            elif isinstance(ct, TranslationTransformation):
                global_translation = list(ct.translation)

    return global_scale, global_translation


def _derive_level_transforms(
    ms: v05.Multiscale,
    global_scale: list[float],
    global_translation: list[float],
) -> tuple[list[AffineTransform], list[float]]:
    """Return (level_transforms, voxel_sizes) where voxel_sizes is the absolute
    physical scale at the finest resolution level (level 0)."""
    n_axes = len(ms.axes)

    per_level: list[tuple[list[float], list[float]]] = []
    for ds in ms.datasets:
        sc = [g * s for g, s in zip(global_scale, ds.scale_transform.scale)]
        tr_raw = ds.translation_transform
        tr_ds = list(tr_raw.translation) if tr_raw is not None else [0.0] * n_axes
        tr = [gs * t + gt for gs, t, gt in zip(global_scale, tr_ds, global_translation)]
        per_level.append((sc, tr))

    s0, t0 = per_level[0]
    transforms: list[AffineTransform] = []
    for sc_k, tr_k in per_level:
        cellier_scale = tuple(sc_k[i] / s0[i] for i in range(n_axes))
        cellier_trans = tuple((tr_k[i] - t0[i]) / s0[i] for i in range(n_axes))
        transforms.append(
            AffineTransform.from_scale_and_translation(
                scale=cellier_scale, translation=cellier_trans
            )
        )
    return transforms, list(s0)


# ---------------------------------------------------------------------------
# OMEZarrImageDataStore
# ---------------------------------------------------------------------------


class OMEZarrImageDataStore(BaseDataStore):
    """Data store for an OME-Zarr v0.5 image read via tensorstore."""

    store_type: Literal["ome_zarr_image"] = "ome_zarr_image"
    zarr_path: str
    multiscale_index: int = 0
    scale_names: list[str]
    level_transforms: list[AffineTransform]
    voxel_sizes: list[float]
    axis_names: list[str]
    axis_units: list[str | None]
    axis_types: list[str]
    anonymous: bool = False
    name: str = "ome zarr image data store"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _ts_stores: list[ts.TensorStore] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._ts_stores = _open_ome_ts_stores(
            self.zarr_path, self.scale_names, anonymous=self.anonymous
        )

    @classmethod
    def from_path(
        cls,
        zarr_path: str,
        *,
        multiscale_index: int = 0,
        series_index: int = 0,
        anonymous: bool = False,
        name: str = "ome zarr image data store",
    ) -> OMEZarrImageDataStore:
        """Construct from an OME-Zarr v0.5 URI or local path."""
        import yaozarrs
        from yaozarrs import v05 as ome_v05

        # Convert bare local paths to file:// URIs.
        parsed = urlparse(zarr_path)
        if not parsed.scheme or (len(parsed.scheme) == 1 and parsed.scheme.isalpha()):
            zarr_path = pathlib.Path(zarr_path).resolve().as_uri()

        _validate_uri_scheme(zarr_path)

        group = yaozarrs.open_group(zarr_path)
        metadata = group.ome_metadata()

        if isinstance(metadata, ome_v05.Bf2Raw):
            zarr_path, group, metadata = cls._resolve_bf2raw(
                zarr_path, group, series_index
            )

        if not isinstance(metadata, ome_v05.Image):
            type_name = type(metadata).__name__ if metadata is not None else "None"
            raise TypeError(
                f"Expected an OME-Zarr Image at {zarr_path!r}, "
                f"got {type_name}. Plates and other types are not supported."
            )

        ms = metadata.multiscales[multiscale_index]

        axis_names = [ax.name for ax in ms.axes]
        axis_units = [getattr(ax, "unit", None) for ax in ms.axes]
        axis_types = [ax.type or "" for ax in ms.axes]

        global_scale, global_translation = _extract_global_transform(ms)

        level_transforms, voxel_sizes = _derive_level_transforms(
            ms, global_scale, global_translation
        )

        scale_names = [ds.path for ds in ms.datasets]

        return cls(
            zarr_path=zarr_path,
            multiscale_index=multiscale_index,
            scale_names=scale_names,
            level_transforms=level_transforms,
            voxel_sizes=voxel_sizes,
            axis_names=axis_names,
            axis_units=axis_units,
            axis_types=axis_types,
            anonymous=anonymous,
            name=name,
        )

    @classmethod
    def _resolve_bf2raw(
        cls,
        zarr_path: str,
        group: Any,
        series_index: int,
    ) -> tuple[str, Any, Any]:
        import yaozarrs
        from yaozarrs import v05 as ome_v05

        series_paths: list[str] | None = None
        if "OME" in group:
            ome_subgroup = group["OME"]
            ome_meta = ome_subgroup.ome_metadata()
            if isinstance(ome_meta, ome_v05.Series):
                series_paths = ome_meta.series

        if series_paths is not None:
            if series_index < 0 or series_index >= len(series_paths):
                raise ValueError(
                    f"series_index={series_index} out of range: "
                    f"Bf2Raw container has {len(series_paths)} series."
                )
            image_path = series_paths[series_index]
        else:
            image_path = str(series_index)

        resolved_path = _join_uri_path(zarr_path, image_path)
        print(
            f"  Bf2Raw container detected — resolving series {series_index} "
            f"at '{image_path}'"
        )

        child_group = yaozarrs.open_group(resolved_path)
        child_metadata = child_group.ome_metadata()

        return resolved_path, child_group, child_metadata

    @property
    def n_levels(self) -> int:
        return len(self._ts_stores)

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        return [tuple(int(d) for d in store.domain.shape) for store in self._ts_stores]

    @property
    def axes(self) -> list[AxisInfo]:
        return [
            AxisInfo(name=n, unit=u, type=t, array_dim=i)
            for i, (n, u, t) in enumerate(
                zip(self.axis_names, self.axis_units, self.axis_types)
            )
        ]

    @property
    def dtype(self) -> np.dtype:
        return self._ts_stores[0].dtype.numpy_dtype

    def get_data(self, request: ChunkRequest) -> np.ndarray:
        """Read a single padded brick, returning a zero-padded float32 array."""
        store = self._ts_stores[request.scale_index]
        store_shape = tuple(int(d) for d in store.domain.shape)

        out_shape = tuple(
            stop - start
            for sel in request.axis_selections
            if isinstance(sel, tuple)
            for start, stop in [sel]
        )
        out = np.zeros(out_shape, dtype=np.float32)

        store_idx: list[int | slice] = []
        dest_starts: list[int] = []
        valid = True

        for axis_i, sel in enumerate(request.axis_selections):
            size = store_shape[axis_i]
            if isinstance(sel, int):
                store_idx.append(max(0, min(sel, size - 1)))
            else:
                start, stop = sel
                c_start = max(start, 0)
                c_stop = min(stop, size)
                if c_stop <= c_start:
                    valid = False
                    break
                store_idx.append(slice(c_start, c_stop))
                dest_starts.append(c_start - start)

        if valid:
            region = np.asarray(
                store[tuple(store_idx)].read().result(),
                dtype=np.float32,
            )
            dest_idx = tuple(slice(d, d + s) for d, s in zip(dest_starts, region.shape))
            out[dest_idx] = region

        return out
