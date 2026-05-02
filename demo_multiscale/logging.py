"""Structured debug logging for the rendering pipeline."""

from __future__ import annotations

import logging
import sys

_PERF_LOGGER = logging.getLogger("cellier.render.perf")
_GPU_LOGGER = logging.getLogger("cellier.render.gpu")
_CACHE_LOGGER = logging.getLogger("cellier.render.cache")
_SLICER_LOGGER = logging.getLogger("cellier.render.slicer")
_CAMERA_LOGGER = logging.getLogger("cellier.render.camera")
_SOURCE_ID_LOGGER = logging.getLogger("cellier.render.source_id")

_CATEGORY_MAP = {
    "perf": _PERF_LOGGER,
    "gpu": _GPU_LOGGER,
    "cache": _CACHE_LOGGER,
    "slicer": _SLICER_LOGGER,
    "camera": _CAMERA_LOGGER,
    "source_id": _SOURCE_ID_LOGGER,
}

_ALL_CATEGORIES = tuple(_CATEGORY_MAP.keys())

_HANDLER_TAG = "_cellier_debug"


def enable_debug_logging(
    categories: tuple[str, ...] = _ALL_CATEGORIES,
    use_rich: bool = True,
    level: int = logging.DEBUG,
) -> None:
    """Activate logging for the requested categories."""
    parent_logger = logging.getLogger("cellier.render")

    for cat in categories:
        logger = _CATEGORY_MAP.get(cat)
        if logger is not None:
            logger.setLevel(level)

    if any(getattr(h, _HANDLER_TAG, False) for h in parent_logger.handlers):
        print(
            f"[cellier] debug logging enabled: categories={set(categories)}",
            file=sys.stderr,
        )
        return

    handler: logging.Handler | None = None
    if use_rich:
        try:
            handler = _make_rich_handler()
        except ImportError:
            handler = None

    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(name)s  %(message)s", datefmt="%H:%M:%S")
        )

    handler.setLevel(level)
    setattr(handler, _HANDLER_TAG, True)
    parent_logger.addHandler(handler)
    parent_logger.setLevel(level)

    print(
        f"[cellier] debug logging enabled: categories={set(categories)}",
        file=sys.stderr,
    )


def disable_debug_logging() -> None:
    """Deactivate debug logging."""
    parent_logger = logging.getLogger("cellier.render")
    for logger in _CATEGORY_MAP.values():
        logger.setLevel(logging.WARNING)
    parent_logger.handlers = [
        h for h in parent_logger.handlers if not getattr(h, _HANDLER_TAG, False)
    ]


_COLOR_MAP = {
    "cellier.render.perf": ("cyan", "PERF"),
    "cellier.render.gpu": ("green", "GPU"),
    "cellier.render.cache": ("yellow", "CACHE"),
    "cellier.render.slicer": ("magenta", "SLICER"),
    "cellier.render.camera": ("blue", "CAMERA"),
    "cellier.render.source_id": ("red", "SOURCE_ID"),
}


def _make_rich_handler() -> logging.Handler:
    from rich.logging import RichHandler

    class CellierRichHandler(RichHandler):
        def emit(self, record: logging.LogRecord) -> None:
            color, label = _COLOR_MAP.get(record.name, ("white", "???"))
            record.msg = f"[{color}]\\[{label}][/{color}] {record.msg}"
            super().emit(record)

    return CellierRichHandler(
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
