"""Component to handle threaded batch loading of chunk data."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QObject, Signal

from demo_multiscale.data_store import ChunkRequest
from demo_multiscale.logging import _PERF_LOGGER, _SLICER_LOGGER

if TYPE_CHECKING:
    from uuid import UUID

# Callable that takes a ChunkRequest and returns an ndarray.
FetchFn = Callable[[ChunkRequest], np.ndarray]

# Callable fired once per batch with (request, data) pairs.
BatchCallback = Callable[[list[tuple[ChunkRequest, np.ndarray]]], None]


class _BatchRelay(QObject):
    """Delivers batch results on the Qt main thread via a queued signal.

    Emitting from a worker thread with AutoConnection (the default) causes
    the connected slot to run on the thread that owns this QObject — i.e.
    the main thread where it was constructed.
    """

    _deliver = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._deliver.connect(self._on_deliver)

    def post(
        self,
        callback: BatchCallback,
        batch: list[tuple[ChunkRequest, np.ndarray]],
    ) -> None:
        self._deliver.emit((callback, batch))

    def _on_deliver(self, payload: object) -> None:
        callback, batch = payload  # type: ignore[misc]
        callback(batch)


class AsyncSlicer:
    """Cancellable threaded batch-fetch service.

    Owns one daemon thread per active ``slice_request_id``.  Within each
    slice, chunk reads are issued concurrently via a shared
    ``ThreadPoolExecutor``.  Completed batches are delivered to the caller's
    callback on the Qt main thread via a ``QObject`` signal.

    Parameters
    ----------
    batch_size :
        Number of chunks to read concurrently per batch.  Higher values
        reduce total load time; lower values give more frequent visual
        feedback during loading.  Default 8 is a good balance for local
        NVMe / SSD storage.
    max_workers :
        Thread-pool size for chunk reads.  Defaults to ``batch_size``.
    """

    def __init__(
        self,
        batch_size: int = 8,
        max_workers: int | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers or batch_size)
        # Per-slice cancellation flags.
        self._cancel_events: dict[UUID, threading.Event] = {}
        # Relay for posting batch results onto the Qt main thread.
        self._relay = _BatchRelay()

    # ── Public API ──────────────────────────────────────────────────────

    def submit(
        self,
        requests: list[ChunkRequest],
        fetch_fn: FetchFn,
        callback: BatchCallback,
        consumer_id: str | None = None,
    ) -> UUID | None:
        """Submit a batch of chunk requests for threaded loading.

        All requests must share the same ``slice_request_id``.  If a thread
        for that ID is already running it is signalled to stop between
        batches before the new thread starts.

        Parameters
        ----------
        requests :
            Ordered list of ``ChunkRequest`` objects, all sharing one
            ``slice_request_id``.  Empty list is a no-op (returns ``None``).
        fetch_fn :
            Synchronous callable ``(ChunkRequest) -> np.ndarray``.
        callback :
            Called once per batch with a list of ``(request, data)`` pairs.
            Always runs on the Qt main thread.
        consumer_id :
            Optional label for logging.
        """
        if not requests:
            return None

        slice_id = requests[0].slice_request_id

        # Signal any running thread for this slice to stop.
        old_event = self._cancel_events.pop(slice_id, None)
        if old_event is not None:
            old_event.set()

        cancel_event = threading.Event()
        self._cancel_events[slice_id] = cancel_event

        thread = threading.Thread(
            target=self._run,
            args=(requests, fetch_fn, callback, slice_id, cancel_event),
            daemon=True,
        )
        thread.start()

        _SLICER_LOGGER.info(
            "task_submitted  requests=%d  slice_id=%s  consumer=%r",
            len(requests),
            slice_id,
            consumer_id,
        )

        return slice_id

    def cancel(self, slice_request_id: UUID | None) -> bool:
        """Cancel in-flight work for ``slice_request_id``.

        No-op if the ID is unknown or the thread has already finished.

        Returns
        -------
        cancelled :
            ``True`` if a running thread was found and signalled;
            ``False`` otherwise.
        """
        if slice_request_id is None:
            return False
        event = self._cancel_events.pop(slice_request_id, None)
        if event is not None:
            event.set()
            return True
        return False

    # ── Internal worker ─────────────────────────────────────────────────

    def _run(
        self,
        requests: list[ChunkRequest],
        fetch_fn: FetchFn,
        callback: BatchCallback,
        slice_id: UUID,
        cancel_event: threading.Event,
    ) -> None:
        """Drive the batched read loop in a daemon thread.

        Splits ``requests`` into batches of ``self._batch_size``, uses the
        shared ``ThreadPoolExecutor`` to read each batch concurrently, then
        posts the results to the Qt main thread via ``_BatchRelay``.

        Cancellation is checked before each batch starts and again after
        ``executor.map`` returns.  Individual in-flight reads cannot be
        interrupted, but the loop exits at the next check point.
        """
        batches = [
            requests[i : i + self._batch_size]
            for i in range(0, len(requests), self._batch_size)
        ]
        n_batches = len(batches)

        _SLICER_LOGGER.info(
            "task_start  slice_id=%s  total_requests=%d  n_batches=%d",
            slice_id,
            len(requests),
            n_batches,
        )

        batch_idx = 0
        batch_times_ms: list[float] = []
        t_fetch_start = time.perf_counter()
        cancelled = False

        try:
            for batch_idx, batch in enumerate(batches):
                if cancel_event.is_set():
                    cancelled = True
                    break

                t_batch = time.perf_counter()
                results: list[np.ndarray] = list(
                    self._executor.map(fetch_fn, batch)
                )
                batch_ms = (time.perf_counter() - t_batch) * 1000
                batch_times_ms.append(batch_ms)

                _PERF_LOGGER.debug(
                    "fetch_batch  %d/%d  bricks=%d  elapsed=%.1fms",
                    batch_idx + 1,
                    n_batches,
                    len(batch),
                    batch_ms,
                )

                if _SLICER_LOGGER.isEnabledFor(logging.INFO):
                    scale_counts: dict[int, int] = {}
                    for req in batch:
                        scale_counts[req.scale_index] = (
                            scale_counts.get(req.scale_index, 0) + 1
                        )
                    _SLICER_LOGGER.info(
                        "batch_done  %d/%d  bricks=%d  scales=%s",
                        batch_idx + 1,
                        n_batches,
                        len(batch),
                        scale_counts,
                    )

                if _SLICER_LOGGER.isEnabledFor(logging.DEBUG):
                    for req, data in zip(batch, results):
                        _SLICER_LOGGER.debug(
                            "  brick_received  id=%s  scale=%d  shape=%s",
                            req.chunk_request_id,
                            req.scale_index,
                            data.shape,
                        )

                if cancel_event.is_set():
                    cancelled = True
                    break

                self._relay.post(callback, list(zip(batch, results)))

        finally:
            self._cancel_events.pop(slice_id, None)

            if batch_times_ms:
                total_ms = (time.perf_counter() - t_fetch_start) * 1000
                _log_fetch_summary(
                    slice_id,
                    len(requests),
                    batch_times_ms,
                    total_ms,
                    cancelled=cancelled,
                )

            if cancelled:
                _SLICER_LOGGER.info(
                    "task_cancelled  slice_id=%s  batches_done=%d/%d",
                    slice_id,
                    batch_idx,
                    n_batches,
                )
            else:
                _SLICER_LOGGER.info(
                    "task_complete  slice_id=%s  total_requests=%d",
                    slice_id,
                    len(requests),
                )


def _log_fetch_summary(
    slice_id: UUID,
    n_requests: int,
    batch_times_ms: list[float],
    total_ms: float,
    *,
    cancelled: bool,
) -> None:
    """Emit an INFO-level fetch timing summary on ``_PERF_LOGGER``."""
    n = len(batch_times_ms)
    mean = sum(batch_times_ms) / n
    min_t = min(batch_times_ms)
    max_t = max(batch_times_ms)
    if n > 1:
        variance = sum((t - mean) ** 2 for t in batch_times_ms) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    status = "cancelled" if cancelled else "complete"
    _PERF_LOGGER.info(
        "fetch_summary  status=%s  slice_id=%s  bricks=%d  batches=%d  "
        "total=%.0fms  per_batch=%.1f±%.1fms  min=%.1fms  max=%.1fms",
        status,
        slice_id,
        n_requests,
        n,
        total_ms,
        mean,
        std,
        min_t,
        max_t,
    )
