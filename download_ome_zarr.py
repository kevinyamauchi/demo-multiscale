"""Download an OME-Zarr store from S3 or HTTPS to a local directory.

Usage:
    uv run download_ome_zarr.py <url> [options]

Examples:
    uv run download_ome_zarr.py https://host/path/data.zarr
    uv run download_ome_zarr.py s3://bucket/path/data.zarr --output ./local.zarr
    uv run download_ome_zarr.py s3://bucket/path/data.zarr --fast-copy
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import math
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import tensorstore as ts
import zarr
import yaozarrs
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Chunk key computation
# ---------------------------------------------------------------------------

def _meta_key(rel_path: str, node: zarr.Array | zarr.Group) -> str:
    """Return the metadata store key for a node."""
    fmt = node.metadata.zarr_format
    if fmt == 3:
        fname = "zarr.json"
    elif isinstance(node, zarr.Array):
        fname = ".zarray"
    else:
        fname = ".zgroup"
    return f"{rel_path}/{fname}" if rel_path else fname


def _child_paths_from_attrs(attrs: dict) -> list[tuple[str, str]]:
    """Inspect OME attrs and return (rel_path, hint) pairs for child nodes.

    *hint* is one of ``"array"``, ``"group"``, or ``"maybe"`` (try it, skip on
    404).  We read from both the ``ome`` sub-key (v0.5) and top-level attrs
    (older bioformats2raw / v0.4) for maximum compatibility.

    Handled OME types
    -----------------
    Image / LabelImage  -- multiscales.datasets[*].path  → arrays
                        -- labels/<name>                  → groups
    Bf2Raw              -- series[*]                      → groups
    Plate               -- plate.wells[*].path            → groups
    Well                -- well.images[*].path            → groups
    LabelsGroup         -- labels[*]                      → groups
    """
    # The OME metadata may live under the "ome" key (v0.5) or directly at the
    # top level (some v0.4 / bioformats2raw stores put "series" at top level).
    ome = attrs.get("ome") or {}
    merged = {**attrs, **ome}   # ome wins on collision

    results: list[tuple[str, str]] = []

    # ── Image / LabelImage ──────────────────────────────────────────────────
    for ms in merged.get("multiscales", []):
        for ds in ms.get("datasets", []):
            path = ds.get("path", "")
            if path:
                results.append((path, "array"))
    # Labels sub-group lives at a sibling "labels/" directory.
    if merged.get("multiscales"):
        results.append(("labels", "maybe"))

    # ── LabelsGroup ─────────────────────────────────────────────────────────
    for label_name in merged.get("labels", []):
        results.append((str(label_name), "group"))

    # ── Bf2Raw / Series ─────────────────────────────────────────────────────
    for series_path in merged.get("series", []):
        results.append((str(series_path).strip("/"), "group"))

    # ── Plate ────────────────────────────────────────────────────────────────
    plate = merged.get("plate") or {}
    for well in plate.get("wells", []):
        path = well.get("path", "")
        if path:
            results.append((path, "group"))

    # ── Well ─────────────────────────────────────────────────────────────────
    well = merged.get("well") or {}
    for img in well.get("images", []):
        path = img.get("path", "")
        if path:
            results.append((path, "group"))

    return results


def _array_chunk_keys(rel_path: str, array: zarr.Array) -> list[str]:
    """Compute every storage key for an array's chunks.

    For zarr v3 with default encoding: keys look like ``c/0/1/2``.
    For zarr v3 with v2 encoding or zarr v2 arrays: keys look like ``0.1.2``.
    Missing chunks (fill-value only) simply won't exist on the server;
    the transfer loop handles that gracefully.
    """
    shape = array.shape
    chunk_shape = array.chunks  # outer chunk shape (= shard shape for sharded arrays)

    fmt = array.metadata.zarr_format

    # Determine separator and whether to prepend the "c" prefix (zarr v3 default enc).
    if fmt == 3:
        enc = array.metadata.chunk_key_encoding
        sep: str = enc.separator        # "/" or "."
        default_enc: bool = enc.name == "default"
    else:
        # zarr v2: dimension_separator defaults to ".", no "c" prefix.
        sep = getattr(array.metadata, "dimension_separator", ".") or "."
        default_enc = False

    # Scalar array (shape == ()): exactly one chunk.
    if not shape:
        if fmt == 3 and default_enc:
            chunk_key = "c"
        elif fmt == 3:
            chunk_key = ""
        else:
            chunk_key = "0"
        return [f"{rel_path}/{chunk_key}" if rel_path else chunk_key]

    n_chunks = [max(1, math.ceil(s / c)) for s, c in zip(shape, chunk_shape)]

    keys: list[str] = []
    for indices in itertools.product(*[range(n) for n in n_chunks]):
        index_part = sep.join(str(i) for i in indices)
        if fmt == 3 and default_enc:
            chunk_key = f"c{sep}{index_part}"
        else:
            chunk_key = index_part
        full_key = f"{rel_path}/{chunk_key}" if rel_path else chunk_key
        keys.append(full_key)

    return keys


# ---------------------------------------------------------------------------
# Key enumeration
# ---------------------------------------------------------------------------

def _enumerate_keys_via_zarr(url: str, storage_options: dict) -> list[str]:
    """Walk the OME-Zarr hierarchy via BFS, returning every store key.

    Never calls ``group.members()``; child discovery is driven entirely by
    OME metadata read from individual ``zarr.json`` / ``.zgroup`` files.
    This makes it work over plain HTTPS where directory listing is impossible.

    Runs synchronously (intended for asyncio.to_thread).
    """
    keys: list[str] = []

    # Queue entries: (absolute_node_url, key_prefix_within_store)
    # key_prefix is the path relative to the zarr root (empty string = root).
    queue: list[tuple[str, str]] = [(url.rstrip("/"), "")]
    visited: set[str] = set()

    while queue:
        node_url, key_prefix = queue.pop(0)

        if key_prefix in visited:
            continue
        visited.add(key_prefix)

        # ── Open the node ────────────────────────────────────────────────────
        # zarr.open() auto-detects Group vs Array from zarr.json node_type.
        # Each call is a single GET request — no listing.
        # Only forward storage_options when non-empty: zarr raises if the kwarg
        # is present but unused (e.g. for plain local paths or some fsspec URIs).
        open_kwargs: dict = {"storage_options": storage_options} if storage_options else {}
        try:
            node = zarr.open(node_url, mode="r", **open_kwargs)
        except Exception:
            # "maybe" nodes (e.g. a labels group that doesn't exist) land here.
            continue

        fmt = node.metadata.zarr_format

        # ── Metadata key ────────────────────────────────────────────────────
        keys.append(_meta_key(key_prefix, node))

        # ── Array ────────────────────────────────────────────────────────────
        if isinstance(node, zarr.Array):
            keys.extend(_array_chunk_keys(key_prefix, node))
            if fmt == 2:
                # v2 arrays may have a companion .zattrs
                keys.append(f"{key_prefix}/.zattrs" if key_prefix else ".zattrs")
            continue

        # ── Group ────────────────────────────────────────────────────────────
        if fmt == 2:
            attr_key = f"{key_prefix}/.zattrs" if key_prefix else ".zattrs"
            meta_key = f"{key_prefix}/.zmetadata" if key_prefix else ".zmetadata"
            keys.append(attr_key)
            keys.append(meta_key)   # may be absent; transfer skips missing keys

        # Discover children from OME metadata
        attrs = dict(node.attrs)
        for child_rel, hint in _child_paths_from_attrs(attrs):
            child_key = f"{key_prefix}/{child_rel}" if key_prefix else child_rel
            child_url = f"{node_url}/{child_rel}"
            if hint == "array":
                # Open directly as array to avoid an extra group-open attempt.
                if child_key in visited:
                    continue
                visited.add(child_key)
                try:
                    arr = zarr.open_array(child_url, mode="r",
                                          **open_kwargs)
                    keys.append(_meta_key(child_key, arr))
                    keys.extend(_array_chunk_keys(child_key, arr))
                    if arr.metadata.zarr_format == 2:
                        keys.append(
                            f"{child_key}/.zattrs" if child_key else ".zattrs"
                        )
                except Exception:
                    pass
            else:
                # "group" or "maybe" — enqueue for BFS expansion.
                queue.append((child_url, child_key))

    return keys


def _enumerate_keys_via_s3fs(bucket: str, path: str, anon: bool) -> list[str]:
    """List every object under s3://<bucket>/<path> using s3fs.

    Runs synchronously (intended for asyncio.to_thread).
    """
    import s3fs  # imported here; optional dep only needed for S3

    fs = s3fs.S3FileSystem(anon=anon)
    root = f"{bucket}/{path.strip('/')}"
    all_paths: list[str] = fs.find(root, detail=False)
    prefix = root + "/"
    # Strip the root prefix; drop directory-like entries (shouldn't occur
    # in S3 but be defensive).
    return [p.removeprefix(prefix) for p in all_paths if not p.endswith("/")]


async def enumerate_keys(url: str, scheme: str, anon: bool) -> list[str]:
    """Dispatch key enumeration based on source scheme."""
    if scheme == "s3":
        print("Enumerating keys via S3 listing...", flush=True)
        parsed = urlparse(url)
        bucket = parsed.netloc
        path = parsed.path.lstrip("/")
        keys = await asyncio.to_thread(_enumerate_keys_via_s3fs, bucket, path, anon)
    else:
        print("Enumerating keys via zarr hierarchy walk...", flush=True)
        storage_options: dict = {}
        keys = await asyncio.to_thread(_enumerate_keys_via_zarr, url, storage_options)

    print(f"  Found {len(keys):,} keys.", flush=True)
    return keys


# ---------------------------------------------------------------------------
# tensorstore KvStore helpers
# ---------------------------------------------------------------------------

async def _open_src_kvstore(url: str, scheme: str) -> ts.KvStore:
    """Open a read-only tensorstore KvStore for S3 sources."""
    assert scheme == "s3", "_open_src_kvstore is S3-only; use _read_http for HTTPS"
    parsed = urlparse(url)
    bucket = parsed.netloc
    # Ensure path ends with "/" so it's treated as a directory prefix.
    path = parsed.path.lstrip("/").rstrip("/") + "/"
    spec: dict = {"driver": "s3", "bucket": bucket, "path": path}
    return await ts.KvStore.open(spec)


async def _transfer_http(
    base_url: str,
    dst_kv: ts.KvStore,
    keys: list[str],
    concurrency: int,
) -> None:
    """Transfer keys from an HTTPS/HTTP source using aiohttp.

    tensorstore's HTTP KvStore driver fails silently on some S3-compatible
    HTTPS endpoints. aiohttp (already used by zarr/fsspec during enumeration)
    works reliably for these servers.
    """
    sem = asyncio.Semaphore(concurrency)
    failures: list[tuple[str, Exception]] = []
    total_bytes = 0
    base = base_url.rstrip("/")

    with tqdm(total=len(keys), desc="Downloading", unit="key") as pbar:
        async with aiohttp.ClientSession() as session:

            async def _fetch_one(key: str) -> None:
                nonlocal total_bytes
                async with sem:
                    try:
                        url = f"{base}/{key}"
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                value = await resp.read()
                                await dst_kv.write(key, value)
                                total_bytes += len(value)
                                pbar.set_postfix(
                                    {"MB": f"{total_bytes / 1_000_000:.1f}"},
                                    refresh=False,
                                )
                            elif resp.status == 404:
                                pass  # fill-value chunk; skip silently
                            else:
                                raise RuntimeError(
                                    f"HTTP {resp.status} for {url}"
                                )
                    except Exception as exc:
                        failures.append((key, exc))
                    finally:
                        pbar.update(1)

            await asyncio.gather(*(_fetch_one(k) for k in keys))

    if failures:
        n = len(failures)
        print(f"\n  ⚠  {n} key(s) failed to transfer:", file=sys.stderr)
        for key, exc in failures[:10]:
            print(f"     {key!r}: {exc}", file=sys.stderr)
        if n > 10:
            print(f"     … and {n - 10} more.", file=sys.stderr)
        raise RuntimeError(f"{n} key(s) failed during transfer.")


async def _open_dst_kvstore(output_dir: Path) -> ts.KvStore:
    """Open a writable tensorstore KvStore for the local destination."""
    path = str(output_dir.resolve()).rstrip("/") + "/"
    return await ts.KvStore.open({"driver": "file", "path": path})


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

async def transfer(
    src_kv: ts.KvStore,
    dst_kv: ts.KvStore,
    keys: list[str],
    concurrency: int,
) -> None:
    """Copy every key from *src_kv* to *dst_kv* concurrently.

    * Missing keys (fill-value chunks absent from the store) are silently
      skipped — they need not be written locally either.
    * Individual key failures are collected and reported after all tasks
      finish, then re-raised as a single RuntimeError.
    * A tqdm progress bar shows completed keys and running byte total.
    """
    sem = asyncio.Semaphore(concurrency)
    failures: list[tuple[str, Exception]] = []
    total_bytes = 0

    with tqdm(total=len(keys), desc="Downloading", unit="key") as pbar:

        async def _copy_one(key: str) -> None:
            nonlocal total_bytes
            async with sem:
                try:
                    result = await src_kv.read(key)
                    # Missing keys return b"" (falsy); present keys return bytes.
                    # Cast to bytes explicitly: HTTP KvStore may return a
                    # tensorstore Buffer rather than plain bytes, which some
                    # write() overloads reject silently.
                    value = bytes(result.value) if result.value else b""
                    if value:
                        nb = len(value)
                        await dst_kv.write(key, value)
                        total_bytes += nb
                        pbar.set_postfix(
                            {"MB": f"{total_bytes / 1_000_000:.1f}"}, refresh=False
                        )
                except Exception as exc:  # noqa: BLE001
                    failures.append((key, exc))
                finally:
                    pbar.update(1)

        await asyncio.gather(*(_copy_one(k) for k in keys))

    if failures:
        n = len(failures)
        print(f"\n  ⚠  {n} key(s) failed to transfer:", file=sys.stderr)
        for key, exc in failures[:10]:
            print(f"     {key!r}: {exc}", file=sys.stderr)
        if n > 10:
            print(f"     … and {n - 10} more.", file=sys.stderr)
        raise RuntimeError(f"{n} key(s) failed during transfer.")


async def fast_copy(src_kv: ts.KvStore, dst_kv: ts.KvStore) -> None:
    """Bulk-copy all keys using tensorstore's experimental_copy_range_to.

    Much faster for S3 sources — no Python-level iteration — but provides
    no progress feedback.
    """
    print("Using fast copy (experimental_copy_range_to)...", flush=True)
    await src_kv.experimental_copy_range_to(dst_kv)
    print("Fast copy complete.", flush=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(output_dir: Path) -> None:
    """Run yaozarrs structural + metadata validation on the downloaded store."""
    print("Validating OME-Zarr store...", flush=True)
    try:
        from yaozarrs._storage import StorageValidationError  # noqa: PLC0415
        yaozarrs.validate_zarr_store(str(output_dir))
        print("✓ Valid OME-Zarr store.", flush=True)
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ == "StorageValidationError":
            print(f"❌ Validation failed:\n{exc}", file=sys.stderr)
            sys.exit(1)
        else:
            # Unexpected exception from yaozarrs — warn but don't abort.
            print(f"⚠  Validation raised an unexpected error: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "url",
        help="Source URL: s3://<bucket>/path/data.zarr  or  https://host/path/data.zarr",
    )
    p.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Local output directory (default: basename of URL)",
    )
    p.add_argument(
        "--concurrency", "-c",
        type=int,
        default=32,
        metavar="N",
        help="Max concurrent chunk transfers (default: 32)",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip OME-Zarr validation after download",
    )
    p.add_argument(
        "--fast-copy",
        action="store_true",
        help=(
            "S3 only: use tensorstore experimental_copy_range_to for bulk "
            "transfer (no progress bar)"
        ),
    )
    p.add_argument(
        "--anon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Anonymous S3 access — use --no-anon for credentialed access (default: --anon)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and replace an existing output directory",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Enumerate keys and check the first few reads, but do not write "
            "anything locally. Useful for diagnosing URL or auth issues."
        ),
    )
    return p


def _default_output_name(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    return name or "downloaded.zarr"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/")
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme not in ("s3", "https", "http"):
        print(
            f"Error: unsupported URL scheme {scheme!r}. "
            "Use s3://, https://, or http://.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path(_default_output_name(url))

    # Guard against clobbering existing data.
    if output_dir.exists():
        if args.overwrite:
            print(f"Removing existing {output_dir} ...", flush=True)
            shutil.rmtree(output_dir)
        else:
            print(
                f"Error: {output_dir} already exists. "
                "Pass --overwrite to replace it.",
                file=sys.stderr,
            )
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. Enumerate ────────────────────────────────────────────────────
        keys = await enumerate_keys(url, scheme, anon=args.anon)

        # ── 2. Open KvStores ────────────────────────────────────────────────
        # src_kv is only needed for S3; HTTPS uses aiohttp directly.
        src_kv = await _open_src_kvstore(url, scheme) if scheme == "s3" else None
        dst_kv = await _open_dst_kvstore(output_dir)

        # ── 3. Dry-run probe ────────────────────────────────────────────────
        if args.dry_run:
            print("  Dry run — probing first 3 keys...", flush=True)
            base = url.rstrip("/")
            async with aiohttp.ClientSession() as session:
                for probe_key in keys[:3]:
                    if scheme == "s3":
                        result = await src_kv.read(probe_key)
                        value = bytes(result.value) if result.value else b""
                    else:
                        async with session.get(f"{base}/{probe_key}") as resp:
                            value = await resp.read() if resp.status == 200 else b""
                    status = f"{len(value):,} bytes" if value else "MISSING (0 bytes)"
                    print(f"    {probe_key!r}: {status}", flush=True)
            print("  Dry run complete. No files written.", flush=True)
            return

        # ── 3. Transfer ─────────────────────────────────────────────────────
        if scheme != "s3":
            # Use aiohttp for HTTPS/HTTP: tensorstore's HTTP KvStore fails
            # silently on many S3-compatible HTTPS endpoints.
            if args.fast_copy:
                print(
                    "Warning: --fast-copy requires S3; "
                    "using aiohttp transfer instead.",
                    file=sys.stderr,
                )
            await _transfer_http(url, dst_kv, keys, args.concurrency)
        elif args.fast_copy:
            await fast_copy(src_kv, dst_kv)
        else:
            await transfer(src_kv, dst_kv, keys, args.concurrency)

        print(f"\n✓  Downloaded to {output_dir}", flush=True)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterrupted — cleaning up partial output ...", file=sys.stderr)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    except Exception as exc:
        print(f"\nFailed: {exc}", file=sys.stderr)
        print("Cleaning up partial output ...", file=sys.stderr)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    # ── 4. Validate ─────────────────────────────────────────────────────────
    if not args.no_validate:
        validate(output_dir)


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()