"""Generate a synthetic anisotropic OME-Zarr v0.5 volume for demo testing.

Usage::

    uv run make_example_data.py --output /tmp/example.ome.zarr \\
        --shape 64 256 256 --voxel-scales 4.0 1.0 1.0 \\
        --n-levels 4 --n-blobs 12 --dtype uint16
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import zarr


def _gaussian_3d(
    shape: tuple[int, int, int],
    center: np.ndarray,
    sigma: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    coords = np.stack([z, y, x], axis=-1).astype(np.float32)
    dist_sq = np.sum(((coords - center) / sigma) ** 2, axis=-1)
    return amplitude * np.exp(-0.5 * dist_sq)


def generate_volume(
    shape: tuple[int, int, int],
    voxel_scales: list[float],
    n_blobs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vol = np.zeros(shape, dtype=np.float32)
    physical_scales = np.array(voxel_scales, dtype=np.float32)
    # sigma in voxels: target ~15 physical units, scaled per axis so blobs look round
    base_sigma_phys = 15.0
    sigma_vox = base_sigma_phys / physical_scales

    for _ in range(n_blobs):
        center = np.array([rng.uniform(0, s) for s in shape], dtype=np.float32)
        amplitude = rng.uniform(0.3, 1.0)
        vol += _gaussian_3d(shape, center, sigma_vox, amplitude)

    return vol


def block_reduce_mean(arr: np.ndarray) -> np.ndarray:
    """2×2×2 mean block reduce."""
    try:
        from skimage.measure import block_reduce

        return block_reduce(arr, block_size=(2, 2, 2), func=np.mean)
    except ImportError:
        # Fallback: manual strided mean
        z, y, x = arr.shape
        z2, y2, x2 = z // 2, y // 2, x // 2
        return (
            arr[:z2*2:2, :y2*2:2, :x2*2:2]
            + arr[1:z2*2:2, :y2*2:2, :x2*2:2]
            + arr[:z2*2:2, 1:y2*2:2, :x2*2:2]
            + arr[:z2*2:2, :y2*2:2, 1:x2*2:2]
            + arr[1:z2*2:2, 1:y2*2:2, :x2*2:2]
            + arr[1:z2*2:2, :y2*2:2, 1:x2*2:2]
            + arr[:z2*2:2, 1:y2*2:2, 1:x2*2:2]
            + arr[1:z2*2:2, 1:y2*2:2, 1:x2*2:2]
        ) / 8.0


def write_ome_zarr(
    output_path: str,
    levels: list[np.ndarray],
    voxel_scales: list[float],
    dtype_str: str,
) -> None:
    dtype = np.dtype(dtype_str)
    if np.issubdtype(dtype, np.integer):
        iinfo = np.iinfo(dtype)
        dtype_max = float(iinfo.max)
    else:
        dtype_max = 1.0

    store = zarr.open_group(output_path, mode="w")

    datasets = []
    for level_idx, arr in enumerate(levels):
        scale_factor = 2.0 ** level_idx
        level_scales = [s * scale_factor for s in voxel_scales]

        # Normalise float to [0, dtype_max]
        arr_norm = arr / max(arr.max(), 1e-8) * dtype_max
        arr_cast = arr_norm.astype(dtype)

        path = str(level_idx)
        store.create_array(
            path,
            data=arr_cast,
            chunks=(min(64, arr_cast.shape[0]), min(64, arr_cast.shape[1]), min(64, arr_cast.shape[2])),
            overwrite=True,
        )

        datasets.append(
            {
                "path": path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": level_scales}
                ],
            }
        )
        print(f"  level {level_idx}: shape={arr_cast.shape}  scale={level_scales}")

    ome_meta = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets,
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            }
        ],
    }
    store.attrs["ome"] = ome_meta
    print(f"\nWrote OME-Zarr to: {output_path}")
    print(f"  axis names : z y x")
    print(f"  voxel_scales: {voxel_scales}")
    print(f"  n_levels   : {len(levels)}")
    print(f"  dtype      : {dtype_str}")
    print(f"\nTo use in demo scripts:")
    print(f"  --zarr-path {output_path} --voxel-scales {' '.join(str(s) for s in voxel_scales)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic OME-Zarr volume")
    parser.add_argument("--output", default="./example.ome.zarr")
    parser.add_argument("--shape", nargs=3, type=int, default=[64, 256, 256], metavar=("Z", "Y", "X"))
    parser.add_argument("--voxel-scales", nargs=3, type=float, default=[4.0, 1.0, 1.0], metavar=("SZ", "SY", "SX"))
    parser.add_argument("--n-levels", type=int, default=4)
    parser.add_argument("--n-blobs", type=int, default=12)
    parser.add_argument("--dtype", default="uint16", choices=["uint8", "uint16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shape = tuple(args.shape)
    rng = np.random.default_rng(args.seed)

    print(f"Generating volume: shape={shape}  voxel_scales={args.voxel_scales}  n_blobs={args.n_blobs}")
    finest = generate_volume(shape, args.voxel_scales, args.n_blobs, rng)

    levels = [finest]
    current = finest
    for i in range(1, args.n_levels):
        current = block_reduce_mean(current)
        levels.append(current)
        print(f"  downsampled level {i}: {current.shape}")

    write_ome_zarr(args.output, levels, args.voxel_scales, args.dtype)


if __name__ == "__main__":
    main()
