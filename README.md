# Demo of multiscale chunked rendering with pygfx

## Run

To run the demo, first make some example data. This will make some gaussian blobs in a 125x500x500 volume, with voxel scales of 4x1x1 (ZxYxX). The data is written to `example.ome.zarr` in the current directory.

```sh
uv run make_example_data.py --output example.ome.zarr --voxel-scales 4.0 1.0 1.0 --shape 125 500 500
```

Then run the 3d viewer demo:

```sh
uv run demo_3d.py --zarr-path ./example.ome.zarr
```