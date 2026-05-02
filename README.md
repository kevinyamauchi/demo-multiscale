# Demo of multiscale chunked rendering with pygfx

https://github.com/user-attachments/assets/3dffa013-4669-45c7-b358-e8e296ac4662

## Run demo

To run the demo, first make some example data. This will make some gaussian blobs in a 125x500x500 volume, with voxel scales of 4x1x1 (ZxYxX). The data is written to `example.ome.zarr` in the current directory.

```sh
uv run make_example_data.py --output example.ome.zarr --voxel-scales 4.0 1.0 1.0 --shape 125 500 500
```

Then run the 3d viewer demo:

```sh
uv run demo_3d.py --zarr-path ./example.ome.zarr
```

## View larger data

Here's an example of a larger dataset from the IDR (idr0066). It is a 3D volume of a mouse brain imaged wth a lightsheet microscope.

You can download the data from the [IDR ome-ngff samples](https://idr.github.io/ome-ngff-samples/) page:

```sh
uv run download_ome_zarr.py "https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpA_VIP_ASLM_on.zarr"
```

Then view it with:

```sh
uv run demo_2d_viewer.py --zarr-path ExpA_VIP_ASLM_on.zarr
```