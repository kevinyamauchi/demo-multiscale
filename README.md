# Demo of multiscale chunked rendering with pygfx

Note: this is a demo repo and not intended to be used.

https://github.com/user-attachments/assets/3dffa013-4669-45c7-b358-e8e296ac4662

(1937, 2048, 2048) image of a mouse brain (idr0066). See the "View larger data section" below for how to download and view this data.

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

Here's an example of a larger dataset from the IDR (idr0066). It is a 3D volume of a mouse brain imaged wth a lightsheet microscope. You might be able to view other ome-zarr v0.5 files, but I haven't really tested it.

You can download the data from the [IDR ome-ngff samples](https://idr.github.io/ome-ngff-samples/) page:

```sh
uv run download_ome_zarr.py "https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0066/ExpA_VIP_ASLM_on.zarr"
```

Then view it with:

```sh
uv run demo_2d_3d.py --zarr-path ExpA_VIP_ASLM_on.zarr
```

The movie above was recorded with iso_thresold=300 and camera_settle=150 (ms). The camera settle time sets how long the camera state has to be constant before reslicing the scene. This is important to keep the viewer responsive when moving around the scene quickly.

## ndv demo
To understand if it would be possible to implement this in ndv, I made a version of the demo that tries to use the ndv architecture/API as much as possible (`demo_2d_3d_ndv.py` and `demo_multiscale_ndv`). You can run it with the same commands as above. For example:

```sh
uv run demo_2d_3d_ndv.py --zarr-path example.ome.zarr
```