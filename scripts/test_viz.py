import argparse
from pathlib import Path

import hvplot.xarray  # noqa
import panel as pn
import xarray as xr


def main():

    # Parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to Echointegration.zarr",
    )
    args = parser.parse_args()
    zarr_path = Path(args.input_file).resolve()

    # Open dataset
    ds: xr.Dataset = xr.open_dataset(zarr_path, engine="zarr")

    # Create echogeam
    echogram = ds.Sv.hvplot.quadmesh(
        x="ping_time",
        y="depth",
        rasterize=True,
        flip_yaxis=True,
        responsive=True,
    )

    # Overlay manual cleaning mask with rasterize; use transparent-to-red cmap
    # so NaN (clean) pixels are invisible and flagged pixels appear as red.
    mask_data = (~ds.mask_clean).astype(float).where(~ds.mask_clean)
    mask = mask_data.hvplot.quadmesh(
        x="ping_time",
        y="depth",
        rasterize=True,
        flip_yaxis=True,
        responsive=True,
        cmap=["#00000000", "#FF000088"],
        colorbar=False,
        clim=(0, 1),
    )

    plot = echogram * mask

    # Create pannel app
    app = pn.panel(plot)
    app.show()


if __name__ == "__main__":
    main()
