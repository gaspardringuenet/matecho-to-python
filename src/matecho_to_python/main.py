import argparse
import logging
import os
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr

# Configure logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s:%(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# Silence ZarrUserWarning on consolidated metadata
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    """Main function: parse CLI args and run matecho_to_zarr"""

    args = _parse_cli()

    ds_path: Path = mat_to_zarr(
        input=args.input,
        float_dtype=np.float32 if args.light else np.float64,
        to_nan_Sv=args.to_nan_Sv,
        zarr_format=args.zarr_format,
    )

    # Open complete Dataset
    logger.info("Opening concatenated Dataset")
    ds = xr.open_dataset(ds_path, engine="zarr")

    # Print Dataset
    logger.info("Printing Dataset")
    print(ds)

    # Optional, plot random echogram
    if args.plot:
        logger.info("Plotting confirmation echogram")
        plot_small_echogram(ds)


def mat_to_zarr(
    input: os.PathLike | list[os.PathLike],
    float_dtype: npt.DTypeLike = np.float64,
    to_nan_Sv: float | None = None,
    zarr_format: int = 2,
) -> Path:
    """Convert Matecho Echointegration.mat to xarray Zarr format compatible with Echostack

    Parameters
    ----------
    input : os.PathLike | list[os.PathLike]
        One or more Matecho Echointegration files. If several files are passed, they must be compatible (same depth and channel coordinates, compatible time coordinate).

    float_dtype: npt.DTypeLike, default=np.float64
        Datatype used to store floats in the Dataset.

    to_nan_Sv: float, optional
        A value of Sv which should be convert to NaN (e.g. -150 dB for manual data removal).

    zarr_format: int, default=2
        The desired zarr format to target (2 or 3) as of zarr v3.

    Returns
    -------
    Path
        Path to the saved Zarr store. Use xr.open_dataset to inspect.
    """

    if isinstance(input, list):  # Normalize to list
        input_lst: list[os.PathLike] = input
    else:
        input_lst: list[os.PathLike] = [input]

    # Create output_path
    input_paths: list[Path] = [Path(e).expanduser().resolve() for e in input_lst]
    output_path: Path = input_paths[0].parent / (input_paths[0].stem + ".zarr")

    # Process files iteratively
    logger.info(f"Processing {len(input_paths)} files → {output_path}")
    for i, filepath in enumerate(input_paths):
        ds = _process_one_file(filepath, float_dtype)

        logger.info("Chunking")
        ds = ds.chunk({"channel": 1, "depth": 100, "ping_time": 5000})

        if to_nan_Sv is not None:
            ds["Sv"] = ds["Sv"].where(ds["Sv"] != to_nan_Sv)

        if i == 0:
            logger.info("Saving as Zarr store")
            ds.to_zarr(output_path, mode="w", zarr_format=2)  # First file: create
        else:
            logger.info("Appending Zarr store on ping_time dim (aligning chunks)")
            ds.to_zarr(
                output_path,
                mode="a",
                append_dim="ping_time",
                align_chunks=True,
                zarr_format=zarr_format,
            )

    logger.info("Done with processing!")

    return output_path


def _parse_cli() -> argparse.Namespace:
    """Parse CLI arguments"""

    parser = argparse.ArgumentParser(
        description="Convert Matecho Echointegration.mat to a Zarr store compatible with Echostack"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Path(s) to the echointegration .mat file(s)",
    )
    parser.add_argument(
        "--light",
        action=argparse.BooleanOptionalAction,
        help="Make the Dataset lighter by using float32 instead of float64 for non-time numerical variables",
    )
    parser.add_argument(
        "--to-nan-Sv",
        type=float,
        required=False,
        help="Sv value to convert to NAN (typically -150 dB for manual data removal in Matecho)",
    )
    parser.add_argument(
        "--zarr-format",
        type=int,
        default=2,
        help="The desired zarr format to target (2 or 3) as of zarr v3. By default 2.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        help="Plot a small confirmation echogram after conversion.",
    )

    return parser.parse_args()


def _process_one_file(file_path: Path, float_dtype: npt.DTypeLike) -> xr.Dataset:
    """Convert a single file"""

    with h5py.File(file_path, "r") as f:
        # Build Dataset
        ds = _convert_matecho_EI(f, float_dtype)

        # Fetch attributes
        attrs_dict = _make_attrs_dict(f)

    # Add attributes
    logger.info("Appending attributes to variables")
    for var, a in attrs_dict.items():
        if var in ds.variables:
            ds[var].attrs = a

    return ds


def _convert_matecho_EI(f: h5py.File, float_dtype: npt.DTypeLike) -> xr.Dataset:
    """Convert a Matecho echointegration (Echointegration.mat) file into
    an xarray Dataset with Echopype-compatible dimensions."""

    logger.info("Starting conversion of .mat data to xr.Dataset")

    ds = xr.Dataset()

    #### Add coordinates ####
    # Channel
    vals_channel = np.asarray(f["FrequencySort"]).squeeze().astype(float_dtype)
    ds.coords["channel"] = ("channel", vals_channel)

    # Depth
    vals_depth = np.asarray(f["depth_surface"]).squeeze().astype(float_dtype)
    ds.coords["depth"] = ("depth", vals_depth)

    # Time
    vals_ping_time = (
        np.asarray(f["Time"])
        .squeeze()
        .astype(dtype="datetime64[s]")
        .astype(dtype="datetime64[ns]")
    )
    ds.coords["ping_time"] = ("ping_time", vals_ping_time)

    # Latitude
    vals_lat = np.asarray(f["Latitude"]).squeeze().astype(float_dtype)
    ds.coords["latitude"] = ("ping_time", vals_lat)

    # Longitude
    vals_lon = np.asarray(f["Longitude"]).squeeze().astype(float_dtype)
    ds.coords["longitude"] = ("ping_time", vals_lon)

    #### Add variables ####
    # Sv variable
    vals_Sv: np.ndarray = np.asarray(f["Sv_surface"]).astype(float_dtype)
    ds["Sv"] = (["channel", "ping_time", "depth"], vals_Sv)

    # sA (NASC)
    vals_sA: np.ndarray = np.asarray(f["Sa_surface"]).astype(float_dtype)
    ds["sA"] = (["channel", "ping_time", "depth"], vals_sA)

    # sA_sum (NASC vertical sum)
    vals_sA_sum: np.ndarray = np.asarray(f["SaSum"]).astype(float_dtype)
    ds["sA_sum"] = (["ping_time", "channel"], vals_sA_sum)

    # Manual cleaning mask variable
    vals_mask: np.ndarray = np.asarray(f["Mask_clean"]).astype("?")  # boolean
    ds["mask_clean"] = (["channel", "ping_time", "depth"], vals_mask)

    # Frequency nominal (for EP compat)
    ds["frequency_nominal"] = (["channel"], vals_channel)

    # Bottom depth
    vals_bottom: np.ndarray = np.asarray(f["Bottom"]).astype(float_dtype)
    ds["bottom"] = (["ping_time", "channel"], vals_bottom)
    ds["bottom"] = ds["bottom"].transpose("channel", "ping_time")

    # Sun position
    vals_sun: np.ndarray = np.asarray(f["SolarAltitude"]).squeeze().astype(float_dtype)
    ds["sun_alt"] = (["ping_time"], vals_sun)

    # Transpose
    ds = ds.transpose("channel", "depth", "ping_time")

    logger.info("Done!")

    return ds


def _make_attrs_dict(f: h5py.File) -> dict:
    """Build a dictionary containing formatted variables attributes"""

    logger.info("Formatting attributes dict using .mat co-variables")

    esu_types_conv: dict[int, str] = {
        0: "nm",
        1: "s",
        2: "pings",
    }

    params: h5py.Group = f["EIParameters"]  # type: ignore

    # Fetch EI parameters
    layer_width = np.asarray(params["LayerWidth"]).squeeze()
    esu_size = np.asarray(params["ESUSize"]).squeeze().item()
    esu_type = np.asarray(params["ESUType"]).squeeze().item()

    low_thresh = np.asarray(params["LowThreshold"]).squeeze()
    high_thresh = np.asarray(params["HighThreshold"]).squeeze()

    attrs_dict = {
        "channel": {
            "units": "Hz",
        },
        "depth": {
            "long_name": "Range distance",
            "unit": "m",
        },
        "ping_time": {
            "axis": "T",
            "long_name": "Ping time",
            "standard_name": "time",
        },
        "Sv": {
            "long_name": "Mean volume backscattering strength (MVBS, mean Sv)",
            "ping_time_interval": str(esu_size) + esu_types_conv[esu_type],
            "range_meter_interval": str(layer_width) + "m",
            "low_threshold": str(low_thresh) + "dB",
            "high_threshold": str(high_thresh) + "dB",
            "units": "dB re 1 m-1",
        },
        "sA": {
            "long_name": "Nautical area scattering coefficient (NASC)",
            "units": "m2 nmi-2",
            "description": "NASC at the EI cell level",
        },
        "sA_sum": {
            "long_name": "Nautical area scattering coefficient (NASC)",
            "units": "m2 nmi-2",
            "description": "NASC at the ESDU level",
        },
        "frequency_nominal": {
            "long_name": "Transducer frequency",
            "standard_name": "sound_frequency",
            "units": "Hz",
            "valid_min": 0.0,
        },
        "latitude": {
            "long_name": "Platform latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "valid_range": (-90.0, 90.0),
        },
        "longitude": {
            "long_name": "Platform longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "valid_range": (-180.0, 180.0),
        },
    }

    logger.info("Done!")

    return attrs_dict


def plot_small_echogram(ds: xr.Dataset) -> None:
    """Plots a small (<= 1000 pings) echogram taken randomly in the Dataset"""

    logger.info("Starting plot of confirmation echogram")

    npings = len(ds.ping_time.values)
    size = min(npings, 1000)
    i = np.random.randint(0, npings - size)
    window = (i, i + size)

    (
        ds.swap_dims({"channel": "frequency_nominal"})
        .drop_vars("channel")
        .Sv.isel(frequency_nominal=0)
        .isel(ping_time=slice(*window))
        .plot.pcolormesh(
            x="ping_time",
            y="depth",
            yincrease=False,
        )
    )
    plt.show()
