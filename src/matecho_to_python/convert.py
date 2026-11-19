import logging

import h5py
import numpy as np
import numpy.typing as npt
import xarray as xr

logger = logging.getLogger(__name__)


def convert_matecho_EI(f: h5py.File, float_dtype: npt.DTypeLike) -> xr.Dataset:
    """Convert a Matecho echointegration (Echointegration.mat) file into
    an xarray Dataset with Echopype-compatible dimensions."""

    logger.debug("Building xr.Dataset")

    ds = xr.Dataset()

    #### Add coordinates ####
    # Channel (echopype uses Vendor channel id, we use a descriptive string)
    vals_channel = np.asarray(f["FrequencySort"]).squeeze().astype(float_dtype)
    channel_str = [f"{int(c / 1e3)} kHz" for c in vals_channel]
    ds.coords["channel"] = ("channel", channel_str)

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

    # Frequency nominal
    ds["frequency_nominal"] = (["channel"], vals_channel)

    # Bottom depth
    vals_bottom: np.ndarray = np.asarray(f["Bottom"]).astype(float_dtype)
    ds["bottom"] = (["ping_time", "channel"], vals_bottom)
    ds["bottom"] = ds["bottom"].transpose("channel", "ping_time")

    # Sun position
    vals_sun: np.ndarray = np.asarray(f["SolarAltitude"]).squeeze().astype(float_dtype)
    ds["sun_alt"] = (["ping_time"], vals_sun)

    # Transpose
    ds = ds.transpose("channel", "ping_time", "depth")

    logger.debug("Done!")

    return ds


def make_attrs_dict(f: h5py.File) -> dict:
    """Build a dictionary containing formatted variables attributes"""

    logger.debug("Formatting attributes dict using .mat co-variables")

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
            "long_name": "Echosounder channel",
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

    logger.debug("Done!")

    return attrs_dict
