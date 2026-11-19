import logging
import os
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import xarray as xr

from .convert import convert_matecho_EI, make_attrs_dict

logger = logging.getLogger(__name__)


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
        with h5py.File(filepath, "r") as f:
            ds = _process_one_file(f, float_dtype)

        logger.debug("Chunking")
        ds = ds.chunk({"channel": 1, "ping_time": 5000, "depth": 100})

        if to_nan_Sv is not None:
            ds["Sv"] = ds["Sv"].where(ds["Sv"] != to_nan_Sv)

        if i == 0:
            logger.info("Saving as Zarr store")
            # First file: create
            ds.to_zarr(output_path, mode="w", zarr_format=zarr_format)
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


def _process_one_file(f: h5py.File, float_dtype: npt.DTypeLike) -> xr.Dataset:
    """Convert a single file"""

    # Build Dataset
    ds = convert_matecho_EI(f, float_dtype)

    # Fetch attributes
    attrs_dict = make_attrs_dict(f)

    # Add attributes
    logger.debug("Appending attributes to variables")
    for var, a in attrs_dict.items():
        if var in ds.variables:
            ds[var].attrs = a

    return ds
