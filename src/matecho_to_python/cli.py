import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from zarr.errors import ZarrUserWarning

from .core import mat_to_zarr

# Configure logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Silence ZarrUserWarning on consolidated metadata
warnings.filterwarnings("ignore", category=ZarrUserWarning)


def parse_cli() -> argparse.Namespace:
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
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level. By default INFO.",
    )

    return parser.parse_args()


def main():
    """Main function: parse CLI args and run matecho_to_zarr"""

    args = parse_cli()

    logging.getLogger("matecho_to_python").setLevel(args.log_level)

    ds_path: Path = mat_to_zarr(
        input=args.input,
        float_dtype=np.float32 if args.light else np.float64,
        to_nan_Sv=args.to_nan_Sv,
        zarr_format=args.zarr_format,
    )

    # Open complete Dataset
    logger.debug("Opening concatenated Dataset")
    ds = xr.open_dataset(ds_path, engine="zarr")

    # Print Dataset
    logger.debug("Printing Dataset")
    if logger.isEnabledFor(logging.DEBUG):
        print(ds)
