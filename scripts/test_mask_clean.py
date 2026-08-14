import argparse
from pathlib import Path

import h5py
import numpy as np


def main():

    # Parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to Echointegration.mat",
    )
    args = parser.parse_args()
    input_path = Path(args.input_file).resolve()

    # Open dataset
    with h5py.File(str(input_path), "r") as f:
        mask_clean: np.ndarray = f["Mask_clean"][()]  # type: ignore

        unique, counts = np.unique(mask_clean, return_counts=True)

        print(dict(zip(unique, counts)))


if __name__ == "__main__":
    main()
