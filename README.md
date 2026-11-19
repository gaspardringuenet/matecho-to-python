# *Matecho* `Echointegration.mat` conversion tool

> Easily re-use Matecho outputs within the acoustic data processing / analysis Python ecosystem

Python CLI tool to convert one or more [Matecho](https://pages.iuem.eu/ird/matecho/getting-started/what-is-matecho.html) `Echointegration.mat` files to a single Zarr store. The tool uses [`xarray`](https://docs.xarray.dev/en/stable/index.html) to produce a Dataset that is compatible with [`Echostack`](https://uw-echospace.github.io/project/echostack/) tools, namely [`Echopype`](https://github.com/echostack-org/echopype).

This is just a format conversion tool, preserving useful Matecho variables, such as `Sa_surface` or `Bottom`.

## Installation

The project supports both pip and uv installs.

```bash
$ pip install git+https://github.com/gaspardringuenet/matecho-to-python.git
```

```bash
$ uv add git+https://github.com/gaspardringuenet/matecho-to-python.git
```

## Example

This example shows how to convert a three-file echointegrated survey into a single Zarr store. Printed logs are removed for clarity.

```bash
user@MacBook-Pro ~/Data/acoustic/EI002_3pings1m
$ tree
.
├── Echointegration_leg01.mat
├── Echointegration_leg02A.mat
└── Echointegration_leg02B.mat

1 directory, 3 files

user@MacBook-Pro ~/Data/acoustic/EI002_3pings1m
$ matecho-to-python \
> Echointegration_leg01.mat \
> Echointegration_leg02A.mat \
> Echointegration_leg02B.mat \
> --to-nan-Sv -150 --light
...

user@MacBook-Pro ~/Data/acoustic/EI002_3pings1m
$ tree -L 1
.
├── Echointegration_leg01.mat
├── Echointegration_leg01.zarr
├── Echointegration_leg02A.mat
└── Echointegration_leg02B.mat

2 directories, 3 files
```

A new `Echointegration_leg01.zarr` is created. Naming uses the first input file and changes then extension, but the data within is that of all three files.

The output dataset is as follows:

```raw
<xarray.Dataset> Size: 32GB
Dimensions:            (channel: 4, ping_time: 1110362, depth: 792)
Coordinates:
  * channel            (channel) <U7 112B '38 kHz' '70 kHz' '120 kHz' '200 kHz'
  * ping_time          (ping_time) datetime64[ns] 9MB 2021-08-28T22:31:56 ......
    latitude           (ping_time) float32 4MB ...
    longitude          (ping_time) float32 4MB ...
  * depth              (depth) float32 3kB 8.5 9.5 10.5 ... 797.5 798.5 799.5
Data variables:
    bottom             (channel, ping_time) float32 18MB ...
    frequency_nominal  (channel) float32 16B ...
    mask_clean         (channel, ping_time, depth) bool 4GB ...
    sA                 (channel, ping_time, depth) float32 14GB ...
    sA_sum             (channel, ping_time) float32 18MB ...
    sun_alt            (ping_time) float32 4MB ...
    Sv                 (channel, ping_time, depth) float32 14GB ...
```

## Python API

The tool also defines a simple Python API.

```python
import xarray as xr

from matecho_to_python import mat_to_zarr, plot_small_echogram

# Conversion
ds_path = mat_to_zarr(["./EI_leg1.mat", "./EI_leg2.mat"])

# Opening dataset & plot validation echogram
ds_MVBS = xr.open_dataset(ds_path, engine="zarr")
plot_small_echogram(ds_MVBS)
```

### Conversion logic

The functions responsible for the conversion logic are:

- `matecho_to_python.convert:convert_matecho_EI` - formatting data arrays into an xarray `Dataset`.
- `matecho_to_python.convert:make_attrs_dict` - hardcoded variable attributes based on the [Matecho documentation](https://pages.iuem.eu/ird/matecho/user-guide/file-descriptions/matlab-data-files/echointegration-mat.html#fields).

## CLI options

```bash
$ matecho-to-python --help
usage: matecho-to-python [-h] [--light | --no-light] [--to-nan-Sv TO_NAN_SV]
                         [--range-lims CHANNEL:DEPTH]
                         [--zarr-format ZARR_FORMAT]
                         [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                         input [input ...]

Convert Matecho Echointegration.mat to a Zarr store compatible with Echostack

positional arguments:
  input                 Path(s) to the echointegration .mat file(s)

options:
  -h, --help            show this help message and exit
  --light, --no-light   Make the Dataset lighter by using float32 instead of
                        float64 for non-time numerical variables
  --to-nan-Sv TO_NAN_SV
                        Sv value to convert to NAN (typically -150 dB for
                        manual data removal in Matecho)
  --range-lims CHANNEL:DEPTH
                        Mask Sv below DEPTH (m) for a channel, e.g. '120
                        kHz:250'. Repeat for several channels.
  --zarr-format ZARR_FORMAT
                        The desired zarr format to target (2 or 3) as of zarr
                        v3. By default 2.
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level. By default INFO.
```

Example masking Sv below 250 m on the 120 kHz channel and below 300 m on the 38 kHz channel (quote values containing spaces):

```bash
$ matecho-to-python Echointegration.mat --range-lims "120 kHz:250" --range-lims "38 kHz:300"
```
