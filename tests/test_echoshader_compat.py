"""Test compatibility of converted object with echoshader's echogram functions"""

import h5py
import holoviews
import numpy as np
import pytest
import xarray as xr
from echoshader.elements.echogram import (
    echogram,
    tricolor_echogram,
)

from matecho_to_python import mat_to_zarr


@pytest.fixture
def mat_ds(tmp_path):

    path = tmp_path / "Echointegration.mat"
    n_channel, n_ping, n_depth = 5, 20, 10
    shape = (n_channel, n_ping, n_depth)
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as f:
        f["FrequencySort"] = np.array([[18e3], [38e3], [70e3], [120e3], [200e3]])
        f["depth_surface"] = np.linspace(1, 100, n_depth)
        f["Time"] = np.arange(n_ping, dtype=np.int64) + 1_700_000_000  # epoch seconds
        f["Latitude"] = np.full(n_ping, 45.0)
        f["Longitude"] = np.full(n_ping, -1.5)
        f["SolarAltitude"] = np.zeros(n_ping)
        f["Sv_surface"] = rng.uniform(-90, -40, shape)
        f["Sa_surface"] = rng.uniform(0, 10, shape)
        f["SaSum"] = rng.uniform(0, 100, (n_ping, n_channel))
        f["Bottom"] = rng.uniform(80, 100, (n_ping, n_channel))
        f["Mask_clean"] = np.zeros(shape, dtype=bool)

        params = f.create_group("EIParameters")
        params["LayerWidth"] = np.array([[5.0]])
        params["ESUSize"] = np.array([[1]])
        params["ESUType"] = np.array([[0]])
        params["LowThreshold"] = np.array([[-90]])
        params["HighThreshold"] = np.array([[-40]])

    return path


@pytest.fixture
def ds(mat_ds):
    path = mat_to_zarr(mat_ds)
    return xr.open_zarr(path)


def test_echogram(ds):

    channel = ds.channel.values[0]

    plot = echogram(
        ds,
        channel=channel,
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.Image)


def test_echogram_with_accessor(ds):

    channel = ds.channel.values[0]

    plot = ds.eshader.echogram(
        channel=channel,
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.Image)


def test_multichannel_echogram(ds):

    channels = ds.sortby("frequency_nominal").channel.values.tolist()

    plot = echogram(
        ds,
        channel=channels,
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.Layout)


def test_tricolor(ds):

    rgb_channels = [
        ds.channel.sel(channel=ds.frequency_nominal == frequency).item()
        for frequency in (120000, 38000, 18000)
    ]

    plot = tricolor_echogram(
        ds,
        channel=rgb_channels,
        vmin=-80,
        vmax=-30,
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.RGB)
