import h5py
import numpy as np
import pytest
import xarray as xr

from matecho_to_python import mat_to_zarr


@pytest.fixture
def mat_ds(tmp_path):

    path = tmp_path / "Echointegration.mat"
    n_channel, n_ping, n_depth = 5, 1000, 500
    shape = (n_channel, n_ping, n_depth)
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as f:
        f["FrequencySort"] = np.array([[18e3], [38e3], [70e3], [120e3], [200e3]])
        f["depth_surface"] = np.linspace(1, 500, n_depth)
        f["Time"] = 1_443_616_800 + np.arange(n_ping, dtype=np.int64) * 11  # epoch seconds, 2015-09-30 12:40 UTC + 11 s/ping
        f["Latitude"] = np.full(n_ping, 45.0)
        f["Longitude"] = np.full(n_ping, -1.5)
        f["SolarAltitude"] = np.zeros(n_ping)
        f["Sv_surface"] = rng.uniform(-90, -40, shape)
        f["Sa_surface"] = rng.uniform(0, 10, shape)
        f["SaSum"] = rng.uniform(0, 100, (n_ping, n_channel))
        f["Bottom"] = rng.uniform(80, 100, (n_ping, n_channel))
        f["Mask_clean"] = np.zeros(shape, dtype=bool)

        params = f.create_group("EIParameters")
        params["LayerWidth"] = np.array([[1]])
        params["ESUSize"] = np.array([[1]])
        params["ESUType"] = np.array([[0]])
        params["LowThreshold"] = np.array([[-90]])
        params["HighThreshold"] = np.array([[-40]])

    return path


@pytest.fixture
def ds(mat_ds):
    path = mat_to_zarr(mat_ds)
    return xr.open_zarr(path)


@pytest.fixture
def da_Sv_fixture(ds):
    return ds.Sv
