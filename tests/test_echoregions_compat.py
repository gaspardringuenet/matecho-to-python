"""Apply echoregions region masking to a synthetic Echointegration.mat converted with mat_to_zarr"""

from pathlib import Path

import echoregions as er
import numpy as np
import pytest
from echoregions.regions2d.regions2d import Regions2D
from xarray import DataArray

DATA_DIR = Path(__file__).parent / "test_data"
CSV_PATH = DATA_DIR / "regions.csv"


@pytest.fixture(scope="function")
def regions2d_fixture() -> Regions2D:
    """
    Regions2D object fixture.

    Returns
    -------
    r2d : Regions2D
        Object containing data of test regions CSV file.
    """
    r2d = er.read_regions_csv(CSV_PATH)
    return r2d


@pytest.fixture
def da_Sv_no_overlap_fixture(da_Sv_fixture: DataArray) -> DataArray:
    """Sv DataArray shifted one day later, so it does not overlap any region in time."""
    return da_Sv_fixture.assign_coords(
        ping_time=da_Sv_fixture.ping_time + np.timedelta64(1, "D")
    )


@pytest.mark.filterwarnings(
    "ignore:All rows in Regions DataFrame have NaN Depth values"
)
def test_mask_empty_no_overlap(
    regions2d_fixture: Regions2D,
    da_Sv_no_overlap_fixture: DataArray,
) -> None:
    """
    Test if mask is empty when there is no overlap between regions and the Sv DataArray.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test regions CSV file.
    da_Sv_no_overlap_fixture : DataArray
        Sv DataArray shifted outside the time range of the regions.
    """
    region_ids = regions2d_fixture.data.region_id.astype(int).to_list()
    da_Sv = da_Sv_no_overlap_fixture.isel(channel=0)

    mask_output, region_points = regions2d_fixture.region_mask(da_Sv, region_ids)
    assert (mask_output["mask_3d"] == 0).all()
    assert region_points.empty

    mask_output_2d, region_points_2d = regions2d_fixture.region_mask(
        da_Sv, region_ids, collapse_to_2d=True
    )
    assert mask_output_2d.isnull().all()
    assert region_points_2d.empty


def test_mask_3d_2d_3d_2d(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Testing if converting 3d-2d-3d-2d masks works on the converted Sv data.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test regions CSV file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    region_id = regions2d_fixture.data.region_id.astype(int).to_list()
    mask_labels = {key: idx for idx, key in enumerate(region_id)}

    mask_3d_ds, _ = regions2d_fixture.region_mask(
        da_Sv_fixture, mask_labels=mask_labels
    )

    assert mask_3d_ds.mask_3d.region_id.values.tolist() == region_id
    assert (mask_3d_ds.mask_3d.sum(dim=["ping_time", "depth"]) > 0).all()
    assert mask_3d_ds.mask_labels.region_id.values.tolist() == region_id

    mask_2d_ds = er.convert_mask_3d_to_2d(mask_3d_ds).compute()
    mask_2d_values = mask_2d_ds.mask_2d.values
    assert set(np.unique(mask_2d_values[~np.isnan(mask_2d_values)])) == set(region_id)

    mask_3d_ds_2nd = er.convert_mask_2d_to_3d(mask_2d_ds)
    assert mask_3d_ds_2nd.equals(mask_3d_ds)

    mask_2d_ds_2nd = er.convert_mask_3d_to_2d(mask_3d_ds_2nd)
    assert mask_2d_ds_2nd.equals(mask_2d_ds)


def test_chunked_region_mask(
    regions2d_fixture: Regions2D, da_Sv_fixture: DataArray
) -> None:
    """
    Testing if chunked region masking and computed region masking operations result in the same array and
    points and checks the array chunks for both 3D and 2D operation subtypes.

    Parameters
    ----------
    regions2d_fixture : Regions2D
        Object containing data of test regions CSV file.
    da_Sv_fixture : DataArray
        DataArray containing Sv data of test zarr file.
    """
    chunk_dict = {"ping_time": 400, "depth": 100}

    mask_3d_ds_chunked, mask_3d_points_chunked = regions2d_fixture.region_mask(
        da_Sv_fixture.chunk(chunk_dict), collapse_to_2d=False
    )
    mask_3d_ds_computed, mask_3d_points_computed = regions2d_fixture.region_mask(
        da_Sv_fixture.compute(), collapse_to_2d=False
    )
    assert mask_3d_ds_chunked.chunksizes["region_id"][0] == 1
    assert mask_3d_ds_chunked.chunksizes["ping_time"][0] == 400
    assert mask_3d_ds_chunked.chunksizes["depth"][0] == 100
    assert mask_3d_ds_chunked.equals(mask_3d_ds_computed)
    assert mask_3d_points_chunked.equals(mask_3d_points_computed)

    mask_2d_ds_chunked, mask_2d_points_chunked = regions2d_fixture.region_mask(
        da_Sv_fixture.chunk(chunk_dict), collapse_to_2d=True
    )
    mask_2d_ds_computed, mask_2d_points_computed = regions2d_fixture.region_mask(
        da_Sv_fixture.compute(), collapse_to_2d=True
    )
    assert mask_2d_ds_chunked.chunksizes["ping_time"][0] == 400
    assert mask_2d_ds_chunked.chunksizes["depth"][0] == 100
    assert mask_2d_ds_chunked.equals(mask_2d_ds_computed)
    assert mask_2d_points_chunked.equals(mask_2d_points_computed)
