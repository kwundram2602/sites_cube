from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sites_cube.aggregate import aggregate_time, to_long, write_per_site


@pytest.fixture
def ds() -> xr.Dataset:
    time = pd.to_datetime(["2023-06-01", "2023-06-10", "2023-06-20", "2023-07-05"])
    red = np.array(
        [[0.1, 0.2], [0.3, np.nan], [0.5, 0.4], [0.7, np.nan]], dtype="float32"
    )
    return xr.Dataset(
        {"red": (("time", "site"), red)}, coords={"time": time, "site": ["A", "B"]}
    )


def test_monthly_median_and_n_obs(ds: xr.Dataset) -> None:
    m = aggregate_time(ds, "1MS", "median")
    np.testing.assert_allclose(m["red"].sel(site="A").values, [0.3, 0.7], rtol=1e-6)
    np.testing.assert_allclose(m["red"].sel(site="B").values[0], 0.3, rtol=1e-6)
    assert np.isnan(m["red"].sel(site="B").values[1])
    np.testing.assert_array_equal(m["red_n_obs"].sel(site="B").values, [2, 0])


def test_monthly_mean(ds: xr.Dataset) -> None:
    m = aggregate_time(ds, "1MS", "mean")
    np.testing.assert_allclose(m["red"].sel(site="A").values, [0.3, 0.7], rtol=1e-6)


def test_unknown_reducer(ds: xr.Dataset) -> None:
    with pytest.raises(ValueError):
        aggregate_time(ds, "1MS", "sum_of_squares")


def test_to_long(ds: xr.Dataset) -> None:
    df = to_long(aggregate_time(ds, "1MS", "max"))
    assert list(df.columns) == ["site", "time", "band", "value", "n_obs"]
    assert len(df) == 4


def test_write_per_site(ds: xr.Dataset, tmp_path: Path) -> None:
    paths = write_per_site(aggregate_time(ds, "1MS", "median"), tmp_path)
    assert sorted(p.name for p in paths) == ["A.csv", "B.csv"]
    df = pd.read_csv(tmp_path / "B.csv")
    assert list(df.columns) == ["time", "red", "red_n_obs"]
    assert df["red_n_obs"].tolist() == [2, 0]
