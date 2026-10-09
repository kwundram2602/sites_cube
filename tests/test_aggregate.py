from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sites_cube.aggregate import (
    BASELINE_LABEL,
    TILE,
    aggregate_time,
    join_labels,
    to_long,
    write_merged,
    write_per_site,
)


@pytest.fixture
def ds() -> xr.Dataset:
    time = pd.to_datetime(["2023-06-01", "2023-06-10", "2023-06-20", "2023-07-05"])
    red = np.array(
        [[0.1, 0.2], [0.3, np.nan], [0.5, 0.4], [0.7, np.nan]], dtype="float32"
    )
    # green is invalid on 2023-06-20 at site A only -> date must be dropped for all bands
    green = red.copy()
    green[2, 0] = np.nan
    return xr.Dataset(
        {"red": (("time", "site"), red), "green": (("time", "site"), green)},
        coords={"time": time, "site": ["A", "B"]},
    )


def test_monthly_median_and_n_obs(ds: xr.Dataset) -> None:
    m = aggregate_time(ds, "1MS", "median")
    # site A June: 0.5 is dropped because green is NaN that day -> median(0.1, 0.3)
    np.testing.assert_allclose(m["red"].sel(site="A").values, [0.2, 0.7], rtol=1e-6)
    np.testing.assert_allclose(m["red"].sel(site="B").values[0], 0.3, rtol=1e-6)
    assert np.isnan(m["red"].sel(site="B").values[1])
    np.testing.assert_array_equal(m["n_obs"].sel(site="A").values, [2, 1])
    np.testing.assert_array_equal(m["n_obs"].sel(site="B").values, [2, 0])


def test_n_obs_zero_for_month_without_dates(ds: xr.Dataset) -> None:
    # August has no time step at all -> n_obs must be 0, like a month with only
    # invalid observations, not a NaN cast to int
    gap = ds.isel(time=[0]).assign_coords(time=pd.to_datetime(["2023-09-02"]))
    m = aggregate_time(xr.concat([ds, gap], "time"), "1MS", "median")
    np.testing.assert_array_equal(m["n_obs"].sel(site="B").values, [2, 0, 0, 1])


def test_monthly_mean(ds: xr.Dataset) -> None:
    m = aggregate_time(ds, "1MS", "mean")
    np.testing.assert_allclose(m["red"].sel(site="A").values, [0.2, 0.7], rtol=1e-6)


def test_unknown_reducer(ds: xr.Dataset) -> None:
    with pytest.raises(ValueError):
        aggregate_time(ds, "1MS", "sum_of_squares")


def test_to_long(ds: xr.Dataset) -> None:
    df = to_long(aggregate_time(ds, "1MS", "max"))
    assert list(df.columns) == ["site", "time", "band", "value", "n_obs"]
    assert len(df) == 8


def test_write_per_site(ds: xr.Dataset, tmp_path: Path) -> None:
    paths = write_per_site(aggregate_time(ds, "1MS", "median"), tmp_path)
    assert sorted(p.name for p in paths) == ["A.csv", "B.csv"]
    df = pd.read_csv(tmp_path / "B.csv")
    assert list(df.columns) == ["time", "red", "green", "n_obs"]
    assert df["n_obs"].tolist() == [2, 0]


def test_to_long_keeps_platform_as_column(ds: xr.Dataset) -> None:
    ds = ds.assign_coords(platform=("time", ["s2a", "s2b", "s2a", "s2b"]))
    df = to_long(ds)
    assert "platform" in df.columns
    assert set(df["band"]) == {"red", "green"}
    assert len(df) == 16


def _labelled(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign_coords(
        {
            TILE: ("time", ["32VNM", "32VNN", "32VNM", "32VNN"]),
            BASELINE_LABEL: ("time", ["04.00", "04.00", "05.10", "05.10"]),
        }
    )


def test_join_labels_keeps_valid_entries_only(ds: xr.Dataset) -> None:
    labelled = _labelled(ds)
    valid = labelled.to_dataarray("band").notnull().all("band")
    key = xr.DataArray(["d1", "d1", "d2", "d2"], dims="time")
    out = join_labels(labelled, valid, key, "group")
    # site B is NaN on 2023-06-10 -> only the first tile on d1
    assert out[TILE].sel(group="d1").values.tolist() == ["32VNM,32VNN", "32VNM"]
    # site A is invalid on 2023-06-20 (green NaN) -> only the second tile on d2
    assert out[TILE].sel(group="d2").values.tolist() == ["32VNN", "32VNM"]
    assert out[BASELINE_LABEL].sel(group="d2", site="A").item() == "05.10"


def test_join_labels_merges_comma_lists() -> None:
    ds = xr.Dataset(
        {"red": (("time", "site"), np.ones((2, 1), dtype="float32"))},
        coords={
            "time": pd.to_datetime(["2023-06-01", "2023-06-02"]),
            "site": ["A"],
            TILE: (("time", "site"), [["A,B"], ["B"]]),
        },
    )
    valid = ds["red"].notnull()
    out = join_labels(ds, valid, xr.DataArray(["m", "m"], dims="time"), "group")
    assert out[TILE].sel(group="m", site="A").item() == "A,B"


def test_aggregate_time_joins_tiles_per_period(ds: xr.Dataset) -> None:
    m = aggregate_time(_labelled(ds), "1MS", "median")
    assert m[TILE].dims == ("time", "site")
    # June, site A: 2023-06-20 is invalid -> 32VNM (06-01) and 32VNN (06-10)
    # July, site B: no valid observation -> empty
    assert m[TILE].sel(site="A").values.tolist() == ["32VNM,32VNN", "32VNN"]
    assert m[TILE].sel(site="B").values.tolist() == ["32VNM", ""]
    assert m[BASELINE_LABEL].sel(site="A").values.tolist() == ["04.00", "05.10"]


def test_aggregate_time_drops_labels_without_sites() -> None:
    cube = xr.Dataset(
        {"red": (("time", "y", "x"), np.ones((2, 1, 1), dtype="float32"))},
        coords={
            "time": pd.to_datetime(["2023-06-01", "2023-06-02"]),
            TILE: ("time", ["A", "B"]),
        },
    )
    assert TILE not in aggregate_time(cube, "1MS", "median").coords


def test_to_long_keeps_labels_as_columns(ds: xr.Dataset) -> None:
    df = to_long(_labelled(ds))
    assert {TILE, BASELINE_LABEL} <= set(df.columns)
    assert set(df["band"]) == {"red", "green"}
    assert len(df) == 16


def test_write_per_site_puts_labels_first(ds: xr.Dataset, tmp_path: Path) -> None:
    ds = _labelled(ds).assign_coords(platform=("time", ["s2a", "s2b", "s2a", "s2b"]))
    write_per_site(ds, tmp_path, dropna=True)
    df = pd.read_csv(tmp_path / "B.csv", dtype={BASELINE_LABEL: str})
    assert list(df.columns) == [
        "time",
        "platform",
        TILE,
        BASELINE_LABEL,
        "red",
        "green",
    ]
    # site B has no value on 2023-06-10 and 2023-07-05
    assert df[TILE].tolist() == ["32VNM", "32VNM"]
    assert df[BASELINE_LABEL].tolist() == ["04.00", "05.10"]


def test_write_merged(ds: xr.Dataset, tmp_path: Path) -> None:
    path = write_merged(_labelled(ds), tmp_path / "sites.csv", dropna=True)
    df = pd.read_csv(path)
    assert list(df.columns) == ["site", "time", TILE, BASELINE_LABEL, "red", "green"]
    assert df["site"].tolist() == ["A"] * 4 + ["B"] * 2
    assert df.groupby("site")["time"].apply(lambda t: t.is_monotonic_increasing).all()


def _with_attrs(ds: xr.Dataset) -> xr.Dataset:
    return _labelled(ds).assign_coords(
        item_id=("time", ["i0", "i1", "i2", "i3"]),
        orbit=("time", ["22", "22", "65", "65"]),
        cloud_cover=("time", np.array([1.0, 2.0, 3.0, 4.0], "float32")),
        sun_elevation=("time", np.array([50.0, 51.0, 52.0, 40.0], "float32")),
        sun_azimuth=("time", np.array([160.0] * 4, "float32")),
        view_incidence=("time", np.array([5.0] * 4, "float32")),
    )


def test_write_per_site_attribute_columns(ds: xr.Dataset, tmp_path: Path) -> None:
    write_per_site(_with_attrs(ds), tmp_path, dropna=True)
    df = pd.read_csv(tmp_path / "A.csv")
    assert list(df.columns) == [
        "time",
        TILE,
        BASELINE_LABEL,
        "item_id",
        "orbit",
        "cloud_cover",
        "sun_elevation",
        "sun_azimuth",
        "view_incidence",
        "red",
        "green",
    ]
    assert df["item_id"].tolist() == ["i0", "i1", "i2", "i3"]


def test_aggregate_time_drops_item_attributes(ds: xr.Dataset) -> None:
    m = aggregate_time(_with_attrs(ds), "1MS", "median")
    assert "item_id" not in m.coords
    assert "sun_elevation" not in m.coords
    assert TILE in m.coords


def test_to_long_keeps_item_attributes(ds: xr.Dataset) -> None:
    df = to_long(_with_attrs(ds))
    assert {"item_id", "orbit", "cloud_cover"} <= set(df.columns)
    assert set(df["band"]) == {"red", "green"}
