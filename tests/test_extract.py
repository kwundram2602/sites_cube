from datetime import UTC, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from sites_cube.aggregate import BASELINE_LABEL, ITEM_ATTRS, TILE
from sites_cube.extract import (
    PLATFORM,
    S2_BASELINE,
    _baseline_property,
    fuse_solar_day,
    item_attributes,
    item_labels,
    platforms_by_time,
    sites_in_bbox,
    to_reflectance,
    window_bbox,
)
from sites_cube.landsat import QA_PIXEL

SCALE = 0.0001


def _ds(dn: list[float], baseline: list[float] | None) -> xr.Dataset:
    time = pd.date_range("2022-01-20", periods=len(dn), freq="5D")
    ds = xr.Dataset(
        {"red": (("time", "site"), np.array(dn, dtype="uint16")[:, None])},
        coords={"time": time, "site": ["A"]},
    )
    if baseline is not None:
        ds[S2_BASELINE] = ("time", np.array(baseline, dtype="float32"))
    return ds


def _red(ds: xr.Dataset, harmonize: bool = True) -> np.ndarray:
    return to_reflectance(ds, 0, SCALE, 0.0, harmonize)["red"].sel(site="A").values


def test_mixed_baselines_have_no_jump() -> None:
    # same surface reflectance 0.15: stored as 1500 before and 2500 from 04.00 on
    np.testing.assert_allclose(
        _red(_ds([1500, 2500, 2500], [3.01, 4.0, 5.1])), [0.15, 0.15, 0.15], rtol=1e-6
    )


def test_nodata_and_values_below_offset_are_nan() -> None:
    red = _red(_ds([0, 0, 1000, 900], [3.01, 5.1, 5.1, 5.1]))
    assert np.isnan(red).all()


def test_no_harmonisation_when_disabled() -> None:
    np.testing.assert_allclose(
        _red(_ds([1500, 2500], [3.01, 5.1]), harmonize=False), [0.15, 0.25], rtol=1e-6
    )


def test_without_baseline_only_scale_is_applied() -> None:
    # e.g. Landsat: no baseline variable, no offset correction
    ds = _ds([1500, 2500], None)
    np.testing.assert_allclose(_red(ds), [0.15, 0.25], rtol=1e-6)
    assert S2_BASELINE not in to_reflectance(ds, 0, SCALE, 0.0, True)


def test_landsat_scale_and_offset() -> None:
    # Collection 2 L2 SR: DN * 2.75e-5 - 0.2, nodata 0, no S2 harmonisation
    ds = _ds([10000, 0, 7000], None)
    red = to_reflectance(ds, 0, 0.0000275, -0.2, False)["red"].sel(site="A").values
    np.testing.assert_allclose(red[[0, 2]], [0.075, -0.0075], rtol=1e-5)
    assert np.isnan(red[1])


def _props_item(**props) -> pystac.Item:
    return pystac.Item(
        id="i",
        geometry=None,
        bbox=None,
        datetime=datetime(2023, 6, 1, tzinfo=UTC),
        properties=props,
    )


def test_baseline_property_per_catalogue() -> None:
    pc = _props_item(**{"s2:processing_baseline": "05.10"})
    cdse = _props_item(**{"constellation": "sentinel-2", "processing:version": "05.10"})
    landsat = _props_item(**{"processing:version": "02.00"})
    assert _baseline_property(pystac.ItemCollection([pc])) == "s2:processing_baseline"
    assert _baseline_property(pystac.ItemCollection([cdse])) == "processing:version"
    assert _baseline_property(pystac.ItemCollection([landsat])) is None


def test_platforms_by_time_matches_item_datetime() -> None:
    items = [
        pystac.Item(
            id=platform,
            geometry=None,
            bbox=None,
            datetime=datetime(2023, 6, 1, 10, minute, 1, 123456, tzinfo=UTC),
            properties={"platform": platform},
        )
        for platform, minute in [("landsat-7", 5), ("landsat-8", 30)]
    ]
    times = np.array(
        ["2023-06-01T10:30:01.123456", "2023-06-01T10:05:01.123456"],
        dtype="datetime64[ns]",
    )
    assert platforms_by_time(pystac.ItemCollection(items), times) == [
        "landsat-8",
        "landsat-7",
    ]


def test_qa_vars_pass_through_reflectance() -> None:
    ds = _ds([10000, 0], None)
    ds[QA_PIXEL] = (("time", "site"), np.array([[21824], [1]], dtype="uint16"))
    out = to_reflectance(ds, 0, 0.0000275, -0.2, False)
    xr.testing.assert_identical(out[QA_PIXEL], ds[QA_PIXEL])
    assert out["red"].dtype == np.float32


def test_fuse_solar_day_drops_platform() -> None:
    ds = _ds([1, 2], None).astype("float32")
    ds = ds.assign_coords({PLATFORM: ("time", ["landsat-7", "landsat-8"])})
    assert PLATFORM not in fuse_solar_day(ds, 10.0).coords


def test_passthrough_vars_are_not_scaled_and_masked_by_footprint() -> None:
    # t0: valid item, t1: site outside the item footprint (all bands nodata)
    ds = _ds([2500, 0], [5.1, 5.1])
    ds["scl"] = (("time", "site"), np.array([[4], [0]], dtype="uint8"))
    ds["cld"] = (("time", "site"), np.array([[0], [0]], dtype="uint8"))
    out = to_reflectance(ds, 0, SCALE, 0.0, True, passthrough=["scl", "cld"])
    np.testing.assert_allclose(out["red"].sel(site="A").values[0], 0.15, rtol=1e-6)
    # unscaled, no S2 offset, 0 % cloud probability stays valid
    np.testing.assert_array_equal(out["scl"].sel(site="A").values[0], 4)
    np.testing.assert_array_equal(out["cld"].sel(site="A").values[0], 0)
    assert out["scl"].dtype == np.float32
    assert np.isnan(out["scl"].sel(site="A").values[1])
    assert np.isnan(out["cld"].sel(site="A").values[1])


def test_passthrough_qa_pixel_is_masked_by_footprint_only() -> None:
    # qa_pixel as quality layer: float32 copy, NaN only outside the footprint
    ds = _ds([10000, 0], None)
    ds[QA_PIXEL] = (("time", "site"), np.array([[22280], [1]], dtype="uint16"))
    out = to_reflectance(ds, 0, 0.0000275, -0.2, False, passthrough=[QA_PIXEL])
    qa = out[QA_PIXEL].sel(site="A").values
    assert qa[0] == 22280
    assert np.isnan(qa[1])


UTM = "EPSG:32632"


def test_window_bbox_none() -> None:
    assert window_bbox(None, UTM, 10) is None


def test_window_bbox_array_is_used_exactly() -> None:
    bbox = [504800, 6859000, 515800, 6865850]
    assert window_bbox(bbox, UTM, 10) == (504800, 6859000, 515800, 6865850)


def test_window_bbox_array_off_grid_raises() -> None:
    with pytest.raises(ValueError, match="multiple of"):
        window_bbox([504805, 6859000, 515800, 6865850], UTM, 10)


def test_window_bbox_vector_file_is_reprojected_and_snapped(tmp_path) -> None:
    path = tmp_path / "bounds.gpkg"
    poly = box(504803, 6859001, 515797, 6865849)
    gpd.GeoDataFrame(geometry=[poly], crs=UTM).to_crs(4326).to_file(path)
    bbox = window_bbox(str(path), UTM, 10)
    assert bbox is not None
    xmin, ymin, xmax, ymax = bbox
    # outward to the 10 m grid, reprojection may move the corners slightly
    assert (xmin, ymin, xmax, ymax) == pytest.approx(
        (504800, 6859000, 515800, 6865850), abs=10
    )
    assert all(v % 10 == 0 for v in (xmin, ymin, xmax, ymax))
    assert xmin <= 504803 and ymin <= 6859001
    assert xmax >= 515797 and ymax >= 6865849


def test_window_bbox_raster_file_is_snapped(tmp_path) -> None:
    path = tmp_path / "bounds.tif"
    profile = {
        "driver": "GTiff",
        "width": 3,
        "height": 2,
        "count": 1,
        "dtype": "uint8",
        "crs": UTM,
        "transform": from_origin(504801, 6859020, 5, 5),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((1, 2, 3), dtype="uint8"))
    # raster extent 504801..504816 / 6859010..6859020
    assert window_bbox(str(path), UTM, 10) == (504800, 6859010, 504820, 6859020)


def test_sites_in_bbox_drops_outside_sites() -> None:
    sites = gpd.GeoDataFrame(
        geometry=[Point(505000, 6860000), Point(600000, 6860000)],
        index=pd.Index(["in", "out"], name="site"),
        crs=UTM,
    ).to_crs(4326)
    kept = sites_in_bbox(sites, UTM, (504800, 6859000, 515800, 6865850))
    assert list(kept.index) == ["in"]


def test_item_labels_by_position() -> None:
    # two tiles of one datastrip share the sensing time
    dt = datetime(2023, 6, 1, 10, 20, tzinfo=UTC)
    items = [
        pystac.Item(
            id=tile,
            geometry=None,
            bbox=None,
            datetime=dt,
            properties={"s2:mgrs_tile": tile, "s2:processing_baseline": b},
        )
        for tile, b in [("32VNN", "05.10"), ("32VNM", "04.00")]
    ]
    assert item_labels(pystac.ItemCollection(items)) == {
        TILE: ["32VNN", "32VNM"],
        BASELINE_LABEL: ["05.10", "04.00"],
    }


def test_item_labels_without_baseline() -> None:
    item = pystac.Item(
        id="l8",
        geometry=None,
        bbox=None,
        datetime=datetime(2023, 6, 1, tzinfo=UTC),
        properties={"landsat:wrs_path": "196", "landsat:wrs_row": "024"},
    )
    assert item_labels(pystac.ItemCollection([item])) == {TILE: ["196024"]}


def test_fuse_solar_day_joins_tiles_of_valid_items() -> None:
    # same overpass, two tiles; site B lies outside the second tile
    time = pd.to_datetime(["2023-06-01T10:20", "2023-06-01T10:20"])
    ds = xr.Dataset(
        {"red": (("time", "site"), np.array([[0.1, 0.2], [0.3, np.nan]], "float32"))},
        coords={
            "time": time,
            "site": ["A", "B"],
            TILE: ("time", ["32VNM", "32VNN"]),
            BASELINE_LABEL: ("time", ["05.10", "05.10"]),
        },
    )
    out = fuse_solar_day(ds, 10.0)
    assert out[TILE].sel(time="2023-06-01").values.tolist() == ["32VNM,32VNN", "32VNM"]
    assert out[BASELINE_LABEL].values.tolist() == [["05.10", "05.10"]]


def test_fuse_solar_day_drops_labels_of_cubes() -> None:
    cube = xr.Dataset(
        {"red": (("time", "y", "x"), np.ones((2, 1, 1), dtype="float32"))},
        coords={
            "time": pd.to_datetime(["2023-06-01", "2023-06-02"]),
            TILE: ("time", ["A", "B"]),
        },
    )
    assert TILE not in fuse_solar_day(cube, 10.0).coords


def test_item_attributes_by_position() -> None:
    cdse = pystac.Item(
        id="S2A_MSIL2A_x",
        geometry=None,
        bbox=None,
        datetime=datetime(2023, 6, 1, tzinfo=UTC),
        properties={
            "sat:relative_orbit": 97,
            "eo:cloud_cover": 0.0,
            "view:sun_elevation": 45.5,
            "view:sun_azimuth": 160.0,
            "view:incidence_angle": 8.1,
        },
    )
    landsat = pystac.Item(
        id="LC08_x",
        geometry=None,
        bbox=None,
        datetime=datetime(2023, 6, 2, tzinfo=UTC),
        properties={"landsat:cloud_cover_land": 12.5, "view:sun_elevation": 50.0},
    )
    items = pystac.ItemCollection([cdse, landsat])
    attrs = item_attributes(items, "eo:cloud_cover")
    assert list(attrs) == list(ITEM_ATTRS)
    assert attrs["item_id"] == ["S2A_MSIL2A_x", "LC08_x"]
    assert attrs["orbit"] == ["97", ""]
    # a real 0 % cloud cover stays 0, a missing value becomes NaN
    assert attrs["cloud_cover"][0] == 0.0
    assert np.isnan(attrs["cloud_cover"][1])
    assert attrs["sun_elevation"] == [45.5, 50.0]
    assert np.isnan(attrs["view_incidence"][1])
    landsat_cloud = item_attributes(items, "landsat:cloud_cover_land")["cloud_cover"]
    assert landsat_cloud[1] == 12.5


def test_fuse_solar_day_drops_item_attributes() -> None:
    ds = _ds([1, 2], None).astype("float32")
    ds = ds.assign_coords(
        item_id=("time", ["a", "b"]),
        sun_elevation=("time", np.array([40.0, 41.0], "float32")),
    )
    out = fuse_solar_day(ds, 10.0)
    assert "item_id" not in out.coords
    assert "sun_elevation" not in out.coords
