from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pystac
import xarray as xr

from sites_cube.extract import S2_BASELINE, _baseline_property, to_reflectance

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
