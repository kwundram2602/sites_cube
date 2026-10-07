import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sites_cube.harmonize import harmonize_landsat, transform_band
from sites_cube.roy2016 import Line


def _ds(values: list[float], platforms: list[str], band: str = "red") -> xr.Dataset:
    time = pd.date_range("2013-06-01", periods=len(values), freq="8D")
    return xr.Dataset(
        {band: (("time", "site"), np.array(values, dtype="float32")[:, None])},
        coords={"time": time, "site": ["A"], "platform": ("time", platforms)},
    )


def test_transform_band_values() -> None:
    red = transform_band(0.10, "red", direction="etm_to_oli")
    nir = transform_band(0.40, "nir08", direction="etm_to_oli", method="rma")
    np.testing.assert_allclose(red, 0.0061 + 0.9047 * 0.10)
    np.testing.assert_allclose(red, 0.09657, atol=1e-5)
    np.testing.assert_allclose(nir, 0.40082, atol=1e-5)


@pytest.mark.parametrize("band", ["blue", "green", "red", "nir08", "swir16", "swir22"])
def test_rma_round_trip_is_identity(band: str) -> None:
    x = np.array([0.02, 0.1, 0.4])
    there = transform_band(x, band, direction="etm_to_oli", method="rma")
    back = transform_band(there, band, direction="oli_to_etm", method="rma")
    np.testing.assert_allclose(back, x, atol=1e-6)


def test_ols_round_trip_is_not_identity() -> None:
    # two separate OLS fits, so back and forth shrinks towards the mean
    there = transform_band(0.4, "nir08", direction="etm_to_oli")
    back = transform_band(there, "nir08", direction="oli_to_etm")
    assert abs(back - 0.4) > 1e-3


def test_only_source_steps_change() -> None:
    ds = _ds([0.10, 0.10, 0.123456], ["landsat-7", "landsat-8", "landsat-9"])
    out = harmonize_landsat(ds)
    red = out["red"].sel(site="A").values
    np.testing.assert_allclose(red[0], 0.09657, atol=1e-6)
    # target steps bit-identical
    assert red[1:].tobytes() == ds["red"].sel(site="A").values[1:].tobytes()
    assert out["red"].dtype == np.float32
    assert out.attrs["harmonized_to"] == "oli"
    assert out.attrs["harmonization_source"] == "Roy et al. 2016, Table 2"


def test_target_etm_transforms_oli() -> None:
    ds = _ds([0.10, 0.10], ["landsat-7", "landsat-8"])
    red = harmonize_landsat(ds, target="etm")["red"].sel(site="A").values
    np.testing.assert_allclose(red, [0.10, 0.0123 + 0.9372 * 0.10], atol=1e-6)


def test_nan_stays_nan() -> None:
    out = harmonize_landsat(_ds([np.nan, 0.1], ["landsat-7", "landsat-7"]))
    assert np.isnan(out["red"].values[0, 0])


def test_bands_without_coefficients_pass_through() -> None:
    ds = _ds([0.1], ["landsat-7"], band="coastal")
    xr.testing.assert_identical(harmonize_landsat(ds)["coastal"], ds["coastal"])


@pytest.mark.parametrize(
    ("mode", "expected"), [("keep", -0.0071), ("clip", 0.0), ("nan", np.nan)]
)
def test_out_of_range(mode: str, expected: float) -> None:
    # blue RMA: -0.0095 + 0.9785 * 0.0025 < 0
    ds = _ds([0.0025], ["landsat-7"], band="blue")
    out = harmonize_landsat(ds, method="rma", out_of_range=mode)  # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(out["blue"].values[0, 0], expected, atol=1e-4)


def test_second_call_raises() -> None:
    out = harmonize_landsat(_ds([0.1], ["landsat-7"]))
    with pytest.raises(ValueError, match="already harmonized"):
        harmonize_landsat(out)


def test_unscaled_dn_raise() -> None:
    with pytest.raises(ValueError, match="unscaled"):
        harmonize_landsat(_ds([10000, 12000], ["landsat-7", "landsat-8"]))


def test_missing_platform_coord_raises() -> None:
    with pytest.raises(KeyError):
        harmonize_landsat(_ds([0.1], ["landsat-7"]).drop_vars("platform"))


def test_mss_platform_raises() -> None:
    with pytest.raises(ValueError, match="landsat-1"):
        harmonize_landsat(_ds([0.1], ["landsat-1"]))


def test_tm_warns_or_raises() -> None:
    ds = _ds([0.1], ["landsat-5"])
    with pytest.warns(UserWarning, match="TM"):
        harmonize_landsat(ds)
    with pytest.raises(ValueError, match="tm_as_etm"):
        harmonize_landsat(ds, tm_as_etm=False)


def test_dask_stays_lazy() -> None:
    ds = _ds([0.1, 0.2], ["landsat-7", "landsat-8"]).chunk({"time": 1})
    out = harmonize_landsat(ds)
    assert dask.is_dask_collection(out["red"])
    np.testing.assert_allclose(out["red"].values[:, 0], [0.09657, 0.2], atol=1e-6)


def test_coefficients_override() -> None:
    out = harmonize_landsat(
        _ds([0.1], ["landsat-7"]), coefficients={"red": Line(0.01, 1.0)}
    )
    np.testing.assert_allclose(out["red"].values[0, 0], 0.11, atol=1e-6)
    assert out.attrs["harmonization_source"] == "user coefficients"
