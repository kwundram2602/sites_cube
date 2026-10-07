import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sites_cube.landsat import QA_PIXEL, QA_RADSAT, mask_landsat

CLEAR = 1 << 6


def _ds(
    qa: list[int], radsat: list[int], platforms: list[str], value: float = 0.1
) -> xr.Dataset:
    n = len(qa)
    time = pd.date_range("2013-06-01", periods=n, freq="8D")
    dims = ("time", "site")
    full = np.full((n, 1), value, dtype="float32")
    return xr.Dataset(
        {
            "blue": (dims, full.copy()),
            "red": (dims, full.copy()),
            "lwir": (dims, full.copy()),
            QA_PIXEL: (dims, np.array(qa, dtype="uint16")[:, None]),
            QA_RADSAT: (dims, np.array(radsat, dtype="uint16")[:, None]),
        },
        coords={"time": time, "site": ["A"], "platform": ("time", platforms)},
    )


def test_clear_pixel_kept_and_qa_dropped() -> None:
    out = mask_landsat(_ds([CLEAR], [0], ["landsat-8"]))
    assert QA_PIXEL not in out and QA_RADSAT not in out
    np.testing.assert_allclose(out["red"].values, 0.1)


@pytest.mark.parametrize("bit", [0, 1, 2, 3, 4, 5])
def test_qa_pixel_bits_mask_all_bands(bit: int) -> None:
    out = mask_landsat(_ds([1 << bit], [0], ["landsat-8"]))
    for band in ("blue", "red", "lwir"):
        assert np.isnan(out[band].values).all()


@pytest.mark.parametrize("bit", [6, 7])
def test_clear_and_water_bits_do_not_mask(bit: int) -> None:
    out = mask_landsat(_ds([1 << bit], [0], ["landsat-7"]))
    assert not np.isnan(out["red"].values).any()


def test_radsat_bit_per_sensor() -> None:
    # bit 2: red on TM/ETM+ (band 3), green on OLI (band 3)
    out = mask_landsat(
        _ds([CLEAR, CLEAR], [1 << 2, 1 << 2], ["landsat-7", "landsat-8"])
    )
    red = out["red"].values[:, 0]
    blue = out["blue"].values[:, 0]
    assert np.isnan(red[0]) and not np.isnan(red[1])
    assert not np.isnan(blue).any()
    # bit 1: blue on OLI (band 2), green on ETM+
    out = mask_landsat(
        _ds([CLEAR, CLEAR], [1 << 1, 1 << 1], ["landsat-7", "landsat-8"])
    )
    blue = out["blue"].values[:, 0]
    assert not np.isnan(blue[0]) and np.isnan(blue[1])


@pytest.mark.parametrize("bit", [9, 11])
def test_radsat_pixel_bits_mask_all_bands(bit: int) -> None:
    out = mask_landsat(_ds([CLEAR], [1 << bit], ["landsat-7"]))
    assert np.isnan(out["red"].values).all() and np.isnan(out["lwir"].values).all()


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_reflectance_outside_unit_range_masked(value: float) -> None:
    out = mask_landsat(_ds([CLEAR], [0], ["landsat-8"], value=value))
    assert np.isnan(out["red"].values).all()


def test_thermal_not_range_masked() -> None:
    ds = _ds([CLEAR], [0], ["landsat-8"]).assign(lwir=(("time", "site"), [[300.0]]))
    np.testing.assert_allclose(mask_landsat(ds)["lwir"].values, 300.0)


def test_snow_kept_when_not_in_flags() -> None:
    snow = 1 << 5
    ds = _ds([snow, 1 << 3], [0, 0], ["landsat-8", "landsat-8"])
    red = mask_landsat(ds, qa_pixel_flags=["cloud"])["red"].values[:, 0]
    assert not np.isnan(red[0]) and np.isnan(red[1])


def test_fill_always_masked() -> None:
    out = mask_landsat(_ds([1], [0], ["landsat-8"]), qa_pixel_flags=[])
    assert np.isnan(out["red"].values).all()


def test_saturation_can_be_disabled() -> None:
    ds = _ds([CLEAR], [(1 << 2) | (1 << 9)], ["landsat-7"])
    assert not np.isnan(mask_landsat(ds, saturation=False)["red"].values).any()


def test_unknown_flag_raises() -> None:
    with pytest.raises(ValueError, match="clouds"):
        mask_landsat(_ds([CLEAR], [0], ["landsat-8"]), qa_pixel_flags=["clouds"])
