"""Linear ETM+/TM <-> OLI surface reflectance harmonisation (Roy et al. 2016).

OLS (default) is what the paper derives its transformations with: correct mean
per pixel, but slopes of 0.85-0.91 shrink the variance of the transformed data
(regression dilution); use it for means, composites and time series regressions.
RMA keeps the distribution and is exactly invertible; use it for change detection
with thresholds or distribution comparisons. See `roy2016` for the caveats.
"""

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from sites_cube.landsat import ETM, OLI, TM
from sites_cube.roy2016 import COEFFICIENTS, SOURCE, Line

Direction = Literal["etm_to_oli", "oli_to_etm"]
Method = Literal["ols", "rma"]

# median reflectance above this means unscaled DN were passed
MAX_MEDIAN_REFLECTANCE = 2.0


def transform_band(
    x: ArrayLike, band: str, *, direction: Direction, method: Method = "ols"
) -> np.ndarray:
    """Transform reflectance of one common-name band with the Roy et al. coefficients."""
    return np.asarray(COEFFICIENTS[(method, direction)][band].apply(np.asarray(x)))


def harmonize_landsat(
    ds: xr.Dataset,
    *,
    target: Literal["oli", "etm"] = "oli",
    method: Method = "ols",
    bands: Sequence[str] | None = None,
    platform_coord: str = "platform",
    out_of_range: Literal["keep", "clip", "nan"] = "nan",
    tm_as_etm: bool = True,
    coefficients: Mapping[str, Line] | None = None,
) -> xr.Dataset:
    """Bring ETM+/TM reflectance to OLI level (`target="oli"`) or OLI to ETM+.

    `ds` holds scaled, masked reflectance (see `landsat.mask_landsat`) with
    common-name variables and a `platform` coordinate along `time`. Only time
    steps of the source sensors change, target steps stay bit-identical;
    variables without coefficients pass through. `out_of_range` handles
    transformed values outside [0, 1]. `coefficients` (band -> Line) overrides
    the paper's values for the chosen direction.
    """
    if "harmonized_to" in ds.attrs:
        raise ValueError(
            f"dataset is already harmonized to {ds.attrs['harmonized_to']}"
        )
    if platform_coord not in ds.coords:
        raise KeyError(f"platform coordinate {platform_coord!r} missing")
    platforms = {str(p) for p in np.unique(ds[platform_coord].values)}
    unknown = platforms - TM - ETM - OLI
    if unknown:
        raise ValueError(f"no harmonisation for platforms {sorted(unknown)}")

    direction: Direction = "etm_to_oli" if target == "oli" else "oli_to_etm"
    sources = set(TM | ETM) if target == "oli" else set(OLI)
    if platforms & TM:
        if not tm_as_etm:
            raise ValueError(
                "TM data present, set tm_as_etm=True to use ETM+ coefficients"
            )
        warnings.warn(
            "TM (Landsat 4/5) is transformed with ETM+ coefficients",
            UserWarning,
            stacklevel=2,
        )

    lines = (
        coefficients if coefficients is not None else COEFFICIENTS[(method, direction)]
    )
    names = [str(b) for b in (bands or ds.data_vars) if str(b) in lines]
    _check_scaled(ds, names)

    is_source = ds[platform_coord].isin(list(sources))
    out = ds.copy()
    for band in names:
        x = ds[band]
        line = lines[band]
        y = line.intercept + line.slope * x
        if out_of_range == "clip":
            y = y.clip(0, 1)
        elif out_of_range == "nan":
            y = y.where((y >= 0) & (y <= 1))
        out[band] = xr.where(is_source, y, x, keep_attrs=True).astype(x.dtype)
    out.attrs |= {
        "harmonized_to": target,
        "harmonization_method": method,
        "harmonization_source": SOURCE if coefficients is None else "user coefficients",
    }
    return out


def _check_scaled(ds: xr.Dataset, bands: Sequence[str]) -> None:
    """Raise if a band looks like raw DN; skipped for dask data to stay lazy."""
    for band in bands:
        if ds[band].chunks is not None:
            continue
        values = ds[band].values
        if not np.isfinite(values).any():
            continue
        if np.nanmedian(values) > MAX_MEDIAN_REFLECTANCE:
            raise ValueError(f"{band}: median > {MAX_MEDIAN_REFLECTANCE}, unscaled DN?")
