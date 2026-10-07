from collections.abc import Iterable
from types import MappingProxyType

import xarray as xr

# STAC asset keys (Planetary Computer landsat-c2-l2)
QA_PIXEL = "qa_pixel"
QA_RADSAT = "qa_radsat"
QA_VARS = (QA_PIXEL, QA_RADSAT)

TM = frozenset({"landsat-4", "landsat-5"})
ETM = frozenset({"landsat-7"})
OLI = frozenset({"landsat-8", "landsat-9"})

# QA_PIXEL flag bits that can be masked; cirrus is OLI only (always 0 on TM/ETM+).
# Fill (bit 0) is always masked, there is no image data.
QA_PIXEL_FILL = 1 << 0
QA_PIXEL_BITS = MappingProxyType(
    {"dilated_cloud": 1, "cirrus": 2, "cloud": 3, "cloud_shadow": 4, "snow": 5}
)
# the filtering of Roy et al. (2016)
DEFAULT_QA_PIXEL_FLAGS = tuple(QA_PIXEL_BITS)
# QA_RADSAT bit of each common-name band: band number - 1, which differs per sensor
RADSAT_BIT_OLI = MappingProxyType(
    {
        "coastal": 0,
        "blue": 1,
        "green": 2,
        "red": 3,
        "nir08": 4,
        "swir16": 5,
        "swir22": 6,
    }
)
RADSAT_BIT_TM_ETM = MappingProxyType(
    {"blue": 0, "green": 1, "red": 2, "nir08": 3, "swir16": 4, "swir22": 6}
)
# QA_RADSAT bits that invalidate the whole pixel: ETM+ dropped pixel 9, OLI occlusion 11
RADSAT_PIXEL_MASK = (1 << 9) | (1 << 11)


def mask_landsat(
    ds: xr.Dataset,
    platform_coord: str = "platform",
    qa_pixel_flags: Iterable[str] = DEFAULT_QA_PIXEL_FLAGS,
    saturation: bool = True,
) -> xr.Dataset:
    """Mask Collection 2 L2 reflectance with QA_PIXEL and QA_RADSAT, drop the QA vars.

    Masks fill and the `qa_pixel_flags` (names of `QA_PIXEL_BITS`) for all bands,
    with `saturation` also saturated bands, dropped (ETM+) and occluded (OLI)
    pixels, and reflectance outside [0, 1] (reflective bands only). The defaults
    are the filtering Roy et al. (2016) applied before fitting the harmonisation
    coefficients. `ds` holds float reflectance plus the raw uint16 QA variables.
    """
    flags = list(qa_pixel_flags)
    unknown = set(flags) - set(QA_PIXEL_BITS)
    if unknown:
        raise ValueError(
            f"unknown QA_PIXEL flags {sorted(unknown)}, use {list(QA_PIXEL_BITS)}"
        )
    qa_mask = QA_PIXEL_FILL | sum(1 << QA_PIXEL_BITS[f] for f in flags)
    qa = ds[QA_PIXEL]
    radsat = ds[QA_RADSAT]
    is_oli = ds[platform_coord].isin(list(OLI))
    pixel_ok = (qa & qa_mask) == 0
    if saturation:
        pixel_ok = pixel_ok & ((radsat & RADSAT_PIXEL_MASK) == 0)
    out = ds.drop_vars(list(QA_VARS))
    for band in out.data_vars:
        ok = pixel_ok
        if band in RADSAT_BIT_OLI:  # reflective band, not thermal (Kelvin)
            ok = ok & (out[band] >= 0) & (out[band] <= 1)
        oli_bit = RADSAT_BIT_OLI.get(str(band))
        tm_bit = RADSAT_BIT_TM_ETM.get(str(band))
        if saturation and oli_bit is not None and tm_bit is not None:
            bit = xr.where(is_oli, 1 << oli_bit, 1 << tm_bit)
            ok = ok & ((radsat & bit) == 0)
        out[band] = out[band].where(ok)
    return out
