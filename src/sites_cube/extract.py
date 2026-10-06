from typing import Literal

import geopandas as gpd
import numpy as np
import odc.stac
import pystac
import xarray as xr

S2_BASELINE_OFFSET_DN = 1000
S2_BASELINE = (
    "s2_processing_baseline"  # odc.stac name of property s2:processing_baseline
)


def load_points(
    items: pystac.ItemCollection,
    bands: dict[str, str],
    sites: gpd.GeoDataFrame,
    crs: str,
    resolution: float,
    buffer_m: float,
    chunks: dict[str, int | Literal["auto"]],
) -> xr.Dataset:
    """Load every item separately over the sites bbox and sample the site pixels.

    Items are not fused here so per-item corrections (S2 baseline offset) can be
    applied before overlapping tiles are merged.
    """
    # 1. sites into the target grid CRS, their extent defines the load window
    pts = sites.to_crs(crs)
    xmin, ymin, xmax, ymax = pts.total_bounds
    # 2. lazy (dask) cube over the sites bbox; nothing is read from the COGs yet
    ds = odc.stac.load(
        items,
        bands=list(bands.values()),  # asset keys, e.g. B04
        groupby="id",  # one time slice per item, no fusing of overlapping tiles
        crs=crs,
        resolution=resolution,
        # buffer keeps edge sites away from the window border
        x=(xmin - buffer_m, xmax + buffer_m),
        y=(ymin - buffer_m, ymax + buffer_m),
        chunks=chunks,
        # per-item baseline as variable (time,), needed for the S2 offset correction
        with_properties=["s2:processing_baseline"] if _is_s2(items) else None,
    )
    # 3. asset keys -> config aliases (B04 -> red)
    ds = ds.rename({v: k for k, v in bands.items()})
    # 4. vectorised point sampling: x/y share the new dim `site`, so each site
    #    gets its own nearest pixel (not the full x*y cross product)
    sel = ds.sel(
        x=xr.DataArray(pts.geometry.x.to_numpy(), dims="site"),
        y=xr.DataArray(pts.geometry.y.to_numpy(), dims="site"),
        method="nearest",
    )
    return (
        # 5. label the site dim with the site ids
        sel.assign_coords(site=pts.index.to_numpy())
        # 6. pixel coords are no longer needed per site
        .drop_vars(["x", "y", "spatial_ref"], errors="ignore")
        # 7. trigger the actual reads, result is a small (time, site) dataset
        .compute()
    )


def _is_s2(items: pystac.ItemCollection) -> bool:
    # only S2 items carry a processing baseline; requesting it for others would fail
    return any("s2:processing_baseline" in i.properties for i in items)


def to_reflectance(
    ds: xr.Dataset,
    nodata: float,
    scale: float,
    offset: float,
    harmonize_s2_offset: bool,
) -> xr.Dataset:
    """Mask nodata, remove S2 baseline offset, apply scale/offset."""
    # 1. spectral bands only, the baseline variable is metadata
    bands = [v for v in ds.data_vars if v != S2_BASELINE]
    # 2. float so NaN can mark nodata pixels (e.g. outside the tile footprint)
    out = ds[bands].astype("float32").where(ds[bands] != nodata)
    if harmonize_s2_offset and S2_BASELINE in ds:
        # 3. baseline >= 04.00 stores DN + 1000; subtract per time step
        #    (baseline is (time,), broadcast over sites)
        out = out - xr.where(ds[S2_BASELINE] >= 4.0, S2_BASELINE_OFFSET_DN, 0)
        # 4. values at/below the offset are invalid after harmonisation
        out = out.where(out > 0)
    # 5. DN -> surface reflectance
    return out * scale + offset


def fuse_solar_day(ds: xr.Dataset, lon: float) -> xr.Dataset:
    """Merge items of the same solar day (overlapping tiles) by nanmean."""
    # 1. UTC -> local solar time (15 deg longitude = 1 h), then cut to the date
    solar_day = (ds["time"] + np.timedelta64(int(lon / 15 * 3600), "s")).dt.floor("D")
    # 2. average all items of one day; a site outside one tile is NaN there and
    #    simply takes the value of the other tile
    fused = ds.groupby(solar_day.rename("day")).mean(skipna=True)
    # 3. restore the `time` dim name expected downstream
    return fused.rename(day="time")
