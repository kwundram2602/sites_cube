from typing import Literal

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import pystac
import xarray as xr

from sites_cube.landsat import QA_VARS

S2_BASELINE_OFFSET_DN = 1000
S2_BASELINE = (
    "s2_processing_baseline"  # odc.stac name of property s2:processing_baseline
)
PLATFORM = "platform"


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
    cube = load_cube(items, bands, sites, crs, resolution, buffer_m, chunks)
    # trigger the actual reads, result is a small (time, site) dataset
    return sample_points(cube, sites, crs).compute()


def load_cube(
    items: pystac.ItemCollection,
    bands: dict[str, str],
    sites: gpd.GeoDataFrame,
    crs: str,
    resolution: float,
    buffer_m: float,
    chunks: dict[str, int | Literal["auto"]],
) -> xr.Dataset:
    """Lazy (time, y, x) cube over the sites bbox, one time slice per item."""
    # 1. sites into the target grid CRS, their extent defines the load window
    xmin, ymin, xmax, ymax = sites.to_crs(crs).total_bounds
    baseline = _baseline_property(items)
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
        # per-item baseline as variable (time,), needed for the S2 offset correction;
        # float so "05.10" compares numerically against 4.0
        with_properties=[{"key": baseline, "dtype": "float32"}] if baseline else None,
    )
    # 3. asset keys -> config aliases (B04 -> red), baseline -> fixed name
    names = {v: k for k, v in bands.items()}
    if baseline:
        names[baseline.replace(":", "_")] = S2_BASELINE
    # 4. platform of every time slice, e.g. to pick the Landsat sensor
    return ds.rename(names).assign_coords(
        {PLATFORM: ("time", platforms_by_time(items, ds["time"].to_numpy()))}
    )


def platforms_by_time(items: pystac.ItemCollection, times: np.ndarray) -> list[str]:
    """Platform of each time slice, matched via the item datetime (one item per slice).

    Not loaded via odc `with_properties`, which only yields numeric variables.
    """
    by_time = {
        pd.Timestamp(i.datetime).tz_convert(None): i.properties["platform"]
        for i in items
        if i.datetime is not None
    }
    return [by_time[pd.Timestamp(t)] for t in times]


def sample_points(ds: xr.Dataset, sites: gpd.GeoDataFrame, crs: str) -> xr.Dataset:
    """Nearest pixel of every site, (time, y, x) -> (time, site)."""
    pts = sites.to_crs(crs)
    # 1. vectorised point sampling: x/y share the new dim `site`, so each site
    #    gets its own nearest pixel (not the full x*y cross product)
    sel = ds.sel(
        x=xr.DataArray(pts.geometry.x.to_numpy(), dims="site"),
        y=xr.DataArray(pts.geometry.y.to_numpy(), dims="site"),
        method="nearest",
    )
    return (
        # 2. label the site dim with the site ids
        sel.assign_coords(site=pts.index.to_numpy())
        # 3. pixel coords are no longer needed per site
        .drop_vars(["x", "y", "spatial_ref"], errors="ignore")
    )


def _baseline_property(items: pystac.ItemCollection) -> str | None:
    """Item property holding the S2 processing baseline, None for non-S2 items.

    PC/Earth Search use `s2:processing_baseline`, CDSE `processing:version`;
    requesting a property the items lack would fail.
    """
    for i in items:
        p = i.properties
        if "s2:processing_baseline" in p:
            return "s2:processing_baseline"
        if p.get("constellation") == "sentinel-2" and "processing:version" in p:
            return "processing:version"
    return None


def to_reflectance(
    ds: xr.Dataset,
    nodata: float,
    scale: float,
    offset: float,
    harmonize_s2_offset: bool,
) -> xr.Dataset:
    """Mask nodata, remove S2 baseline offset, apply scale/offset.

    Landsat QA variables are passed through unchanged for `landsat.mask_landsat`.
    """
    # 1. spectral bands only, the baseline and QA variables are metadata
    bands = [v for v in ds.data_vars if v != S2_BASELINE and v not in QA_VARS]
    # 2. float so NaN can mark nodata pixels (e.g. outside the tile footprint)
    out = ds[bands].astype("float32").where(ds[bands] != nodata)
    if harmonize_s2_offset and S2_BASELINE in ds:
        # 3. baseline >= 04.00 stores DN + 1000; subtract per time step
        #    (baseline is (time,), broadcast over sites)
        out = out - xr.where(ds[S2_BASELINE] >= 4.0, S2_BASELINE_OFFSET_DN, 0)
        # 4. values at/below the offset are invalid after harmonisation
        out = out.where(out > 0)
    # 5. DN -> surface reflectance
    out = out * scale + offset
    return out.assign({v: ds[v] for v in QA_VARS if v in ds})


def fuse_solar_day(ds: xr.Dataset, lon: float) -> xr.Dataset:
    """Merge items of the same solar day (overlapping tiles) by nanmean."""
    # 1. UTC -> local solar time (15 deg longitude = 1 h), then cut to the date
    solar_day = (ds["time"] + np.timedelta64(int(lon / 15 * 3600), "s")).dt.floor("D")
    # 2. platform is a string per item and has no mean, drop it
    ds = ds.drop_vars(PLATFORM, errors="ignore")
    # 3. average all items of one day; a site outside one tile is NaN there and
    #    simply takes the value of the other tile
    fused = ds.groupby(solar_day.rename("day")).mean(skipna=True)
    # 4. restore the `time` dim name expected downstream
    return fused.rename(day="time")
