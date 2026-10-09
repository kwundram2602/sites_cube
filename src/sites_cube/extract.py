import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import pystac
import rasterio
import xarray as xr
from rasterio.warp import transform_bounds

from sites_cube.landsat import QA_VARS

S2_BASELINE_OFFSET_DN = 1000
S2_BASELINE = (
    "s2_processing_baseline"  # odc.stac name of property s2:processing_baseline
)
PLATFORM = "platform"
# raster bounds files, everything else is read as vector
RASTER_SUFFIXES = {".tif", ".tiff", ".vrt"}

Bbox = tuple[float, float, float, float]


def window_bbox(
    value: Sequence[float] | str | None, crs: str, resolution: float
) -> Bbox | None:
    """Fixed load window in `crs` from `load.bbox`, None = derive it from the sites.

    An array `[xmin, ymin, xmax, ymax]` must lie on the `resolution` grid, so the
    window matches the source pixels exactly. A bounds file (vector or raster) is
    reprojected and snapped outward to the grid, hand-drawn extents rarely align.
    """
    if value is None:
        return None
    if isinstance(value, str):
        bounds = _file_bounds(Path(value), crs)
        return (
            math.floor(bounds[0] / resolution) * resolution,
            math.floor(bounds[1] / resolution) * resolution,
            math.ceil(bounds[2] / resolution) * resolution,
            math.ceil(bounds[3] / resolution) * resolution,
        )
    xmin, ymin, xmax, ymax = (float(v) for v in value)
    off_grid = [v for v in (xmin, ymin, xmax, ymax) if v % resolution]
    if off_grid:
        raise ValueError(
            f"load.bbox {off_grid} not a multiple of the resolution {resolution}"
        )
    return xmin, ymin, xmax, ymax


def _file_bounds(path: Path, crs: str) -> Bbox:
    if path.suffix.lower() in RASTER_SUFFIXES:
        with rasterio.open(path) as src:
            return transform_bounds(src.crs, crs, *src.bounds)
    xmin, ymin, xmax, ymax = gpd.read_file(path).to_crs(crs).total_bounds
    return xmin, ymin, xmax, ymax


def sites_in_bbox(sites: gpd.GeoDataFrame, crs: str, bbox: Bbox) -> gpd.GeoDataFrame:
    """Sites inside `bbox` (in `crs`); outside ones would sample the border pixel."""
    pts = sites.to_crs(crs)
    xmin, ymin, xmax, ymax = bbox
    inside = pts.geometry.x.between(xmin, xmax) & pts.geometry.y.between(ymin, ymax)
    return sites[inside.to_numpy()]


def load_points(
    items: pystac.ItemCollection,
    bands: dict[str, str],
    sites: gpd.GeoDataFrame,
    crs: str,
    resolution: float,
    buffer_m: float,
    chunks: dict[str, int | Literal["auto"]],
    bbox: Bbox | None = None,
) -> xr.Dataset:
    """Load every item separately over the window and sample the site pixels.

    Items are not fused here so per-item corrections (S2 baseline offset) can be
    applied before overlapping tiles are merged.
    """
    cube = load_cube(items, bands, sites, crs, resolution, buffer_m, chunks, bbox)
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
    bbox: Bbox | None = None,
) -> xr.Dataset:
    """Lazy (time, y, x) cube over `bbox` or the buffered sites bbox, one slice per item."""
    # 1. load window in the target grid CRS: fixed bbox (on the pixel grid, no
    #    buffer) or the buffered extent of the sites
    if bbox is None:
        xmin, ymin, xmax, ymax = sites.to_crs(crs).total_bounds
        xmin, ymin = xmin - buffer_m, ymin - buffer_m
        xmax, ymax = xmax + buffer_m, ymax + buffer_m
    else:
        xmin, ymin, xmax, ymax = bbox
    baseline = _baseline_property(items)
    # 2. lazy (dask) cube over the sites bbox; nothing is read from the COGs yet
    ds = odc.stac.load(
        items,
        bands=list(bands.values()),  # asset keys, e.g. B04
        groupby="id",  # one time slice per item, no fusing of overlapping tiles
        crs=crs,
        resolution=resolution,
        # the sites buffer keeps edge sites away from the window border
        x=(xmin, xmax),
        y=(ymin, ymax),
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
    passthrough: Sequence[str] = (),
) -> xr.Dataset:
    """Mask nodata, remove S2 baseline offset, apply scale/offset.

    Landsat QA variables are passed through unchanged for `landsat.mask_landsat`.
    `passthrough` variables (quality layers such as SCL or cloud probability) are
    neither scaled nor nodata-masked; they become float32 and NaN only where all
    spectral bands are nodata, i.e. outside the item footprint.
    """
    # 1. spectral bands only, the baseline, QA and quality variables are metadata
    skip = {S2_BASELINE, *QA_VARS, *passthrough}
    bands = [v for v in ds.data_vars if v not in skip]
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
    out = out.assign({v: ds[v] for v in QA_VARS if v in ds and v not in passthrough})
    # 6. quality layers: 0 can be a valid value (0 % cloud), so mask by footprint
    footprint = (ds[bands] != nodata).to_dataarray("band").any("band")
    return out.assign(
        {v: ds[v].astype("float32").where(footprint) for v in passthrough}
    )


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
