from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox


def write_periods(
    ds: xr.Dataset,
    geobox: GeoBox,
    out_dir: Path,
    baselines: pd.Series | None = None,
) -> list[Path]:
    """One GeoTIFF per time step: data variables, then `n_obs` if present, as float32.

    `ds` is a (time, y, x) cube on `geobox`, either reduced per period (with
    `n_obs`, file `YYYY-MM-DD.tif`, periods without valid observation skipped) or
    one slice per item (without `n_obs`, file named by sensing time, slices
    without any value skipped).

    `baselines` maps a time step to its S2 processing baseline(s) (see
    `stac.join_baselines`), written as metadata tag `PROCESSING_BASELINE`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_item = "n_obs" not in ds
    names = [str(v) for v in ds.data_vars if v != "n_obs"]
    if not per_item:
        names.append("n_obs")
    profile = {
        "driver": "GTiff",
        "height": geobox.height,
        "width": geobox.width,
        "count": len(names),
        "dtype": "float32",
        "crs": str(geobox.crs),
        "transform": geobox.affine,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
    }
    paths = []
    # by position: per-item times can repeat (tiles with the same sensing time)
    for i, t in enumerate(ds["time"].to_numpy()):
        step = ds.isel(time=i)
        # (band, y, x) in the band order of `names`
        data = np.stack(
            [step[n].transpose("y", "x").to_numpy().astype("float32") for n in names]
        )
        if per_item:
            if np.isnan(data).all():
                continue
            stem = f"{pd.Timestamp(t):%Y-%m-%dT%H%M%S}"
            path = _free_path(out_dir, stem, paths)
        else:
            if int(step["n_obs"].max()) == 0:
                continue
            path = out_dir / f"{pd.Timestamp(t):%Y-%m-%d}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
            dst.descriptions = tuple(names)
            if baselines is not None and (b := baselines.get(pd.Timestamp(t))):
                dst.update_tags(PROCESSING_BASELINE=b)
        paths.append(path)
    return paths


def _free_path(out_dir: Path, stem: str, taken: list[Path]) -> Path:
    """`stem.tif`, or `stem_1.tif`, `stem_2.tif`, ... if written in this run already.

    Files of earlier runs are overwritten, not counted.
    """
    path = out_dir / f"{stem}.tif"
    n = 0
    while path in taken:
        n += 1
        path = out_dir / f"{stem}_{n}.tif"
    return path
