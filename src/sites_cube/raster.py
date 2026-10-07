from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox


def write_periods(ds: xr.Dataset, geobox: GeoBox, out_dir: Path) -> list[Path]:
    """One GeoTIFF per time step: spectral bands, then `n_obs`, as float32.

    `ds` is a reduced (time, y, x) cube on `geobox`; periods without any valid
    observation are skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [v for v in ds.data_vars if v != "n_obs"] + ["n_obs"]
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
    for t in ds["time"].to_numpy():
        step = ds.sel(time=t)
        if int(step["n_obs"].max()) == 0:
            continue
        # (band, y, x) in the band order of `names`
        data = np.stack(
            [step[n].transpose("y", "x").to_numpy().astype("float32") for n in names]
        )
        path = out_dir / f"{pd.Timestamp(t):%Y-%m-%d}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
            dst.descriptions = tuple(names)
        paths.append(path)
    return paths
