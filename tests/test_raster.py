import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox

from sites_cube.raster import write_periods


def test_one_tif_per_period_with_data(tmp_path) -> None:
    geobox = GeoBox.from_bbox(
        (500000, 6860000, 500090, 6860060), "EPSG:32632", resolution=30
    )
    ny, nx = geobox.shape
    red = np.full((2, ny, nx), 0.1, dtype="float32")
    red[1] = np.nan  # second period: no observation at all
    red[0, 0, 0] = 0.25
    n_obs = np.zeros((2, ny, nx), dtype="int32")
    n_obs[0] = 3
    ds = xr.Dataset(
        {"red": (("time", "y", "x"), red), "n_obs": (("time", "y", "x"), n_obs)},
        coords={"time": pd.to_datetime(["2023-06-01", "2023-07-01"])},
    )

    paths = write_periods(ds, geobox, tmp_path)

    assert [p.name for p in paths] == ["2023-06-01.tif"]
    with rasterio.open(paths[0]) as src:
        assert src.count == 2
        assert src.descriptions == ("red", "n_obs")
        assert src.crs.to_epsg() == 32632
        assert src.transform == geobox.affine
        np.testing.assert_allclose(src.read(1), red[0])
        np.testing.assert_array_equal(src.read(2), 3)
