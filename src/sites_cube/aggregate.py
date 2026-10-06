from pathlib import Path

import pandas as pd
import xarray as xr

REDUCERS = {"median", "mean", "min", "max", "std", "first", "last"}


def aggregate_time(ds: xr.Dataset, freq: str, reducer: str) -> xr.Dataset:
    """Resample to `freq` with `reducer`; adds `<band>_n_obs` valid-observation counts."""
    if reducer not in REDUCERS:
        raise ValueError(f"reducer {reducer!r} not in {sorted(REDUCERS)}")
    resampled = ds.resample(time=freq)
    reduced = getattr(resampled, reducer)(skipna=True)
    counts = resampled.count().rename({v: f"{v}_n_obs" for v in ds.data_vars})
    return xr.merge([reduced, counts])


def to_long(ds: xr.Dataset) -> pd.DataFrame:
    """Long table: site, time, band, value, n_obs (n_obs only if present)."""
    # 1. spectral bands only, the `<band>_n_obs` count variables are handled separately
    bands = [v for v in ds.data_vars if not str(v).endswith("_n_obs")]
    df = (
        ds[bands]
        # 2. (time, site) grid -> one row per time/site, one column per band
        .to_dataframe()
        # 3. move the `time` and `site` index levels into regular columns
        .reset_index()
        # 4. wide -> long: band columns become rows with a `band` and a `value` column
        .melt(id_vars=["site", "time"], var_name="band")
    )
    # 5. count variables exist only after `aggregate_time`, not for daily data
    n_obs_vars = [f"{b}_n_obs" for b in bands if f"{b}_n_obs" in ds]
    if n_obs_vars:
        n = (
            ds[n_obs_vars]
            # 6. `red_n_obs` -> `red`, so the band names match the value table
            .rename({f"{b}_n_obs": b for b in bands if f"{b}_n_obs" in ds})
            # 7. same reshaping as above, the count goes into column `n_obs`
            .to_dataframe()
            .reset_index()
            .melt(id_vars=["site", "time"], var_name="band", value_name="n_obs")
        )
        # 8. attach each value's observation count
        df = df.merge(n, on=["site", "time", "band"])
    # 9. one contiguous time series per site and band
    return df.sort_values(["site", "band", "time"]).reset_index(drop=True)


def write_per_site(ds: xr.Dataset, out_dir: Path, decimals: int = 4) -> list[Path]:
    """One wide CSV per site: `time` rows, one column per band (+ `<band>_n_obs`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for site in ds["site"].to_numpy():
        df = ds.sel(site=site).drop_vars("site").to_dataframe().round(decimals)
        # file names must not contain path separators
        path = out_dir / f"{str(site).replace('/', '_')}.csv"
        df.to_csv(path, index_label="time")
        paths.append(path)
    return paths
