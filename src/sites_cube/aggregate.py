from pathlib import Path

import pandas as pd
import xarray as xr

REDUCERS = {"median", "mean", "min", "max", "std", "first", "last"}


def aggregate_time(ds: xr.Dataset, freq: str, reducer: str) -> xr.Dataset:
    """Resample to `freq` with `reducer`; adds `n_obs`, the number of valid observations.

    An observation counts only if all bands are valid, so every band of a period
    is reduced from the same set of dates.
    """
    if reducer not in REDUCERS:
        raise ValueError(f"reducer {reducer!r} not in {sorted(REDUCERS)}")
    valid = ds.to_dataarray("band").notnull().all("band")
    masked = ds.where(valid)
    reduced = getattr(masked.resample(time=freq), reducer)(skipna=True)
    # periods without any time step come back as NaN; casting that to int would
    # yield INT32_MIN, so they get 0 like periods with only invalid observations
    n_obs = valid.resample(time=freq).sum().fillna(0).astype("int32")
    return reduced.assign(n_obs=n_obs)


def to_long(ds: xr.Dataset) -> pd.DataFrame:
    """Long table: site, time, band, value, n_obs (n_obs only if present)."""
    # 1. spectral bands only, the `n_obs` count is handled separately
    bands = [v for v in ds.data_vars if v != "n_obs"]
    df = (
        ds[bands]
        # 2. (time, site) grid -> one row per time/site, one column per band
        .to_dataframe()
        # 3. move the `time` and `site` index levels into regular columns
        .reset_index()
        # 4. wide -> long: band columns become rows with a `band` and a `value` column
        .melt(id_vars=["site", "time"], var_name="band")
    )
    # 5. the count exists only after `aggregate_time`, not for daily data
    if "n_obs" in ds:
        # 6. one count per site and period, attached to every band row
        n = ds["n_obs"].to_dataframe().reset_index()
        df = df.merge(n[["site", "time", "n_obs"]], on=["site", "time"])
    # 7. one contiguous time series per site and band
    return df.sort_values(["site", "band", "time"]).reset_index(drop=True)


def write_per_site(ds: xr.Dataset, out_dir: Path, decimals: int = 4) -> list[Path]:
    """One wide CSV per site: `time` rows, one column per band (+ `n_obs`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for site in ds["site"].to_numpy():
        df = ds.sel(site=site).drop_vars("site").to_dataframe().round(decimals)
        # file names must not contain path separators
        path = out_dir / f"{str(site).replace('/', '_')}.csv"
        df.to_csv(path, index_label="time")
        paths.append(path)
    return paths
