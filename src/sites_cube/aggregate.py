from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REDUCERS = {"median", "mean", "min", "max", "std", "first", "last"}
# item labels: per item (time,), comma-joined per site after fusing/aggregating
TILE = "tile"
BASELINE_LABEL = "processing_baseline"
LABELS = (TILE, BASELINE_LABEL)
# scene properties of every item, per-item tables only (dropped on fusing)
ITEM_ATTRS = (
    "item_id",
    "orbit",
    "cloud_cover",
    "sun_elevation",
    "sun_azimuth",
    "view_incidence",
)
# per-item metadata, written before the bands
METADATA = ("platform", *LABELS, *ITEM_ATTRS)


def aggregate_time(ds: xr.Dataset, freq: str, reducer: str) -> xr.Dataset:
    """Resample to `freq` with `reducer`; adds `n_obs`, the number of valid observations.

    An observation counts only if all bands are valid, so every band of a period
    is reduced from the same set of dates.
    """
    if reducer not in REDUCERS:
        raise ValueError(f"reducer {reducer!r} not in {sorted(REDUCERS)}")
    labelled = ds
    ds = ds.drop_vars([*LABELS, *ITEM_ATTRS], errors="ignore")
    valid = ds.to_dataarray("band").notnull().all("band")
    masked = ds.where(valid)
    reduced = getattr(masked.resample(time=freq), reducer)(skipna=True)
    # periods without any time step come back as NaN; casting that to int would
    # yield INT32_MIN, so they get 0 like periods with only invalid observations
    n_obs = valid.resample(time=freq).sum().fillna(0).astype("int32")
    reduced = reduced.assign(n_obs=n_obs)
    if "site" not in ds.dims or not any(c in labelled.coords for c in LABELS):
        return reduced
    # period label of every time step, as assigned by the resampling above
    times = pd.DatetimeIndex(ds["time"].to_numpy())
    period = np.empty(len(times), dtype="datetime64[ns]")
    for label, pos in (
        pd.Series(range(len(times)), index=times).resample(freq).apply(list).items()
    ):
        period[pos] = label
    labels = join_labels(labelled, valid, xr.DataArray(period, dims="time"), "time")
    return reduced.assign_coords(
        labels.reindex(time=reduced["time"], fill_value="").data_vars
    )


def join_labels(
    ds: xr.Dataset, valid: xr.DataArray, key: xr.DataArray, dim: str
) -> xr.Dataset:
    """Distinct `LABELS` of the valid observations per group and site, comma-joined.

    `key` holds the group of every time step (e.g. solar day or period), `valid`
    whether a (time, site) observation counts. Labels may already be comma lists
    (fused days), they are split and merged again. Groups without a valid
    observation at a site get "". Returns one (`dim`, site) variable per label.
    """
    sites = valid["site"].to_numpy()
    keys = np.asarray(key)
    groups = pd.Index(pd.unique(keys), name=dim)
    t, s = np.nonzero(valid.transpose("time", "site").to_numpy())
    grid = pd.MultiIndex.from_product([groups, sites], names=[dim, "site"])
    out = {}
    for c in (c for c in LABELS if c in ds.coords):
        label = ds[c].broadcast_like(valid).transpose("time", "site").to_numpy()
        joined = (
            pd.DataFrame({dim: keys[t], "site": sites[s], c: label[t, s]})
            .groupby([dim, "site"])[c]
            .agg(_join)
            .reindex(grid, fill_value="")
        )
        out[c] = ((dim, "site"), joined.to_numpy().reshape(len(groups), len(sites)))
    return xr.Dataset(out, coords={dim: groups, "site": sites})


def _join(values: pd.Series) -> str:
    return ",".join(sorted({p for v in values for p in str(v).split(",") if p}))


def to_long(ds: xr.Dataset) -> pd.DataFrame:
    """Long table: site, time, band, value, n_obs (n_obs only if present)."""
    # 1. spectral bands only, the `n_obs` count is handled separately
    bands = [v for v in ds.data_vars if v != "n_obs"]
    # per-item data carries platform, tile and baseline, kept as columns
    ids = ["site", "time"] + [c for c in METADATA if c in ds.coords]
    df = (
        ds[bands]
        # 2. (time, site) grid -> one row per time/site, one column per band
        .to_dataframe()
        # 3. move the `time` and `site` index levels into regular columns
        .reset_index()
        # 4. wide -> long: band columns become rows with a `band` and a `value` column
        .melt(id_vars=ids, var_name="band")
    )
    # 5. the count exists only after `aggregate_time`, not for daily data
    if "n_obs" in ds:
        # 6. one count per site and period, attached to every band row
        n = ds["n_obs"].to_dataframe().reset_index()
        df = df.merge(n[["site", "time", "n_obs"]], on=["site", "time"])
    # 7. one contiguous time series per site and band
    return df.sort_values(["site", "band", "time"]).reset_index(drop=True)


def write_per_site(
    ds: xr.Dataset, out_dir: Path, decimals: int = 4, dropna: bool = False
) -> list[Path]:
    """One wide CSV per site: `time` rows, metadata, one column per band (+ `n_obs`).

    `dropna` drops rows without any value, e.g. per-item data where the site lies
    outside the item footprint.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for site in ds["site"].to_numpy():
        df = ds.sel(site=site).drop_vars("site").to_dataframe()
        df = _table(df, ds, decimals, dropna)
        # file names must not contain path separators
        path = out_dir / f"{str(site).replace('/', '_')}.csv"
        df.to_csv(path, index_label="time")
        paths.append(path)
    return paths


def write_merged(
    ds: xr.Dataset, path: Path, decimals: int = 4, dropna: bool = False
) -> Path:
    """All sites in one wide CSV: `site`, `time`, metadata, bands, see `write_per_site`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _table(ds.to_dataframe().reset_index("site"), ds, decimals, dropna)
    # stable sort keeps the item order of tiles sharing a sensing time
    df = df.rename_axis("time").reset_index().sort_values("site", kind="stable")
    df = df[["site", "time", *(c for c in df.columns if c not in ("site", "time"))]]
    df.to_csv(path, index=False)
    return path


def _table(
    df: pd.DataFrame, ds: xr.Dataset, decimals: int, dropna: bool
) -> pd.DataFrame:
    """Round, drop empty rows, order the columns: metadata, rest, bands."""
    bands = [str(v) for v in ds.data_vars]
    df = df.round(decimals)
    if dropna:
        df = df.dropna(how="all", subset=bands)
    meta = [c for c in METADATA if c in df.columns]
    rest = [c for c in df.columns if c not in meta and c not in bands]
    return df[rest + meta + bands]
