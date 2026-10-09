from collections import defaultdict
from datetime import timedelta

import geopandas as gpd
import pandas as pd
import planetary_computer
import pystac
import pystac_client
from pystac_client.stac_api_io import StacApiIO
from shapely.geometry import mapping, shape
from urllib3 import Retry

SIGNERS = {"planetary_computer": planetary_computer.sign_inplace}
# retry rate limits (CDSE answers 429) and transient server errors with
# exponential backoff, honouring Retry-After; POST searches are retried too
RETRY = Retry(
    total=8,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=None,
    respect_retry_after_header=True,
)
# items per search page, fewer requests than the server default
PAGE_SIZE = 100
# share of an item's footprint not covered by newer items of the same scene
# above which it counts as a separate datastrip part, not a reprocessing
MIN_NEW_AREA = 0.05
# columns of `items_table`, also for searches without any item
ITEM_COLUMNS = [
    "id",
    "datetime",
    "platform",
    "tile",
    "orbit",
    "cloud_cover",
    "processing_baseline",
]
# scene cloud cover property per collection, `max_cloud_cover` is applied to it
CLOUD_COVER_PROPERTY = {
    "sentinel-2-l2a": "eo:cloud_cover",
    # land-only share, so coastal scenes are not dropped for clouds over the sea
    "landsat-c2-l2": "landsat:cloud_cover_land",
}


def cloud_cover_property(collection: str) -> str:
    return CLOUD_COVER_PROPERTY.get(collection, "eo:cloud_cover")


def open_client(url: str, sign: str | None) -> pystac_client.Client:
    modifier = SIGNERS[sign] if sign else None
    return pystac_client.Client.open(
        url, modifier=modifier, stac_io=StacApiIO(max_retries=RETRY)
    )


def build_filter(
    collection: str, max_cloud_cover: float | None, platforms: list[str] | None
) -> dict | None:
    """CQL2-JSON filter for scene cloud cover and platforms, None if unfiltered.

    CQL2 instead of the `query` extension: CDSE rejects `query` with `in`.
    """
    args = []
    if max_cloud_cover is not None:
        args.append(
            {
                "op": "<",
                "args": [
                    {"property": cloud_cover_property(collection)},
                    max_cloud_cover,
                ],
            }
        )
    if platforms:
        args.append({"op": "in", "args": [{"property": "platform"}, list(platforms)]})
    if not args:
        return None
    return args[0] if len(args) == 1 else {"op": "and", "args": args}


def search_items(
    client: pystac_client.Client,
    collection: str,
    sites: gpd.GeoDataFrame,
    start: str,
    end: str,
    max_cloud_cover: float | None,
    platforms: list[str] | None = None,
) -> pystac.ItemCollection:
    """Search all items intersecting the convex hull of the sites."""
    aoi = sites.to_crs(4326).union_all().convex_hull
    search = client.search(
        collections=[collection],
        intersects=mapping(aoi),
        datetime=f"{start}/{end}",
        filter=build_filter(collection, max_cloud_cover, platforms),
        filter_lang="cql2-json",
        limit=PAGE_SIZE,
    )
    return search.item_collection()


def dedupe_items(items: pystac.ItemCollection) -> pystac.ItemCollection:
    """Drop older processings of a scene, but keep split datastrip parts.

    Some catalogues (e.g. Planetary Computer S2) hold original and reprocessed
    versions of the same acquisition side by side. Others split one datastrip
    into two products, so the same tile appears twice with the same time but
    different footprints. Within each (platform, time, tile) group items are
    taken newest first and kept only if they add footprint area.
    """
    groups: dict[tuple, list[pystac.Item]] = defaultdict(list)
    for item in items:
        groups[(item.properties.get("platform"), item.datetime, tile(item))].append(
            item
        )
    keep = []
    for group in groups.values():
        covered = None
        for item in sorted(group, key=_rank, reverse=True):
            geom = shape(item.geometry)
            if covered is None:
                covered = geom
            elif geom.difference(covered).area > MIN_NEW_AREA * geom.area:
                covered = covered.union(geom)
            else:
                continue
            keep.append(item)
    return pystac.ItemCollection(keep)


def _rank(item: pystac.Item) -> tuple[float, str, str]:
    p = item.properties
    generated = p.get("s2:generation_time") or p.get("processing:datetime") or ""
    return (float(baseline(p) or 0), generated, item.id)


def baseline(p: dict) -> str | None:
    """S2 processing baseline, PC/Earth Search (`s2:`) or CDSE (`processing:version`)."""
    if "s2:processing_baseline" in p:
        return p["s2:processing_baseline"]
    if p.get("constellation") == "sentinel-2":
        return p.get("processing:version")
    return None


def tile(item: pystac.Item) -> str | None:
    """S2 MGRS tile or Landsat path/row, from the STAC extension or `grid:code`."""
    p = item.properties
    if "s2:mgrs_tile" in p:
        return p["s2:mgrs_tile"]
    if "grid:code" in p:  # e.g. Earth Search: "MGRS-32UPU"
        return p["grid:code"].split("-", 1)[-1]
    if "landsat:wrs_path" in p:
        return p["landsat:wrs_path"] + p.get("landsat:wrs_row", "")
    return None


def items_table(
    items: pystac.ItemCollection, lon: float, cloud_property: str = "eo:cloud_cover"
) -> pd.DataFrame:
    """One row per item with the metadata relevant for completeness checks.

    `solar_day` shifts UTC by the site longitude, so overlapping tiles of the
    same overpass fall on the same day. `cloud_cover` is read from
    `cloud_property`, the property the scene filter uses.
    """
    rows = []
    for item in items:
        p = item.properties
        rows.append(
            {
                "id": item.id,
                "datetime": item.datetime,
                "platform": p.get("platform"),
                "tile": tile(item),
                "orbit": p.get("sat:relative_orbit"),
                "cloud_cover": p.get(cloud_property),
                "processing_baseline": baseline(p),
            }
        )
    df = pd.DataFrame(rows, columns=ITEM_COLUMNS)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    df.insert(2, "solar_day", _solar_day(pd.DatetimeIndex(df["datetime"]), lon))
    return df.sort_values("datetime").reset_index(drop=True)


def count_report(table: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Summarise collection size: totals, duplicates, per tile, per year/month."""
    if table.empty:
        by_month = pd.DataFrame(columns=["year", "month", "n_items", "n_dates"])
        return "items total:            0", by_month
    dup = table.duplicated(["solar_day", "tile", "platform"], keep=False)
    by_month = (
        table.assign(year=table["solar_day"].dt.year, month=table["solar_day"].dt.month)
        .groupby(["year", "month"])
        .agg(n_items=("id", "size"), n_dates=("solar_day", "nunique"))
        .reset_index()
    )
    by_year = by_month.groupby("year")[["n_items", "n_dates"]].sum()
    lines = [
        f"items total:            {len(table)}",
        f"unique acquisition days:{table['solar_day'].nunique():>6}",
        f"items sharing day+tile+platform (duplicates): {int(dup.sum())}",
        f"date range:             {table['datetime'].min():%Y-%m-%d} .. {table['datetime'].max():%Y-%m-%d}",
        "items per tile:",
        table["tile"].value_counts().sort_index().to_string(),
    ]
    if table["processing_baseline"].notna().any():
        lines += [
            "items per processing baseline:",
            table["processing_baseline"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string(),
        ]
    lines += [
        "items / days per year:",
        by_year.to_string(),
    ]
    return "\n".join(lines), by_month


def baselines_by_time(items: pystac.ItemCollection) -> pd.Series:
    """S2 processing baseline per item, indexed by naive UTC datetime; empty for non-S2."""
    pairs = [
        (pd.Timestamp(i.datetime).tz_convert(None), b)
        for i in items
        if i.datetime is not None and (b := baseline(i.properties)) is not None
    ]
    return pd.Series(
        [b for _, b in pairs],
        index=pd.DatetimeIndex([t for t, _ in pairs]),
        dtype="object",
    )


def join_baselines(
    baselines: pd.Series, freq: str | None = None, lon: float = 0.0
) -> pd.Series:
    """Distinct baselines per time step, comma-joined, e.g. "04.00,05.10".

    Without `freq` per item datetime (tiles can share one). With `freq` per
    period of the solar days, labelled like `aggregate.aggregate_time`.
    """
    if baselines.empty:
        return pd.Series(dtype="object")
    if freq is None:
        key = baselines.index
    else:
        days = pd.Series(
            baselines.to_numpy(),
            index=_solar_day(pd.DatetimeIndex(baselines.index), lon),
        )
        baselines, key = days, pd.Grouper(freq=freq)
    joined = baselines.groupby(key).agg(lambda s: ",".join(sorted(set(s))))
    # periods without any item come back as empty string from resampling
    return joined[joined != ""]


def _solar_day(times: pd.DatetimeIndex, lon: float) -> pd.DatetimeIndex:
    """Local solar date of naive UTC times, shifting by the site longitude."""
    return (times + timedelta(hours=lon / 15)).normalize()
