from collections import defaultdict
from datetime import timedelta

import geopandas as gpd
import pandas as pd
import planetary_computer
import pystac
import pystac_client
from shapely.geometry import mapping, shape

SIGNERS = {"planetary_computer": planetary_computer.sign_inplace}
# share of an item's footprint not covered by newer items of the same scene
# above which it counts as a separate datastrip part, not a reprocessing
MIN_NEW_AREA = 0.05


def open_client(url: str, sign: str | None) -> pystac_client.Client:
    modifier = SIGNERS[sign] if sign else None
    return pystac_client.Client.open(url, modifier=modifier)


def search_items(
    client: pystac_client.Client,
    collection: str,
    sites: gpd.GeoDataFrame,
    start: str,
    end: str,
    max_cloud_cover: float | None,
) -> pystac.ItemCollection:
    """Search all items intersecting the convex hull of the sites."""
    aoi = sites.to_crs(4326).union_all().convex_hull
    query = (
        {"eo:cloud_cover": {"lt": max_cloud_cover}}
        if max_cloud_cover is not None
        else None
    )
    search = client.search(
        collections=[collection],
        intersects=mapping(aoi),
        datetime=f"{start}/{end}",
        query=query,
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
        groups[(item.properties.get("platform"), item.datetime, _tile(item))].append(
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
    return (
        float(p.get("s2:processing_baseline") or 0),
        p.get("s2:generation_time", ""),
        item.id,
    )


def _tile(item: pystac.Item) -> str | None:
    """S2 MGRS tile or Landsat path/row, from the STAC extension or `grid:code`."""
    p = item.properties
    if "s2:mgrs_tile" in p:
        return p["s2:mgrs_tile"]
    if "grid:code" in p:  # e.g. Earth Search: "MGRS-32UPU"
        return p["grid:code"].split("-", 1)[-1]
    if "landsat:wrs_path" in p:
        return p["landsat:wrs_path"] + p.get("landsat:wrs_row", "")
    return None


def items_table(items: pystac.ItemCollection, lon: float) -> pd.DataFrame:
    """One row per item with the metadata relevant for completeness checks.

    `solar_day` shifts UTC by the site longitude, so overlapping tiles of the
    same overpass fall on the same day.
    """
    rows = []
    for item in items:
        p = item.properties
        rows.append(
            {
                "id": item.id,
                "datetime": item.datetime,
                "platform": p.get("platform"),
                "tile": _tile(item),
                "orbit": p.get("sat:relative_orbit"),
                "cloud_cover": p.get("eo:cloud_cover"),
                "processing_baseline": p.get("s2:processing_baseline"),
            }
        )
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert(None)
    df.insert(
        2, "solar_day", (df["datetime"] + timedelta(hours=lon / 15)).dt.normalize()
    )
    return df.sort_values("datetime").reset_index(drop=True)


def count_report(table: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Summarise collection size: totals, duplicates, per tile, per year/month."""
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
        "items / days per year:",
        by_year.to_string(),
    ]
    return "\n".join(lines), by_month
