from datetime import timedelta

import geopandas as gpd
import pandas as pd
import planetary_computer
import pystac
import pystac_client
from shapely.geometry import mapping

SIGNERS = {"planetary_computer": planetary_computer.sign_inplace}


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
    """Keep only the newest processing of each scene (same platform, time and tile).

    Some catalogues (e.g. Planetary Computer S2) hold original and reprocessed
    versions of the same acquisition side by side.
    """
    newest: dict[tuple, pystac.Item] = {}
    for item in items:
        p = item.properties
        key = (
            p.get("platform"),
            p["datetime"],
            p.get("s2:mgrs_tile"),
            p.get("landsat:wrs_path"),
            p.get("landsat:wrs_row"),
        )
        rank = (
            p.get("s2:processing_baseline", ""),
            p.get("s2:generation_time", ""),
            item.id,
        )
        if key not in newest or rank > _rank(newest[key]):
            newest[key] = item
    return pystac.ItemCollection(newest.values())


def _rank(item: pystac.Item) -> tuple[str, str, str]:
    p = item.properties
    return (
        p.get("s2:processing_baseline", ""),
        p.get("s2:generation_time", ""),
        item.id,
    )


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
                "datetime": p["datetime"],
                "platform": p.get("platform"),
                "tile": p.get("s2:mgrs_tile")
                or p.get("landsat:wrs_path", "") + p.get("landsat:wrs_row", ""),
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
