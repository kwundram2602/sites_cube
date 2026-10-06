import sys
from pathlib import Path

import odc.stac

from sites_cube import aggregate, extract, stac
from sites_cube.compute import dask_context
from sites_cube.config import load_config
from sites_cube.sites import read_sites


def main() -> None:
    cfg = load_config(sys.argv[1:])
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.collection

    sites = read_sites(cfg.sites.path, cfg.sites.layer, cfg.sites.id_field)
    lon = sites.to_crs(4326).union_all().centroid.x
    client = stac.open_client(cfg.stac.url, cfg.stac.sign)
    print(
        f"{len(sites)} sites, {cfg.stac.url} / {name}, {cfg.time.start} .. {cfg.time.end}"
    )

    # collection size with and without the scene cloud filter, for comparison with GEE
    all_items = stac.search_items(
        client, name, sites, cfg.time.start, cfg.time.end, None
    )
    report, _ = stac.count_report(stac.items_table(all_items, lon))
    print(f"\n== all items, no cloud filter (raw catalogue) ==\n{report}")
    unique = stac.dedupe_items(all_items)
    report, _ = stac.count_report(stac.items_table(unique, lon))
    print(f"\n== all items, no cloud filter, newest processing only ==\n{report}")

    items = unique
    if cfg.max_cloud_cover is not None:
        items = stac.dedupe_items(
            stac.search_items(
                client, name, sites, cfg.time.start, cfg.time.end, cfg.max_cloud_cover
            )
        )
    table = stac.items_table(items, lon)
    report, by_month = stac.count_report(table)
    print(
        f"\n== eo:cloud_cover < {cfg.max_cloud_cover}, newest processing only ==\n{report}"
    )
    table.to_csv(out_dir / f"items_{name}.csv", index=False)
    by_month.to_csv(out_dir / f"item_counts_{name}.csv", index=False)

    if cfg.count_only:
        return

    print(f"\nloading {len(items)} items at {len(sites)} sites ...")
    # GDAL cloud settings; odc captures them into the task graph, so workers get them too
    odc.stac.configure_rio(cloud_defaults=True)
    with dask_context(cfg.dask):
        raw = extract.load_points(
            items,
            dict(cfg.bands),
            sites,
            cfg.load.crs,
            cfg.load.resolution,
            cfg.load.buffer_m,
            dict(cfg.load.chunks),
        )
    refl = extract.to_reflectance(
        raw, cfg.nodata, cfg.scale, cfg.offset, cfg.harmonize_s2_offset
    )
    daily = extract.fuse_solar_day(refl, lon)
    aggregate.to_long(daily).to_parquet(out_dir / f"raw_{name}.parquet", index=False)

    reducer = cfg.aggregate.reducer
    monthly = aggregate.aggregate_time(daily, cfg.aggregate.freq, reducer)
    site_dir = out_dir / name / f"monthly_{reducer}"
    paths = aggregate.write_per_site(monthly, site_dir)
    print(f"wrote {len(paths)} site CSVs to {site_dir}")
