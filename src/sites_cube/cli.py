import os
import sys
from pathlib import Path
from typing import Any

import odc.stac

from sites_cube import aggregate, extract, raster, stac
from sites_cube.compute import dask_context
from sites_cube.config import load_config
from sites_cube.sites import read_sites


def main() -> None:
    cfg = load_config(sys.argv[1:])
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.collection
    platforms = list(cfg.platforms) if cfg.get("platforms") else None
    cloud_property = stac.cloud_cover_property(name)

    sites = read_sites(cfg.sites.path, cfg.sites.layer, cfg.sites.id_field)
    lon = sites.to_crs(4326).union_all().centroid.x
    client = stac.open_client(cfg.stac.url, cfg.stac.sign)
    print(
        f"{len(sites)} sites, {cfg.stac.url} / {name}, {cfg.time.start} .. {cfg.time.end}"
        + (f", platforms {platforms}" if platforms else "")
    )

    # collection size with and without the scene cloud filter, for comparison with GEE
    all_items = stac.search_items(
        client, name, sites, cfg.time.start, cfg.time.end, None, platforms
    )
    report, _ = stac.count_report(stac.items_table(all_items, lon, cloud_property))
    print(f"\n== all items, no cloud filter (raw catalogue) ==\n{report}")
    unique = stac.dedupe_items(all_items)
    report, _ = stac.count_report(stac.items_table(unique, lon, cloud_property))
    print(f"\n== all items, no cloud filter, newest processing only ==\n{report}")

    items = unique
    if cfg.max_cloud_cover is not None:
        items = stac.dedupe_items(
            stac.search_items(
                client,
                name,
                sites,
                cfg.time.start,
                cfg.time.end,
                cfg.max_cloud_cover,
                platforms,
            )
        )
    table = stac.items_table(items, lon, cloud_property)
    report, by_month = stac.count_report(table)
    print(
        f"\n== {cloud_property} < {cfg.max_cloud_cover}, newest processing only ==\n{report}"
    )
    table.to_csv(out_dir / f"items_{name}.csv", index=False)
    by_month.to_csv(out_dir / f"item_counts_{name}.csv", index=False)

    if cfg.count_only:
        return

    print(f"\nloading {len(items)} items at {len(sites)} sites ...")
    # GDAL cloud settings plus per-catalogue options (e.g. CDSE S3 endpoint);
    # odc captures them into the task graph, so workers get them too
    gdal_opts: dict[str, Any] = {
        k: str(v) for k, v in (cfg.stac.get("gdal") or {}).items()
    }
    odc.stac.configure_rio(cloud_defaults=True, **gdal_opts)
    # the rasterio env is thread-local and some worker threads open files without
    # it (CDSE reads then went to AWS); GDAL falls back to process env vars, which
    # the LocalCluster workers inherit
    os.environ.update(gdal_opts)
    load_args = (
        items,
        dict(cfg.bands),
        sites,
        cfg.load.crs,
        cfg.load.resolution,
        cfg.load.buffer_m,
        dict(cfg.load.chunks),
    )
    write_raster = cfg.get("raster", {}).get("enabled", False)
    with dask_context(cfg.dask):
        if write_raster:
            # read the whole window once; the points are sampled from memory
            cube = extract.load_cube(*load_args).compute()
            raw = extract.sample_points(cube, sites, cfg.load.crs)
        else:
            raw = extract.load_points(*load_args)
    refl = extract.to_reflectance(
        raw, cfg.nodata, cfg.scale, cfg.offset, cfg.harmonize_s2_offset
    )
    daily = extract.fuse_solar_day(refl, lon)
    aggregate.to_long(daily).to_parquet(out_dir / f"raw_{name}.parquet", index=False)

    reducer = cfg.aggregate.reducer
    freq = cfg.aggregate.freq
    reduced = aggregate.aggregate_time(daily, freq, reducer)
    site_dir = out_dir / name / f"sites_{reducer}_{freq}"
    paths = aggregate.write_per_site(reduced, site_dir)
    print(f"wrote {len(paths)} site CSVs to {site_dir}")

    if write_raster:
        # same chain as the points: reflectance -> solar-day mean -> period reducer
        refl_cube = extract.to_reflectance(
            cube, cfg.nodata, cfg.scale, cfg.offset, cfg.harmonize_s2_offset
        )
        periods = aggregate.aggregate_time(
            extract.fuse_solar_day(refl_cube, lon), freq, reducer
        )
        raster_dir = out_dir / name / f"raster_{reducer}_{freq}"
        tifs = raster.write_periods(periods, cube.odc.geobox, raster_dir)
        print(f"wrote {len(tifs)} GeoTIFFs to {raster_dir}")
