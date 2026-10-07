from datetime import UTC, datetime

import pystac
from shapely.geometry import box, mapping

from sites_cube.stac import (
    build_filter,
    cloud_cover_property,
    dedupe_items,
    items_table,
)

DT = datetime(2023, 6, 1, 10, 20, 31, tzinfo=UTC)


def _item(id_: str, geom, **props) -> pystac.Item:
    return pystac.Item(
        id=id_,
        geometry=mapping(geom),
        bbox=list(geom.bounds),
        datetime=DT,
        properties={"platform": "sentinel-2a", **props},
    )


def _ids(items: pystac.ItemCollection) -> set[str]:
    return {i.id for i in items}


def test_reprocessed_duplicate_keeps_newest_baseline() -> None:
    tile = box(9, 50, 10, 51)
    items = [
        _item(
            "old", tile, **{"s2:mgrs_tile": "32UPU", "s2:processing_baseline": "04.00"}
        ),
        _item(
            "new", tile, **{"s2:mgrs_tile": "32UPU", "s2:processing_baseline": "05.10"}
        ),
    ]
    assert _ids(dedupe_items(pystac.ItemCollection(items))) == {"new"}


def test_baseline_compared_numerically() -> None:
    tile = box(9, 50, 10, 51)
    items = [
        _item(
            "a", tile, **{"s2:mgrs_tile": "32UPU", "s2:processing_baseline": "05.09"}
        ),
        _item(
            "b", tile, **{"s2:mgrs_tile": "32UPU", "s2:processing_baseline": "05.10"}
        ),
    ]
    assert _ids(dedupe_items(pystac.ItemCollection(items))) == {"b"}


def test_split_datastrip_halves_are_both_kept() -> None:
    # two halves of the same tile with a small overlap, same baseline
    props = {"s2:mgrs_tile": "32UPU", "s2:processing_baseline": "05.10"}
    items = [
        _item("north", box(9, 50.45, 10, 51), **props),
        _item("south", box(9, 50, 10, 50.55), **props),
    ]
    assert _ids(dedupe_items(pystac.ItemCollection(items))) == {"north", "south"}


def test_grid_code_tiles_are_not_merged() -> None:
    # Earth Search style: no s2:mgrs_tile, tile only in grid:code
    items = [
        _item("upu", box(9, 50, 10, 51), **{"grid:code": "MGRS-32UPU"}),
        _item("uqu", box(10, 50, 11, 51), **{"grid:code": "MGRS-32UQU"}),
    ]
    unique = dedupe_items(pystac.ItemCollection(items))
    assert _ids(unique) == {"upu", "uqu"}
    assert set(items_table(unique, lon=10.0)["tile"]) == {"32UPU", "32UQU"}


def test_filter_s2_cloud_only() -> None:
    assert build_filter("sentinel-2-l2a", 30, None) == {
        "op": "<",
        "args": [{"property": "eo:cloud_cover"}, 30],
    }


def test_filter_landsat_uses_land_cloud_cover_and_platforms() -> None:
    assert build_filter("landsat-c2-l2", 30, ["landsat-8", "landsat-9"]) == {
        "op": "and",
        "args": [
            {"op": "<", "args": [{"property": "landsat:cloud_cover_land"}, 30]},
            {
                "op": "in",
                "args": [{"property": "platform"}, ["landsat-8", "landsat-9"]],
            },
        ],
    }


def test_filter_without_filters_is_none() -> None:
    assert build_filter("landsat-c2-l2", None, None) is None
    assert build_filter("landsat-c2-l2", None, []) is None


def test_cdse_reprocessed_duplicate_keeps_newest_processing() -> None:
    # CDSE: baseline in processing:version, generation time in processing:datetime
    tile = box(9, 50, 10, 51)

    def cdse(id_: str, version: str, generated: str) -> pystac.Item:
        return _item(
            id_,
            tile,
            **{
                "constellation": "sentinel-2",
                "grid:code": "MGRS-32UNC",
                "processing:version": version,
                "processing:datetime": generated,
            },
        )

    items = [
        # id order would pick "z_old", so the properties must decide
        cdse("z_old", "05.10", "2024-09-10T11:17:18Z"),
        cdse("a_new", "05.10", "2024-09-26T22:09:31Z"),
        cdse("z_older_baseline", "05.09", "2024-12-01T00:00:00Z"),
    ]
    unique = dedupe_items(pystac.ItemCollection(items))
    assert _ids(unique) == {"a_new"}
    assert items_table(unique, lon=10.0)["processing_baseline"].tolist() == ["05.10"]


def test_processing_version_of_non_s2_is_no_baseline() -> None:
    item = _item("ls", box(9, 50, 10, 51), **{"processing:version": "02.00"})
    assert (
        items_table(pystac.ItemCollection([item]), lon=10.0)["processing_baseline"]
        .isna()
        .all()
    )


def test_unknown_collection_falls_back_to_eo_cloud_cover() -> None:
    assert cloud_cover_property("some-other-collection") == "eo:cloud_cover"


def test_landsat_wrs_tiles_are_not_merged() -> None:
    def ls(id_: str, geom, row: str) -> pystac.Item:
        item = _item(
            id_,
            geom,
            **{
                "landsat:wrs_path": "194",
                "landsat:wrs_row": row,
                "eo:cloud_cover": 50.0,
                "landsat:cloud_cover_land": 10.0,
            },
        )
        item.properties["platform"] = "landsat-8"
        return item

    items = [ls("r24", box(8, 51, 11, 53), "024"), ls("r25", box(8, 50, 11, 52), "025")]
    unique = dedupe_items(pystac.ItemCollection(items))
    assert _ids(unique) == {"r24", "r25"}
    table = items_table(unique, lon=10.0, cloud_property="landsat:cloud_cover_land")
    assert set(table["tile"]) == {"194024", "194025"}
    assert (table["cloud_cover"] == 10.0).all()
