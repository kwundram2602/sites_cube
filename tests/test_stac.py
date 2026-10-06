from datetime import UTC, datetime

import pystac
from shapely.geometry import box, mapping

from sites_cube.stac import dedupe_items, items_table

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
