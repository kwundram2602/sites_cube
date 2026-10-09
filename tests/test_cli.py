from datetime import UTC, datetime

import pystac
import pytest
from omegaconf import OmegaConf

from sites_cube.cli import check_assets, check_config, gdal_options


def _item(id_: str, assets: list[str]) -> pystac.Item:
    item = pystac.Item(
        id=id_,
        geometry=None,
        bbox=None,
        datetime=datetime(2016, 5, 24, tzinfo=UTC),
        properties={},
    )
    for key in assets:
        item.add_asset(key, pystac.Asset(href=f"s3://x/{id_}/{key}.jp2"))
    return item


def test_quality_requires_aggregation_off() -> None:
    cfg = OmegaConf.create({"quality": {"scl": "SCL"}, "aggregate": {"enabled": True}})
    with pytest.raises(ValueError, match="quality"):
        check_config(cfg)


def test_quality_without_aggregation_is_valid() -> None:
    check_config(
        OmegaConf.create({"quality": {"scl": "SCL"}, "aggregate": {"enabled": False}})
    )
    check_config(OmegaConf.create({"quality": None, "aggregate": {"enabled": True}}))
    # configs without the new keys keep working: aggregation on, no quality
    check_config(OmegaConf.create({"aggregate": {"freq": "1MS"}}))


def test_quality_alias_without_asset_key() -> None:
    cfg = OmegaConf.create(
        {"quality": {"scl:SCL": None}, "aggregate": {"enabled": False}}
    )
    with pytest.raises(ValueError, match="asset key"):
        check_config(cfg)


def test_gdal_options_decode_single_threaded_by_default() -> None:
    cfg = OmegaConf.create(
        {"stac": {"gdal": {"AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu"}}}
    )
    opts = gdal_options(cfg)
    assert opts["GDAL_NUM_THREADS"] == "1"
    assert opts["AWS_S3_ENDPOINT"] == "eodata.dataspace.copernicus.eu"
    assert gdal_options(OmegaConf.create({"stac": {"gdal": None}})) == {
        "GDAL_NUM_THREADS": "1"
    }


def test_gdal_options_config_overrides_default() -> None:
    cfg = OmegaConf.create({"stac": {"gdal": {"GDAL_NUM_THREADS": "ALL_CPUS"}}})
    assert gdal_options(cfg)["GDAL_NUM_THREADS"] == "ALL_CPUS"


def test_check_assets_drops_quality_missing_in_some_items(capsys) -> None:
    items = pystac.ItemCollection(
        [
            _item("full", ["B08_10m", "SCL_20m", "CLD_20m"]),
            _item("no_cld", ["B08_10m", "SCL_20m"]),
        ]
    )
    quality = check_assets(
        items, {"nir": "B08_10m"}, {"scl": "SCL_20m", "cld": "CLD_20m"}
    )
    assert quality == {"scl": "SCL_20m"}
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "cld" in out and "CLD_20m" in out
    assert "no_cld" in out


def test_check_assets_keeps_complete_quality_silently(capsys) -> None:
    items = pystac.ItemCollection([_item("a", ["B08_10m", "SCL_20m"])])
    assert check_assets(items, {"nir": "B08_10m"}, {"scl": "SCL_20m"}) == {
        "scl": "SCL_20m"
    }
    assert capsys.readouterr().out == ""


def test_check_assets_missing_spectral_band_raises() -> None:
    items = pystac.ItemCollection([_item("a", ["B08_10m"]), _item("b", [])])
    with pytest.raises(ValueError, match="B08_10m.*b"):
        check_assets(items, {"nir": "B08_10m"}, {})
