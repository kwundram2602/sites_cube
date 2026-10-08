import pytest
from omegaconf import OmegaConf

from sites_cube.cli import check_config, gdal_options


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
