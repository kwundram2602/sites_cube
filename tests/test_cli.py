import pytest
from omegaconf import OmegaConf

from sites_cube.cli import check_config


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
