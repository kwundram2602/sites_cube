from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG = Path("config/sentinel2.yaml")


def load_config(argv: list[str] | None = None) -> DictConfig:
    """Load the YAML config and merge `key=value` CLI overrides on top.

    A `config=<path>` override selects a different base file.
    """
    cli = OmegaConf.from_cli(argv or [])
    path = Path(cli.pop("config", DEFAULT_CONFIG))
    cfg = OmegaConf.merge(OmegaConf.load(path), cli)
    assert isinstance(cfg, DictConfig)
    return cfg
