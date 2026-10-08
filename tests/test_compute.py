import numpy as np
import pytest
import rasterio
from omegaconf import OmegaConf
from rasterio.errors import RasterioIOError

from sites_cube.cli import gdal_options
from sites_cube.compute import retry_reads


def _failing(n_failures: int, exc: Exception):
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] <= n_failures:
            raise exc
        return "data"

    return fn, calls


def test_retry_reads_succeeds_after_failures() -> None:
    fn, calls = _failing(2, RasterioIOError("Read failed"))
    assert retry_reads(fn, attempts=3) == "data"
    assert calls["n"] == 3


def test_retry_reads_reraises_after_last_attempt() -> None:
    fn, calls = _failing(3, RasterioIOError("Read failed"))
    with pytest.raises(RasterioIOError):
        retry_reads(fn, attempts=3)
    assert calls["n"] == 3


def test_retry_reads_ignores_unrelated_errors() -> None:
    fn, calls = _failing(1, ValueError("bug"))
    with pytest.raises(ValueError):
        retry_reads(fn, attempts=3)
    assert calls["n"] == 1


def test_truncated_jp2_read_raises(tmp_path) -> None:
    """Multi-threaded JP2 decoding turns a truncated stream into zero pixels."""
    path = tmp_path / "tiles.jp2"
    data = np.random.default_rng(0).integers(1, 10000, (512, 512), dtype="uint16")
    profile = {"driver": "JP2OpenJPEG", "width": 512, "height": 512, "count": 1}
    profile |= {"dtype": "uint16", "blockxsize": 128, "blockysize": 128}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    truncated = f"/vsisubfile/0_{path.stat().st_size // 2},{path}"
    opts = gdal_options(OmegaConf.create({"stac": {"gdal": None}}))
    with (
        rasterio.Env(**opts),
        rasterio.open(truncated) as src,
        pytest.raises(RasterioIOError),
    ):
        src.read(1)
