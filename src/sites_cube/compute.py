from collections.abc import Callable, Iterator
from contextlib import contextmanager

import dask
from dask.distributed import Client, LocalCluster
from omegaconf import DictConfig
from rasterio.errors import RasterioIOError, WarpOperationError


@contextmanager
def dask_context(cfg: DictConfig) -> Iterator[None]:
    """Run the enclosed `.compute()` calls on a LocalCluster or the threaded scheduler."""
    if not cfg.distributed:
        with dask.config.set(scheduler="threads", num_workers=cfg.num_threads):
            yield
        return
    with (
        LocalCluster(
            n_workers=cfg.n_workers,
            threads_per_worker=cfg.threads_per_worker,
            memory_limit=cfg.memory_limit,
        ) as cluster,
        Client(cluster) as client,
    ):
        print(f"dask dashboard: {client.dashboard_link}")
        yield


def retry_reads[T](fn: Callable[[], T], attempts: int) -> T:
    """Call `fn`, repeating it after failed raster reads (e.g. truncated downloads).

    A retry recomputes everything `fn` reads, fine for the small site windows.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (RasterioIOError, WarpOperationError) as e:
            if attempt == attempts:
                raise
            print(f"read failed ({e}), retry {attempt}/{attempts - 1} ...")
    raise ValueError("attempts must be >= 1")
