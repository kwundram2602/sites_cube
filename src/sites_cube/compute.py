from collections.abc import Iterator
from contextlib import contextmanager

import dask
from dask.distributed import Client, LocalCluster
from omegaconf import DictConfig


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
