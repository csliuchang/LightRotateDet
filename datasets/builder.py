from utils import Registry, build_from_cfg, get_dist_info
from samper import GroupSampler
from functools import partial
import numpy as np
import random
from torch.utils.data import DataLoader

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):
    """Build Pytorch Dataloader.


    Returns:
        Dataloader: A Pytorch dataloader
    """
    rank, world_size = get_dist_info()
    if dist:
        # add DDP
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu
        pass
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,

    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs
    )

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate(batch, samples_per_gpu=1):
    """
    A batch collator that does nothing.
    """
    return batch
