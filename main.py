import argparse
from utils import Config
import os
from models import build_detector
from trainer.train import Train
from datasets import build_dataset
import time
import torch
import random
import numpy as np
import logging
import os.path as osp
import copy


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--config', default='./config/rretinanet/train_retinanet_msra.json', help='train config file path')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    args = parse_args()
    cfg = Config.fromjson(args.config)
    # add network params
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # set random seeds
    meta = dict()
    if args.seed is not None:
        logger = logging.getLogger(__name__)
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['time_str'] = timestamp
    # build_dataset
    cfg.dataset.update({"stage": "train"})
    datasets = [build_dataset(cfg.dataset)]
    cfg.dataset.pop("stage")
    if len(cfg.workflow) == 2:
        cfg.dataset.update({"stage": "val"})
        val_dataset = copy.deepcopy(cfg.dataset)
        datasets.append(build_dataset(val_dataset))
    # build model
    cfg.model.backbone.in_channels = cfg.input_channel
    cfg.model.backbone.input_size = (cfg.input_width, cfg.input_height)
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # build datasets
    trainer = Train(
        cfg,
        datasets,
        model,
        meta,
        distributed=False
    )
    trainer.run()


if __name__ == "__main__":
    import sys
    import pathlib
    main()