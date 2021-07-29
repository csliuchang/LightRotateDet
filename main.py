from tools.train import TrainerBase
import argparse
from utils import Config
import os
from models import build_detector
from trainer.trainer import Trainer
from datasets import build_dataset, build_dataloader


def main(cfg):
    # build model
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # build datasets
    datasets = build_dataset(cfg.data_root)
    train_loader = build_dataloader()
    trainer = Trainer(
        cfg,
        model,
        datasets,
        train_loader=train_loader,
    )



if __name__ == "__main__":
    import sys
    import pathlib
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='./config/rretinanet/rretinanet_yolov5s.json', help='train config file path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    main(cfg)