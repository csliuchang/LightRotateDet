from tools.trainer import TrainerBase
import argparse
from utils import Config
import os
from models import build_detector



class Train(TrainerBase):
    def __init__(self, cfg):
        super(Train, self).__init__(cfg)
        # build model
        self.model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)


    def _train_epoch(self):
        pass


    def _val_epoch(self):
        pass





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
    Train(cfg)