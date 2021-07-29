import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from pipelines import Compose

__all__ = "BaseDataset"


class BaseDataset(Dataset):
    """
    A base datasets for rotate detection
    """

    def __init__(self, data_root, pipeline, stage='train'):
        self.stage = stage
        self.img_ids = None
        self.data_root = data_root
        if self.stage == "train":
            self.ann_file = os.path.join(data_root, 'txt_train.txt')
        elif self.stage == 'val':
            self.ann_file = os.path.join(data_root, 'txt_val.txt')
        else:
            self.ann_file = os.path.join(data_root, 'txt_test.txt')
        self.data_infos = self.load_annotations(self.ann_file)
        self.load_pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        if self.stage == 'test':
            return self.prepare_test_img(index)
        while True:
            data = self.prepare_train_img(index)
            if data is None:
                index = self._rand_another(index)
                continue
            return data

    def get_ann_info(self, index):
        return self.data_infos[index]['ann']

    def load_image(self, index):
        img_pre_path = self.data_infos[index]['filename']
        img_all_path = os.path.join(self.data_root, img_pre_path)
        return cv2.imread(img_all_path, cv2.IMREAD_UNCHANGED)

    def load_annotations(self, ann_file):
        raise NotImplementedError

    def prepare_train_img(self, index):
        img_info = self.load_image[index]
        ann_info = self.get_ann_info[index]
        results = dict(img_info=img_info, ann_info=ann_info)
        return self.load_pipeline(results)

    def prepare_test_img(self, index):
        img_info = self.load_image[index]
        ann_info = self.get_ann_info[index]
        results = dict(img_info=img_info, ann_info=ann_info)
        return results

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, index):
        pool = np.where(self.flag == self.flag[index])[0]
        return np.random.choice(pool)


