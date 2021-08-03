from .builder import DATASETS
from torch.utils.data import Dataset
import numpy as np
import os
import bisect
import cv2
from .pipelines import Compose
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

__all__ = "BaseDataset"


class BaseDataset(Dataset):
    """
    A _base datasets for rotate detection
    """

    def __init__(self, data_root, pipeline, train_file=None, val_file=None, test_mode=False, stage='train'):
        self.stage = stage
        self.img_ids = None
        self.test_mode = test_mode
        self.data_root = data_root
        if self.stage == "train":
            self.ann_file = os.path.join(data_root, train_file)
        elif self.stage == 'val':
            self.ann_file = os.path.join(data_root, val_file)
        else:
            self.ann_file = os.path.join(data_root, 'txt_test.txt')
        self.data_infos = self.load_annotations(self.ann_file)
        self.load_pipeline = Compose(pipeline)
        if not self.test_mode:
            self._set_group_flag()

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
        img_info, filename = cv2.imread(img_all_path, cv2.IMREAD_UNCHANGED), img_pre_path.split('/')[-1]
        ori_image_shape = img_info.shape[:2]
        return img_info, filename, ori_image_shape

    def load_annotations(self, ann_file):
        raise NotImplementedError

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)


    def prepare_train_img(self, index):
        img_info, filename, ori_image_shape = self.load_image(index)
        ann_info = self.get_ann_info(index)
        results = dict(filename=filename, img_info=img_info, ann_info=ann_info, ori_image_shape=ori_image_shape)
        return self.load_pipeline(results)

    def prepare_test_img(self, index):
        img_info = self.load_image(index)
        ann_info = self.get_ann_info(index)
        results = dict(img_info=img_info, ann_info=ann_info)
        return results


    def _rand_another(self, index):
        pool = np.where(self.flag == self.flag[index])[0]
        return np.random.choice(pool)



@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def get_cat_ids(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)



