import cv2
import numpy as np

from ..builder import PIPELINES
from engine.parallel import DataContainer as DC
import torch


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class RResize(object):
    """
        Resize images & rotated bbox
        Inherit Resize pipeline class to handle rotated bboxes
    """
    def __init__(self, img_scale):
        self.scale = img_scale
        self.resize_height, self.resize_width = self.scale

    def _resize_img(self, results):

        image = results['img_info']
        image = cv2.resize(image, [self.resize_width, self.resize_height], interpolation=cv2.INTER_LINEAR)
        results['img_info'] = image
        results['image_shape'] = [self.resize_width, self.resize_height]

    def _resize_bboxes(self, results):
        original_height, original_width = results['ori_image_shape']
        bboxes = results['ann_info']['bboxes']
        width_ratio = float(self.resize_width) / original_width
        height_ratio = float(self.resize_height) / original_height
        new_bbox = []
        for bbox in bboxes:
            bbox[0] = int(bbox[0] * width_ratio)
            bbox[2] = int(bbox[2] * width_ratio)
            bbox[4] = int(bbox[4] * width_ratio)
            bbox[6] = int(bbox[6] * width_ratio)
            bbox[1] = int(bbox[1] * height_ratio)
            bbox[3] = int(bbox[3] * height_ratio)
            bbox[5] = int(bbox[5] * height_ratio)
            bbox[7] = int(bbox[7] * height_ratio)
            new_bbox.append(bbox)
        new_bbox = np.array(new_bbox, dtype=np.float32)
        results['ann_info']['bboxes'] = new_bbox

    def __call__(self, results):
        self._resize_img(results)
        self._resize_bboxes(results)
        return results



@PIPELINES.register_module()
class Collect(object):
    """
    Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_image_shape', 'image_shape'),
                 fields=(dict(key='img', stack=True), dict(key='gt_bboxes'),
                         dict(key='gt_labels'))):
        self.fields = fields
        self.keys = keys
        self.meta_keys = meta_keys


    def __call__(self, results):
        data = {}
        img_meta = {}
        img = np.ascontiguousarray(results['img_info'].transpose(2, 0, 1)).astype(np.float32)
        results['img'] = to_tensor(img)
        results['gt_bboxes'] = np.array(results['ann_info']['bboxes'], dtype=np.float32)
        results['gt_labels'] = np.array(results['ann_info']['labels'], dtype=np.int64)
        results['gt_masks'] = np.ones(shape=results['image_shape'], dtype=np.uint8) * 0.
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'
