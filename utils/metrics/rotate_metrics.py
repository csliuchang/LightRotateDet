from shapely.geometry import Polygon



def combine_predicts_gt(predicts, gt):
    pass















class RotateDetEval(object):
    def __init__(self):
        pass

    def val_batch_measure(self, batch, output, box_thresh):
        """
        Evaluate val datasets with batch
        """



    def __call__(self, collection, box_thresh=0.4):
        return self.val_batch_measure(batch, output, box_thresh)