from ..builder import PIPELINES



@PIPELINES.register_module()
class RResize(object):
    """
        Resize images & rotated bbox
        Inherit Resize pipeline class to handle rotated bboxes
    """
    def __init__(self, ):
        pass


    def _resize_bboxes(self, results):
        pass

