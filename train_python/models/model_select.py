import sys

from libs.models.classical_seg import Deeplab, LRASPP

sys.path.insert(0, '..')


class SelectTransforModel:
    def __init__(self, config):

        self.model_dict = {
            'deeplab': Deeplab(numClasses=config.classes, weights='COCO_WITH_VOC_LABELS_V1'),
            'LRASPP': LRASPP(numClasses=config.classes, weights='COCO_WITH_VOC_LABELS_V1'),
        }

    def __call__(self, trans_models):

        model = self.model_dict[trans_models]

        return model
