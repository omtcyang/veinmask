from pycocotools.coco import COCO

from .registry import DATASETS
from .coco_seg import Coco_Seg_Dataset


@DATASETS.register_module
class SBDDataset(Coco_Seg_Dataset):
    CLASSES = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def load_annotations(self, ann_file):
        print(ann_file)
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)

        self.count = {0: 0,
                      1: 96 * 96,
                      2: 96 * 96 + 48 * 48,
                      3: 96 * 96 + 48 * 48 + 24 * 24,
                      4: 96 * 96 + 48 * 48 + 24 * 24 + 12 * 12}
        return img_infos

    def get_ceiling(self, pid):
        if pid < 9216:
            return 0
        elif pid < 11520:
            return 1
        elif pid < 12096:
            return 2
        elif pid < 12240:
            return 3
        return 4
