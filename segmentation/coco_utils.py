import copy
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image

import os

from pycocotools import mask as coco_mask

from transforms import Compose


class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def get_coco(root, image_set, transforms):
    PATHS = {
        "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]

    transforms = Compose([
        FilterAndRemapCocoCategories(CAT_LIST, remap=True),
        ConvertCocoPolysToMask(),
        transforms
    ])

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset

coco_classes = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 
 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 
 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 
 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 
 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 
 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 
 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
  59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 
  67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 
  77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 
  84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

def return_class_names(class_ids=[]):
    if not class_ids:
        return coco_classes
    else:
        return {id:coco_classes[id] for id in class_ids}

def random_n_classes(num_classes):
    catIds = list(coco_classes.keys())
    if num_classes == 81:
        catIds.append(0)
        return catIds
    
    np.random.seed(1)
    # with replacements...
    indices = np.random.choice(80, num_classes-1, replace=False).tolist()
    n_classes = [catIds[i] for i in indices]
    n_classes.append(0)
    return n_classes