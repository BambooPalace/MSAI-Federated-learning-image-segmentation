#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import torch
from torchvision import datasets

from transforms import Compose
from train import get_transform
from coco_utils import ConvertCocoPolysToMask, FilterAndRemapCocoCategories, random_n_classes
from sampling import *

import numpy as np
from PIL import Image


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'pedestrain':
        # 170 images with 2 classes
        path2data = os.path.join(args.root, 'data/PennFudanPed')
        train_dataset = PennFudanDataset(path2data, get_transform(train=True))
        test_dataset = PennFudanDataset(path2data, get_transform(train=False))
        torch.manual_seed(args.seed)
        split_idx=len(train_dataset)//5 * 4
        idxs = torch.randperm(len(train_dataset)).tolist()
        # torch.save(idxs, 'idxs.pt')#check if same idxs for different runs: YES
        train_dataset = torch.utils.data.Subset(train_dataset, idxs[:split_idx])
        test_dataset = torch.utils.data.Subset(test_dataset, idxs[split_idx:])
        
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = coco_iid(train_dataset, args.num_users)
        else:
            raise AttributeError('current iid options are not available for pedestrain dataset')        

    elif args.dataset == 'coco':
        if args.data == 'val2017':
            path2data = os.path.join(args.root, 'data/coco/val2017')
            path2ann = os.path.join(args.root, 'data/coco/annotations/instances_val2017.json')
        elif args.data == 'train2017':
            path2data = os.path.join(args.root, 'data/coco/train2017')
            path2ann = os.path.join(args.root, 'data/coco/annotations/instances_train2017.json')        
    
        # path2data = r"C:\Users\cgong002\Google Drive\data\coco\val2017" #local
        # path2ann = r"C:\Users\cgong002\Google Drive\data\coco\annotations\instances_val2017.json" #local
        
        if args.num_classes == 81:
            catIds = random_n_classes(args.num_classes)
        elif args.num_classes == 21:
            catIds = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        augmentation = False if args.dp else True
        train_dataset = datasets.CocoDetection(path2data, path2ann, transforms=Compose([FilterAndRemapCocoCategories(catIds, remap=True), 
                                                                                    ConvertCocoPolysToMask(),
                                                                                    get_transform(train=augmentation)]))
        test_dataset = datasets.CocoDetection(path2data, path2ann, transforms=Compose([FilterAndRemapCocoCategories(catIds, remap=True), 
                                                                            ConvertCocoPolysToMask(),
                                                                            get_transform(train=False)]))
         # split train and test indice
        torch.manual_seed(args.seed)
        split_idx=len(train_dataset)//5 * 4
        idxs = torch.randperm(len(train_dataset)).tolist()
        # torch.save(idxs, 'idxs.pt')#check if same idxs for different runs: YES
        train_dataset = torch.utils.data.Subset(train_dataset, idxs[:split_idx])
        test_dataset = torch.utils.data.Subset(test_dataset, idxs[split_idx:])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = coco_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = coco_noniid_unequal(train_dataset, args.num_users, data=args.data)
            else:
                # Chose euqal splits for every user
                user_groups = coco_noniid(train_dataset, args.num_users, data=args.data)

    else:
        exit('Unrecognized dataset')

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    No of classes     : {args.num_classes}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
        if args.unequal:
            print(' Unequal')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


