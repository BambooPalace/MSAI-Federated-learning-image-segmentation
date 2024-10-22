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


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'coco':
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
        elif args.num_classes == 2:
            catIds = [0,1]
        augmentation = False if args.dp else True
        train_dataset = datasets.CocoDetection(path2data, path2ann, transforms=Compose([FilterAndRemapCocoCategories(catIds, remap=True), 
                                                                                    ConvertCocoPolysToMask(),
                                                                                    get_transform(train=augmentation)]))
        test_dataset = datasets.CocoDetection(path2data, path2ann, transforms=Compose([FilterAndRemapCocoCategories(catIds, remap=True), 
                                                                            ConvertCocoPolysToMask(),
                                                                            get_transform(train=False)]))          
         # split train and test indice
        n = len(train_dataset)
        torch.manual_seed(args.seed)             
        idxs = torch.randperm(n).tolist()
        idxs = idxs[:int(n*args.sample_rate)]
        split_idx= len(idxs)//5 * 4         
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
                user_groups = coco_noniid_unequal(train_dataset, args.num_users, args.data)
            else:
                # Chose euqal splits for every user
                user_groups = coco_noniid(train_dataset, args.num_users, args.data)

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


