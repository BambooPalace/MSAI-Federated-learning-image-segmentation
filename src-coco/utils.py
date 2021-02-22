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



def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'coco':
        # path2data = os.path.join(args.root, 'data/coco/val2017')
        # path2ann = os.path.join(args.root, 'data/coco/annotations/instances_val2017.json')
        path2data = r"C:\Users\cgong002\Google Drive\data\coco\val2017" #use single quotes seems to cause error
        path2ann = r"C:\Users\cgong002\Google Drive\data\coco\annotations\instances_val2017.json"
        
        if args.num_classes == 81:
        catIds = random_n_classes(args.num_classes)
        elif args.num_classes == 21:
            catIds = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
        train_dataset = datasets.CocoDetection(path2data, path2ann, transforms=Compose([FilterAndRemapCocoCategories(catIds, remap=True), 
                                                                                    ConvertCocoPolysToMask(),
                                                                                    get_transform(train=True)]))
        test_dataset = datasets.CocoDetection(path2data, path2ann, transforms=Compose([FilterAndRemapCocoCategories(catIds, remap=True), 
                                                                            ConvertCocoPolysToMask(),
                                                                            get_transform(train=False)]))
        # split train and test indice
        torch.manual_seed(args.seed)
        split_idx=4000
        idxs = torch.randperm(len(train_dataset)).tolist()
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
                user_groups = coco_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = coco_noniid(train_dataset, args.num_users)

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
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


