#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import time
import numpy as np
import torch
from torchvision import datasets, transforms


def convert_coco_mask_to_top_class(dataset):
    # return the numpy array of top class of each img
    targets = [] 
    for (_, target) in dataset:
        classes = np.unique(target.numpy()) # a sorted array
         # remove background class 0, 255
        if len(classes) and classes[0] == 0:
            classes = classes[1:]
        if len(classes) and classes[-1] == 255:
            classes = classes[:-1]

        if len(classes) == 0:
            targets.append(0)
        elif len(classes) == 1:
            targets.append(classes[0])
        else:
            pixels_per_class = []
            for c in classes:
                pixels = len(np.where(target==c)[0])
                pixels_per_class.append(pixels)
            # get the top class with most pixels
            top_class = classes[np.argmax(pixels_per_class)]
            targets.append(top_class) 
    return np.array(targets)
        
def coco_iid(dataset, num_users):    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def coco_noniid(dataset, num_users, data):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    timer = time.time()
    # 4,000 training imgs -->  200 shards
    num_shards = 200 if data == 'val2017' else 2000 # default 200
    num_imgs = len(dataset) // num_shards 
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}    
    total_shards = num_shards * num_imgs
    idxs = np.arange(total_shards)
    labels_path = './save/labels{}.pt'.format(data)
    if os.path.exists(labels_path):
        labels = torch.load(labels_path)[:total_shards]
    else:
        labels = convert_coco_mask_to_top_class(dataset)[:total_shards]
        torch.save(labels, labels_path)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):        
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    print('Time consumed to get user indices: ', (time.time()-timer)//60)
    # torch.save(dict_users, './save/dict_users.pt') # save to check
    return dict_users

def coco_noniid_unequal(dataset, num_users, data):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    timer = time.time()
    # 4000 training imgs --> 1000 shards
    num_shards = 1000 if data == 'val2017' else 10000 # default 1000
    num_imgs = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    total_shards = num_shards * num_imgs
    idxs = np.arange(total_shards)
    labels_path = './save/labels{}.pt'.format(data)
    if os.path.exists(labels_path):
        labels = torch.load(labels_path)[:total_shards]
    else:
        labels = convert_coco_mask_to_top_class(dataset)[:total_shards]
        torch.save(labels, labels_path)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1 
    max_shard = 30 # original is 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    print('Time consumed to get user indices: ', (time.time()-timer)//60)
    return dict_users