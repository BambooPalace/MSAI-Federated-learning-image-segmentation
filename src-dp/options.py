#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=1,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E, default is 10")
    parser.add_argument('--local_bs', type=int, default=1,
                        help="local batch size: B, default=8, local gpu can only set 1")
    parser.add_argument('--num_workers', type=int, default=1,
                        help='test colab gpu num_workers=1 is faster')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')

    # model arguments
    parser.add_argument('--model', type=str, default='fcn_mobilenetv2', \
                        choices=['fcn_mobilenetv2', 'deeplabv3_mobilenetv2'], help='model name')    

    # other arguments
    parser.add_argument('--dataset', type=str, default='coco', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=81, help="number \
                        of classes")
    parser.add_argument('--cpu_only', action='store_true', help="indicate to use cpu only")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('-aux', '--aux_lr_param', type=int, default=2, help='times of normal learning rate used for auxiliary classifier ')
    parser.add_argument('--lr_scheduler', default='lambda', choices=['lambda', 'step'], help='learning rate scheduler')
    parser.add_argument('--checkpoint', type=str, default=None, help='full file name of the checkpoint')
    parser.add_argument('--save_frequency', type=int, default=1, help='number of epochs to save checkpoint')
    parser.add_argument('--root', type=str, default='./', help='home directory')
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--data', type=str, default='val2017', choices=['val2017', 'train2017'], help='val has 5k images while train has 118k')
    parser.add_argument('--local_test_frac', default=1.0, type=float, help='frac of num_users for local testing')
    parser.add_argument('--dp', action='store_true')
    parser.add_argument('--virtual_bs', default=4, type=int, help='the bs for noise addition, to save memory')
    args = parser.parse_args()
    return args
