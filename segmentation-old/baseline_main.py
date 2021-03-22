#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, deeplabv3_mobilenet_v3_large, fcn_resnet50

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import fcn_mobilenetv2, deeplabv3_mobilenetv3
from train import train_one_epoch, evaluate, criterion

def make_model(args):
    if args.model == 'fcn_mobilenetv2':
        # Convolutional neural netorks
        global_model = fcn_mobilenetv2(num_classes=args.num_classes, aux_loss=bool(args.aux_lr))

    elif args.model == 'deeplabv3_mobilenetv2':
        global_model = deeplabv3_mobilenetv2(num_classes=args.num_classes, aux_loss=bool(args.aux_lr))

    elif args.model == 'deeplabv3_mobilenetv3':
        global_model = deeplabv3_mobilenet_v3_large(num_classes=args.num_classes, aux_loss=bool(args.aux_lr), 
                                                    pretrained=args.pretrained)


    elif args.model == 'lraspp_mobilenetv3':        
        global_model = lraspp_mobilenet_v3_large(num_classes=args.num_classes, pretrained=args.pretrained)
        # no aux classifier, no dropout layer
        global_model.aux_classifier = None   

    #resnet for test only as too many params 
    elif args.model == 'fcn_resnet50':
        global_model = fcn_resnet50(num_classes=args.num_classes, pretrained=True)
        
    else:
        exit('Error: unrecognized model')

    if args.activation == 'tanh': # test tanh for DP-SGD
        global_model = convert_relu_tanh(global_model)
    if args.freeze_backbone: # test for DP-SGD
        for p in global_model.backbone.parameters():
            p.requires_grad = False

    return global_model

def get_exp_name(args):
    exp_name = 'baseline_{}_{}_c{}_e{}_B[{}]_lr[{}x{}]_{}_{}_weight{}'.\
                format(args.data, args.model, args.num_classes, args.epochs, 
                # args.frac, args.iid, args.unequal,args.local_ep, 
                args.local_bs, args.lr, args.aux_lr, args.lr_scheduler, args.optimizer, args.weight)    
                    
    return exp_name

if __name__ == '__main__':
    args = args_parser()
    device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
    torch.manual_seed(args.seed)

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.local_bs, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    # BUILD MODEL
    global_model = make_model(args)
    exp_name = get_exp_name(args)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    if args.aux_lr > 1:
        params_to_optimize = [
        {"params": [p for p in global_model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in global_model.classifier.parameters() if p.requires_grad]}]
        if global_model.aux_classifier:
            params = [p for p in global_model.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": args.lr * args.aux_lr}) #multiplier default is 10
    else:
        params_to_optimize = [p for p in global_model.parameters() if p.requires_grad]

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr,
                                     momentum=args.momentum, weight_decay=0.0001)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr,
                                     weight_decay=1e-4)

    scheduler_dict = {
        'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1),
        'lambda':torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(train_loader)*max(1,args.epochs))) ** 0.9)
    }
    lr_scheduler = scheduler_dict[args.lr_scheduler]

    start_ep = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(
            os.path.join( args.root, 'save/checkpoints', args.checkpoint),
            map_location=device)
        global_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_ep = checkpoint['epoch']   

    print(args.epochs,' epochs of training starts:')
    lines = ['Options:', str(args)]
    for epoch in tqdm(range(start_ep, args.epochs)):
        logger = train_one_epoch(global_model, criterion, optimizer, train_loader, lr_scheduler, 
                                device, epoch, print_freq=1000, background_weight=args.weight)
        lines.append(logger)
        lr_scheduler.step()
        if epoch % args.save_frequency == 0 or epoch == args.epochs-1:
            torch.save(
                {
                    'model': global_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch+1,
                    'model_name': args.model
                },
                os.path.join(args.root, 'save/checkpoints', exp_name +'.pth')
                )
    print('Training ends')
    
    print('Testing:')
    timer = time.time()
    if not args.train_only:
        confmat = evaluate(global_model, test_loader, device, num_classes=args.num_classes)
        print(confmat)
        lines.append('Confusion matrix on test dataset: ')
        lines.append(str(confmat))
    print('Test time: ', time.time() - timer)


    # logging
    
    path = os.path.join(args.root, 'save/logs', exp_name + '_log.txt')
    with open(path, 'w') as w:
        for line in lines:
            w.write(line + '\n')
