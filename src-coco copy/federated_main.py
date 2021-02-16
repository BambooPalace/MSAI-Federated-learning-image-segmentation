#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details
from models import fcn_mobilenetv2, deeplabv3_mobilenetv3
from train import train_one_epoch, evaluate, criterion

if __name__ == '__main__':
    start_time = time.time()

    # define paths

    args = args_parser()
    exp_details(args)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
    print('device: ', device)
    torch.manual_seed(args.seed)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    test_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    # BUILD MODEL
    if args.model == 'fcn_mobilenetv2':
        # Convolutional neural netorks
        global_model = fcn_mobilenetv2(num_classes=args.num_classes, aux_loss=True)
    elif args.model == 'deeplabv3_mobilenetv2':
        global_model = deeplabv3_mobilenetv3(num_classes=args.num_classes, aux_loss=True)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, local_test_accuracy, local_test_iou = [], [], []
    print_every = 2

    # Load checkpoint
    start_ep = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(
            os.path.join( args.root, 'save/checkpoints', args.checkpoint),
            map_location=device)
        global_model.load_state_dict(checkpoint['model'])
        start_ep = checkpoint['epoch']   

    # Global rounds
    print(args.epochs,' epochs of training starts:')
    log = ['Options:', str(args)]
    for epoch in tqdm(range(start_ep, args.epochs)):
        local_weights, local_losses = [], []
        print('\n | Global Training Round : {} |\n'.format(epoch+1))
        #
        log.append('\n | Global Training Round : {} |\n'.format(epoch+1))

        print('training global model on {} of {} users locally'.format(args.frac, args.num_users))
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # Local training
        for idx in idxs_users:
            print('user idx : ', idx)

            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        print('weight averaging')
        global_weights = average_weights(local_weights)
        # update global weights
        global_model.load_state_dict(global_weights)
        # save global model to checkpoint
        exp_name = 'fed_{}_{}_e{}_C[{}]_iid[{}]_uneq[{}]_E[{}]_B[{}]'.\
                    format(args.dataset, args.model, args.epochs, args.frac,\
                         args.iid, args.unequal, args.local_ep, args.local_bs)
        if epoch % args.save_frequency == 0 or epoch == args.epochs-1:
            torch.save(
                {
                    'model': global_model.state_dict(),
                    'epoch': epoch,
                    'exp_name': exp_name
                },
                os.path.join(args.root, 'save/checkpoints', exp_name+'.pth')
            )
        print('global model weights save to checkpoint')

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg test accuracy over all users at every epoch
        list_acc, list_iou = [], []
        global_model.eval()
        print('test global model on {} users'.format(args.num_users))
        for c in tqdm(range(args.num_users)):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])            
            acc, iou = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_iou.append(iou)
        local_test_accuracy.append(sum(list_acc)/len(list_acc))
        local_test_iou.append(sum(list_iou)/len(list_iou))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Local Test Accuracy: {:.2f}% \n'.format(local_test_accuracy[-1]))
            print('Local Test IoU: {:.2f}% \n'.format(local_test_iou[-1]))
            #
            log.append(' \nAvg Training Stats after {epoch+1} global rounds:')
            log.append('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            log.append('Local Test Accuracy: {:.2f}% \n'.format(local_test_accuracy[-1]))
            log.append('Local Test IoU: {:.2f}% \n'.format(local_test_iou[-1]))

    # Inference on test dataset after completion of training
    test_acc, test_iou = test_inference(args, global_model, test_loader)

    print(' \n Results after {} global rounds of training:'.format(args.epochs))
    print("|---- Avg local test Accuracy: {:.2f}%".format(local_test_accuracy[-1]))
    print("|---- Global Test Accuracy: {:.2f}%".format(test_acc))
    print("|---- Global Test IoU: {:.2f}%".format(test_iou))
    #
    log.append(' \n Results after {} global rounds of training:'.format(args.epochs))
    log.append("|---- Avg Train Accuracy: {:.2f}%".format(local_test_accuracy[-1]))
    log.append("|---- Test Accuracy: {:.2f}%".format(test_acc))
    log.append("|---- Global Test IoU: {:.2f}%".format(test_iou))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    #
    log.append('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(os.path.join(args.root, 'save', exp_name+'_loss.png'))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(local_test_accuracy)), local_test_accuracy, color='k', label='local test accuracy')
    plt.plot(range(len(local_test_iou)), local_test_iou, color='b', label='local test IoU')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.legend()
    plt.savefig(os.path.join(args.root, 'save', exp_name+'_metrics.png'))

    # Logging
    filename = os.path.join(args.root, 'save', exp_name+'_log.txt')
    with open(filename, 'w') as w:
        for line in log:
            w.write(line + '\n')