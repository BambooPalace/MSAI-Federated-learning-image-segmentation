#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('./save/logs')

    args = args_parser()
    exp_details(args)

    device = 'cpu' if torch.cuda.is_available() else 'cuda'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)

        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    lines = ['Options:', str(args), 'Global model:', str(global_model)]
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print('\n | Global Training Round : {} |\n'.format(epoch+1))
        lines.append('\n | Global Training Round : {} |\n'.format(epoch+1))

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            print('user idx : ', idx)
            # lines.append(f'user idx is {idx}')

            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            lines.append(' \nAvg Training Stats after {epoch+1} global rounds:')
            lines.append('Training Loss : {}'.format(np.mean(np.array(train_loss))))
            lines.append('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(' \n Results after {} global rounds of training:'.format(args.epochs))
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    lines.append(' \n Results after {} global rounds of training:'.format(args.epochs))
    lines.append("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    lines.append("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/fed_{}_{}_e{}_C[{}]_iid[{}]_uneq[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.unequal,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    lines.append('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_e{}_C[{}]_iid[{}]_uneq[{}]_E[{}]_B[{}].png'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.unequal,
               args.local_ep, args.local_bs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_e{}_C[{}]_iid[{}]_uneq[{}]_E[{}]_B[{}].png'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.unequal,
               args.local_ep, args.local_bs))

    # Logging
    filename = './save/fed_{}_{}_e{}_C[{}]_iid[{}]_uneq[{}]_E[{}]_B[{}].txt'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.unequal,
               args.local_ep, args.local_bs)
    with open(filename, 'w') as w:
        for line in lines:
            w.write(line + '\n')