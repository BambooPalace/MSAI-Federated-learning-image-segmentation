#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar


if __name__ == '__main__':
    args = args_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netorks
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

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    print('training starts:')
    lines = ['Options:', str(args), 'Global model:', str(global_model)]
    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch+1, batch_idx * len(images), len(trainloader.dataset),
                #     100. * batch_idx / len(trainloader), loss.item()))
                # Why these lines not working??
                lines.append('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(\
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),\
                    100. * batch_idx / len(trainloader), loss.item()))

            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        lines.append('\nTrain loss: {:.6f}'.format(loss_avg))
        epoch_loss.append(loss_avg)

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('./save/bsl_{}_{}_e{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
    lines.append('Test on {} samples'.format(len(test_dataset)))
    lines.append("Test Accuracy: {:.2f}%".format(100*test_acc))

    # logging
    filename = './save/bsl_{}_{}_e{}.txt'.format(args.dataset, args.model,
                                                 args.epochs)
    with open(filename, 'w') as w:
        for line in lines:
            w.write(line + '\n')
