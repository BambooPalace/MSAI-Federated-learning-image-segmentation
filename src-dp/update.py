#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from train import criterion, evaluate

from opacus import PrivacyEngine

NOISE_MULTIPLIER = 0.38
MAX_GRAD_NORM = 1.2
DELTA = 1e-5
DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        # pytorch warning and suggest below 
        return image.clone().detach().float(), label.clone().detach()


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
        # Default criterion set to NLL loss function

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, and test (80, 20)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        # mod1: add num_workers, to see if can speed up training. ANS is no for cifar
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.virtual_bs, num_workers=self.args.num_workers, shuffle=True, drop_last=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max(len(idxs_test)//10,1), num_workers=self.args.num_workers, shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round, log):
        def print_log(string, log=log):
            ''' print string and append to log[list] '''
            print(string)
            log.append(string)
            
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer and lr_scheduler for the local updates
        args = self.args
        if args.aux_lr > 1:
            params_to_optimize = [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]}]
            if model.aux_classifier:
                params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params, "lr": args.lr * args.aux_lr}) #multiplier default is 10
        else:
            params_to_optimize = [p for p in model.parameters() if p.requires_grad]

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr,
                                        momentum=args.momentum, weight_decay=0.0001)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr,
                                        weight_decay=1e-4)
        #PRIVACY ENGINE
        VIRTUAL_STEP = args.local_bs // args.virtual_bs
        if args.dp:
            privacy_engine = PrivacyEngine(
                model,
                # batch_size = args.local_bs,
                # sample_size = len(self.trainloader)*args.local_bs,
                sample_rate = VIRTUAL_STEP / len(self.trainloader),   
                alphas = DEFAULT_ALPHAS,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
            )
            privacy_engine.attach(optimizer)   

        scheduler_dict = {
            'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5),
            'lambda':torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(self.trainloader)*max(1,args.local_ep))) ** 0.9)
        }
        lr_scheduler = scheduler_dict[args.lr_scheduler]                                                                             

        # training
        start_time = time.time()
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)                
                
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                # virtual step to save memory for noise addition
                if (iter+1) % VIRTUAL_STEP == 0 or (iter+1) == len(self.trainloader):
                    optimizer.step() # update params
                    optimizer.zero_grad()                    
                else:
                    optimizer.virtual_step() # sum per-sample gradients into one and save for later, discard gradients

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            lr_scheduler.step()
            if args.dp:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(DELTA)          
                print_log(f'local epoch: {iter}, epsilon={epsilon:.2f}, delta={DELTA}')                  

            if self.args.verbose:
                string = '| Global Round : {} | Local Epoch : {} | {} images\tLoss: {:.6f}'.format(
                    global_round, iter+1, len(self.trainloader.dataset),loss.item())
                print_log(string)            

        # after training, print logs
        strings = [
            '| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}'.format(
            global_round, self.args.local_ep, len(self.trainloader.dataset), loss.item()),
            '\n'
            'Run Time: {0:0.4f}'.format(time.time()-start_time),
            ]
        print_log(''.join(strings))
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        confmat = evaluate(model, self.testloader, self.device, self.args.num_classes)
        return confmat.acc_global, confmat.iou_mean


def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """
    device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
    confmat = evaluate(model, testloader, device, args.num_classes)
    return confmat.acc_global, confmat.iou_mean, str(confmat)
