#IMPORT
import torch
from torch.utils.data import DataLoader
from torchvision import models

from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details
# from models import fcn_mobilenetv2, deeplabv3_mobilenetv3
from train import train_one_epoch, evaluate, criterion

def test_fcn_resnet(args):
    # DATA
    train_dataset, test_dataset, user_groups = get_dataset(args)
    test_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    #MODEL
    print('loading pretrained model')
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.to('cuda')

    #TEST    
    print('testing on pretrained model')
    test_acc, test_iou, confmat = test_inference(args, model, test_loader)
    print(confmat)

if __name__ == '__main__':
    
    args = args_parser()
    test_fcn_resnet(args)
    