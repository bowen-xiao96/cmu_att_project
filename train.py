import os, sys
import numpy as np

import cPickle

import json
import copy
import argparse
import torch
from torch.autograd import Variable
import re

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import Trainer

from model_builder import AttentionNetwork
from utils import *
from show import *

def get_parser():
    parser = argparse.ArgumentParser(description='The script to train the combine net')

    # General settings
    parser.add_argument('--gpu', default=None, type=str, action='store',
                        help='the id of gpu')
    parser.add_argument('--expId', default=None, type=str, action='store',
                        help='the ID of experiment')
    parser.add_argument('--save_dir', default='/data2/simingy/model/', type=str, action='store',
                        help='the master directory of saving the model')
    parser.add_argument('--load_file', default=None, type=str, action='store',
                        help='the name of file loading the model')
    parser.add_argument('--data_path', default='/data2/simingy/data/', type=str, action='store',
                        help='the path of loading data')
    parser.add_argument('--save_att_map', default=0, type=int, action='store',
                        help='whether save attention map')
    parser.add_argument('--print_fe', default=0, type=int, action='store',
                        help='whether print familiarity effect related numbers')
    parser.add_argument('--task', default='cifar10', type=str, action='store',
                        help='the dataset task')

    # Network settings
    parser.add_argument('--network_config', default=None, type=str, action='store',
                        help='the name of config file')
    parser.add_argument('--batch_size', default=256, type=int, action='store',
                        help='batch size')
    parser.add_argument('--weight_init', default=None, type=str, action='store',
                        help='the way to initialize the weights of network')
    parser.add_argument('--print_loss', default=0, type=int, action='store',
                        help='whether print training loss')
    parser.add_argument('--grad_clip', default=0.0, type=float, action='store',
                        help='whether do gradient clipping')
    parser.add_argument('--init_weight', default='vgg', type=str, action='store',
                        help='How to initialize network weights')
    parser.add_argument('--fix_load_weight', default=0, type=int, action='store',
                        help='whethter fix the loaded weights')


    # Learning rate settings
    parser.add_argument('--init_lr', default=0.1, type=float, action='store',
                        help='the initial learning rate')
    parser.add_argument('--adjust_lr', default=1, type=int, action='store',
                        help='Whether adjust learning rate automatically')
    parser.add_argument('--lr_decay', default=0.5, type=float, action='store',
                        help='the rate of decaying learning rate')
    parser.add_argument('--lr_freq', default=30, type=int, action='store',
                        help='the internal to decay the learning rate')
    parser.add_argument('--optim', default=0, type=int, action='store',
                        help='which optimizer 0: SGD, 1: Adam')


    return parser


def attention_model_training(args):
    

    # Build Model

    network_cfg = postprocess_config(json.load(open(os.path.join('network_configs', args.network_config))))
    if args.task == 'cifar10':
        net = AttentionNetwork(network_cfg, args)
    elif args.task == 'imagenet':
        net = AttentionNetwork(network_cfg, args, num_class=1000)


    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.cuda()

    # Initialize weights
    if args.init_weight == 'vgg':
        vgg_init(net) 
    elif args.init_weight == 'xavier':
        xavier_init(net)

    # Loss function
    criterion = nn.CrossEntropyLoss().cuda()

    start = 0

    # Load File
    if args.load_file is not None:
        prefix = (args.load_file).split('.')[-1]
        if prefix == 'pth':
            pretrained_dict = torch.load(args.load_file)
        elif prefix == 'pkl':
            pretrained_dict = torch.load(args.load_file)[-1]
        net_dict = net.state_dict()
        net_list = list(net.state_dict().keys())
        pre_list = list(pretrained_dict.keys())
        print("net size: {} pre size: {}".format(len(net_list), len(pre_list)))
        print(pre_list)
        print(net_list)
        if args.fix_load_weight == 1:
            for i,p in enumerate(net.parameters()):
                if i < (len(pre_list) - 2):
                    p.requires_grad = False

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        start = (int) ((re.findall(r"\d+", args.load_file))[-1])

    # Optimizer
    if args.optim == 0:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.init_lr, momentum=0.9, weight_decay=5e-4)
    elif args.optim == 1:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.init_lr, weight_decay=5e-5)
    
    # Import Dataset
    if args.task == 'cifar10':
        train_loader, test_loader = get_dataloader(
            cifar10_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=4,
        )
    elif args.task =='imagenet':
        train_loader, test_loader = get_Imagenetloader(
            imn_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=4,
        )

    if args.print_fe == 1:
        train_it_one(net, criterion, optimizer)
        print("****************************************************************")
        test_it(net, criterion, optimizer)
    else:
        Trainer.start(
                model=net,
                optimizer=optimizer,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                criterion=criterion,
                max_epoch=300,
                lr_sched=None,
                display_freq=50,
                output_dir=args.save_dir+args.expId,
                save_every=20,
                max_keep=20,
                save_model_data='/data2/simingy/model_data/'+args.expId
            )           
    
def main():
    parser = get_parser()

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    attention_model_training(args)

if __name__ == '__main__':
    main()
