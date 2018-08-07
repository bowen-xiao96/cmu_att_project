import os, sys
import numpy as np

import cPickle

import json
import copy
import argparse
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

from pay_attention import AttentionNetwork
from utils import *

def get_parser():
    parser = argparse.ArgumentParser(description='The script to train the combine net')

    # General settings
    parser.add_argument('--gpu', default=None, type=str, action='store',
                        help='the id of gpu')
    parser.add_argument('--save_dir', default=None, type=str, action='store',
                        help='the directory of saving the model')
    parser.add_argument('--load_file', default=None, type=str, action='store',
                        help='the name of file loading the model')
    parser.add_argument('--data_path', default='/mnt/fs1/siming/data/', type=str, action='store',
                        help='the path of loading data')

    # Network settings
    parser.add_argument('--network_config', default=None, type=str, action='store',
                        help='the name of config file')
    parser.add_argument('--weight_init', default=None, type=str, action='store',
                        help='the way to initialize the weights of network')
    parser.add_argument('--print_loss', default=0, type=int, action='store',
                        help='whether print training loss')
    parser.add_argument('--grad_clip', default=0.0, type=float, action='store',
                        help='whether do gradient clipping')

    # Learning rate settings
    parser.add_argument('--init_lr', default=0.1, type=float, action='store',
                        help='the initial learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float, action='store',
                        help='the rate of decaying learning rate')
    parser.add_argument('--lr_freq', default=30, type=int, action='store',
                        help='the internal to decay the learning rate')

    return parser

def attention_model_training(args):
    

    network_cfg = postprocess_config(json.load(open(os.path.join('network_configs', args.network_config))))

    net = AttentionNetwork(network_cfg)
    net.cuda()
    print(net)
    
    if args.weight_init == 'xavier':
        net.apply(xavier_init)

    if args.load_file is not None:
        net.load_state_dict(torch.load(args.load_file))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=5e-4)

    # Import Dataset

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    trainloader = torch.utils.data.DataLoader(
                    datasets.CIFAR10(root=args.data_path, train=True, transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                normalize,
                                ]), download=True),
                                batch_size=128, shuffle=True,
                                num_workers=4, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
                    datasets.CIFAR10(root=args.data_path, train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                                ])),
                                batch_size=128, shuffle=False,
                                num_workers=4, pin_memory=True)

    for epoch in range(300):
        
        adjust_learning_rate(optimizer, epoch, args.init_lr, args.lr_decay, args.lr_freq)
        '''Training Stage'''
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
        
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if args.grad_clip != 0:
                nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

            running_loss += loss.data[0]
            if args.print_loss != 0:
                if i % 20 == 0:
                    print("Training Loss:", running_loss / 20)
                    running_loss = 0.0

        '''Test Stage'''  
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            labels = Variable(labels.cuda())
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        print("Epoch %d, Test accuracy: %f, lr=%f" % (epoch, float(correct) / total, float(current_lr)))

        if args.save_dir is not None:
            os.system('mkdir -p %s' % args.save_dir) 
            if epoch % 30 == 0:
                save_path = os.path.join(args.save_dir, 'model-' + str(epoch) + '.pth')
                torch.save(net.state_dict(), save_path)

    if args.save_file is not None:
        save_path = os.path.join(args.save_dir, 'model-last.pth')
        torch.save(net.state_dict(), save_path)


    
def main():
    parser = get_parser()

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    attention_model_training(args)

if __name__ == '__main__':
    main()

