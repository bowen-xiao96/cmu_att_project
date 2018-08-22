# test on imagenet high resolution images

import os, sys
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms
from utils import *
from model_builder import AttentionNetwork
import heapq
import pickle

torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)


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
    parser.add_argument('--dataset', default='imagenet', type=str, action='store',
                        help='the dataset name')
    parser.add_argument('--task', default='recurrent_att', type=str, action='store',
                        help='the task name')
    # Network settings
    parser.add_argument('--network_config', default=None, type=str, action='store',
                        help='the name of config file')
    parser.add_argument('--batch_size', default=256, type=int, action='store',
                        help='batch size')
    parser.add_argument('--weight_init', default=None, type=str, action='store',
                        help='the way to initialize the weights of network')
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

    # Attention Settings
    parser.add_argument('--save_att_map', default=0, type=int, action='store',
                        help='whether save attention map')
    parser.add_argument('--print_fe', default=0, type=int, action='store',
                        help='whether print familiarity effect related numbers')
    parser.add_argument('--att_channel', default=1, type=int, action='store',
                        help='how many score maps do you want to add in the attention\
                                recurrent model')
    parser.add_argument('--att_unroll_count', default=1, type=int, action='store',
                        help='how many time do you want to unroll')



    return parser

def get_imagenet_images():
    # sample some imagenet images
    # format: raw PIL images resized to 224 * 224
    #root_dir = '/mnt/fs0/feigelis/imagenet-data/raw-data/train'
    root_dir = '/data2/leelab/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train/'
    labels = ('n03954731', 'n02690373', 'n02105855')

    images = list()
    image_count = 30
    ratio_thresh = 0.8

    for label in labels:
        count = 0
        image_list = os.listdir(os.path.join(root_dir, label))
        for f in image_list:
            fullname = os.path.join(root_dir, label, f)
            img = Image.open(fullname).convert('RGB')
            w, h = img.size
            s = float(min(w, h)) / max(w, h)
            if s >= ratio_thresh:
                count += 1
                images.append(img.resize((224, 224), resample=Image.LANCZOS))

            if count == image_count:
                break

    return images

def get_cifar10_images():
    root_dir = '/data2/simingy/data/cifar-10-batches-py/'
    train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ]
    for fentry in train_list:
        f = fentry[0]
        files = os.path.join(root_dir, f)
        fo = open(files, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        train_data = entry['data']

        fo.close()

    train_data = train_data.reshape((10000, 3, 32, 32))
    train_data = train_data.transpose((0, 2, 3, 1))
    images = []
    for i in range(60):
        images.append(train_data[i])
    
    return images

def extract_attention_maps(model, x, size=(32, 32)):
    feature_maps = list()
    score_maps = list()

    for i, layer in enumerate(model.backbone):
        x = layer(x)

        if i in [13]:
            feature_maps.append(x)
    
    #x = F.max_pool2d(x, kernel_size=7, stride=7)   
    x = x.view(x.size(0), -1)

    for i, layer in enumerate(model.fclayers):
        x = layer(x)

    features = list()

    for i, feature_map in enumerate(feature_maps):
        if len(model.attention[i]) == 2:
            new_x = model.attention[i][0](x)
        else:
            new_x = x
        
        score = model.attention[i][-1](feature_map + new_x.view(new_x.size(0), -1, 1, 1))
        old_shape = score.size()
  
        score = F.softmax(score.view(old_shape[0], -1), dim=1).view(old_shape)

        score = F.upsample(score, size=size, mode='bilinear')

        score_maps.append(score.data.cpu().numpy())

    score_maps = np.concatenate(score_maps, axis=1)
    return score_maps
    

def extract_recurrent_attention_v2_maps(model, x, size=(32, 32), unroll_count=1):
    feature_maps = list()
    score_maps = list()

    feature_maps = list()
    for i, layer in enumerate(model.backbone):
        x = layer(x)

        # after relu layer
        if i in [15]:
            feature_maps.append(x)
        if i in [29]:
            break

    features = list()
    for i, feature_map in enumerate(feature_maps):
        if unroll_count > 1:
            recurrent_buf = list()
            recurrent_buf.append(feature_map)

        for j in range(unroll_count):
            
            if unroll_count > 1:
                feature_map = recurrent_buf[-1]

            for k, layer in enumerate(model.att_recurrent_b[i]):
                x = layer(x)

            old_shape = x.size()

            score = F.softmax(x.view(old_shape[0], -1), dim=1).view(old_shape)

            x = model.att_recurrent_f[i](torch.cat([score, feature_map], dim=1))
            if unroll_count > 1:
                recurrent_buf.append(x)

            # Save score maps
            # upsample the spatial map
            score = F.upsample(score, size=size, mode='bilinear')
            #print(score.shape)
            score_maps.append(score.data.cpu().numpy())

            for k in range(16, 30):
                x = model.backbone[k](x)
        
    score_maps = np.concatenate(score_maps, axis=1)
    return score_maps

def extract_recurrent_attention_maps(model, x, size=(224, 224), unroll_count=1):
    feature_maps = list()
    score_maps = list()

    feature_maps = list()
    for i, layer in enumerate(model.backbone):
        x = layer(x)

        # after relu layer
        if i in [13]:
            feature_maps.append(x)

    # global feature
    #x = F.avg_pool2d(x, kernel_size=7, stride=7)
    x = x.view(x.size(0), -1)
    x = model.fclayers[0](x)
    x = model.fclayers[1](x)
    x = model.fclayers[2](x)

    features = list()
    for i, feature_map in enumerate(feature_maps):
        if unroll_count > 1:
            recurrent_buf = list()
            recurrent_buf.append(feature_map)

        for j in range(unroll_count):
            
            if unroll_count > 1:
                feature_map = recurrent_buf[-1]

            if len(model.att_recurrent_b[i]) == 2:
                # project the global feature
                new_x = model.att_recurrent_b[i][0](x)
            else:
                new_x = x

            # attention score map (do 1x1 convolution on the addition of feature map and the global feature)
            score = model.att_recurrent_b[i][-1](feature_map + new_x.view(new_x.size(0), -1, 1, 1))
            old_shape = score.size()
            score = F.softmax(
                score.view(old_shape[0], -1), dim=1
            ).view(old_shape)
            print(j)
            nump = score.data.cpu().numpy()
            nump = np.reshape(nump, (nump.shape[0], -1))
            for k in range(nump.shape[0]):
                nump[k].sort()
                #print(nump[k, -3:])

            #score_ = score.expand(old_shape[0], 1, old_shape[2], old_shape[3])

            x = model.att_recurrent_f[i](torch.cat([score, feature_map], dim=1))
            if unroll_count > 1:
                recurrent_buf.append(x)

            # Save score maps
            # upsample the spatial map
            score = F.upsample(score, size=size, mode='bilinear')
            #print(score.shape)
            score_maps.append(score.data.cpu().numpy())

            for k in range(14, len(model.backbone)):
                x = model.backbone[k](x)
            
            #print(j)
            #print(x.shape)
            #x = F.avg_pool2d(x, kernel_size=7, stride=7)
            x = x.view(x.size(0), -1)          
            for k in range(len(model.fclayers) - 1):
                x = model.fclayers[k](x)

    score_maps = np.concatenate(score_maps, axis=1)
    return score_maps


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    mean = np.array([0.49139968, 0.48215827, 0.44653124])
    std = np.array([0.24703233, 0.24348505, 0.26158768])
    normalize = transforms.Normalize(mean, std)

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    network_cfg = postprocess_config(json.load(open(os.path.join('network_configs', args.network_config))))

    model_path = args.load_file
    _, _, pretrained_dict = torch.load(model_path)
    print(pretrained_dict.keys())
    if args.dataset == 'cifar10':
        model = AttentionNetwork(network_cfg, args)
    elif args.dataset == 'imagenet':
        model = AttentionNetwork(network_cfg, args, num_class=1000)
    print(model.state_dict().keys())
    
    load_parallel_flag = load_parallel(pretrained_dict)
    if load_parallel_flag == 1:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
    else:
        new_state_dict = pretrained_dict

    # load params
    print(new_state_dict.keys())
    model.load_state_dict(new_state_dict)
    model.eval()
    model.cuda()

    #print(model)
    if args.dataset == 'imagenet':
        all_images = get_imagenet_images()
    elif args.dataset == 'cifar10':
        all_images = get_cifar10_images()

    batch_size = 4
    batch_count = int(math.ceil(float(len(all_images)) / batch_size))
    score_maps = list()

    for i in range(batch_count):
        images = all_images[i * batch_size: (i + 1) * batch_size]
        images = A.Variable(torch.stack([transform(img) for img in images]).cuda())

        # get score map
        if args.dataset == 'imagenet':
            if args.task == 'recurrent_att':
                score_maps.append(extract_recurrent_attention_maps(model, images, size=(224,224), unroll_count=args.att_unroll_count))
            elif args.task == 'attention':
                score_maps.append(extract_attention_maps(model, images, size=(224,224)))
        elif args.dataset == 'cifar10':
            if args.task == 'recurrent_att':
                score_maps.append(extract_recurrent_attention_maps(model, images, size=(32, 32), unroll_count=args.att_unroll_count))
            elif args.task == 'attention':
                score_maps.append(extract_attention_maps(model, images, size=(32, 32)))
            elif args.task == 'recurrent_att_v2':
                score_maps.append(extract_recurrent_attention_v2_maps(model, images, size=(32, 32), unroll_count=args.att_unroll_count))


    score_maps = np.concatenate(score_maps, axis=0)
    all_images = np.stack([np.array(img) for img in all_images])
    
    print("Save it!")
    os.system('mkdir -p %s' % os.path.join(args.save_dir, args.expId))
    np.savez(os.path.join(args.save_dir, args.expId, 'imagenet.npz'), images=all_images, score_maps=score_maps)
