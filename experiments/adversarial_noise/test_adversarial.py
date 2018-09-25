import os, sys
import numpy as np
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

from torchvision.models import vgg16

from attack import FGSM_attack

sys.path.insert(0, '/data2/bowenx/attention/pay_attention')
from dataset.imagenet.get_imagenet_dataset import get_dataloader
from model.get_model import get_model
from utils.metric import accuracy

if __name__ == '__main__':
    assert len(sys.argv) > 4
    GPU_ID = int(sys.argv[1])
    model_name = sys.argv[2]
    weight_file = sys.argv[3]
    eps = float(sys.argv[4])

    if GPU_ID == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

    # prapare model
    model = get_model(model_name)
    new_loss = model_name == 'multiple_recurrent_newloss'
    if model_name != 'vgg':
        # recurrent model, need multiple GPUs
        model = nn.DataParallel(model)

    # load weights
    state_dict = torch.load(weight_file)
    if isinstance(state_dict, tuple):
        state_dict = state_dict[-1]

    model.load_state_dict(state_dict)
    model.eval()
    model = model.cuda()
    del state_dict

    imagenet_dir = '/data2/simingy/data/Imagenet'
    batch_size = 128 if model_name == 'vgg' else 16
    test_loader = get_dataloader(imagenet_dir, batch_size, 8)

    pred = list()
    gt = list()

    for x, y in test_loader:
        scores = FGSM_attack(model, x, y, eps)
        pred.append(scores)
        gt.append(y)

    # calculate accuracy drop and fooling rate
    pred = torch.cat(pred, dim=0)
    gt = torch.cat(gt, dim=0)

    top_1_original, top_5_original = accuracy(pred[:, 0, :], gt)
    top_1_fool, top_5_fool = accuracy(pred[:, 1, :], gt)

    pred_original = A.Variable(pred[:, 0, :].cuda())
    pred_fool = A.Variable(pred[:, 1, :].cuda())
    pred_original = F.softmax(pred_original, dim=-1)
    pred_fool = F.softmax(pred_fool, dim=-1)
    ave_drop = torch.mean(
        torch.index_select(pred_original - pred_fool, -1, gt)
    )
    ave_drop = ave_drop.data.cpu()[0]

    print(top_1_original, top_5_original, top_1_fool, top_5_fool, ave_drop)
