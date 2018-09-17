import os, sys
import numpy as np
import h5py
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, '/data2/bowenx/attention/pay_attention')

from torchvision.models import vgg16
from model.multiple_recurrent_l import *
from util.metric import accuracy

CLASSES = ('aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train')
LEVEL = ('ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE')


class VehicleOcclusion(Dataset):
    def __init__(self, root_dir, level):
        self.root_dir = root_dir
        self.level = LEVEL[level - 1]
        self.classes = list(CLASSES)

        if level not in (1, 5, 9):
            self.classes.pop(-1)

        # load imagenet labels mapping
        mapping = dict()
        with open('map_clsloc.txt', 'r') as f_in:
            for line in f_in.readlines():
                line = line.strip()
                if not line: continue

                class_tag, class_id, _ = line.split()
                mapping[class_tag] = int(class_id) - 1  # start from 1 instead of 0

        # load all files and store in data list
        self.data = list()
        for i, c in enumerate(self.classes):
            class_root = os.path.join(root_dir, '%sLEVEL%s' % (c, self.level))
            for f in os.listdir(class_root):
                class_tag = f.split('_')[0]
                if class_tag in mapping:
                    class_id = mapping[class_tag]
                    self.data.append((os.path.join(class_root, f), class_id))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose([
            # transforms.Resize(256, interpolation=Image.LANCZOS),
            # transforms.CenterCrop(224),
            transforms.Resize((224, 224), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            normalize
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        filename, class_id = self.data[item]

        with h5py.File(filename, 'r') as f:
            # directly load image
            img = np.array(f['record'].get('img'))
            img = Image.fromarray(img.transpose((2, 1, 0)), mode='RGB')

        img = self.transform(img)
        return img, class_id


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    assert len(sys.argv) > 3
    # mode=0: original vgg16 model
    # mode=1: recurrent gating model

    model_file = sys.argv[1]
    mode = int(sys.argv[2])

    # occ_level: from 1 to 9
    occ_level = int(sys.argv[3])

    # load model
    if mode == 0:
        model = vgg16(num_classes=1000, init_weights=False)
        state_dict = torch.load(model_file)
    else:
        # recurrent gating model
        # such hyperparameters are fixed
        connections = (
            (13, 8, 256, 128, 2),
            (20, 15, 512, 256, 2)
        )
        model = MultipleRecurrentModel(network_cfg, connections, 5, 1000)
        model = nn.DataParallel(model)
        _, _, state_dict = torch.load(model_file)

    # print(model)
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    del state_dict

    # load dataset
    root_dir = r'/data2/bowenx/dataset/occ'
    dataset = VehicleOcclusion(root_dir, occ_level)
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    pred = list()
    gt = list()

    for x, y in dataloader:
        x = A.Variable(x.cuda(), volatile=True)
        y = A.Variable(y.cuda(), volatile=True)

        output = model(x)
        pred.append(output.data.cpu())
        gt.append(y.data.cpu())

    pred = torch.cat(pred, dim=0)
    gt = torch.cat(gt, dim=0)

    prec1, prec5 = accuracy(pred, gt, topk=(1, 5))
    print(prec1[0], prec5[0])
