import os, sys
import h5py
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

sys.path.insert(0, '/data2/bowenx/attention/pay_attention')
from model.multiple_recurrent_l import *
from cnn_pretrained import get_one_network_meta

# read hdf5 input file
with h5py.File('/data2/leelab/data/cnn_feature_extraction_input.hdf5', 'r') as f_in:
    # shape: (540, 3, 224, 224)
    # already preprocessed in pytorch format
    input_imgs = np.array(f_in.get('crcns_pvc-8_large').get('vgg16').get('half'))

# load model
connections = (
    (13, 8, 256, 128, 2),
    (20, 15, 512, 256, 2)
)
model = MultipleRecurrentModel(network_cfg, connections, 5, 1000)
_, _, state_dict = torch.load(r'/data2/bowenx/attention/pay_attention/experiments/multiple_recurrent_l2/best.pkl')
state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
model.load_state_dict(state_dict)
model.eval()
model = model.cuda()
del state_dict

# load network metadata
_, slice_dict, layers, _ = get_one_network_meta('vgg16', 24)
idx_dict = {
    1: 'conv1_1',
    3: 'conv1_2',
    4: 'pool1',
    6: 'conv2_1',
    8: 'conv2_2',
    9: 'pool2',
    11: 'conv3_1',
    13: 'conv3_2',
    15: 'conv3_3',
    16: 'pool3',
    18: 'conv4_1',
    20: 'conv4_2',
    22: 'conv4_3',
    23: 'pool4',
    25: 'conv5_1',
    27: 'conv5_2',
    29: 'conv5_3',
    30: 'pool5',
}

# unroll_count * layer_count
features = [[list() for j in range(len(idx_dict) + 2)] for i in range(model.unroll_count)]


def extract_feat(model, x):
    # assert model and x (images) are all already on GPU
    # x is pytorch Variable object
    buf = [list() for _ in model.backbone]

    for i in range(model.min_end_layer):
        # do not actually include the starting layer
        x = model.backbone[i](x)

        if i in idx_dict:
            offset = layers.index(idx_dict[i])
            slice_r, slice_c = slice_dict[idx_dict[i]]
            features[0][offset].append(x[:, :, slice_r, slice_c].data.cpu().numpy())

    buf[model.min_end_layer].append(x)

    for i in range(model.unroll_count):
        if i == 0:
            # the first unrolling step
            # just straightforward forward computation
            for j in range(model.min_end_layer, model.layer_count):
                x = model.backbone[j](x)

                if j in idx_dict:
                    offset = layers.index(idx_dict[j])
                    slice_r, slice_c = slice_dict[idx_dict[j]]
                    features[0][offset].append(x[:, :, slice_r, slice_c].data.cpu().numpy())

                # put it into the buffer if there is feedback connection
                for point_to in model.point_to[j]:
                    buf[point_to].append(x)

            # make final predictions
            x = x.view(x.size(0), -1)

            for j in range(len(model.classifier)):
                x = model.classifier[j](x)

                if j == 1:
                    features[0][-2].append(x.data.cpu().numpy())
                elif j == 4:
                    features[0][-1].append(x.data.cpu().numpy())

        else:
            pos = 0

            low, high = buf[model.min_end_layer][-2:]
            x = model.gating[pos](low, high)
            buf[model.min_end_layer].append(x)
            pos += 1

            if model.min_end_layer in idx_dict:
                offset = layers.index(idx_dict[model.min_end_layer])
                slice_r, slice_c = slice_dict[idx_dict[model.min_end_layer]]
                features[i][offset].append(x[:, :, slice_r, slice_c].data.cpu().numpy())

            for j in range(model.min_end_layer, model.layer_count):
                # forward computation of this layer
                if j in model.end_layers and j != model.min_end_layer:
                    low, high = x, buf[j][-1]
                    x = model.gating[pos](low, high)
                    pos += 1

                x = model.backbone[j](x)

                if j in idx_dict and j != model.min_end_layer:
                    offset = layers.index(idx_dict[j])
                    slice_r, slice_c = slice_dict[idx_dict[j]]
                    features[i][offset].append(x[:, :, slice_r, slice_c].data.cpu().numpy())

                # put it into the buffer if there is feedback connection
                for point_to in model.point_to[j]:
                    buf[point_to].append(x)

            # make final predictions
            x = x.view(x.size(0), -1)

            for j in range(len(model.classifier)):
                x = model.classifier[j](x)

                if j == 1:
                    features[i][-2].append(x.data.cpu().numpy())
                elif j == 4:
                    features[i][-1].append(x.data.cpu().numpy())


# process each image at a time
img_count = input_imgs.shape[0]
for i in range(img_count):
    img = input_imgs[i: i + 1, ...]
    img = A.Variable(torch.from_numpy(img).cuda())

    extract_feat(model, img)

# save all features in hdf5 file
with h5py.File('features.h5', 'w') as f_out:
    for i in range(model.unroll_count):
        for j in range(len(idx_dict) + 2):
            dataset_name = 'crcns_pvc-8_large/vgg-GR%d/half/%d' % (i + 1, j)

            if i > 0 and j < layers.index(idx_dict[model.min_end_layer]):
                # the layers before recurrent are always the same
                f_out.create_dataset(dataset_name, data=np.concatenate(features[0][j], axis=0))
            else:
                assert len(features[i][j]) == img_count
                f_out.create_dataset(dataset_name, data=np.concatenate(features[i][j], axis=0))
