import math

import torch
import torch.nn as nn

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

receptive_field = {
    'conv1_1': 3,
    'conv1_2': 5,
    'pool1': 6,
    'conv2_1': 10,
    'conv2_2': 14,
    'pool2': 16,
    'conv3_1': 24,
    'conv3_2': 32,
    'conv3_3': 40,
    'pool3': 44,
    'conv4_1': 60,
    'conv4_2': 76,
    'conv4_3': 92,
    'pool4': 100,
    'conv5_1': 132,
    'conv5_2': 164,
    'conv5_3': 196,
    'pool5': 212
}


def initialize_vgg(model):
    for m in model.modules():
        # leave fc layers default initialized

        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal(m.weight.data, mean=0.0, std=math.sqrt(2.0 / n))
            nn.init.constant(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight.data, 1.0)
            nn.init.constant(m.bias.data, 0.0)


def extract_multiple_recurrent_network_features(model, x, layers):
    # for each unrolling count, we extract the features of the given layers
    features = [dict() for _ in range(model.unroll_count)]
    buf = [list() for _ in model.backbone]

    for i in range(model.min_end_layer):
        x = model.backbone[i](x)
        if i in layers:
            features[0][i] = x.data.cpu().numpy()

    buf[model.min_end_layer].append(x)

    for i in range(model.unroll_count):
        if i == 0:
            for j in range(model.min_end_layer, model.layer_count):
                x = model.backbone[j](x)
                if j in layers:
                    features[0][j] = x.data.cpu().numpy()

                for point_to in model.point_to[j]:
                    buf[point_to].append(x)

            x = x.view(x.size(0), -1)

            for j in range(len(model.classifier)):
                x = model.classifier[j](x)

                # for classifier layers, we recalculate layer index
                real_idx = len(model.backbone) + j
                if real_idx in layers:
                    features[0][real_idx] = x.data.cpu().numpy()

        else:
            pos = 0

            # we do gating stuff here at the beginning of each unrolling step
            low, high = buf[model.min_end_layer][-2:]
            x = model.gating[pos](low, high)
            buf[model.min_end_layer].append(x)
            pos += 1

            for j in range(model.min_end_layer, model.layer_count):
                if j in model.end_layers and j != model.min_end_layer:
                    low, high = x, buf[j][-1]
                    x = model.gating[pos](low, high)
                    pos += 1

                x = model.backbone[j](x)
                # the feature is defined as after the backbone layer (instead of before)
                if j in layers:
                    features[i][j] = x.data.cpu().numpy()

                for point_to in model.point_to[j]:
                    buf[point_to].append(x)

            x = x.view(x.size(0), -1)

            for j in range(len(model.classifier)):
                x = model.classifier[j](x)
                real_idx = len(model.backbone) + j
                if real_idx in layers:
                    features[i][real_idx] = x.data.cpu().numpy()

    return features


def extract_vgg_network_features(model, x, layers):
    features = dict()

    for i in range(len(model.features)):
        x = model.features[i](x)
        if i in layers:
            features[i] = x.data.cpu().numpy()

    x = x.view(x.size(0), -1)

    for j in range(len(model.classifier)):
        x = model.classifier[j](x)
        real_idx = len(model.features) + j
        if real_idx in layers:
            features[real_idx] = x.data.cpu().numpy()

    return [features]
