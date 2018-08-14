# test on imagenet high resolution images

import os, sys
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from pay_attention import *


def get_imagenet_images():
    # sample some imagenet images
    # format: raw PIL images resized to 224 * 224
    root_dir = r'/data2/leelab/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train'
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


def extract_attention_maps(model, x, size=(224, 224)):
    feature_maps = list()
    score_maps = list()

    feature_maps = list()
    for i, layer in enumerate(model.backbone):
        x = layer(x)

        # after relu layer
        if i in model.attention_layers:
            feature_maps.append(x)

    # global feature
    x = F.avg_pool2d(x, kernel_size=7, stride=7)
    x = x.view(x.size(0), -1)
    x = model.fc1(x)

    features = list()
    for i, feature_map in enumerate(feature_maps):
        if len(model.attention[i]) == 2:
            # project the global feature
            new_x = model.attention[i][0](x)
        else:
            new_x = x

        # attention score map (do 1x1 convolution on the addition of feature map and the global feature)
        score = model.attention[i][-1](feature_map + new_x.view(new_x.size(0), -1, 1, 1))
        old_shape = score.size()
        score = F.softmax(
            score.view(old_shape[0], -1), dim=1
        ).view(old_shape)

        # upsample the spatial map
        score = F.upsample(score, size=size, mode='bilinear')
        score_maps.append(score.data.cpu().numpy())

    score_maps = np.concatenate(score_maps, axis=1)
    return score_maps


if __name__ == '__main__':
    mean = np.array([0.49139968, 0.48215827, 0.44653124])
    std = np.array([0.24703233, 0.24348505, 0.26158768])
    normalize = transforms.Normalize(mean, std)

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    model_path = r'/data2/bowenx/attention/cifar/attention/best.pkl'
    _, _, state_dict = torch.load(model_path)
    model = AttentionNetwork(cfg, attention_layers, 10)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    del state_dict

    all_images = get_imagenet_images()
    batch_size = 16
    batch_count = int(math.ceil(float(len(all_images)) / batch_size))
    score_maps = list()

    for i in range(batch_count):
        images = all_images[i * batch_size: (i + 1) * batch_size]
        images = A.Variable(torch.stack([transform(img) for img in images]).cuda())

        # get score map
        score_maps.append(extract_attention_maps(model, images))

    score_maps = np.concatenate(score_maps, axis=0)
    all_images = np.stack([np.array(img) for img in all_images])

    np.savez('imagenet.npz', images=all_images, score_maps=score_maps)
