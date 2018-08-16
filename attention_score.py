import os, sys
import random
import pickle

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pay_attention import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
IMAGE_SIZE = 224


# training the model on these selected images
# below is a monkey patch
def forward(model, x):
    feature_maps = list()
    for i, layer in enumerate(model.backbone):
        x = layer(x)

        # after relu layer
        if i in model.attention_layers:
            feature_maps.append(x)

    # global feature
    x = x.view(x.size(0), -1)
    x = model.fc1(x)

    score_maps = list()
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

        # upsample the score map
        score_maps.append(F.upsample(score, (IMAGE_SIZE, IMAGE_SIZE), mode='bilinear'))

        # weighted sum the feature map
        weighted_sum = torch.sum(torch.sum(score * feature_map, dim=3), dim=2)
        features.append(weighted_sum)

    scores = model.classifier(torch.cat(features, dim=1))

    # output is a 3 x batch_size x IMAGE_SIZE x IMAGE_SIZE map
    score_maps = torch.squeeze(torch.stack(score_maps))
    return scores, score_maps


# image transform
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean, std)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


def train_without_novel(input_data, num_epoch, batch_size, save_step):
    img_count = len(input_data)
    imgs, labels = zip(*input_data)
    imgs = np.stack([train_transform(img).numpy() for img in imgs])
    labels = np.array(labels, dtype=np.int64)

    # set up model and train on these images
    model = AttentionNetwork(cfg, attention_layers, 10, avg_pool=7)
    model.forward = forward
    initialize_vgg(model)

    model.train()
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )

    output_data = list()
    for i in range(num_epoch):
        idx = np.array(random.sample(range(img_count), batch_size))
        x = A.Variable(torch.from_numpy(imgs[idx, ...]).cuda())
        y = A.Variable(torch.from_numpy(labels[idx]).cuda())

        pred, scores = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % save_step == 0:
            output_data.append((pred.data.cpu().numpy(), scores.data.cpu().numpy()))

    with open('without_novel.pkl', 'wb') as f_out:
        pickle.dump(output_data, f_out, pickle.HIGHEST_PROTOCOL)


def train_with_novel(familiar_data, novel_data, num_epoch, familiar_size, novel_size, save_step):
    random.shuffle(novel_data)
    novel_data_pos = 0

    familiar_data = [
        (train_transform(img).numpy(), label) for img, label in familiar_data
    ]

    novel_data = [
        (train_transform(img).numpy(), label) for img, label in novel_data
    ]

    # set up model and train on these images
    model = AttentionNetwork(cfg, attention_layers, 10, avg_pool=7)
    model.forward = forward
    initialize_vgg(model)

    model.train()
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )

    output_data = list()
    for i in range(num_epoch):
        f_sample = random.sample(familiar_data, familiar_size)
        n_sample = novel_data[novel_data_pos: novel_data_pos + novel_size]
        novel_data_pos += novel_size

        x, y = zip(*(f_sample + n_sample))
        x = A.Variable(torch.from_numpy(np.stack(x)).cuda())
        y = A.Variable(torch.from_numpy(np.array(y, dtype=np.int64)).cuda())

        pred, scores = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % save_step == 0:
            output_data.append((pred.data.cpu().numpy(), scores.data.cpu().numpy()))

    with open('with_novel.pkl', 'wb') as f_out:
        pickle.dump(output_data, f_out, pickle.HIGHEST_PROTOCOL)
