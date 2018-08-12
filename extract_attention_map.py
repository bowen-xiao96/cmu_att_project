import numpy as np

import torch
import torch.nn.functional as F
import torch.autograd as A

from pay_attention import *


def extract_attention_maps(model, x, size=(32, 32)):
    feature_maps = list()
    score_maps = list()

    feature_maps = list()
    for i, layer in enumerate(model.backbone):
        x = layer(x)

        # after relu layer
        if i in model.attention_layers:
            feature_maps.append(x)

    # global feature
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

    size = 32
    sample_count = 200
    cifar10_dir = '/data2/bowenx/attention/cifar'

    model_path = r'/data2/bowenx/attention/cifar/attention/best.pkl'
    _, _, state_dict = torch.load(model_path)
    model = AttentionNetwork(cfg, attention_layers, 10)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    del state_dict

    for flag in ('train', 'test'):
        dataset = CIFAR10(
            cifar10_dir,
            train=True if flag == 'train' else False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=sample_count,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: zip(*batch),
            pin_memory=True,
            drop_last=True
        )

        for x, _ in dataloader:
            x_var = A.Variable(
                torch.stack([transform(img) for img in x]).cuda()
            )

            score_maps = extract_attention_maps(model, x_var)
            images = np.stack([np.array(img) for img in x])
            np.savez(flag + '.npz', images=images, score_maps=score_maps)
            break
