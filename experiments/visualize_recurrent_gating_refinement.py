# test the model on cifar-10 images
import os, sys
import types
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def prediction_forward(model, x):
    stream_a = model.stream_a(x) + 0.1

    # do spatial normalization on all channels
    spatial_sum = torch.sum(torch.sum(stream_a, dim=3, keepdim=True), dim=2, keepdim=True)
    stream_a = stream_a / spatial_sum

    stream_d = model.stream_d(x)

    # mix together and classify
    output = stream_a * stream_d

    # perform spatial average
    spatial_ave = torch.mean(torch.mean(output, dim=3), dim=2)
    # perform softmax over channels
    channel_softmax = F.softmax(output, dim=1)

    return spatial_ave, channel_softmax


def model_forward(model, x):
    recurrent_buf = list()

    # layers before recurrent
    for i, layer in enumerate(model.backbone):
        x = layer(x)

        if i == model.start_idx - 1:
            # the input of conv3_1
            recurrent_buf.append(x)
        elif i == model.end_idx:
            break

    intermediate_pred = list()
    intermediate_score_map = list()

    # do recurrent
    for i in range(model.unroll_count):
        prev = recurrent_buf[-1]

        x = F.upsample(x, scale_factor=2, mode='bilinear')

        # gated mixing
        gate = model.gating(torch.cat((x, prev), dim=1))
        x = gate * model.projection(x) + (1.0 - gate) * prev

        # make prediction
        pred, score_map = model.intermediate_classifier(x)
        score_map = F.upsample(score_map, scale_factor=4, mode='bilinear')
        intermediate_pred.append(pred)
        intermediate_score_map.append(score_map)

        # push result into the buffer
        recurrent_buf.append(x)

        # remaining layers
        for j in range(model.start_idx, model.end_idx + 1):
            x = model.backbone[j](x)

    for i in range(model.end_idx + 1, len(model.backbone)):
        x = model.backbone[i](x)

    # batch_size * unroll_count * C
    intermediate_pred = torch.stack(intermediate_pred, dim=1)
    x = x.view(x.size(0), -1)
    # batch_size * unroll_count * C * H * W
    intermediate_score_map = torch.stack(intermediate_score_map, dim=1)
    final_pred = model.classifier(x)

    return intermediate_pred, intermediate_score_map, final_pred


if __name__ == '__main__':
    assert len(sys.argv) > 1
    flag = int(sys.argv[1])

    if flag == 0:
        sys.path.insert(0, '/data2/bowenx/attention/pay_attention')

        from model.recurrent_gating_refinement import *
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        from utils.visualize_scoremap import draw_image

        model = RecurrentGatingRefinementModel(network_cfg, 5, True, 10)

        # monkey patches
        model.forward = types.MethodType(model_forward, model)
        model.intermediate_classifier.forward = types.MethodType(
            prediction_forward, model.intermediate_classifier
        )

        # load network parameters
        model_path = r'/data2/bowenx/attention/pay_attention/experiments/recurrent_gating_refinement_cifar_5/best.pkl'
        _, _, state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        model.cuda()
        del state_dict

        cifar_path = '/data2/bowenx/attention/pay_attention/dataset/cifar'
        train_dataset = CIFAR10(cifar_path, train=True)
        test_dataset = CIFAR10(cifar_path, train=True)

        mean = np.array([0.49139968, 0.48215827, 0.44653124])
        std = np.array([0.24703233, 0.24348505, 0.26158768])
        normalize = transforms.Normalize(mean, std)

        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        output = dict()

        sample_num = 50
        for tag, dataset in (('train', train_dataset), ('test', test_dataset)):
            samples = random.sample(range(len(dataset)), sample_num)
            output[flag] = list()

            for j in samples:
                img, label = dataset[j]
                # img is a PIL image

                img_var = A.Variable(torch.unsqueeze(transform(img), dim=0).cuda())
                intermediate_pred, intermediate_score_map, final_pred = \
                    [torch.squeeze(x).data.cpu().numpy() for x in model(img_var)]

                intermediate_score_map = np.transpose(intermediate_score_map, (1, 0, 2, 3))
                canvas = draw_image(
                    [img for _ in range(intermediate_score_map.shape[0])],
                    intermediate_score_map,
                    size=32,
                    padding=5
                )

                output[flag].append((canvas, intermediate_pred, final_pred, label))

        with open('visualization.pkl', 'wb') as f_out:
            pickle.dump(output, f_out, pickle.HIGHEST_PROTOCOL)

    elif flag == 1:
        # visual results on local host
        import pickle
        import matplotlib.pyplot as plt

        with open('visualization.pkl', 'rb') as f_in:
            data = pickle.load(f_in)

        for tag in ('train', 'test'):
            for i, (canvas, intermediate_pred, final_pred, label) in enumerate(data[tag]):
                print('Image: %d' % i)

                # print prediction result
                print(intermediate_pred[:, label])
                print(final_pred[label])

                # draw visualization
                plt.clf()
                plt.imshow(canvas)
                plt.show()
