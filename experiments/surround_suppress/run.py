# for one model, plot the curves of surround suppression effect
# on stimulus with different orientation and frequency

import os, sys
import numpy as np
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as A
from torchvision import transforms
from torchvision.models import vgg16
from pattern import sinewave_pattern

sys.path.insert(0, '/data2/bowenx/attention/pay_attention')
from utils.model_tools import *
from model.multiple_recurrent_l import *

# create model
assert len(sys.argv) > 3
GPU_ID = int(sys.argv[1])
model_file = sys.argv[2]
# mode=0: original vgg16 pytorch model
# mode=1: recurrent gating model
mode = int(sys.argv[3])
tag = 'vgg16' if mode == 0 else 'recurrent'
if GPU_ID == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

# create model
if mode == 0:
    model = vgg16(num_classes=1000, init_weights=False)
else:
    connections = (
        (13, 8, 256, 128, 2),
        (20, 15, 512, 256, 2)
    )
    model = MultipleRecurrentModel(network_cfg, connections, 5, 1000, gating_module=GatingModule)

state_dict = torch.load(model_file)
if isinstance(state_dict, tuple):
    state_dict = state_dict[-1]

state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
model.load_state_dict(state_dict)
model.eval()
model = model.cuda()
print(model)
del state_dict

# image normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean, std)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# layers to analyze
# only consider layers with recurrent connections
layer = 15
rf = receptive_field[idx_dict[layer]]
feature_map_size = 56  # should be even
central_start_idx, central_end_idx = feature_map_size // 2 - 1, feature_map_size // 2 + 1

orientation_level = 6
min_freq, max_freq, freq_step = 10, 40, 5
min_size, max_size, size_step = 0, 60, 5

# we focus on the layers with recurrent connections
top_n = 5
total_count = 0

for f in range(min_freq, max_freq, freq_step):
    freq = float(f)

    for o in range(orientation_level):
        orientation = o * np.pi / orientation_level

        # we record the top 5 responses of all central neurons
        # and plot the average of them as the stimulus goes larger
        sizes = list(range(min_size, max_size, size_step))
        stimulus = [sinewave_pattern(size, freq, orientation) for size in sizes]

        # find the closest size of the stimulus to the receptive field
        diff = [abs(rf - size) for size in sizes]
        idx = diff.index(min(diff))
        stimulus_rf = stimulus[idx]

        stimulus_count = len(stimulus)
        stimulus = torch.stack([transform(s) for s in stimulus])
        stimulus = A.Variable(stimulus.cuda(), volatile=True)

        if mode == 0:
            all_features = extract_vgg_network_features(model, stimulus, (layer, ))
        else:
            all_features = extract_multiple_recurrent_network_features(model, stimulus, (layer, ))

        unroll_count = len(all_features)
        response_curves = list()

        for i in range(unroll_count):
            feat = all_features[i][layer]  # already numpy array
            feat = feat[:, :, central_start_idx: central_end_idx, central_start_idx: central_end_idx]
            feat = np.reshape(feat, (stimulus_count, -1))
            response_curve = list()
            for j in range(stimulus_count):
                response = np.sort(feat[j])
                response_curve.append(np.mean(response[-top_n:]))

            response_curves.append(np.array(response_curve))

        sizes = np.array(sizes)

        # plot images and save to disk
        root_dir = os.path.join(tag, 'img_%d' % total_count)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        total_count += 1

        for i in range(unroll_count + 1):
            plt.clf()
            if i == unroll_count:
                plt.axis('off')
                plt.imshow(stimulus_rf)
            else:
                plt.plot(sizes, response_curves[i])

            plt.title('layer: %d, o: %.3f, f: %d' % (layer, o, f))
            filename = os.path.join(root_dir, 'stimulus.jpg' if i == unroll_count else '%d.jpg' % i)
            plt.savefig(filename)
