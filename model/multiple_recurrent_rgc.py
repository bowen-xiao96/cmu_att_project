import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

sys.path.insert(0, '/home/simingy/cmu_att_project/')
from model.gate import *

network_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class MultipleRecurrentModel_RGC(nn.Module):
    def __init__(self, network_cfg, connections, unroll_count, num_class,
                 gating_module=GatingModule, last_dim=7, dropout=0.5, recurrent_count=1, rgc=0):
        super(MultipleRecurrentModel_RGC, self).__init__()

        # connections: (high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor)
        # high_layer, low_layer: layer index
        # scale_factor: 2**n

        # ** data flows from high_layer to low_layer (recurrent) **
        # ** currently, there should not be two connections pointing to a same low_layer **

        self.unroll_count = unroll_count
        self.memory_idx = list()
        self.recurrent_cnt = recurrent_count

        # backbone network
        self.backbone = nn.ModuleList()
        input_dim = 3
        for v in network_cfg:
            if v == 'M':
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.backbone.append(nn.Conv2d(input_dim, v, kernel_size=3, padding=1))
                self.backbone.append(nn.ReLU(inplace=True))
                input_dim = v

        # we use fully connected layer as classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * last_dim * last_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_class)
        )

        # set up recurrent connections
        self.end_layers = set([c[1] for c in connections])
        self.min_end_layer = min(self.end_layers)
        self.layer_count = len(self.backbone)

        self.gating = nn.ModuleList()
        self.point_to = [list() for _ in self.backbone]

        # sort connections according to low_layer
        # since there should not be two connections pointing to a same low_layer
        # the modules in classifiers and gating modules are also in this order (low_layer)
        connections = sorted(connections, key=lambda c: c[1])

        for high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor in connections:
            # can modify the settings below
            # kernel_size, stride, padding, output_padding

            if scale_factor == 1:
                gating = gating_module(low_layer_dim, high_layer_dim, (3, 1, 1, 0), 1, 3, dropout=0.5)
            elif scale_factor == 2 and rgc == 0:
                gating = gating_module(low_layer_dim, high_layer_dim, (3, 2, 1, 1), 1, 3, dropout=0.5)
            elif scale_factor == 2 and rgc == 1:
                gating = gating_module(low_layer_dim, high_layer_dim, (3, 2, 1, 1))
            else:
                # TODO: long connection across two blocks
                raise NotImplementedError

            self.gating.append(gating)
            self.point_to[high_layer].append(low_layer)
            self.memory_idx.append(low_layer)

    def forward(self, x):
        # each unrolling step, the model makes prediction at the final classifier

        # the input to each layer (not output)
        buf = [list() for _ in self.backbone]
        memory_buf = [list() for _ in range(self.recurrent_cnt)]
        memory_cnt = 0

        # the intermediate predictions of the network
        # the output should be a batch_size * unroll_count * num_class Tensor
        intermediate_pred = list()

        # layers before recurrent start
        for i in range(self.min_end_layer):
            # do not actually include the starting layer
            x = self.backbone[i](x)
            if i in self.memory_idx:
                memory_buf[memory_cnt].append(x)
                memory_cnt += 1

        buf[self.min_end_layer].append(x)
       
        # do recurrent stuff
        for i in range(self.unroll_count):
            if i == 0:
                # the first unrolling step
                # just straightforward forward computation
                for j in range(self.min_end_layer, self.layer_count):
                    x = self.backbone[j](x)

                    # put it into the buffer if there is feedback connection
                    for point_to in self.point_to[j]:
                        buf[point_to].append(x)

                    if j in self.memory_idx:
                        memory_buf[memory_cnt].append(x)
                        memory_cnt += 1


                # make final predictions
                x = x.view(x.size(0), -1)
                pred = self.classifier(x)
                intermediate_pred.append(pred)

            else:
                # we need to apply recurrent to the input of each recurrent block
                # and make intermediate predictions using the prediction module

                # record which gating_models to use
                pos = 0
                memory_cnt = 0

                low, high = buf[self.min_end_layer][-2:]
                low_memory = memory_buf[memory_cnt][-1]
                #print("low_memory:", low_memory.size())
                x, low_memory_update = self.gating[pos](low, high, low_memory)

                memory_buf[memory_cnt].append(low_memory_update)
                memory_cnt += 1

                buf[self.min_end_layer].append(x)
                pos += 1

                for j in range(self.min_end_layer, self.layer_count):
                    # forward computation of this layer
                    if j in self.end_layers and j != self.min_end_layer:
                        low, high = x, buf[j][-1]
                        low_memory = memory_buf[memory_cnt][-1]
                        x, low_memory_update = self.gating[pos](low, high, low_memory)
                        memory_buf[memory_cnt].append(low_memory_update)
                        memory_cnt += 1
                        pos += 1

                    x = self.backbone[j](x)

                    # put it into the buffer if there is feedback connection
                    for point_to in self.point_to[j]:
                        buf[point_to].append(x)

                # make final predictions
                x = x.view(x.size(0), -1)
                pred = self.classifier(x)
                intermediate_pred.append(pred)

        # collect predictions
        # batch_size * unroll_count * num_class
        intermediate_pred = torch.stack(intermediate_pred, dim=1)
        return intermediate_pred


if __name__ == '__main__':
    # high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor
    connections = (
        #(6, 3, 128, 64, 2),
        (13, 8, 256, 128, 2),
        (20, 15, 512, 256, 2),
        #(27, 22, 512, 512, 2)
    )

    # works for imagenet images
    model = MultipleRecurrentModel_RGC(network_cfg, connections, 5, 1000, recurrent_count=2)
    print(model)
    model_input = A.Variable(torch.zeros(5, 3, 224, 224))
    model_output = model(model_input)
    print(model_output.size())
