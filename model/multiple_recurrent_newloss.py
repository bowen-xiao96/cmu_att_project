import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

sys.path.insert(0, '/data2/bowenx/attention/pay_attention')
from model.gate import *

network_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class MultipleRecurrentModel(nn.Module):
    def __init__(self, network_cfg, connections, unroll_count, num_class, loss_scheme,
                 gating_module=GatingModule, last_dim=7, dropout=0.5):
        super(MultipleRecurrentModel, self).__init__()

        # connections: (high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor)
        # high_layer, low_layer: layer index
        # scale_factor: 2**n

        # loss_scheme: (mode, params...)
        # mode can be 'each', 'final' and 'refine'
        # if mode is 'each' or 'refine', we need a param controlling the weight of losses

        # ** data flows from high_layer to low_layer (recurrent) **
        # ** currently, there should not be two connections pointing to a same low_layer **

        self.unroll_count = unroll_count

        self.loss_type = loss_scheme[0]
        if self.loss_type in ('each', 'refine'):
            # loss_scheme[1] is the gamma parameter
            gamma = float(loss_scheme[1])

            # like (gamma**5, gamma**4, gamma**3, gamma**2, gamma)
            weights = list()
            for i in range(unroll_count):
                if i == 0:
                    weights.append(gamma)
                else:
                    weights.append(weights[-1] * gamma)

            if self.loss_type == 'refine':
                weights.pop(-1)

            weights = torch.FloatTensor(list(reversed(weights)))
            weights /= torch.sum(weights)
            print('Loss weights for different time steps:')
            print(weights)

            self.gamma_weights_ = weights
            self.register_buffer('gamma_weights', self.gamma_weights_)

            if self.loss_type == 'refine':
                # we need another alpha weight to control the balance of the two losses
                # loss_scheme[2] is the alpha parameter
                alpha = float(loss_scheme[2])

                self.alpha_weights_ = torch.FloatTensor([alpha])
                self.register_buffer('alpha_weights', self.alpha_weights_)

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
        self.max_start_layer = max([c[0] for c in connections])
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
                gating = gating_module(low_layer_dim, high_layer_dim, (3, 1, 1, 0), 3, 1, dropout=0.5)
            elif scale_factor == 2:
                gating = gating_module(low_layer_dim, high_layer_dim, (3, 2, 1, 1), 3, 1, dropout=0.5)
            else:
                # TODO: long connection across two blocks
                raise NotImplementedError

            self.gating.append(gating)
            self.point_to[high_layer].append(low_layer)

    def forward(self, x, y):
        # output prediction result as well as loss
        if self.loss_type in ('each', 'refine'):
            # the input to each layer (not output)
            buf = [list() for _ in self.backbone]

            # the intermediate predictions of the network
            # the output should be a batch_size * unroll_count * num_class Tensor
            intermediate_pred = list()

            # layers before recurrent start
            for i in range(self.min_end_layer):
                # do not actually include the starting layer
                x = self.backbone[i](x)

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

                    # make final predictions
                    x = x.view(x.size(0), -1)
                    pred = self.classifier(x)
                    intermediate_pred.append(pred)

                else:
                    # we need to apply recurrent to the input of each recurrent block
                    # and make intermediate predictions using the prediction module

                    # record which gating_models to use
                    pos = 0

                    low, high = buf[self.min_end_layer][-2:]
                    x = self.gating[pos](low, high)
                    buf[self.min_end_layer].append(x)
                    pos += 1

                    for j in range(self.min_end_layer, self.layer_count):
                        # forward computation of this layer
                        if j in self.end_layers and j != self.min_end_layer:
                            low, high = x, buf[j][-1]
                            x = self.gating[pos](low, high)
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

            # calculate loss function
            if self.loss_type == 'each':
                # batch_size * unroll_count * num_class
                old_size = intermediate_pred.size()
                # flatten into (-1, num_class)
                batch_size, time_step = old_size[:2]
                new_size = torch.Size([batch_size * time_step] + list(old_size[2:]))

                loss_1 = F.cross_entropy(
                    intermediate_pred.view(new_size),
                    y.repeat(time_step, 1).transpose(0, 1).contiguous().view(-1),
                    reduce=False
                )
                # average over batches
                loss = torch.sum(loss_1 * self.gamma_weights.repeat(batch_size)) / batch_size

                return intermediate_pred, loss

            elif self.loss_type == 'refine':
                # first is the standard cross entropy loss at the final time step
                batch_size = intermediate_pred.size(0)
                loss_1 = F.cross_entropy(intermediate_pred[:, -1, :], y)

                # then is the so-called refinement loss
                margin = 0.05  # hard-coded margin parameter
                prob = F.softmax(intermediate_pred, dim=-1)
                diff = prob[:, :-1, y] - prob[:, 1:, y] + margin
                diff = F.relu(diff, inplace=True)

                loss_2 = torch.sum(diff.view(-1) * self.gamma_weights.repeat(batch_size)) / batch_size
                loss = loss_1 + self.alpha_weights * loss_2
                return intermediate_pred, loss

        else:
            # loss_type is 'final'
            # in this case, we only need prediction at the final time step
            # no need to do redundant computation at intermediate time steps

            # the input to each layer (not output)
            buf = [list() for _ in self.backbone]

            intermediate_pred = list()

            # layers before recurrent start
            for i in range(self.min_end_layer):
                # do not actually include the starting layer
                x = self.backbone[i](x)

            buf[self.min_end_layer].append(x)

            # do recurrent stuff
            for i in range(self.unroll_count):
                if i == 0:
                    # the first unrolling step
                    # just straightforward forward computation
                    for j in range(self.min_end_layer, self.max_start_layer + 1):
                        x = self.backbone[j](x)

                        # put it into the buffer if there is feedback connection
                        for point_to in self.point_to[j]:
                            buf[point_to].append(x)

                else:
                    # we need to apply recurrent to the input of each recurrent block
                    # and make intermediate predictions using the prediction module

                    # record which gating_models to use
                    pos = 0

                    low, high = buf[self.min_end_layer][-2:]
                    x = self.gating[pos](low, high)
                    buf[self.min_end_layer].append(x)
                    pos += 1

                    for j in range(self.min_end_layer, self.max_start_layer + 1):
                        # forward computation of this layer
                        if j in self.end_layers and j != self.min_end_layer:
                            low, high = x, buf[j][-1]
                            x = self.gating[pos](low, high)
                            pos += 1

                        x = self.backbone[j](x)

                        # put it into the buffer if there is feedback connection
                        for point_to in self.point_to[j]:
                            buf[point_to].append(x)

                    # if this is final unrolling time step, make predictions using the remaining layers
                    if i == self.unroll_count - 1:
                        for j in range(self.max_start_layer + 1, self.layer_count):
                            x = self.backbone[j](x)

                        x = x.view(x.size(0), -1)
                        x = self.classifier(x)
                        intermediate_pred.append(x)

            # calculate loss and return final prediction result
            final_pred = intermediate_pred[0]
            loss = F.cross_entropy(final_pred, y)
            return final_pred, loss


if __name__ == '__main__':
    # high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor
    connections = (
        (6, 3, 128, 64, 2),
        (13, 8, 256, 128, 2),
        (20, 15, 512, 256, 2),
        (27, 22, 512, 512, 2)
    )

    # works for imagenet images
    model = MultipleRecurrentModel(network_cfg, connections, 5, 1000, ('final', ))
    print(model)

    # some all-zero input
    model_input = A.Variable(torch.zeros(5, 3, 224, 224))
    model_label = A.Variable(torch.zeros(5).long())
    model_output, model_loss = model(model_input, model_label)

    # check for the shape of the output tensors
    print(model_output.size())
    print(model_loss.size())
