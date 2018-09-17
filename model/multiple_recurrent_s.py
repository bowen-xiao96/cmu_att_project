import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

from gate import *

network_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class MultipleRecurrentModel(nn.Module):
    def __init__(self, network_cfg, connections, unroll_count, num_class, last_dim=1, dropout=0.5):
        super(MultipleRecurrentModel, self).__init__()

        self.unroll_count = unroll_count

        self.backbone = nn.ModuleList()
        input_dim = 3
        for v in network_cfg:
            if v == 'M':
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.backbone.append(nn.Conv2d(input_dim, v, kernel_size=3, padding=1))
                self.backbone.append(nn.ReLU(inplace=True))
                input_dim = v

        self.classifier = nn.Sequential(
            nn.Linear(512 * last_dim * last_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_class)
        )

        self.end_layers = set([c[1] for c in connections])
        self.min_end_layer = min(self.end_layers)
        self.layer_count = len(self.backbone)

        self.gating = nn.ModuleList()
        self.point_to = [list() for _ in self.backbone]

        connections = sorted(connections, key=lambda c: c[1])

        for high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor in connections:
            if scale_factor in (1, 2):
                gating = GatingModule(low_layer_dim, high_layer_dim, scale_factor, 3, 1, dropout=0.5)
            else:
                raise NotImplementedError

            self.gating.append(gating)
            self.point_to[high_layer].append(low_layer)

    def forward(self, x):
        buf = [list() for _ in self.backbone]

        intermediate_pred = list()

        for i in range(self.min_end_layer):
            x = self.backbone[i](x)

        buf[self.min_end_layer].append(x)

        for i in range(self.unroll_count):
            if i == 0:
                for j in range(self.min_end_layer, self.layer_count):
                    x = self.backbone[j](x)

                    for point_to in self.point_to[j]:
                        buf[point_to].append(x)

                x = x.view(x.size(0), -1)
                pred = self.classifier(x)
                intermediate_pred.append(pred)

            else:
                pos = 0

                low, high = buf[self.min_end_layer][-2:]
                x = self.gating[pos](low, high)
                buf[self.min_end_layer].append(x)
                pos += 1

                for j in range(self.min_end_layer, self.layer_count):
                    if j in self.end_layers and j != self.min_end_layer:
                        low, high = x, buf[j][-1]
                        x = self.gating[pos](low, high)
                        pos += 1

                    x = self.backbone[j](x)

                    for point_to in self.point_to[j]:
                        buf[point_to].append(x)

                x = x.view(x.size(0), -1)
                pred = self.classifier(x)
                intermediate_pred.append(pred)

        intermediate_pred = torch.stack(intermediate_pred, dim=1)
        return intermediate_pred


if __name__ == '__main__':
    # high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor
    connections = (
        (6, 3, 128, 64, 2),
        (13, 8, 256, 128, 2),
        (20, 15, 512, 256, 2),
        (27, 22, 512, 512, 2)
    )

    model = MultipleRecurrentModel(network_cfg, connections, 5, 10)
    print(model)
    model_input = A.Variable(torch.zeros(5, 3, 32, 32))
    model_output = model(model_input)
    print(model_output.size())
