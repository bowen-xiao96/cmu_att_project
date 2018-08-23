import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

network_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


# network make predictions at each time step
# for the structure of the classifier, see
# https://arxiv.org/pdf/1803.02563.pdf


class PredictionModule(nn.Module):
    def __init__(self, input_dim, attention_map_dim, spatial_reduce, dropout=0.5):
        super(PredictionModule, self).__init__()
        self.spatial_reduce = spatial_reduce

        self.stream_a = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(input_dim, attention_map_dim, kernel_size=1),
            nn.Dropout(p=dropout),
            nn.Softplus(beta=1.0)
        )

        self.stream_d = nn.Conv2d(input_dim, attention_map_dim, kernel_size=1)

    def forward(self, x):
        stream_a = self.stream_a(x) + 0.1

        # do spatial normalization on all channels
        spatial_sum = torch.sum(torch.sum(stream_a, dim=3, keepdim=True), dim=2, keepdim=True)
        stream_a = stream_a / spatial_sum

        stream_d = self.stream_d(x)

        # mix together and classify
        output = stream_a * stream_d

        # if spatial_reduce is on, we output the mean value of the spatial map
        if self.spatial_reduce:
            output = torch.mean(torch.mean(output, dim=3), dim=2)

        return output


class GatingModule(nn.Module):
    def __init__(self,
                 channel_low,
                 channel_high,
                 upsample,
                 upsample_rate,
                 gating_kernel):
        super(GatingModule, self).__init__()

        # this module is slightly different from the previous one
        # we first re-project the high level feature map
        # and use this re-projection to generate the gate and to mix

        if upsample == 'interpolate':
            self.upsampling = nn.Sequential(
                nn.Upsample(scale_factor=upsample_rate, mode='bilinear'),
                nn.Conv2d(channel_high, channel_low, kernel_size=1)
            )
        else:
            # `upsample` param will be a list specifying the params of the deconvolution layer
            ks, s, p, op = upsample
            self.upsampling = nn.ConvTranspose2d(
                channel_high, channel_high, kernel_size=ks, stride=s, padding=p, output_padding=op
            )

        padding = gating_kernel // 2
        self.gating = nn.Sequential(
            nn.Conv2d(channel_low * 2, channel_low, kernel_size=gating_kernel, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, low, high):
        high = self.upsampling(high)
        gate = self.gating(torch.cat((high, low), dim=1))
        return gate * high + (1.0 - gate) * low


class MultipleRecurrentModel(nn.Module):
    def __init__(self, network_cfg, connections, unroll_count, spatial_reduce, num_class, avg_pool=1, dropout=0.5):
        super(MultipleRecurrentModel, self).__init__()
        # connections: (start_layer, end_layer, start_layer_dim, end_layer_dim, scale_factor)
        # ** data flows from start_layer to end_layer **
        # scale_factor: 2**n
        # ** currently, there should not be two connections pointing to a same end_layer **

        self.unroll_count = unroll_count

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

        self.backbone.append(nn.AvgPool2d(kernel_size=avg_pool, stride=avg_pool))

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_class)
        )

        # set up recurrent connections
        self.end_layers = set([c[1] for c in connections])
        self.min_end_layer = min(self.end_layers)
        self.max_start_layer = max([c[0] for c in connections])

        self.gating_models = nn.ModuleList()
        self.point_to = [list() for _ in self.backbone]

        # sort connections according to end_layer
        connections = sorted(connections, key=lambda c: c[1])
        for start_layer, end_layer, start_layer_dim, end_layer_dim, scale_factor in connections:
            # can modify the settings below
            self.gating_models.append(
                # end layer is the low layer
                GatingModule(end_layer_dim, start_layer_dim, 'interpolate', scale_factor, 3)
            )

            self.point_to[start_layer].append(end_layer)

    def forward(self, x):
        # the input to each layer (not output)
        buf = [list() for _ in self.backbone]
        initial_buf = list()

        # layers before recurrent
        for i in range(self.min_end_layer):
            # do not actually include the starting layer
            x = self.backbone[i](x)

        initial_buf.append(x)

        # do recurrent
        for i in range(self.unroll_count):
            pos = 0  # record which gating_models to use

            if i != 0:
                # not the first unrolling step
                # we need to apply recurrent to the input of the recurrent block
                low, high = initial_buf[-2:]
                x = self.gating_models[pos](low, high)
                initial_buf.append(x)
                pos += 1

            for j in range(self.min_end_layer, self.max_start_layer + 1):
                # forward computation of this layer
                if j in self.end_layers and j != self.min_end_layer:
                    low, high = x, buf[j][-1]
                    x = self.gating_models[pos](low, high)
                    pos += 1

                x = self.backbone[j](x)

                # put it into a buffer if there is feedback connection
                for point_to in self.point_to[j]:
                    if point_to == self.min_end_layer:
                        initial_buf.append(x)
                    else:
                        buf[point_to].append(x)

        # remaining layers
        for i in range(self.max_start_layer + 1, len(self.backbone)):
            x = self.backbone[i](x)

        x = x.view(x.size(0), -1)
        return self.classifier(x)
