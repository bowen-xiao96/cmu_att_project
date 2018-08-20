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


class RecurrentGatingRefinementModel(nn.Module):
    def __init__(self, network_cfg, unroll_count, spatial_reduce, num_class, avg_pool=1, dropout=0.5):
        super(RecurrentGatingRefinementModel, self).__init__()
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

        # recurrent connection
        # start at conv3_1, end at conv4_3
        self.start_idx = 10  # conv of conv3_1
        self.end_idx = 22  # relu after conv4_3

        self.projection = nn.Conv2d(512, 128, kernel_size=1)

        self.gating = nn.Sequential(
            nn.Conv2d(512 + 128, 128, kernel_size=3, padding=1),  # alternative: 1x1 convolution
            nn.Sigmoid()
        )

        self.intermediate_classifier = PredictionModule(128, num_class, spatial_reduce, dropout=dropout)

    def forward(self, x):
        recurrent_buf = list()

        # layers before recurrent
        for i, layer in enumerate(self.backbone):
            x = layer(x)

            if i == self.start_idx - 1:
                # the input of conv3_1
                recurrent_buf.append(x)
            elif i == self.end_idx:
                break

        intermediate_pred = list()

        # do recurrent
        for i in range(self.unroll_count):
            prev = recurrent_buf[-1]

            x = F.upsample(x, scale_factor=2, mode='bilinear')

            # gated mixing
            gate = self.gating(torch.cat((x, prev), dim=1))
            x = gate * self.projection(x) + (1.0 - gate) * prev

            # make prediction
            intermediate_pred.append(self.intermediate_classifier(x))

            # push result into the buffer
            recurrent_buf.append(x)

            # remaining layers
            for j in range(self.start_idx, self.end_idx + 1):
                x = self.backbone[j](x)

        for i in range(self.end_idx + 1, len(self.backbone)):
            x = self.backbone[i](x)

        # b1, ..., b1, b2, ..., b2
        intermediate_pred = torch.stack(intermediate_pred, dim=1)
        x = x.view(x.size(0), -1)
        final_pred = self.classifier(x)

        return intermediate_pred, final_pred


if __name__ == '__main__':
    model = RecurrentGatingRefinementModel(network_cfg, 5, False, 10)
    model.train()

    model_input = A.Variable(torch.zeros(3, 3, 32, 32))
    first_output, second_output = model(model_input)
    print(first_output.size())
    print(second_output.size())
