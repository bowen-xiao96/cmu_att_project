import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

cfg = [64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 'M', 512, 'M']


class VGG16Modified(nn.Module):
    def __init__(self, cfg, num_class, avg_pool=1, dropout=0.5):
        super(VGG16Modified, self).__init__()

        backbone = list()
        input_dim = 3
        for v in cfg:
            if v == 'M':
                backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                backbone.append(nn.Conv2d(input_dim, v, kernel_size=3, padding=1))
                backbone.append(nn.ReLU(inplace=True))
                input_dim = v

        backbone.append(nn.AvgPool2d(kernel_size=avg_pool, stride=avg_pool))

        self.backbone = nn.Sequential(*backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
