import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A


class GatingModule(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, gating_kernel, reprojection_kernel, dropout=0.5):
        super(GatingModule, self).__init__()

        # use a upsampled version of high level feature map
        # (can be either interpolation and deconvolution)
        # to concatenate with the low level feature map and generate a gate
        # use this gate to mix the low level feature map and reprojected high level feature map

        if isinstance(upsample, int):
            # upsample is an int specifying the scale_factor
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_high, kernel_size=1),
                nn.ReLU(inplace=True)
            )

        else:
            # upsample param will be a tuple specifying the params of the deconvolution layer
            # kernel_size, stride, padding, output_padding
            ks, s, p, op = upsample

            # add a ReLU layer after each convolution layer
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channel_high, channel_high, kernel_size=ks, stride=s, padding=p, output_padding=op),
                nn.ReLU(inplace=True)
            )

        padding_k = gating_kernel // 2
        self.gating = nn.Sequential(
            nn.Conv2d(channel_low + channel_high, channel_low, kernel_size=gating_kernel, padding=padding_k),
            nn.Sigmoid()
        )

        padding_p = reprojection_kernel // 2
        self.reproject = nn.Sequential(
            nn.Conv2d(channel_high, channel_low, kernel_size=reprojection_kernel, padding=padding_p),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, low, high):
        upsampled_high = self.upsample(high)
        reprojected_high = self.reproject(upsampled_high)

        # concatenate along the channel and generate gating
        gate = self.gating(torch.cat((upsampled_high, low), dim=1))
        output = gate * reprojected_high + low
        return output
