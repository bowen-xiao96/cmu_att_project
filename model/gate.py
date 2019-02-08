import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A


# original module
# horizontal gate top-down
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
                nn.Conv2d(channel_high, channel_high, kernel_size=3),
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


# gating scheme 1
# top-down gate horizontal
class GatingModule1(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, gating_kernel, horizontal_kernel, dropout=0.5):
        super(GatingModule1, self).__init__()

        if isinstance(upsample, int):
            # upsample is an int specifying the scale_factor
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_high, kernel_size=3),
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
            nn.Conv2d(channel_high, channel_low, kernel_size=gating_kernel, padding=padding_k),
            nn.Sigmoid()
        )

        padding_h = horizontal_kernel // 2
        self.horizontal = nn.Sequential(
            nn.Conv2d(channel_low + channel_high, channel_low, kernel_size=horizontal_kernel, padding=padding_h),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, low, high):
        upsampled_high = self.upsample(high)

        gate = self.gating(upsampled_high)
        horizontal = self.horizontal(torch.cat((upsampled_high, low), dim=1))
        output = gate * horizontal + low
        return output


# gating scheme 2
# horizontal gate horizontal
class GatingModule2(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, gating_kernel, horizontal_kernel, dropout=0.5):
        super(GatingModule2, self).__init__()

        if isinstance(upsample, int):
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_high, kernel_size=3),
                nn.ReLU(inplace=True)
            )

        else:
            ks, s, p, op = upsample

            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channel_high, channel_high, kernel_size=ks, stride=s, padding=p, output_padding=op),
                nn.ReLU(inplace=True)
            )

        padding_h = horizontal_kernel // 2
        self.horizontal = nn.Sequential(
            nn.Conv2d(channel_low + channel_high, channel_low, kernel_size=horizontal_kernel, padding=padding_h),
            nn.ReLU(inplace=True)
        )

        padding_k = gating_kernel // 2
        self.gating = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=gating_kernel, padding=padding_k),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, low, high):
        upsampled_high = self.upsample(high)

        horizontal = self.horizontal(torch.cat((upsampled_high, low), dim=1))
        gate = self.gating(horizontal)
        output = gate * self.dropout(horizontal) + low
        return output


# gating scheme 3
# cross gating
class GatingModule3(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, gating_kernel_h, gating_kernel_l, dropout=0.5):
        super(GatingModule3, self).__init__()

        # note that we change the dimensionality in the upsampling process
        if isinstance(upsample, int):
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_low, kernel_size=3),
                nn.ReLU(inplace=True)
            )

        else:
            ks, s, p, op = upsample

            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channel_high, channel_low, kernel_size=ks, stride=s, padding=p, output_padding=op),
                nn.ReLU(inplace=True)
            )

        # gate generated by high-level layer gating low-level layer
        padding_h = gating_kernel_h // 2
        self.gating_h = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=gating_kernel_h, padding=padding_h),
            nn.Sigmoid()
        )
        self.dropout_h = nn.Dropout(p=dropout)

        # gate generated by low-level layer gating high-level layer
        padding_l = gating_kernel_l // 2
        self.gating_l = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=gating_kernel_l, padding=padding_l),
            nn.Sigmoid()
        )
        self.dropout_l = nn.Dropout(p=dropout)

    def forward(self, low, high):
        upsampled_high = self.upsample(high)

        gated_l = self.gating_h(upsampled_high) * self.dropout_h(low)
        gated_h = self.gating_l(low) * self.dropout_l(upsampled_high)
        output = gated_h + gated_l + low
        return output


# gate scheme 4
# horizontal gate top-down and top-down gate horizontal
class GatingModule4(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, gating_kernel, horizontal_kernel, dropout=0.5):
        super(GatingModule4, self).__init__()

        # note that we change the dimensionality in the upsampling process
        if isinstance(upsample, int):
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_low, kernel_size=3),
                nn.ReLU(inplace=True)
            )

        else:
            ks, s, p, op = upsample

            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channel_high, channel_low, kernel_size=ks, stride=s, padding=p, output_padding=op),
                nn.ReLU(inplace=True)
            )

        padding_h = horizontal_kernel // 2
        self.horizontal = nn.Sequential(
            nn.Conv2d(channel_low + channel_low, channel_low, kernel_size=horizontal_kernel, padding=padding_h),
            nn.ReLU(inplace=True)
        )

        padding_g = gating_kernel // 2

        # gate generated by horizontal layer gating high-level layer
        self.gating_horizontal = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=gating_kernel, padding=padding_g),
            nn.Sigmoid()
        )
        self.dropout_horizontal = nn.Dropout(p=dropout)

        # gate generated by high-level layer gating horizontal layer
        self.gating_high = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=gating_kernel, padding=padding_g),
            nn.Sigmoid()
        )
        self.dropout_high = nn.Dropout(p=dropout)

    def forward(self, low, high):
        upsampled_high = self.upsample(high)

        horizontal = self.horizontal(torch.cat((upsampled_high, low), dim=1))
        gated_horizontal = self.gating_high(upsampled_high) * self.dropout_high(horizontal)
        gated_high = self.gating_horizontal(horizontal) * self.dropout_horizontal(upsampled_high)
        outout = gated_high + gated_horizontal + low
        return outout

# gating scheme 5
# baseline concat
class GatingModule5(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, hidden_gate_kernel=1, hidden_gs_kernel=1, memory_gate_kernel=1, memory_gs_kernel=1, dropout=0.5):
        super(GatingModule5, self).__init__()

        if isinstance(upsample, int):
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_high, kernel_size=3),
                nn.ReLU(inplace=True)
            )

        else:
            ks, s, p, op = upsample

            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channel_high, channel_high, kernel_size=ks, stride=s, padding=p, output_padding=op),
                nn.ReLU(inplace=True)
            )

        padding_h = hidden_gate_kernel // 2
        self.hidden_gate = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=hidden_gate_kernel, padding=padding_h),
            nn.Sigmoid()
        )

        padding_h_gs = hidden_gs_kernel // 2
        self.hidden_gs = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=hidden_gs_kernel, padding=padding_h_gs),
            nn.Sigmoid()
        )

        padding_m = memory_gate_kernel // 2
        self.memory_gate = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=memory_gate_kernel, padding=padding_m),
            nn.Sigmoid()
        )

        padding_m_gs = memory_gs_kernel // 2
        self.memory_gs = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=memory_gs_kernel, padding=padding_m_gs),
            nn.Sigmoid()
        )


        self.dropout = nn.Dropout(p=dropout)
        self.align_d = nn.Conv2d(channel_low+channel_high, channel_low, kernel_size=1, padding=0)

    def forward(self, low, high, memory_low):

        upsampled_high = self.upsample(high)       
        x_input = self.align_d(torch.cat((upsampled_high, low), dim=1))
        
        c_output = (1 - self.hidden_gate(low)) * x_input + (1 - self.memory_gs(memory_low)) * memory_low
        
        output = (1 - self.memory_gate(memory_low)) * x_input + (1 - self.hidden_gs(low)) * low

        return output, c_output

# gate scheme 6
# baseline model use add operation replace the gate
class GatingModule6(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, gating_kernel, reprojection_kernel, dropout=0.5):
        super(GatingModule6, self).__init__()

        # use a upsampled version of high level feature map
        # (can be either interpolation and deconvolution)
        # to concatenate with the low level feature map and generate a gate
        # use this gate to mix the low level feature map and reprojected high level feature map

        if isinstance(upsample, int):
            # upsample is an int specifying the scale_factor
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_high, kernel_size=3),
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
        output = gate + low
        return output

# gate scheme 7
# baseline model use multiply operation replace the gate
class GatingModule7(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, gating_kernel, reprojection_kernel, dropout=0.5):
        super(GatingModule7, self).__init__()

        # use a upsampled version of high level feature map
        # (can be either interpolation and deconvolution)
        # to concatenate with the low level feature map and generate a gate
        # use this gate to mix the low level feature map and reprojected high level feature map

        if isinstance(upsample, int):
            # upsample is an int specifying the scale_factor
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_high, kernel_size=3),
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
        output = gate * low
        return output

# gating scheme 8
# top-down gate real horizontal
class GatingModule8(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, gating_kernel, horizontal_kernel, dropout=0.5):
        super(GatingModule8, self).__init__()

        if isinstance(upsample, int):
            # upsample is an int specifying the scale_factor
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_high, kernel_size=3),
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
            nn.Conv2d(channel_high, channel_low, kernel_size=gating_kernel, padding=padding_k),
            nn.Sigmoid()
        )
        
        padding_h = horizontal_kernel // 2
        self.horizontal = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=horizontal_kernel, padding=padding_h),
            nn.ReLU(inplace=True),
        )
        padding_c = 3 // 2
        self.contextual = nn.Sequential(
            nn.Conv2d(channel_low + channel_high, channel_low, kernel_size=horizontal_kernel, padding=padding_h),
            nn.Tanh()
        )

    def forward(self, low, high):
        upsampled_high = self.upsample(high)

        gate = self.gating(upsampled_high)

        horizontal = self.horizontal(low)

        contextual = self.contextual(torch.cat((upsampled_high, horizontal), dim=1))
        output = gate * contextual + low

        return output

# gating scheme 8 control
# top-down gate real horizontal
class GatingModule8_Ctl(nn.Module):
    def __init__(self, channel_low, channel_high, upsample, gating_kernel, horizontal_kernel, dropout=0.5):
        super(GatingModule8_Ctl, self).__init__()

        if isinstance(upsample, int):
            # upsample is an int specifying the scale_factor
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample, mode='bilinear'),
                nn.Conv2d(channel_high, channel_high, kernel_size=3),
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
            nn.Conv2d(channel_high, channel_low, kernel_size=gating_kernel, padding=padding_k),
            nn.Sigmoid()
        )
        
        padding_h = horizontal_kernel // 2
        self.horizontal = nn.Sequential(
            nn.Conv2d(channel_low, channel_low, kernel_size=horizontal_kernel, padding=padding_h),
            nn.ReLU(inplace=True),
        )
        padding_c = 3 // 2
        self.contextual = nn.Sequential(
            nn.Conv2d(channel_low + channel_high, channel_low, kernel_size=horizontal_kernel, padding=padding_h),
            nn.Tanh()
        )

    def forward(self, low, high):
        upsampled_high = self.upsample(high)

        gate = self.gating(upsampled_high)

        horizontal = self.horizontal(low)

        contextual = self.contextual(torch.cat((upsampled_high, horizontal), dim=1))
        output = contextual + low

        return output
