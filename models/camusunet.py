import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convs(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, mode='bilinear'):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.conv = DoubleConv(in_channels+out_channels, out_channels)

    def forward(self, x1, x2):
        out = self.up(x1)
        out = torch.cat((x2, out), dim=1)
        out = self.conv(out)

        return out


class CamusUnet(nn.Module):
    def __init__(self, n_channels, n_classes, ch_sizes=[32, 32, 64, 128, 128, 128]):
        super().__init__()

        self.sizes = ch_sizes

        self.conv1 = DoubleConv(n_channels, self.sizes[0])

        down_layers = []

        for i in range(len(self.sizes)-1):
            down_layers.append(DownSample(self.sizes[i], self.sizes[i+1]))

        self.down = nn.Sequential(*down_layers)

        self.conv2 = DoubleConv(self.sizes[-1], self.sizes[-1])

        up_layers = []

        for i in range(1, len(self.sizes))[::-1]:
            up_layers.append(UpSample(self.sizes[i], self.sizes[i-1]))

        self.up = nn.Sequential(*up_layers)

        self.fc = nn.Conv2d(self.sizes[0], n_classes, 1)

        self.activate = nn.Softmax(dim=1)

    def forward(self, x):
        # BxCxHxW : B, 1, 256, 256

        # encoder path
        enc_1 = self.conv1(x)  # B, 32, 256, 256
        enc_2 = self.down[0](enc_1)  # B, 32, 128, 128
        enc_3 = self.down[1](enc_2)  # B, 64, 64, 64
        enc_4 = self.down[2](enc_3)  # B, 128, 32, 32
        enc_5 = self.down[3](enc_4)  # B, 128, 16, 16

        bridge = self.down[4](enc_5)  # B, 128, 8, 8

        # decoder path
        out = self.up[0](bridge, enc_5)  # B, 128, 16, 16
        out = self.up[1](out, enc_4)  # B, 128, 32, 32
        out = self.up[2](out, enc_3)  # B, 64, 64, 64
        out = self.up[3](out, enc_2)  # B, 32, 128, 128
        out = self.up[4](out, enc_1)  # B, 32, 256, 256

        f_out = self.fc(out)  # B, n_classes, 224, 224
#         f_out = self.activate(f_out)

        return f_out
