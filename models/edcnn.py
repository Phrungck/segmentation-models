import torch
from torch import nn

# implementation of Degerli's EDcnn


class Up(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(
            in_c+out_c, out_c, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x1, x2):

        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        out = self.up(out)

        return out


class EDcnn(nn.Module):

    def __init__(self, n_channels=1, n_classes=2, filters=[32, 64, 128, 256, 512, 1024]):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.filters = filters
        n = len(self.filters)

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        # Encoder
        self.down_layers.append(
            nn.Sequential(
                nn.Conv2d(n_channels, self.filters[0], kernel_size=(
                    3, 3), padding=(1, 1)),
                nn.BatchNorm2d(self.filters[0]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        )

        for i in range(n-2):
            self.down_layers.append(
                nn.Sequential(
                    nn.Conv2d(self.filters[i], self.filters[i+1],
                              kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(self.filters[i+1]),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
            )

        self.down_layers.append(
            nn.Sequential(
                nn.Conv2d(self.filters[-2], self.filters[-1],
                          kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(self.filters[-1]),
                nn.ReLU()
            )
        )

        self.down_layers = nn.Sequential(*self.down_layers)

        # Decoder
        self.up_layers.append(
            Up(self.filters[-1], self.filters[-2]))

        for i in range(n-2, -1, -1):
            self.up_layers.append(
                Up(self.filters[i], self.filters[i-1])
            )

        self.up_layers = nn.Sequential(*self.up_layers)

        self.fc = nn.Conv2d(
            self.filters[0], self.n_classes, kernel_size=(1, 1))

    def forward(self, x):

        out1 = self.down_layers[0](x)  # B,32,128,128
        out2 = self.down_layers[1](out1)  # B,64,64,64
        out3 = self.down_layers[2](out2)  # B,128,32,32
        out4 = self.down_layers[3](out3)  # B,256,16,16
        out5 = self.down_layers[4](out4)  # B,512,8,8

        bridge = self.down_layers[5](out5)  # B,1024,8,8

        out = self.up_layers[0](bridge, out5)  # B,512,16,16
        out = self.up_layers[1](out, out4)  # B,256,32,32
        out = self.up_layers[2](out, out3)  # B,128,64,64
        out = self.up_layers[3](out, out2)  # B,64,128,128
        out = self.up_layers[4](out, out1)  # B,32,256,256

        out = self.fc(out)

        return out
