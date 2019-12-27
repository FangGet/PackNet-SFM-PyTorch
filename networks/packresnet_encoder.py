import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels,stride=stride,  kernel_size=3, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, need_short=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, in_channels)
        self.norm2d1 = nn.GroupNorm(16, in_channels, 1e-10)
        self.nolin = nn.ELU(inplace=True)

        self.conv2 = conv3x3(in_channels, out_channels)
        self.norm2d2 = nn.GroupNorm(16, out_channels, 1e-10)

        self.need_short = need_short
        if self.need_short:
            self.conv3 = conv1x1(in_channels, out_channels)
            self.norm2d3 = nn.GroupNorm(16, out_channels, 1e-10)
        self.dropout = nn.Dropout2d(p=0.3)



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm2d1(out)
        out = self.nolin(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2d2(out)

        if self.need_short:
            residual = self.conv3(residual)
            residual = self.norm2d3(residual)

        out += residual
        out = self.nolin(out)

        return out

def make_layer(in_channels, block, out_channels, blocks, downscale=4):

    layers = []
    layers.append(block(in_channels, out_channels, need_short=True))
    in_channels = out_channels * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channels, out_channels))

    layers.append(downsampleX(out_channels, out_channels))

    return nn.Sequential(*layers)


class downsampleX(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(downsampleX, self).__init__()
        self.pack = PackBlock(in_channel, out_channel, need_last_nolin=True)

    def forward(self, x):
        return self.pack(x)

class PackResNetEncoder(nn.Module):

    def __init__(self, num_input_images=1):
        super(PackResNetEncoder, self).__init__()

        conv_planes = [64, 64, 64, 128, 128]
        self.num_ch_enc = conv_planes
        self.downsample1 = downsampleX(conv_planes[0], conv_planes[1])

        self.conv1 = nn.Conv2d(num_input_images * 3, conv_planes[0], kernel_size=5, stride=1, padding=2, bias=False)
        self.norm2d1 = nn.GroupNorm(16, conv_planes[0], 1e-10)
        self.nolin1 = nn.ELU(inplace=True)

        self.conv2 = nn.Conv2d(conv_planes[0], conv_planes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.norm2d2 = nn.GroupNorm(16, conv_planes[0], 1e-10)
        self.nolin2 = nn.ELU(inplace=True)

        residuals = [2,2,2,2]
        block = BasicBlock

        self.layers1 = make_layer(conv_planes[0], block, conv_planes[0+1], blocks=residuals[0])
        self.layers2 = make_layer(conv_planes[1], block, conv_planes[1 + 1], blocks=residuals[1])
        self.layers3 = make_layer(conv_planes[2], block, conv_planes[2 + 1], blocks=residuals[2])
        self.layers4 = make_layer(conv_planes[3], block, conv_planes[3 + 1], blocks=residuals[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        self.features.append(x) #0
        x = self.conv1(x)
        x = self.norm2d1(x)
        x = self.nolin1(x) #1

        x = self.conv2(x)
        x = self.norm2d2(x)
        x = self.nolin2(x) #2
        self.features.append(self.downsample1(x)) #3
        self.features.append(self.layers1(self.features[-1])) #5
        self.features.append(self.layers2(self.features[-1])) #7
        self.features.append(self.layers3(self.features[-1])) #9
        self.features.append(self.layers4(self.features[-1])) #11

        return self.features



