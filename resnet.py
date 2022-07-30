from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from typing import List


"""
Conv3d
input size: (N, C_in, D, H, W)
output size: (N, C_out, D_out, H_out, W_out)

N       For mini batch (or how many sequences do we want to feed at one go)
Cin     For the number of channels in our input (if our image is rgb, this is 3)
D       For depth or in other words the number of images/frames in one input sequence 
        (if we are dealing videos, this is the number of frames)
H       For the height of the image/frame
W       For the width of the image/frame
"""


def conv3x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels,
                     out_channels,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(channels, channels)
        self.bn2 = nn.BatchNorm3d(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = conv3x3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm3d(channels)
        self.conv3 = conv1x1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 sub_blocks: List[int],
                 block_in_channels: List[int],
                 model_input_channels=3,
                 out_dim=10):
        super().__init__()

        self.in_channels = block_in_channels[0]

        self.conv1 = nn.Conv3d(model_input_channels,
                               self.in_channels,
                               kernel_size=(7, 7, 7),
                               stride=(2, 2, 2),
                               padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_block(block,
                                       sub_blocks[0],
                                       block_in_channels[0])
        self.block2 = self._make_block(block,
                                       sub_blocks[1],
                                       block_in_channels[1],
                                       stride=2)
        self.block3 = self._make_block(block,
                                       sub_blocks[2],
                                       block_in_channels[2],
                                       stride=2)
        self.block4 = self._make_block(block,
                                       sub_blocks[3],
                                       block_in_channels[3],
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_in_channels[3] * block.expansion, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_block(self, block, sub_blocks: int, channels: int, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_channels, channels * block.expansion, stride),
                nn.BatchNorm3d(channels * block.expansion))

        _sub_blocks = [block(in_channels=self.in_channels,
                             channels=channels,
                             stride=stride,
                             downsample=downsample)]

        self.in_channels = channels * block.expansion
        for i in range(1, sub_blocks):
            _sub_blocks.append(block(self.in_channels, channels))

        return nn.Sequential(*_sub_blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    # model = ResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], n_classes=20)
    model = ResNet(Bottleneck, [2, 2, 2, 2], [64, 128, 256, 512], out_dim=20)
    model.cuda()
    print(model)

    from torchinfo import summary
    summary(model, (16, 3, 64, 128, 128), depth=5)
    # (batch_size, input_channel, frame_time, frame_width, frame_height)
    # FRAME_WIDTH = 480
    # FRAME_HEIGHT = 270
    # FRAME_LENGTH = 150
