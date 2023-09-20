import unittest

import numpy as np
import torch
import torch.nn as nn

from ml.nn.utils.regularization import stochastic_depth

CHANNEL_EXPANSION = 4
LOW_PERFORMANCE_MODELS = ['ResNet18', 'ResNet32']
HIGH_PERFORMANCE_MODELS = ['ResNet50', 'ResNet101', 'ResNet152']
# STOCHASTIC_DEPTH = True
STOCHASTIC_DEPTH = False

# DROP_BLOCK_EPOCH_START = 0
DROP_BLOCK_EPOCH_START = 80
P_DROP_BLOCK = 0.1
# P_DROP_BLOCK = 0.0
P_DROP_BLOCK_FCTR = 1.1
P_DROP_BLOCK_MAX = 0.5

# DROP_OUT_EPOCH_START = 0
DROP_OUT_EPOCH_START = 20
P_DROPOUT = 0.2
# P_DROPOUT = 0.0
P_DROPOUT_FCTR = 1.1
P_DROPOUT_MAX = 0.5

EPOCH = 0


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):

        super().__init__()
        # - Same output size O = (I + 2P - K) / S + 1
        # - As the kernel = 3, stride = 1 and padding = 1 - the O == I
        # => O = (I + 2*1 - 3) / 1 + 1 = (I - 1) / 1 + 1 = I - 1 + 1 = I
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # - If stride == 1 - it stays the same
        # - If the stride == 2 - the output is halved
        # => O = (I + 2*1 - 3) / 2 + 1 = (I - 1) / 2 + 1 = I/2 - 1/2 + 1 = I / 2 + 1
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.ELU()

        self.identity_downsample = identity_downsample
        self.stochastic_depth = stochastic_depth


    def forward(self, x):
        x_identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            x_identity = self.identity_downsample(x_identity)

        x += x_identity

        x = self.activation(x)

        return x

class BottleNeckResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channel_expansion: int = 4, identity_downsample=None, stride=1):

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // CHANNEL_EXPANSION,
            kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels // CHANNEL_EXPANSION)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels // CHANNEL_EXPANSION,
            out_channels=out_channels // CHANNEL_EXPANSION,
            kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // CHANNEL_EXPANSION)

        self.conv3 = nn.Conv2d(
            in_channels=out_channels // CHANNEL_EXPANSION,
            out_channels=out_channels ,
            kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.activation = nn.ELU()
        self.identity_downsample = identity_downsample


    def forward(self, x):
        x_identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.training and P_DROP_BLOCK > 0.0 and EPOCH > DROP_BLOCK_EPOCH_START:
            p = P_DROP_BLOCK * P_DROP_BLOCK_FCTR
            p = p if p < P_DROP_BLOCK_MAX else P_DROP_BLOCK_MAX
            x = nn.Dropout2d(p=p)(x)

        if self.identity_downsample is not None:
            x_identity = self.identity_downsample(x_identity)

        x += x_identity
        x = self.activation(x)

        return x


class ResNet(nn.Module):
    def __init__(self, architecture: str, block, n_blocks_in_layer: np.ndarray or list, image_channels, output_size, prediction_layer: torch.nn.modules.activation):
        super().__init__()
        self.epoch = 0
        self.architecture = architecture

        self.channel_expansion = 1
        if self.architecture in HIGH_PERFORMANCE_MODELS:
            self.channel_expansion = CHANNEL_EXPANSION

        # self.in_channels = 64
        self.block = block
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.n_blocks_in_layer = n_blocks_in_layer
        self.image_channels = image_channels
        self.out_channels = [64, 128, 256, 512]
        self.strides = [1, 2, 2, 2]
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.convs = self._get_convs()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_channels[-1] * self.channel_expansion, output_size)
        self.pred_layer = prediction_layer

    def _add_conv_layer(self, layers: list, layer_index: int, n_blocks: int, in_channels: int, out_channels: int, stride: int):
        in_chnls, out_chnls = in_channels, out_channels
        if self.architecture in HIGH_PERFORMANCE_MODELS:
            out_chnls = out_channels * CHANNEL_EXPANSION

        for blk_idx in range(1, n_blocks + 1):
            # - Add the blocks
            # - O = [(I + 2*P - K) / S + 1]
            # - If the current layer is the first, or the block is not the first in a layer - no need to downsample the input
            if (self.architecture not in HIGH_PERFORMANCE_MODELS) \
            and (layer_index == 0 or blk_idx != 1):
                layers.append(
                    self.block(
                        in_channels=in_chnls,
                        out_channels=out_chnls,
                        stride=1,
                        identity_downsample=None
                    )
                )

            # - If the current layer is not the first, and the block is the first in a layer - we need to downsample the input
            else:
                # - Update the out channels to be double than the input
                layers.append(
                    self.block(
                        in_channels=in_chnls,
                        out_channels=out_chnls,
                        stride=stride,
                        identity_downsample=nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_chnls,
                                out_channels=out_chnls,
                                kernel_size=1, stride=stride, padding=0),
                            nn.BatchNorm2d(out_chnls)
                        )
                    )
                )

                # - Update the in channels for the next layer to match the output channels from the previous layer
                in_chnls = out_chnls
        return in_chnls

    def _get_convs(self):
        # - First layer - down sampling
        lyrs = [
            nn.Conv2d(in_channels=self.image_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            self.activation,
            self.maxpool
        ]

        in_chnls = self.out_channels[0]
        out_chnls = in_chnls
        for lyr_idx, (n_blks, out_chnls, strd) in enumerate(zip(self.n_blocks_in_layer, self.out_channels, self.strides)):
            in_chnls = self._add_conv_layer(
                layers=lyrs,
                layer_index=lyr_idx,
                n_blocks=n_blks,
                in_channels=in_chnls,
                out_channels=out_chnls,
                stride=strd
            )

        return nn.Sequential(*lyrs)

    def forward(self, x):
        # - Update the epochs for scheduling
        global EPOCH
        EPOCH = self.epoch

        x = self.convs(x)

        if self.training and P_DROP_BLOCK > 0.0 and EPOCH > DROP_BLOCK_EPOCH_START:
            p = P_DROP_BLOCK * P_DROP_BLOCK_FCTR
            p = p if p < P_DROP_BLOCK_MAX else P_DROP_BLOCK_MAX
            x = nn.Dropout2d(p=p)(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        if self.training and P_DROPOUT > 0 and EPOCH > DROP_BLOCK_EPOCH_START:
            p = P_DROPOUT * P_DROPOUT_FCTR
            p = p if p < P_DROPOUT_MAX else P_DROPOUT_MAX
            x = nn.Dropout(p=p)(x)

        if self.pred_layer is not None:
            x = self.pred_layer(x)

        return x


def get_resnet18(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(
        architecture='ResNet18',
        block=ResBlock,
        n_blocks_in_layer=[2, 2, 2, 2],
        image_channels=image_channels,
        output_size=output_size,
        prediction_layer=prediction_layer
    )


def get_resnet34(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(
        architecture='ResNet34',
        block=ResBlock,
        n_blocks_in_layer=[3, 4, 6, 3],
        image_channels=image_channels,
        output_size=output_size,
        prediction_layer=prediction_layer
    )


def get_resnet50(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(
        architecture='ResNet50',
        block=BottleNeckResBlock,
        n_blocks_in_layer=[3, 4, 6, 3],
        image_channels=image_channels,
        output_size=output_size,
        prediction_layer=prediction_layer
    )


def get_resnet101(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(
        architecture='ResNet101',
        block=BottleNeckResBlock,
        n_blocks_in_layer=[3, 4, 23, 3],
        image_channels=image_channels,
        output_size=output_size,
        prediction_layer=prediction_layer
    )


def get_resnet152(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(
        architecture='ResNet152',
        block=BottleNeckResBlock,
        n_blocks_in_layer=[3, 8, 36, 3],
        image_channels=image_channels,
        output_size=output_size,
        prediction_layer=prediction_layer
    )



class TestNets(unittest.TestCase):
    def test_resnet18(self):
        print(f'Testing ResNet18...')
        net = get_resnet18()
        x = torch.randn(2, 3, 224, 224)
        y = net(x).to('cuda')
        self.assertEqual(y.shape, torch.Size([2, 1000]))

    def test_resnet34(self):
        print(f'Testing ResNet34...')
        net = get_resnet34()
        x = torch.randn(2, 3, 224, 224)
        y = net(x).to('cuda')
        self.assertEqual(y.shape, torch.Size([2, 1000]))

    def test_resnet50(self):
        print(f'Testing ResNet50...')
        net = get_resnet50()
        x = torch.randn(2, 3, 224, 224)
        y = net(x).to('cuda')
        self.assertEqual(y.shape, torch.Size([2, 1000]))

    def test_resnet101(self):
        print(f'Testing ResNet101...')
        net = get_resnet101()
        x = torch.randn(2, 3, 224, 224)
        y = net(x).to('cuda')
        self.assertEqual(y.shape, torch.Size([2, 1000]))

    def test_resnet152(self):
        print(f'Testing ResNet152...')
        net = get_resnet152()
        x = torch.randn(2, 3, 224, 224)
        y = net(x).to('cuda')
        self.assertEqual(y.shape, torch.Size([2, 1000]))


if __name__ == '__main__':
    unittest.main()
