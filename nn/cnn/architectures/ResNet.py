import numpy as np
import torch
import torch.nn as nn

from ml.nn.utils.regularization import stochastic_depth

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


class SimpleResBlock(nn.Module):
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

        # print(f'x shape = {x.shape}')
        # print(f'identity shape = {x_identity.shape}')

        x += x_identity

        x = self.activation(x)

        return x

class BottleNeckResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):

        super().__init__()
        self.channel_expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels * self.channel_expansion,
            kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.channel_expansion)
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
    def __init__(self, block, n_blocks_in_layer: np.ndarray or list, image_channels, output_size, prediction_layer: torch.nn.modules.activation):
        super().__init__()
        self.epoch = 0

        self.in_channels = 64
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
        # convs_out_shape = self.convs(torch.randn((1, self.image_channels, 256, 256))).shape
        # convs_out_size = convs_out_shape[0] * convs_out_shape[1] * convs_out_shape[2] * convs_out_shape[3]

        # self.logit_activation = nn.Sigmoid()

        # ResNet layers
        self.layer1 = self._make_layer(block=block, num_residual_blocks=n_blocks_in_layer[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block=block, num_residual_blocks=n_blocks_in_layer[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block=block, num_residual_blocks=n_blocks_in_layer[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block=block, num_residual_blocks=n_blocks_in_layer[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_size)
        # self.fc = nn.Linear(convs_out_size, output_size)
        # self.fc = nn.Linear(512 * 4, output_size)
        self.pred_layer = prediction_layer

    def _get_convs(self):
        # - First layer - down sampling
        conv_blks = [
            nn.Conv2d(in_channels=self.image_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            self.activation,
            self.maxpool
        ]

        in_chnls = self.out_channels[0]
        out_chnls = in_chnls
        for lyr_idx, (n_blks, out_chnls, strd) in enumerate(zip(self.n_blocks_in_layer, self.out_channels, self.strides)):
            # print(n_blks, out_chnls, strd)

            # - For each layer
            for blk_idx in range(1, n_blks + 1):

                # - Add the blocks
                # - O = [(I + 2*P - K) / S + 1]
                # - If the current layer is the first, or the block is not the first in a layer - no need to downsample the input
                if lyr_idx == 0 or blk_idx != 1:
                    conv_blks.append(
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
                    conv_blks.append(
                        self.block(
                            in_channels=in_chnls,
                            out_channels=out_chnls,
                            stride=strd,
                            identity_downsample= nn.Sequential(
                                nn.Conv2d(in_channels=in_chnls, out_channels=out_chnls, kernel_size=1, stride=2, padding=0),
                                nn.BatchNorm2d(out_chnls)
                            )
                        )
                    )

                    # - Update the in channels for the next layer to match the output channels from the previous layer
                    in_chnls = out_chnls

        return nn.Sequential(*conv_blks)



    def forward(self, x):
        # - Update the epochs for scheduling
        global EPOCH
        EPOCH = self.epoch

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.activation(x)
        #
        # x = self.maxpool(x)
        x = self.convs(x)

        # print(x.shape)
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

    # def forward(self, x):
    #     # - Update the epochs for scheduling
    #     global EPOCH
    #     EPOCH = self.epoch
    #
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.activation(x)
    #
    #     x = self.maxpool(x)
    #
    #     x = self.layer1(x)
    #
    #     # if self.training and STOCHASTIC_DEPTH:
    #     #     x = apply_stochastic_depth(x, x_identity, survival_prop=0.5, training=self.training)
    #
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     if self.training and P_DROP_BLOCK > 0.0 and EPOCH > DROP_BLOCK_EPOCH_START:
    #         p = P_DROP_BLOCK * P_DROP_BLOCK_FCTR
    #         p = p if p < P_DROP_BLOCK_MAX else P_DROP_BLOCK_MAX
    #         x = nn.Dropout2d(p=p)(x)
    #
    #     x = self.avgpool(x)
    #     x = x.reshape(x.shape[0], -1)
    #     x = self.fc(x)
    #
    #     if self.training and P_DROPOUT > 0 and EPOCH > DROP_BLOCK_EPOCH_START:
    #         p = P_DROPOUT * P_DROPOUT_FCTR
    #         p = p if p < P_DROPOUT_MAX else P_DROPOUT_MAX
    #         x = nn.Dropout(p=p)(x)
    #
    #     if self.pred_layer is not None:
    #         x = self.pred_layer(x)
    #
    #     return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

        layers.append(block(
            in_channels=self.in_channels, out_channels=out_channels,
            identity_downsample=identity_downsample, stride=stride))
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(in_channels=self.in_channels, out_channels=out_channels))

        return nn.Sequential(*layers)


def get_resnet18(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(block=SimpleResBlock, n_blocks_in_layer=[2, 2, 2, 2], image_channels=image_channels, output_size=output_size,
                  prediction_layer=prediction_layer)


def get_resnet34(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(block=SimpleResBlock, n_blocks_in_layer=[3, 4, 6, 3], image_channels=image_channels, output_size=output_size,
                  prediction_layer=prediction_layer)


def get_resnet50(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(block=BottleNeckResBlock, n_blocks_in_layer=[3, 4, 6, 3], image_channels=image_channels, output_size=output_size,
                  prediction_layer=prediction_layer)


def get_resnet101(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(block=BottleNeckResBlock, n_blocks_in_layer=[3, 4, 23, 3], image_channels=image_channels, output_size=output_size,
                  prediction_layer=prediction_layer)


def get_resnet152(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(block=BottleNeckResBlock, n_blocks_in_layer=[3, 8, 36, 3], image_channels=image_channels, output_size=output_size,
                  prediction_layer=prediction_layer)


def test_resnet50():
    net = get_resnet50()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)


def test_resnet101():
    net = get_resnet101()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)


def test_resnet152():
    net = get_resnet152()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)


if __name__ == '__main__':
    pass
