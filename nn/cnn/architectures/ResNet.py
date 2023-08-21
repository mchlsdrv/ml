import torch
import torch.nn as nn
from ml.nn.utils.regularization import stochastic_depth, apply_stochastic_depth

STOCHASTIC_DEPTH = False

class ResBlock(nn.Module):
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

        if self.identity_downsample is not None:
            x_identity = self.identity_downsample(x_identity)

        x += x_identity
        x = self.activation(x)

        if self.training and STOCHASTIC_DEPTH:
            x = apply_stochastic_depth(x, x_identity, survival_prop=0.5, training=self.training)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, output_size, prediction_layer: torch.nn.modules.activation):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ELU()
        self.logit_activation = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block=block, num_residual_blocks=layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block=block, num_residual_blocks=layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block=block, num_residual_blocks=layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block=block, num_residual_blocks=layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, output_size)
        self.pred_layer = prediction_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        # x = self.logit_activation(x)
        # x = self.activation(x)

        if self.pred_layer is not None:
            x = self.pred_layer(x)

        return x

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


def get_resnet50(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(block=ResBlock, layers=[3, 4, 6, 3], image_channels=image_channels, output_size=output_size,
                  prediction_layer=prediction_layer)


def get_resnet101(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(block=ResBlock, layers=[3, 4, 23, 3], image_channels=image_channels, output_size=output_size,
                  prediction_layer=prediction_layer)


def get_resnet152(image_channels=3, output_size=1000, prediction_layer=None):
    return ResNet(block=ResBlock, layers=[3, 8, 36, 3], image_channels=image_channels, output_size=output_size,
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
