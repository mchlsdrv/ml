import torch
import torch.nn as nn
from math import ceil

from ml.nn.utils.regularization import stochastic_depth, apply_stochastic_depth

base_model = [
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_vals = {  # alpha, beta, gamma ; depth  = alpha ** phi | width = betta ** phi | resolution = gamma ** phi
    'b0': (0, 244, 0.2),
    'b1': (0.5, 240, 0.2),
    'b2': (1., 260, 0.3),
    'b3': (2, 300, 0.3),
    'b4': (3, 380, 0.4),
    'b5': (4, 456, 0.4),
    'b6': (5, 528, 0.5),
    'b7': (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        # if we set groups = 1 - it's a normal conv,
        # if we set groups = in_channels - it becomes Depth-wise conv
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        return self.activation(self.batch_norm(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    """
    Calculating attention scores for each of the channels, i.e., how much we should prioritize the values in it
    """
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.squeeze_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=reduced_dim, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.squeeze_excitation(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4,
                 survival_prob=0.8):
        super().__init__()
        self.survival_prop = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduce_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.conv = nn.Sequential(
            CNNBlock(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=hidden_dim),
            SqueezeExcitation(in_channels=hidden_dim, reduced_dim=reduce_dim),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    # def stochastic_depth(self, x):
    #     """
    #     Randomly skips certain layers
    #     :param x: Input Tensor
    #     :return:
    #     """
    #     if not self.training:
    #         return x
    #
    #     binary_tensor = torch.randn(x.shape[0], 1, 1, 1, device=x.device) > self.survival_prop
    #     return torch.div(x, self.survival_prop) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        x = self.conv(x)

        if self.use_residual:
            x = apply_stochastic_depth(x=x, inputs=inputs, survival_prop=self.survival_prop, training=self.training)

        return x


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super().__init__()
        width_fctr, depth_fctr, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_fctr)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_fctr, depth_fctr, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

    @staticmethod
    def calculate_factors(version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_vals[version]
        depth_fctr = alpha ** phi
        width_fctr = beta ** phi
        return width_fctr, depth_fctr, drop_rate

    @staticmethod
    def create_features(width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        feats = [CNNBlock(in_channels=3, out_channels=channels, kernel_size=3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                feats.append(
                    InvertedResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k = 1 => pad = 0 ; if k = 3 => pad = 1 ; if k = 5 => pad = 2
                    )
                )
                in_channels = out_channels

        feats.append(
            CNNBlock(in_channels=in_channels, out_channels=last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*feats)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    version = 'b7'
    phi, res, drop_rate = phi_vals.get(version)
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res), device=device)
    mdl = EfficientNet(
        version=version,
        num_classes=num_classes
    ).to(device)
    print(mdl)
    print(mdl(x).shape)


test()
