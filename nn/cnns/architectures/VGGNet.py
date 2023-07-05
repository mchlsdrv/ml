import torch
import torch.nn as nn  # All the neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.functional as F  # For all the functions which don't have any parameters
from torch.utils.data import DataLoader  # Gives easier dataset managment and creates mini-batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

ARCHS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGNet(nn.Module):
    def __init__(self, architecture, in_channels=3, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels
        self.activation = nn.ReLU
        self.conv_layers = self.create_convs(architecture)
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)

        return x

    def create_convs(self, blocks):
        layers = []
        in_channels = self.in_channels

        for blck in blocks:
            if isinstance(blck, int):
                out_channels = blck

                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(blck),
                    self.activation()
                ]

                in_channels = blck
            elif isinstance(blck, str) and blck == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                ]

        return nn.Sequential(*layers)


dvc = 'cuda' if torch.cuda.is_available() else 'cpu'
mdl = VGGNet(architecture=ARCHS.get('VGG16'), in_channels=3, num_classes=1000).to(dvc)
x = torch.randn(1, 3, 224, 224).to(dvc)
print(mdl(x).shape)
