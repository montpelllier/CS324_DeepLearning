from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class CNN(nn.Module):

    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()

        # Pool Layer
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_pos = [0, 1, 3, 5, 7]
        # Batch Normalization Layer
        self.bn_list = [nn.BatchNorm2d(64), nn.BatchNorm2d(128), nn.BatchNorm2d(256), nn.BatchNorm2d(256),
                        nn.BatchNorm2d(512), nn.BatchNorm2d(512), nn.BatchNorm2d(512), nn.BatchNorm2d(512)]
        # Relu Layer
        self.relu = nn.ReLU()
        # Convolution Layer
        self.conv_list = [nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)]
        # Linear Layer
        self.linear = nn.Linear(512, n_classes)

    def forward(self, x):
        """
        Performs forward pass of the input.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        # conv -> ReLU -> pool
        for i, (conv, bn) in enumerate(zip(self.conv_list, self.bn_list)):
            x = self.relu(bn(conv(x)))
            if i in self.pool_pos:
                x = self.pool(x)
        # Resize x
        x = x.view(-1, 512)
        x = self.linear(x)

        out = x
        return out
