from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
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
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn_list = [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6, self.bn7, self.bn8]
        # self.bn_list = [nn.BatchNorm2d(64), nn.BatchNorm2d(128), nn.BatchNorm2d(256), nn.BatchNorm2d(256),
        #                 nn.BatchNorm2d(512), nn.BatchNorm2d(512), nn.BatchNorm2d(512), nn.BatchNorm2d(512)]
        # Relu Layer
        self.relu = nn.ReLU()
        # Convolution Layer
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8]
        # self.conv_list = [nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
        #                   nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        #                   nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        #                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        #                   nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        #                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        #                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        #                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)]
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
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            # x = self.relu(bn(conv(x)))
            if i in self.pool_pos:
                x = self.pool(x)
        # Resize x
        x = x.view(-1, 512)
        x = self.linear(x)

        out = x
        return out


class CNN1(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN1, self).__init__()

        # Convolution Layer：input_channels = 3, output_channels = 6, kernal_size = 5 * 5
        self.conv1 = nn.Conv2d(n_channels, 6, 5)
        # Pooling Layer: kernal_size = 2 * 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Linear Layer: in_features = 400, out_features = 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        # conv -> ReLU -> pool
        # After the operations, the size of x is: 14 * 14
        x = self.pool(F.relu(self.conv1(x)))
        # After the operations, the size of x is: 5 * 5
        x = self.pool(F.relu(self.conv2(x)))
        # Resize x
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
