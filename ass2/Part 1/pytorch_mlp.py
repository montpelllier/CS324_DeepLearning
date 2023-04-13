from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        self.in_size = n_inputs

        input_layer = [nn.Linear(n_inputs, n_hidden[0]), nn.ReLU()]
        output_layer = [nn.Linear(n_hidden[-1], n_classes), nn.Softmax(dim=-1)]
        hidden_layer = []
        for i in range(len(n_hidden) - 1):
            hidden_layer.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
            hidden_layer.append(nn.ReLU())
            if n_inputs > 1000:
                # 减少过拟合
                hidden_layer.append(nn.Dropout(0.2))

        self.layers = input_layer + hidden_layer + output_layer
        self.layers = nn.Sequential(*self.layers)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        # resize
        x = x.view(-1, self.in_size)
        out = self.layers(x)
        return out
