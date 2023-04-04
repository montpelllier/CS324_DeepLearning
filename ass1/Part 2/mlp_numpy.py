from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input_data vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        input_layer = [Linear(n_inputs, n_hidden[0]), ReLU()]
        output_layer = [Linear(n_hidden[-1], n_classes), SoftMax()]
        hidden_layer = []
        for i in range(len(n_hidden) - 1):
            hidden_layer.append(Linear(n_hidden[i], n_hidden[i + 1]))
            hidden_layer.append(ReLU())

        self.layers = input_layer + hidden_layer + output_layer
        self.loss_fc = CrossEntropy()

    def forward(self, x):
        """
        Predict network output from input_data by passing it through several layers.
        Args:
            x: input_data to the network
        Returns:
            x: output of the network
        """
        for layer in self.layers:
            x = layer(x)
        out = x
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        dx = dout
        return dx

    def predict(self, x):
        flag = False
        for layer in self.layers:
            x = layer(x, flag)
        out = x
        return out

    __call__ = forward
