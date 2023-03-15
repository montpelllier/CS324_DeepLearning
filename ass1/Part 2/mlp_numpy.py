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
        self.activation = ReLU()
        self.loss_fc = CrossEntropy()
        n_hidden = n_hidden[0]
        self.layers = []
        # for i in range(len(n_hidden)):
        #     self.layers.append(Linear())
        self.fc1 = Linear(n_inputs, n_hidden)
        self.fc2 = Linear(n_hidden, n_classes)
        self.softmax = SoftMax()

    def predict(self, x):
        flag = False
        x = self.activation.forward(self.fc1.forward(x, flag), flag)
        x = self.fc2(x, flag)
        out = self.softmax(x)
        return out

    def forward(self, x):
        """
        Predict network output from input_data by passing it through several layers.
        Args:
            x: input_data to the network
        Returns:
            out: output of the network
        """
        x = self.activation(self.fc1(x))
        # print("first layer:", x)
        x = self.fc2(x)
        # print("second layer:", x)
        out = self.softmax(x)
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        # dx = self.softmax.backward(dout)
        dx = self.fc2.backward(dout)
        dx = self.activation.backward(dx)
        dx = self.fc1.backward(dx)
        return dx

    __call__ = forward
