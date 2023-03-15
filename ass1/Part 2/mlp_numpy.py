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
        # self.activation = ReLU()
        self.loss_fc = CrossEntropy()
        self.layers = []
        for i in range(len(n_hidden)):
            if i == 0:
                self.layers.append(Linear(n_inputs, n_hidden[i]))
            else:
                self.layers.append(Linear(n_hidden[i-1], n_hidden[i]))
            self.layers.append(ReLU())
        self.layers.append(Linear(n_hidden[-1], n_classes))
        # for layer in self.layers:
        #     print(layer.in_features, layer.out_features)
        self.softmax = SoftMax()

    def predict(self, x):
        flag = False
        # x = self.activation.forward(self.fc1.forward(x, flag), flag)
        # x = self.fc2(x, flag)
        # out = self.softmax(x)
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out, flag)
            # if i != len(self.layers) - 1:
            #     out = self.activation(out, flag)
            # else:
        out = self.softmax(out)
        return out

    def forward(self, x):
        """
        Predict network output from input_data by passing it through several layers.
        Args:
            x: input_data to the network
        Returns:
            out: output of the network
        """
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            # if i != len(self.layers)-1:
            #     out = self.activation(out)
            # else:
            out = self.softmax(out)
        # x = self.activation(self.fc1(x))
        # x = self.fc2(x)
        # out = self.softmax(x)
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        # dx = self.softmax.backward(dout)
        dx = self.layers[-1].backward(dout)
        for i in range(len(self.layers)-2, -1, -1):
            # dx = self.activation.backward(dx)
            dx = self.layers[i].backward(dx)
        # dx = self.fc2.backward(dout)
        # dx = self.activation.backward(dx)
        # dx = self.fc1.backward(dx)
        return dx

    __call__ = forward
