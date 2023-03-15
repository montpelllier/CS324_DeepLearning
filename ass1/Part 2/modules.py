import numpy as np
import torch
from numpy import nan


class Linear(object):

    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input_data dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and std = 0.0001.
        2) Initialize biases self.params['bias'] with 0.
        3) Initialize gradients with zeros.
        """
        self.x = None
        self.in_features = in_features
        self.out_features = out_features

        self.params = {'weight': np.random.randn(in_features, out_features) * 0.0001,
                       'bias': np.zeros(self.out_features)}
        self.grads = {'weight': np.random.randn(in_features, out_features) * 0.0001,
                      'bias': np.zeros(self.out_features)}

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input_data).
        Args:
            x: input_data to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object and use them in the backward
        pass computation. This is true for *all* forward methods of *all* modules in this class.
        """
        # torch.nn.Linear()
        # torch.autograd.backward()
        self.x = x
        out = np.dot(x, self.params['weight']) + self.params['bias']
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input_data of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to layer parameters in
        self.grads['weight'] and self.grads['bias'].
        """
        # print("dout: ", dout)
        # print(self.grads['bias'].shape, self.x.shape)
        self.grads['weight'] = np.dot(self.x.transpose(), dout)
        # print(self.grads['weight'].shape)
        self.grads['bias'] = np.mean(dout, axis=0)
        # print(self.grads['bias'].shape)
        dx = np.dot(dout, self.params['weight'].transpose())
        return dx

    __call__ = forward


class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input_data to the module
        Returns:
            out: output of the module
        """
        self.x = x
        out = np.maximum(0, x)
        # print("RELU", x)
        # print("OUT", out)
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input_data of the module
        """
        dx = dout
        dx[self.x > 0] = 1
        dx[self.x <= 0] = 0
        # print(np.sum(dx))
        return dx

    __call__ = forward


class SoftMax(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input_data to the module
        Returns:
            out: output of the module
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        """
        # torch.nn.Softmax
        # print(np.max(x))
        x -= np.max(x)
        x = np.exp(x)
        # print(x)
        for i in range(len(x)):
            # x[i] -= max(x[i])
            # x[i] = np.exp(x[i])
            x[i] /= sum(x[i])
            if sum(x[i]) is nan:
                print(x[i])
            # try:
            #     x[i] /= sum(x[i])
            # except Exception as e:
            #     print(e)
            #     print(x[i])
            #     exit(0)

        self.x = x
        out = x
        # print("out", out)
        return out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input_data of the module
        """
        # https://zhuanlan.zhihu.com/p/37740860
        # dx = np.diag(self.x) - np.dot(self.x.transpose(), self.x)
        # dx = np.dot(dx, dout.transpose()).transpose()
        # print("dx", dx)
        return dout

    __call__ = forward


class CrossEntropy(object):

    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input_data to the module
            y: labels of the input_data
        Returns:
            out: cross entropy loss
        """
        x += 1e-7
        # print("log", np.log(x))
        out = -np.sum(np.dot(y.transpose(), np.log(x)))
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input_data to the module
            y: labels of the input_data
        Returns:
            dx: gradient of the loss with respect to the input_data x.
        """
        # x += 1e-6
        # print(x)
        dx = y - x
        print("dx", dx)
        return dx

    __call__ = forward
