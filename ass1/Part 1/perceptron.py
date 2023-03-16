import numpy as np
from matplotlib import pyplot as plt
from numpy import random


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.w = np.array([0, 0])
        self.n = n_inputs
        self.epoch = max_epochs
        self.rate = learning_rate

    def forward(self, input_data):
        """
        Predict label from input 
        Args:
            input_data: array of dimension equal to n_inputs.
        """
        label = np.sign(np.dot(self.w, np.transpose(input_data)))
        return label

    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        for _ in range(int(self.epoch)):
            tmp = np.column_stack((training_inputs, labels))
            np.random.shuffle(tmp)
            training_inputs = tmp[:, :-1]
            labels = tmp[:, -1]
            predict = self.forward(training_inputs)
            if np.dot(predict, labels) <= 0:
                self.w = self.w + self.rate * np.dot(labels, training_inputs)


if __name__ == '__main__':
    # generate dataset
    mean1, mean2, = [10, 16], [-6, -3]
    cov1 = [[3, 0], [0, 4]]
    cov2 = [[2, 0], [0, 5]]
    x1, y1 = random.multivariate_normal(mean1, cov1, 100).T
    x2, y2 = random.multivariate_normal(mean2, cov2, 100).T
    # split into train set and test set
    train_data = np.array([[x1[i], y1[i]] for i in range(80)] + [[x2[i], y2[i]] for i in range(80)])
    train_label = np.array([1 for i in range(80)] + [-1 for i in range(80)])
    test_data = np.array([[x1[i], y1[i]] for i in range(80, 100)] + [[x2[i], y2[i]] for i in range(80, 100)])
    test_label = np.array([1 for i in range(20)] + [-1 for i in range(20)])
    # start training
    perceptron = Perceptron(160)
    perceptron.train(train_data, train_label)
    # output result
    res = perceptron.forward(test_data)
    print("accuracy: ", np.dot(test_label, res) / 40)
    # draw
    x = np.array([-10, 10])
    y = -perceptron.w[0] / perceptron.w[1] * x
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.plot(x, y)
    plt.show()
