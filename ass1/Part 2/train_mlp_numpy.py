from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from mlp_numpy import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def oneHot(labels, dim=None):
    if dim is None:
        dim = np.max(labels) + 1
    res = np.zeros((len(labels), dim))
    for i, label in enumerate(labels):
        res[i, label] = 1
    return res


def accuracy(predictions, labels):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    correct = 0
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if predictions[i][j] == np.max(predictions[i]):
                if labels[i][j] == 1:
                    correct += 1
                continue
    res = correct / len(predictions)
    # res = np.mean((np.argmax(predictions, axis=1) == labels))
    return res


def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    args = parser.parse_args()
    dim_hidden = list(map(int, args.dnn_hidden_units.split(',')))
    freq = args.eval_freq
    lr = args.learning_rate
    max_step = args.max_steps
    size = 1000
    data, label = make_moons(n_samples=size, noise=0.05)
    # plt.scatter(data[:, 0], data[:, 1], s=10, c=label)
    # plt.show()
    bound = int(0.8 * size)
    train_data, test_data = data[:bound], data[bound:]
    train_label, test_label = oneHot(label[:bound]), oneHot(label[bound:])
    train_data, test_data = np.array(train_data), np.array(test_data)

    module = MLP(n_inputs=2, n_hidden=dim_hidden, n_classes=2)
    a, l = [], []
    for t in range(max_step):
        pred = module(train_data)
        if t % freq == 0:
            # print(pred)
            # print(train_label)
            loss = module.loss_fc(pred, train_label)
            acc = accuracy(module.predict(test_data), test_label)
            print("In round {}, the loss is {}.".format(t, loss))
            print("accuracy: ", acc)
            l.append(loss)
            a.append(acc)

        grad = module.loss_fc.backward(pred, train_label)
        module.backward(grad)

        for layer in module.layers:
            layer.update(lr)

    plt.figure()
    plt.plot(l, 'b-')
    plt.ylabel("loss function")
    plt.figure()
    plt.plot(a, 'b-')
    plt.ylabel("accuracy")
    plt.show()


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
