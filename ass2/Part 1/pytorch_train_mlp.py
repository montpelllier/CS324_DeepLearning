from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn

from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    _, one_hot = torch.max(predictions.data, 1)
    one_hot = nn.functional.one_hot(one_hot)
    acc = (one_hot == targets).all(dim=1).float().mean()
    return acc


def train(epoch, hidden_list, freq, lr, sgd, train_set, test_set):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_acc_list, test_acc_list, loss_list = [], [], []
    module = MLP(2, hidden_list, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr)

    for t in range(epoch):
        x = train_x
        y = train_y

        outputs = module(x)
        loss = module.criterion(outputs, y)
        # update paras
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if t % freq == 0:
            train_acc = accuracy(module(train_x), train_y)
            test_acc = accuracy(module(test_x), test_y)
            print("In round {}, the loss is {}, the test accuracy is {}, and the train accuracy is {}.".format(t, loss,
                                                                                                               test_acc,
                                                                                                               train_acc))

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            loss_list.append(float(loss))

    plt.figure()
    plt.title("Pytorch MLP Accuracy")
    plt.plot(train_acc_list, label="train accuracy")
    plt.plot(test_acc_list, label="test accuracy")
    plt.ylabel("accuracy")
    plt.legend()
    plt.figure()
    plt.title("Pytorch MLP Loss")
    plt.plot(loss_list, 'b-')
    plt.ylabel("loss function")
    plt.show()


def main():
    """
    Main function
    """
    # handle arguments
    args = parser.parse_args()
    dim_hidden = list(map(int, args.dnn_hidden_units.split(',')))
    freq = args.eval_freq
    lr = args.learning_rate
    max_step = args.max_steps
    # sgd = args.sgd
    sgd = False
    # generate dataset
    size = 1000
    X, y = make_moons(n_samples=size, noise=0.05)
    X, y = torch.tensor(X).float(), torch.tensor(y)
    # split into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y_train = nn.functional.one_hot(y_train, num_classes=2).float()
    y_test = nn.functional.one_hot(y_test, num_classes=2).float()
    # train
    train(max_step, dim_hidden, freq, lr, sgd, (X_train, y_train), (X_test, y_test))


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
