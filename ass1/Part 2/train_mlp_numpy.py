from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

import modules
from mlp_numpy import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 3e-1
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
SGD_DEFAULT = False
FLAGS = None


def oneHot(labels, dim=None):
    if dim is None:
        dim = np.max(labels) + 1
    one_hot = np.zeros((len(labels), dim))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


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
    # print(predictions)
    for i, pred in enumerate(predictions):
        if np.argmax(pred) == np.argmax(labels[i]):
            correct += 1
    return correct / len(predictions)


def train(epoch, in_size, hidden_list, out_size, freq, lr, sgd, train_set, test_set):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_acc_list, test_acc_list, loss_list = [], [], []
    module = MLP(in_size, hidden_list, out_size)

    # start training
    for t in range(epoch):
        if sgd:
            rand_i = np.random.randint(len(train_x))
            x = train_x[rand_i:rand_i + 1]
            y = train_y[rand_i:rand_i + 1]
        else:
            x = train_x
            y = train_y

        pred = module(x)
        # print("pred", pred)
        # print("y", y)
        grad = module.loss_fc.backward(pred, y)

        module.backward(grad)

        if t % freq == 0:
            train_acc = accuracy(module.predict(train_x), train_y)
            test_acc = accuracy(module.predict(test_x), test_y)
            loss = module.loss_fc(pred, y)
            print("In round {}, the loss is {}, the accuracy is {}.".format(t, loss, test_acc))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            loss_list.append(loss)

        for layer in module.layers:
            if isinstance(layer, modules.Linear):
                layer.update(lr)

    plt.figure()
    plt.plot(train_acc_list, label="train accuracy")
    plt.plot(test_acc_list, label="test accuracy")
    plt.ylabel("accuracy")
    plt.legend()
    plt.figure()
    plt.plot(loss_list, color='blue')
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
    sgd = args.sgd
    # generate dataset
    size = 1000
    # data, label = make_moons(n_samples=size, noise=0.05)
    data, label = make_circles(n_samples=size, noise=0.04, factor=0.7)
    # data, label = make_blobs(n_samples=size, n_features=16, centers=6)
    n_in, n_out = len(data[0]), max(label) + 1
    # split into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)
    y_train, y_test = oneHot(y_train), oneHot(y_test)
    # draw
    plt.scatter(data[:, 0], data[:, 1], s=10, c=label)
    plt.show()
    # train
    train(max_step, n_in, dim_hidden, n_out, freq, lr, sgd, (X_train, y_train), (X_test, y_test))


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
    parser.add_argument('--sgd', type=bool, default=SGD_DEFAULT, help='stochastic gradient descent')
    FLAGS, unparsed = parser.parse_known_args()
    main()
