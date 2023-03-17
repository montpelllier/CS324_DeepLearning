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
    for i, pred in enumerate(predictions):
        if np.argmax(pred) == np.argmax(labels[i]):
            correct += 1
    return correct / len(predictions)


def train(epoch, hidden_list, freq, lr, sgd, train_set, test_set):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    train_x, train_y = train_set
    test_x, test_y = test_set
    train_acc_list, test_acc_list, loss_list = [], [], []
    module = MLP(2, hidden_list, 2)
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
            layer.update(lr)

    plt.figure()
    plt.plot(train_acc_list, label="train accuracy")
    plt.plot(test_acc_list, label="test accuracy")
    plt.ylabel("accuracy")
    plt.legend()
    plt.figure()
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
    sgd = args.sgd
    # generate dataset
    size = 1000
    data, label = make_moons(n_samples=size, noise=0.05)
    # split into train set and test set
    bound = int(0.8 * size)
    train_data, test_data = data[:bound], data[bound:]
    train_label, test_label = oneHot(label[:bound]), oneHot(label[bound:])
    train_data, test_data = np.array(train_data), np.array(test_data)
    # draw
    plt.scatter(data[:, 0], data[:, 1], s=10, c=label)
    plt.show()
    # train
    train(max_step, dim_hidden, freq, lr, sgd, (train_data, train_label), (test_data, test_label))


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
