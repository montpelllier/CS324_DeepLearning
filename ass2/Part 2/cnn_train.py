from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime

import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms

import cnn_model

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 50
EVAL_FREQ_DEFAULT = 10
DATA_DIR_DEFAULT = './data'
OPTIMIZER_DEFAULT = 'ADAM'

FLAGS = None


def get_acc(model, data_loader):
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(epoch_num: int, optimizer_name, learning_rate, train_loader, freq, test_loader):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    model = cnn_model.CNN(3, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer_name = optimizer_name.upper()
    if optimizer_name == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'RPROP':
        optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
    else:
        return
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_acc_list, test_acc_list, loss_list = [], [], []

    epoch = 0
    flag = True
    while flag:
        for X, y in train_loader:
            epoch += 1
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss)
            # print("epoch", epoch)
            if epoch % freq == 0:
                # 测试模型
                print("testing model")
                # print(len(test_loader)+len(train_loader))
                test_acc = get_acc(model, test_loader)
                test_acc_list.append(test_acc)
                train_acc = get_acc(model, train_loader)
                train_acc_list.append(train_acc)
                print(
                    'Epoch [{}/{}], Loss: {:.4f}, Test Acc.: {:.4f} %, Train Acc.{:.4f} %.'.format(epoch,
                                                                                                   epoch_num,
                                                                                                   loss,
                                                                                                   test_acc * 100,
                                                                                                   train_acc * 100))
                if epoch == epoch_num:
                    print("finish training")
                    flag = False
                    break

    plt.figure()
    plt.title("Pytorch CNN Accuracy with " + optimizer_name)
    plt.plot(train_acc_list, label="train accuracy")
    plt.plot(test_acc_list, label="test accuracy")
    plt.ylabel("accuracy")
    plt.legend()
    plt.figure()
    plt.title("Pytorch CNN Loss with " + optimizer_name)
    plt.plot(loss_list, 'b-')
    plt.ylabel("loss function")
    plt.show()


def main():
    """
    Main function
    """
    # handle arguments
    args = parser.parse_args()
    freq = args.eval_freq
    optimizer = args.optimizer
    lr = args.learning_rate
    max_step = args.max_steps
    batch_size = args.batch_size
    data_dir = args.data_dir

    # load and transform the dataset.
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    print("start training")
    train(max_step, optimizer, lr, train_loader, freq, test_loader)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT, help='Name of optimizer')
    FLAGS, unparsed = parser.parse_known_args()

    main()
