from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime

import torch
from matplotlib import pyplot as plt
from torchvision import transforms, datasets

from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '512, 256, 128'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 200
BATCH_DEFAULT = 4

FLAGS = None


def get_acc(model, data_loader, device):
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(epoch, hidden_list, freq, lr, train_loader, test_loader):
    """
    Performs training and evaluation of MLP model.
    """
    # use GPU if available
    if torch.cuda.is_available():
        print("using GPU", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("using CPU")
        device = torch.device("cpu")

    train_acc_list, test_acc_list, loss_list = [], [], []

    model = MLP(32 * 32 * 3, hidden_list, 10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr)
    # start training
    e = 0
    flag = True
    while flag:
        for data in train_loader:
            e += 1
            # if e > 1500:
            #     optimizer = torch.optim.SGD(model.parameters(), lr/10)
            # else:
            #     optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(float(loss))
            if e % freq == 0:
                train_acc = get_acc(model, train_loader, device)
                test_acc = get_acc(model, test_loader, device)
                print("In round {}, Loss={}, Test Acc={:.4f} %, Train Acc={:.4f} %.".format(e,
                                                                                            loss,
                                                                                            test_acc * 100,
                                                                                            train_acc * 100))
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

            if e == epoch:
                print("finish training")
                flag = False
                break

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
    batch_size = args.batch_size
    # generate dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    test_set = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=0)

    # train
    t = datetime.datetime.now()
    train(max_step, dim_hidden, freq, lr, train_loader, test_loader)
    print("cost total:", datetime.datetime.now() - t)


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
    parser.add_argument('--batch_size', type=int, default=BATCH_DEFAULT, help='Batch size')
    FLAGS, unparsed = parser.parse_known_args()
    main()
