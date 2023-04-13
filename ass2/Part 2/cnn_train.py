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
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
DATA_DIR_DEFAULT = './data'
OPTIMIZER_DEFAULT = 'ADAM'

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


def train(epoch_num: int, optimizer_name, learning_rate, train_loader, freq, test_loader):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # use GPU if available
    if torch.cuda.is_available():
        print("using GPU", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("using CPU")
        device = torch.device("cpu")

    model = cnn_model.CNN(3, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_name = optimizer_name.upper()

    # chose optimizer
    print("try to use optimizer", optimizer_name)
    if optimizer_name == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'RPROP':
        optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
    else:
        print("unsupported optimizer!")
        raise RuntimeError()

    train_acc_list, test_acc_list, loss_list = [], [], []

    epoch = 0
    flag = True
    while flag:
        for feature, label in train_loader:
            epoch += 1
            feature, label = feature.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(feature)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.cpu().item())

            if epoch % freq == 0:
                # test model
                test_acc = get_acc(model, test_loader, device)
                test_acc_list.append(test_acc)
                train_acc = get_acc(model, train_loader, device)
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

    # draw curve of acc and loss
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
                                               shuffle=True, num_workers=0)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=0)

    t = datetime.datetime.now()
    train(max_step, optimizer, lr, train_loader, freq, test_loader)
    print("cost total:", datetime.datetime.now() - t)


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
