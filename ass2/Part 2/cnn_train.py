from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime

import torch
import torchvision
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


def train(epoch_num: int, learning_rate, train_loader, freq, test_loader):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    model = cnn_model.CNN(3, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(len(train_loader))
    for epoch in range(epoch_num):
        running_loss = 0
        for i, data in enumerate(train_loader):
            start_time = datetime.datetime.now()

            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()/len(train_loader)
            if i % 100 == 99:
                print('Epoch [{}/{}], Loss: {:.4f}, Index [{}/{}]'.format(epoch + 1, epoch_num, running_loss, i, len(train_loader)))

            end_time = datetime.datetime.now()
            delta = (end_time-start_time)
            print(delta)

        print(running_loss)

        if epoch % freq == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epoch_num, running_loss))

    # 测试模型
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('测试集准确率为: {} %'.format(100 * correct / total))


def main():
    """
    Main function
    """
    # handle arguments
    args = parser.parse_args()
    freq = args.eval_freq
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

    train(max_step, lr, train_loader, freq, test_loader)


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
    FLAGS, unparsed = parser.parse_known_args()

    main()
