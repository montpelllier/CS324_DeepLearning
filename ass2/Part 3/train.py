from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import *


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model that we are going to use
    model = VanillaRNN(seq_length=config.input_length, input_dim=config.input_dim, hidden_dim=config.num_hidden,
                       output_dim=config.num_classes, batch_size=config.batch_size, device=device)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Set up the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # Create list to store data
    acc_list = []
    loss_list = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        output = model.forward(batch_inputs)
        loss = criterion(output, batch_targets)
        loss.backward()
        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        loss = loss.item()
        acc_in = np.argmax(output.cpu().detach().numpy(), axis=1) == batch_targets.cpu().detach().numpy()
        accuracy = np.sum(acc_in) / batch_targets.shape[0]
        loss_list.append(loss)
        acc_list.append(accuracy)

        if step % 10 == 0:
            print(
                'Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f} %.'.format(step,
                                                                          config.train_steps,
                                                                          loss,
                                                                          accuracy * 100))

        if step == config.train_steps:
            break

    print('Done training.')
    # draw curve of acc and loss
    plt.figure()
    plt.title("Pytorch CNN Accuracy")
    plt.plot(acc_list, label="accuracy")
    plt.ylabel("accuracy")
    plt.legend()

    plt.figure()
    plt.title("Pytorch CNN Loss")
    plt.plot(loss_list, 'b-')
    plt.ylabel("loss function")
    plt.show()


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train(config)
