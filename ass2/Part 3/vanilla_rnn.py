from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size, device):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # create variables for weights and biases
        self.W_hx = nn.Parameter(torch.empty((hidden_dim, input_dim)).to(device), requires_grad=True)
        self.W_hh = nn.Parameter(torch.empty((hidden_dim, hidden_dim)).to(device), requires_grad=True)
        self.W_yh = nn.Parameter(torch.empty((output_dim, hidden_dim)).to(device), requires_grad=True)
        self.b_h = nn.Parameter(torch.empty((hidden_dim, 1)).to(device), requires_grad=True)
        self.b_y = nn.Parameter(torch.empty((output_dim, 1)).to(device), requires_grad=True)
        self.h = nn.Parameter(torch.empty((hidden_dim, batch_size)).to(device))
        # initialize weights and biases with xavier and zeros initialization
        nn.init.xavier_uniform_(self.W_hx)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.W_yh)
        nn.init.zeros_(self.b_h)
        nn.init.zeros_(self.b_y)

    def forward(self, x):
        nn.init.zeros_(self.h)  # initialize
        h_t = self.h
        y = None

        for t in range(self.seq_length):
            x_t = x[:, t].view(1, -1)
            h_t = torch.tanh(self.W_hx @ x_t + self.W_hh @ h_t + self.b_h)  # calculate hidden state
            y = self.W_yh @ h_t + self.b_y  # calculate output
        return y.T
