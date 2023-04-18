from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        # initialization of weights and biases
        self.W_hx = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.Tensor(hidden_dim))
        self.W_yh = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        self.b_y = nn.Parameter(torch.Tensor(output_dim))

        # initialize weights and biases with xavier initialization
        nn.init.xavier_uniform_(self.W_hx)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.W_yh)
        nn.init.zeros_(self.b_h)
        nn.init.zeros_(self.b_y)

    def forward(self, x):
        # Implementation here ...
        h_t = torch.zeros(self.batch_size, self.hidden_dim)  # initialize hidden state
        for t in range(self.seq_length):
            x_t = x[:, t, :]
            h_t = torch.tanh(x_t @ self.W_hx + h_t @ self.W_hh + self.b_h)  # calculate hidden state
            y_t = h_t @ self.W_yh + self.b_y  # calculate output
        return y_t

    # add more methods here if needed
