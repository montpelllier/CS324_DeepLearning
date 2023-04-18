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
        self.W_ph = nn.Parameter(torch.empty((output_dim, hidden_dim)).to(device), requires_grad=True)
        self.b_h = nn.Parameter(torch.empty((hidden_dim, 1)).to(device), requires_grad=True)
        self.b_o = nn.Parameter(torch.empty((output_dim, 1)).to(device), requires_grad=True)
        self.h = nn.Parameter(torch.empty((hidden_dim, batch_size)).to(device))
        # initialize weights and biases with xavier and zeros initializations
        nn.init.xavier_uniform_(self.W_hx)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.W_ph)
        nn.init.zeros_(self.b_h)
        nn.init.zeros_(self.b_o)

    def forward(self, x):
        # initialize
        nn.init.zeros_(self.h)
        h_t, o_t = self.h, None

        for t in range(self.seq_length):
            x_t = x[:, t].view(1, -1)
            h_t = torch.tanh(self.W_hx @ x_t + self.W_hh @ h_t + self.b_h)  # calculate hidden state
            o_t = self.W_ph @ h_t + self.b_o  # calculate output
        y = torch.softmax(o_t.T, dim=1)
        return y
