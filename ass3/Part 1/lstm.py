from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.num_hidden = hidden_dim
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.W_gx = nn.Parameter(torch.randn((hidden_dim, input_dim)), requires_grad=True)
        self.W_gh = nn.Parameter(torch.randn((hidden_dim, hidden_dim)), requires_grad=True)
        self.bias_g = nn.Parameter(torch.zeros((hidden_dim, 1)), requires_grad=True)

        self.W_ix = nn.Parameter(torch.randn((hidden_dim, input_dim)), requires_grad=True)
        self.W_ih = nn.Parameter(torch.randn((hidden_dim, hidden_dim)), requires_grad=True)
        self.bias_i = nn.Parameter(torch.zeros((hidden_dim, 1)), requires_grad=True)

        self.W_fx = nn.Parameter(torch.randn((hidden_dim, input_dim)), requires_grad=True)
        self.W_fh = nn.Parameter(torch.randn((hidden_dim, hidden_dim)), requires_grad=True)
        self.bias_f = nn.Parameter(torch.zeros((hidden_dim, 1)), requires_grad=True)

        self.W_ox = nn.Parameter(torch.randn((hidden_dim, input_dim)), requires_grad=True)
        self.W_oh = nn.Parameter(torch.randn((hidden_dim, hidden_dim)), requires_grad=True)
        self.bias_o = nn.Parameter(torch.randn((hidden_dim, 1)), requires_grad=True)

        self.W_ph = nn.Parameter(torch.randn((output_dim, hidden_dim)), requires_grad=True)
        self.bias_p = nn.Parameter(torch.zeros((output_dim, 1)), requires_grad=True)

    def forward(self, x):
        # Implementation here ...
        h_tmin1 = torch.zeros(self.num_hidden, self.batch_size)
        c_tmin1 = torch.zeros(self.num_hidden, self.batch_size)

        for t in range(self.seq_length):
            g = torch.tanh(self.W_gx @ x[:, t].view(1, -1) + self.W_gh @ h_tmin1 + self.bias_g)
            i = nn.functional.sigmoid(self.W_ix @ x[:, t].view(1, -1) + self.W_ih @ h_tmin1 + self.bias_i)
            f = nn.functional.sigmoid(self.W_fx @ x[:, t].view(1, -1) + self.W_fh @ h_tmin1 + self.bias_f)
            o = nn.functional.sigmoid(self.W_ox @ x[:, t].view(1, -1) + self.W_oh @ h_tmin1 + self.bias_o)
            c = g * i + c_tmin1 * f
            h = torch.tanh(c) * o

            h_tmin1 = h
            c_tmin1 = c

        p = (self.W_ph @ h + self.bias_p).transpose(1, 0)
        y = nn.functional.softmax(p)

        return y
