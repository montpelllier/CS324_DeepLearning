from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.nn import Parameter, Module


class LSTM(Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size, device):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        # create variables for weights and biases
        self.W_gx = Parameter(torch.randn((hidden_dim, input_dim)).to(device), requires_grad=True)
        self.W_gh = Parameter(torch.randn((hidden_dim, hidden_dim)).to(device), requires_grad=True)
        self.b_g = Parameter(torch.zeros((hidden_dim, 1)).to(device), requires_grad=True)

        self.W_ix = Parameter(torch.randn((hidden_dim, input_dim)).to(device), requires_grad=True)
        self.W_ih = Parameter(torch.randn((hidden_dim, hidden_dim)).to(device), requires_grad=True)
        self.b_i = Parameter(torch.zeros((hidden_dim, 1)).to(device), requires_grad=True)

        self.W_fx = Parameter(torch.randn((hidden_dim, input_dim)).to(device), requires_grad=True)
        self.W_fh = Parameter(torch.randn((hidden_dim, hidden_dim)).to(device), requires_grad=True)
        self.b_f = Parameter(torch.zeros((hidden_dim, 1)).to(device), requires_grad=True)

        self.W_ox = Parameter(torch.randn((hidden_dim, input_dim)).to(device), requires_grad=True)
        self.W_oh = Parameter(torch.randn((hidden_dim, hidden_dim)).to(device), requires_grad=True)
        self.b_o = Parameter(torch.randn((hidden_dim, 1)).to(device), requires_grad=True)

        self.W_ph = Parameter(torch.randn((output_dim, hidden_dim)).to(device), requires_grad=True)
        self.b_p = Parameter(torch.zeros((output_dim, 1)).to(device), requires_grad=True)

        self.h = torch.zeros(hidden_dim, batch_size).to(device)
        self.c = torch.zeros(hidden_dim, batch_size).to(device)

    def forward(self, x):
        # initialize
        h_t = self.h
        c_t = self.c
        # train hidden layers
        for t in range(self.seq_length):
            g = torch.tanh(self.W_gx @ x[:, t].view(1, -1) + self.W_gh @ h_t + self.b_g)
            i = torch.sigmoid(self.W_ix @ x[:, t].view(1, -1) + self.W_ih @ h_t + self.b_i)
            f = torch.sigmoid(self.W_fx @ x[:, t].view(1, -1) + self.W_fh @ h_t + self.b_f)
            o = torch.sigmoid(self.W_ox @ x[:, t].view(1, -1) + self.W_oh @ h_t + self.b_o)
            c = g * i + c_t * f
            h = torch.tanh(c) * o

            h_t = h
            c_t = c
        # output
        p = (self.W_ph @ h_t + self.b_p).transpose(1, 0)
        y = torch.softmax(p, dim=-1)
        return y
