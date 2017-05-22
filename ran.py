import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn._functions.rnn import Recurrent, StackedRNN


class RAN(nn.Module):

    def __init__(self, input_size, hidden_size, nlayers=1, dropout=0.5):
        super().__init__()
        if nlayers > 1:
            raise NotImplementedError("TODO: nlayers > 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.w_cx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_ic = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_ix = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_fc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_fx = nn.Parameter(torch.Tensor(hidden_size, input_size))

        self.b_cx = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ic = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ix = nn.Parameter(torch.Tensor(hidden_size))
        self.b_fc = nn.Parameter(torch.Tensor(hidden_size))
        self.b_fx = nn.Parameter(torch.Tensor(hidden_size))

        self.weights = self.w_cx, self.w_ic, self.w_ix, self.w_fc, self.w_fx
        for w in self.weights:
            init.xavier_uniform(w)

        self.biases = self.b_cx, self.b_ic, self.b_ix, self.b_fc, self.b_fx
        for b in self.biases:
            b.data.fill_(0)

    def forward(self, input, hidden):
        layer = (Recurrent(RANCell), )
        func = StackedRNN(layer, self.nlayers, dropout=self.dropout)
        hidden, output = func(input, hidden, ((self.weights, self.biases), ))
        return output, hidden


def RANCell(input, hidden, weights, biases):
    w_cx, w_ic, w_ix, w_fc, w_fx = weights
    b_cx, b_ic, b_ix, b_fc, b_fx = biases

    ctilde_t = F.linear(input, w_cx, b_cx)
    i_t = F.sigmoid(F.linear(hidden, w_ic, b_ic) + F.linear(input, w_ix, b_ix))
    f_t = F.sigmoid(F.linear(hidden, w_fc, b_fc) + F.linear(input, w_fx, b_fx))
    c_t = i_t * ctilde_t + f_t * hidden
    h_t = F.tanh(c_t)

    return h_t
