import torch.nn as nn
import torch
from utils.math import *

class Value(nn.Module):
    def __init__(self, n, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.n = n
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim*n
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        # self.value_head.weight.data.mul_(0.1)
        # self.value_head.bias.data.mul_(0.0)
        set_init(self.affine_layers)
        set_init([self.value_head])

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        # print(x.shape)
        # exit()
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value
