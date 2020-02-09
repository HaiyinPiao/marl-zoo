import os
import sys
import torch.nn as nn
import torch
from utils.math import *
from utils.args import *

# sys.path.append(os.getcwd()+'/../transformer-encoder/')
# sys.path.append(os.getcwd()+'/../transformer-encoder/transformer/')

# import transformer.Constants as Constants
# from transformer.Layers import EncoderLayer
# from transformer.Models import Transformer, Encoder

log_protect = 1e-5
multinomial_protect = 1e-10

class DiscretePolicy(nn.Module):
    def __init__(self, n, state_dim, action_num, hidden_size=[128], activation='tanh'):
        super().__init__()
        self.n = n
        self.state_dim = state_dim
        self.action_num = action_num

        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        # utilizing Transformer Encoder as hidden for Relational-MARL.
        if args.rrl is True:
            # self.encoder_stacks = Encoder(d_model=state_dim, d_inner=64, d_word_vec=state_dim, n_position=self.n,
            #     n_layers=2, n_head=6, d_k=16, d_v=16, dropout=0.05)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=state_dim, nhead=1)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        # mlp as hidden.
        # only use 1 layer.
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim if args.rrl is True else state_dim*n
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh) )
            last_dim = nh
        set_init(self.affine_layers)

        self.action_hiddens = nn.ModuleList()
        self.action_heads = nn.ModuleList()
        for i in range(n):
            self.action_hiddens.append( nn.Linear(last_dim, int(last_dim/2)) )
            self.action_heads.append( nn.Linear(int(last_dim/2), action_num ) )
        
        set_init(self.action_hiddens)
        set_init(self.action_heads)

    def forward(self, x):
        action_prob = []
        # print(x.shape)

        # utilizing Transformer Encoder as hidden for Relational-MARL.
        if args.rrl is True:
            # x, _ = self.encoder_stacks.forward(x, src_mask = None)
            x = self.transformer_encoder(x)
            # # max aggregation.
            # print(x.shape)

            # x = (torch.max(x,1)).values
            x = torch.sum(x,1)
            # print(x.shape)
        # mlp as hidden.
        x = x.view(x.shape[0],-1)
        # print(x.shape)
        # exit()

        for l in self.affine_layers:
            x = self.activation(l(x))        

        for i in range(self.n):
            a = self.activation(self.action_hiddens[i](x))
            # print(self.action_heads[i](a))
            # print(a)
            a = torch.softmax(self.action_heads[i](a), dim=1)
            # print(a)
            # exit()
            action_prob.append(a)
        # print(action_prob)
        # exit()
        return action_prob

    def select_action(self, x):
        actions = []
        action_prob = self.forward(x)
        for i in range(self.n):
            action_prob[i] += multinomial_protect
            action = action_prob[i].multinomial(1)
            actions.append(action.item())
        # print(actions)
        # exit()
        return actions

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        # print(action_prob[0].shape)

        action_prob = torch.stack(action_prob)
        action_prob.transpose_(0,1)

        actions = actions.unsqueeze(2)
        # actions.transpose_(0,1)
        # print(actions.shape)
        
        # action_prob = torch.tensor(action_prob)
        # print(action_prob.shape)
        # print(x.shape, actions.shape)
        # print(action_prob.shape, actions.shape)
        # print(action_prob[0], actions[0])

        # ******************gather all action heads probs********************
        choice = action_prob.gather(2, actions.long())+log_protect
        # print(choice.shape, choice)
        choice = torch.prod(choice, dim=1)
        # print(choice.shape, choice)
        # exit()
        return torch.log(choice)

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}

