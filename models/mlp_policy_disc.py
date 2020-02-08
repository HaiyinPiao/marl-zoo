import torch.nn as nn
import torch
from utils.math import *


class DiscretePolicy(nn.Module):
    def __init__(self, n, state_dim, action_num, hidden_size=(64, 32), activation='tanh'):
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

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim*n
        for i,nh in enumerate(hidden_size):
            self.affine_layers.append(nn.Linear(last_dim, nh) )
            last_dim = nh

        self.action_hiddens = nn.ModuleList()
        self.action_heads = nn.ModuleList()
        for i in range(n):
            self.action_hiddens.append( nn.Linear(last_dim, int(last_dim/2)) )
            self.action_heads.append( nn.Linear(int(last_dim/2), action_num ) )
        
        set_init(self.affine_layers)
        set_init(self.action_hiddens)
        set_init(self.action_heads)

    def forward(self, x):
        action_prob = []
        # print(x)
        x = x.view(x.shape[0],-1)
        # print(x)
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
        choice = action_prob.gather(2, actions.long())
        # print(choice.shape, choice)
        choice = torch.prod(choice, dim=1)
        # print(choice.shape, choice)
        # exit()
        return torch.log(choice)

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}
