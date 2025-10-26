import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
class FCN_Model(nn.Module):
    
    def __init__(self, x_dim = 8, y_dim = 6, hidden_dim = 17, n_h_layers = 4):
        super().__init__()
        
        self.x_dim = x_dim
        self.y_dim = y_dim        
        self.hidden_dim = hidden_dim
        self.n_h_layers = n_h_layers
        self.w = nn.Parameter(data=torch.tensor([0.]))
        
        self.modules = []
        self.modules.append(nn.Linear(self.x_dim, self.hidden_dim))
        self.modules.append(nn.ReLU())
        assert self.n_h_layers > 1, 'Wrong number of hidden layers need > 1'
        for i in range(self.n_h_layers - 2):
            self.modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.modules.append(nn.BatchNorm1d(self.hidden_dim))
            self.modules.append(nn.ReLU())
            self.modules.append(nn.Dropout(0.2))
        self.modules.append(nn.Linear(self.hidden_dim, self.y_dim))
        
        self.linear_relu_stack = nn.Sequential(*self.modules)               

    def print_weights(self,nlayer = 0): 
        lw = self.modules[nlayer].weight.clone()
        lb = self.modules[nlayer].bias.clone()
        print(lw)
        print(lb)

    def forward(self, x):
        y_pred = self.linear_relu_stack(x)
        return y_pred
