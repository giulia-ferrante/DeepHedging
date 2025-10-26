import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

'''Small feedforward subnet used inside the main RNN model'''
class SubModel(nn.Module):  
    def __init__(self, sid = 1, in_dim = 4, out_dim = 2, hidden_dim = 17, n_h_slayers = 4):
        super().__init__()
        
        self.sid = sid
        self.in_dim = in_dim
        self.out_dim = out_dim        
        self.hidden_dim = hidden_dim
        self.n_h_slayers = n_h_slayers

        # Build a small sequential feedforward network
        self.modules = []
        self.modules.append(nn.Linear(self.in_dim, self.hidden_dim))
        self.modules.append(nn.ReLU())

        assert self.n_h_slayers > 1, 'Need at least 2 hidden subnet layers'
        for i in range(self.n_h_slayers - 2):
            self.modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))            
            self.modules.append(nn.BatchNorm1d(self.hidden_dim))
            self.modules.append(nn.ReLU())
            self.modules.append(nn.Dropout(0.2))
        self.modules.append(nn.Linear(self.hidden_dim, self.out_dim))
        self.linear_relu_stack = nn.Sequential(*self.modules) # Combine all layers into one sequential module
                
    def forward(self, x):
        return self.linear_relu_stack(x)
        
    def print_weights(self,nlayer = 0): 
        lw = self.modules[nlayer].weight.clone()
        lb = self.modules[nlayer].bias.clone()
        print(lw)
        print(lb)

'''Large RNN-like model built from sequential SubModels.
    Each timestep has its own small subnet'''
class RNN_Model(nn.Module): 
    
    def __init__(self, n = 30, x_dim = 8, y_dim = 6, hidden_dim = 17, n_h_slayers = 4):
        super().__init__()
        
        assert n > 0, 'At least 1 timestep required'
        self.n = n
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.out_dim = x_dim - y_dim 
        self.in_dim = 2 * self.out_dim     
        self.hidden_dim = hidden_dim
        self.n_h_slayers = n_h_slayers
        self.w = nn.Parameter(data=torch.tensor([0.])) 
            
        # Create a list of small subnets, one for each timestep    
        self.sub_models = [SubModel(i, self.in_dim, self.out_dim, self.hidden_dim, self.n_h_slayers) 
                           for i in range(self.n)] 
        self.seq = nn.Sequential(*self.sub_models) 

    def forward(self, x):
        
        self.output = x[:, :self.y_dim].detach().clone() 
        y = self.sub_models[0](x[:, :2*self.out_dim]) 
        self.output[:, :self.out_dim] = y  
        
        for i in range(1, self.n):
            x1 = torch.cat((y, x[:, (i+1)*self.out_dim:(i+2)*self.out_dim]), dim=1)
            y = self.sub_models[i](x1)
            self.output[:, i*self.out_dim:(i+1)*self.out_dim] = y

        return self.output


