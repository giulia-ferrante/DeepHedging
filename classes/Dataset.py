import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

class Dataset():

    def __init__(self, n_samples = 1024, x_dim = 8, y_dim = 8, device = 'cpu'):

        self.n_samples = n_samples
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.device = device
    
    def get_randn_dataset(self):

        X = torch.randn(self.n_samples, self.x_dim).to(self.device)
        Y = torch.randn(self.n_samples, self.y_dim).to(self.device)
        return X, Y
    
    def get_easy_dataset(self):
        X = torch.randn(self.n_samples, self.x_dim).to(self.device)
        Y = torch.randn(self.n_samples, self.y_dim).to(self.device)
        for i in range(self.n_samples):
          a = X[i][0]
          b = X[i][1]
          for j in range(0, self.y_dim-2,2):
            Y[i][j] = a + X[i][j+2]
            Y[i][j+1] = b + X[i][j+3]
            a= Y[i][j]
            b= Y[i][j+1]
        return X, Y
