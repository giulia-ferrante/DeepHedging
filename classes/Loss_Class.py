import torch
import torch.nn as nn
import numpy as np

from classes.Heston import Heston

class Loss_Class(nn.Module):
    
    def __init__(self, n = 30, K = 100, batch_size = 32, alpha = 0.5, device = 'cpu'):
        super(Loss_Class, self).__init__()  
        self.n = n
        self.K = K
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device  

    ''' Initialize Heston simulation values 
    and intermediate tensors for the loss computation'''    
    def init_he(self):
        self.S1, self.S2 = self.he.get_S1_S2_loss()
        self.S1_S2 = torch.zeros(self.batch_size,2*(self.n + 1)).to(self.device)
        start = self.batch_id*self.batch_size
        end = (self.batch_id + 1)*self.batch_size
        self.S1_S2[:,0::2] = self.S1[start:end,:]
        self.S1_S2[:,1::2] = self.S2[start:end,:]        
        self.liabilities = torch.zeros(self.batch_size).to(self.device)
        self.sum_delta_S = torch.zeros(self.batch_size).to(self.device)
        self.S1T = torch.zeros(self.batch_size).to(self.device)
        self.S1T[:] = self.S1[start:end,-1]

    '''Compute batch loss for the model'''
    def forward(self, delta_pred, batch_x, batch_id, he, model: torch.nn.Module):

        self.he = he
        self.model = model
        self.delta_pred = delta_pred
        self.batch_x = batch_x
        self.batch_id = batch_id
        self.init_he()      
        self.loss = torch.zeros(self.batch_size).to(self.device)
      
        self.sum_delta_S = (self.batch_x[:,0]*(self.S1_S2[:,2] - self.S1_S2[:,0])) 
        self.sum_delta_S += (self.batch_x[:,1]*(self.S1_S2[:,3] - self.S1_S2[:,1])) 
        for t in range(0, (self.n - 2)*2 ,2):            
            self.sum_delta_S += (self.delta_pred[:,t]*(self.S1_S2[:,t+4] - self.S1_S2[:,t+2])) 
            self.sum_delta_S += (self.delta_pred[:,t+1]*(self.S1_S2[:,t+5] - self.S1_S2[:,t+3])) 
        self.liabilities = torch.clamp(self.S1T[:] - self.K, min=0)
        self.loss = (self.model.w + torch.clamp(self.liabilities[:] - self.sum_delta_S[:] - self.model.w, min=0)/(1-self.alpha))
        return self.loss.mean()
            
