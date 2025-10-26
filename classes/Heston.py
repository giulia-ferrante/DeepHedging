import numpy as np
import torch
import math
import matplotlib.pyplot as plt

class Heston():
    
    def __init__(self, 
        s1_0=100,       # Initial price of asset S1
        v0=0.04,        # Initial variance
        mu=0,           # Risk-free interest rate
        k=1.0,          # Mean reversion speed of variance
        theta=0.04,     # Long-term average variance
        sigma=2.0,      # Volatility of the variance
        rho=-0.7,       # Correlation between asset and variance
        n=30,           # Number of time steps
        device = 'cpu'):

        self.s1_0 = s1_0
        self.v0 = v0
        self.mu = mu
        self.k = k
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.n = n
        self.T = n/365.0
        self.device = device
        
        self.out_dim = 2  # Number of assets (S1 and S2)
        self.dt = self.T / self.n

    ''' Simulate asset S1 and variance v using the Heston model with discrete time steps.'''
    def simulate_heston(self):

        self.S1 = np.zeros(self.n+1)
        self.v = np.zeros(self.n+1)
        self.S1[0] = self.s1_0
        self.v[0] = self.v0

        for t in range(1, self.n + 1):
            # Generate two correlated standard normal variables
            self.Z1 = np.random.normal()
            self.Z2 = self.rho * self.Z1 + np.sqrt(1 - self.rho**2) * np.random.normal()

            # Asset price update 
            self.S1[t] = self.S1[t-1] * np.exp((self.mu - 0.5 * self.v[t-1]) * self.dt + np.sqrt(self.v[t-1] * self.dt) * self.Z1)

            # Variance update with truncation to avoid negatives
            self.v[t] = self.v[t-1] + self.k * (self.theta - self.v[t-1]) * self.dt + self.sigma * np.sqrt(self.v[t-1] * self.dt) * self.Z2
            self.v[t]= max(self.v[t], 0)

    '''Compute S2 as the discrete-time integral of the variance process.'''
    def compute_S2(self):
        
        self.S2 = np.zeros(self.n+1)
        realized=0.0  # Integral of variance already realized

        for ik in range(self.n + 1):
            if ik > 0 :
              realized = np.sum(self.v[:ik]) * self.dt
            else: 
              realized = 0.0

            expected_future = ((self.v[ik] - self.theta)/self.k) * (1 - np.exp(-self.k*(self.T - ik * self.dt))) \
                              + self.theta*(self.T - ik * self.dt)
            self.S2[ik] = realized + expected_future #Total variance (realized + expected)       
     
    def delta_true(self): #STILL TO DO
        
        self.Y = torch.randn(self.n_samples, self.out_dim * (self.n + 1)).to(self.device)
        return self.Y

    def set_S1_S2(self, n_samples = 16): #(S1^0, S1^1,..., S1^30)*n_samples & (S2^0, S2^1,..., S2^30)*n_samples
        
        self.n_samples = n_samples
        self.S1L = torch.empty(self.n_samples, self.n + 1).to(self.device)
        self.S2L = torch.empty(self.n_samples, self.n + 1).to(self.device)  
        self.X = torch.zeros(self.n_samples, self.out_dim*(self.n + 2)).to(self.device)

        for i in range(self.n_samples):
            self.simulate_heston()
            self.compute_S2()
            
            self.X[i][:self.out_dim] = 0.0  
            for j in range(1, self.n + 2): 
                self.X[i][self.out_dim*j] = math.log(self.S1[j-1])  #Data input RNN network
                self.X[i][self.out_dim*j+1] = self.v[j-1]           #Data input RNN network
               
            self.S1L[i] = torch.from_numpy(self.S1) #Data input Loss function
            self.S2L[i] = torch.from_numpy(self.S2) #Data input Loss function
        
        self.Y = self.delta_true() 
        
    def get_S1_S2_loss(self): 
    
        return self.S1L, self.S2L 
     
    def get_dataset(self, n_samples = 16):
        
        self.set_S1_S2(n_samples)                        
        return self.X, self.Y
