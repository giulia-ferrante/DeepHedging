import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

from classes.Dataset import Dataset
from classes.RNN import RNN_Model
from classes.FCN import FCN_Model
from classes.Heston import Heston
from classes.Loss_Class import Loss_Class

import matplotlib.pyplot as plt

N_SAMPLES = 2048 
BATCH_SIZE = 256
N_EPOCHS = 128
OUT_DIM = 2           # Delta size
IN_DIM = OUT_DIM * 2 
TIMESTEPS = 30 
X_DIM = OUT_DIM * (TIMESTEPS + 2) 
Y_DIM = OUT_DIM * (TIMESTEPS + 1) 
HIDDEN_DIM = 17        
N_H_SLAYERS = 3       # Number og layers inside RNN subnet network  
N_H_LAYERS = 4        # Number og layers inside FCN network  

_PRE_TRAINED = False     # True => pre-trained network (load existing model parameters)
_TRAIN       = True      # True => Train else Inference

_MODEL = 'RNN'
#_MODEL = 'FCN'
_HESTON = True
_PLOT_ERROR = True

def plot_error_pred(X, he, model, filename=None):

    loss = Loss_Class(n=TIMESTEPS, batch_size=N_SAMPLES, alpha = 0.5, device=device)
    y_pred = model(X).to(device)
    loss_mean = loss.forward(delta_pred=y_pred, batch_x=X, batch_id=0, he=he, model=model)
    err_delta = (loss_mean - loss.liabilities[:] + loss.sum_delta_S[:]).cpu().detach().numpy()

    mean_err = np.mean(err_delta)
    std_err = np.std(err_delta)
    bins = np.linspace(mean_err - 3*std_err, mean_err + 3*std_err, 50)

    plt.figure(figsize=(8,5))
    plt.hist(err_delta, bins=bins, color='orange', edgecolor='black', linewidth=1.2)
    plt.xlabel('Hedging Error')
    plt.ylabel(f'Number of Samples (out of {N_SAMPLES})')
    plt.title(f'Distribution of Final Hedging Errors')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_loss(eloss):
    nloss = np.asarray(eloss, dtype=np.float32)
    plt.plot(nloss, color='black')
    plt.xlabel('Epoch')
    plt.ylabel(f'Loss')
    plt.title('Average Loss Function')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("HW_Accelerator:",device)
    print()
    torch.set_default_device(device)
    torch.autograd.set_detect_anomaly(True) 

    if (_MODEL == 'RNN'):
        model = RNN_Model(n = TIMESTEPS, x_dim = X_DIM, y_dim = Y_DIM, hidden_dim = HIDDEN_DIM, n_h_slayers = N_H_SLAYERS).to(device)
    elif (_MODEL == 'FCN'):
        model = FCN_Model(x_dim = X_DIM, y_dim = Y_DIM, hidden_dim = HIDDEN_DIM, n_h_layers = N_H_LAYERS).to(device)

    print(model)
    summary(model)

    if _PRE_TRAINED or not _TRAIN: 
        if (_MODEL == 'RNN'):
            model = torch.load('model_RNN.pth',weights_only=False)
        elif (_MODEL == 'FCN'):
            model = torch.load('model_FCN.pth',weights_only=False)

    if _TRAIN:
        
        N_BATCH = N_SAMPLES//BATCH_SIZE
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        if _HESTON:
            loss_function = Loss_Class(n = TIMESTEPS, batch_size = BATCH_SIZE, device = device)
        else:
            loss_function = nn.MSELoss()
        if _HESTON:
            he = Heston(n = TIMESTEPS, device = device) 
            X,Y = he.get_dataset(n_samples = N_SAMPLES)
        else:
            ds = Dataset(n_samples = N_SAMPLES, x_dim = X_DIM, y_dim = Y_DIM, device = device)
            X,Y = ds.get_easy_dataset()    

        eloss = []

        for epoch in range(N_EPOCHS):
            total_loss = 0
            for i in range(N_BATCH):
                batch_x = X[i*BATCH_SIZE:(i+1)*BATCH_SIZE].detach().clone()
                batch_y = Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE].detach().clone()                
                optimizer.zero_grad()
                y_pred = model(batch_x)
                if _HESTON:
                    loss = loss_function(delta_pred = y_pred, batch_x = batch_x, batch_id = i, he = he, model = model)
                else:
                    loss = loss_function(y_pred, batch_y)

                total_loss += loss.item()
                loss.backward()
                optimizer.step()                
                
            average_loss = total_loss / N_BATCH
            print(f'Epoch {epoch + 1}/{N_EPOCHS}, Average Loss: {average_loss}')
            eloss.append(average_loss)

        if (_MODEL == 'RNN'):
            torch.save(model, 'model_RNN.pth')
        elif (_MODEL == 'FCN'):
            torch.save(model, 'model_FCN.pth')

        plot_loss(eloss)
        if (_PLOT_ERROR): plot_error_pred(X,he,model)
                
    else: # Inference

        if _HESTON:
            he = Heston(n = TIMESTEPS, device = device)
            X,Y = he.get_dataset(n_samples = N_SAMPLES)
        else:
            ds = Dataset(n_samples = N_SAMPLES, x_dim = X_DIM, y_dim = Y_DIM, device = device) 
            X,Y = ds.get_easy_dataset()

        y_pred = model(X).to(device)        
        print("Prediction: ", y_pred[0])
        if (_PLOT_ERROR): plot_error_pred(X,he,model)
    