import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from TimeSeriesDecomposition.Model import *
from TimeSeriesDecomposition.get_stock_data import *
from TimeSeriesDecomposition.SyntheticDataGeneration import *
def train_data(model,lr,epoches_trend,epoches_seasonality,batch_size,x_train,y_train):
    optimizer=torch.optim.Adam(params=model.parameters(),lr=lr)
    Lambda=0
    l=[]
    for epoch in range(epoches_trend):
        l.append(0)
        for b in range(0,x_train.shape[0],batch_size):
            loss=((y_train[b:b+batch_size,:]-model(x_train[b:b+batch_size,:]))**2).mean()+\
                Lambda*(torch.abs(model.L1.L1.weight).mean()+torch.abs(model.L1.L3.weight).mean()+\
                        torch.abs(model.L2.L1.weight).mean()+torch.abs(model.L2.L3.weight).mean()+\
                        torch.abs(model.L3.weight).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l[-1]+=loss.detach()
        print(f'epoch:{epoch}/{epoches_trend} loss: {l[-1]}')
    plt.plot(l)
    plt.show()
    l=[]
    model.ind=1
    Lambda=1
    epoches=5000
    for epoch in range(epoches_seasonality):
        l.append(0)
        for b in range(0,x_train.shape[0],batch_size):
            loss=((y_train[b:b+batch_size,:]-model(x_train[b:b+batch_size,:])[-1])**2).mean()+\
                Lambda*(torch.abs(model.seasonality.L1.weight).mean()+torch.abs(model.seasonality.L3.weight).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l[-1]+=loss.detach()
        print(f'epoch:{epoch}/{epoches_seasonality} loss: {l[-1]}')
    plt.plot(l)
    plt.show()

if __name__=='__main__':
    model=NeuralDecomposition()
    # n=1000
    # x_train,y_train=torch.linspace(0,1,n,requires_grad=True).view(-1,1),get_synthetic_data(n).requires_grad_(True).view(-1,1)
    PATH=...
    x_train,y_train=get_stock_data(PATH)
    inputs={'model':model,
            'lr':0.01,
            'epoches_trend':2000,
            'epoches_seasonality':5000,
            'batch_size':20,
            'x_train':x_train,
            'y_train':y_train}
    train_data(**inputs)

    plt.plot(y_train.detach())
    trend,seasonality,total=model(x_train)
    plt.plot(trend.detach())
    # plt.plot(seasonality.detach())
    plt.plot(total.detach())
    plt.legend(['original data','trend','final_data'])
    plt.show()
