import torch
from collections import OrderedDict
class SineActivation(torch.nn.Module):
    def forward(self,x):
        return torch.sin(x)
class NeuralDecomposition(torch.nn.Module):
    def __init__(self,N=40,k=3):
        super().__init__()
        self.L1=torch.nn.Sequential(
            OrderedDict({'L1':torch.nn.Linear(1,k),
            'L2':torch.nn.Softplus(threshold=10000),
            'L3':torch.nn.Linear(k,1,bias=False)}))
        self.L2=torch.nn.Sequential(
            OrderedDict({'L1':torch.nn.Linear(1,k),
            'L2':torch.nn.Sigmoid(),
            'L3':torch.nn.Linear(k,1,bias=False)}))
        self.L3=torch.nn.Linear(1,1)
        self.trend=lambda x:self.L1(x)+self.L2(x)+self.L3(x)
        self.seasonality=torch.nn.Sequential(OrderedDict({'L1':torch.nn.Linear(1,N),
                                             'L2':SineActivation(),
                                             'L3':torch.nn.Linear(N,1,bias=False)}))
        for i in range(N):
            torch.nn.init.normal_(self.seasonality.L1.weight[i:i+1,:],(2*torch.pi*(i+1))//2,0.1)
        for i in range(N):
            torch.nn.init.normal_(self.seasonality.L1.bias[i:i+1],(torch.pi*(i+1))//2,0.1)
        for i in range(N):
            torch.nn.init.normal_(self.seasonality.L3.weight[i:i+1],0,0.5)
        self.ind=0
    def forward(self,x):
        if self.ind==0:
            trend=self.trend(x)
            return trend
        if self.ind==1:
            with torch.no_grad():
                trend=self.trend(x)
            seasonality=self.seasonality(x)
            return trend,seasonality,trend+seasonality