import pandas as pd
from datetime import datetime
import torch
def get_stock_data(PATH):
    data=pd.read_csv(PATH)
    data.columns=list(map(lambda a:a.lower().lstrip().rstrip(),data.columns))
    data.date=data.date.apply(lambda a:datetime.strptime(a,'%d-%b-%Y'))
    data['t']=(data.date-datetime.strptime('13-Nov-2024','%d-%b-%Y')).apply(lambda a:float(str(a).split()[0]))
    data=data[['t','close']].to_numpy()
    x_train,y_train=torch.tensor(data[:,0][::-1].copy(),dtype=torch.float32).view(-1,1),torch.tensor(data[:,1][::-1].copy(),dtype=torch.float32).view(-1,1)
    x_train=(x_train-x_train.min())/(x_train.max()-x_train.min())
    y_train=10*(y_train-y_train.min())/(y_train.max()-y_train.min())
    x_train=x_train.requires_grad_(True)
    y_train=y_train.requires_grad_(True)
    return x_train,y_train