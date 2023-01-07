import torch
import numpy as np
import pandas as pd
import torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt

start_time=time.time()

path="spec15.xlsx"
spec=pd.read_excel(path)
spec=torch.Tensor(spec.to_numpy()).cuda()
path1="shape15.xlsx"
shape=pd.read_excel(path1)
shape=torch.Tensor(shape.to_numpy()).cuda()
#print(spec1)

class Meta(nn.Module):
    def __init__(self):
       super(Meta,self).__init__()
       self.module=Sequential(

        nn.Linear(64,200),
        nn.ReLU(),
        nn.Linear(200, 400),
        nn.ReLU(),
        nn.Linear(400, 128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU()
       )
    def forward(self,x):
        x=self.module(x)
        return x
model=torch.load("wrMoudle")
with torch.no_grad():
    output=model(spec[14][0:64]).cuda()

output=output.tolist()
print(output)


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

output=standardization(output)
print(output)

n='\n'
a="1"
b="0"
t=0
maxmin=0

f=open("预测到的.txt",'w')
for i in output:
    if i>maxmin:
        f.write(a)
    else:
        f.write(b)
    t=t+1
    if t%8==0:
        f.write(n)
f.close()

x=[]
for i in output:
    x.append(i)

y=[]
for i in range(400,591,3):
    y.append(i)
end_time=time.time()-start_time
# print(end_time)
# plt.plot(y,output)
# plt.xlabel("wavelength(nm)")
# plt.xlim((0,700))
# plt.ylabel("abs(T)")
# plt.show()





