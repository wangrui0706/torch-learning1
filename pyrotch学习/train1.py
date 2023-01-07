import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Sequential
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from PIL import Image

spec_dir="20-20spec.xlsx"
shape_dir="20-20shape.xlsx"

# spec=pd.read_excel(spec_dir)
# shape=pd.read_excel(shape_dir)
#
# spec=np.array(spec,dtype=np.float32)
# shape=np.array(shape,dtype=np.float32)
#
# spec=spec[0]
# shape=shape[0]

class data(Dataset):
    def __init__(self,spec_path,shape_path):
       df1=pd.read_excel(spec_path).astype('float32')
       df2 = pd.read_excel(shape_path).astype('float32')
       self.n_sample=11
       self.spec=df1.values#[:,0:100]
       df2=df2.values
       df2=df2.reshape(11,400)
       self.shape=df2#.reshape((1,15,64))#[:,0:64]
    def __getitem__(self,index):
        #print(self.labels[index],self.features[index])
        return self.spec[index],self.shape[index]
    def __len__(self):
        return self.n_sample

wrdataset=data(spec_dir,shape_dir)
dataloader=DataLoader(wrdataset,15,shuffle=True)

class Meta(nn.Module):
    def __init__(self):
       super(Meta,self).__init__()
       self.module=Sequential(
        #nn.Conv2d(1,10,3),

        nn.Linear(100,200),
        nn.ReLU(),
        nn.Linear(200, 400),
        nn.ReLU(),
        nn.Linear(400, 128),
        nn.ReLU(),
        nn.Linear(128,400),
        nn.ReLU(),
       )
    def forward(self,x):
        x=self.module(x)
        return x

wr=Meta()
wr=wr.cuda()

loss_fn=nn.MSELoss()
loss_fn=loss_fn.cuda()

learning_rate=0.01
optimizer=torch.optim.SGD(wr.parameters(),lr=learning_rate)

epoch=200000
total_train_step=1
x=[]
y=[]
print("start------------------")
start_time=time.time()
for i in range(epoch):
    print("NO {}.次训练".format(total_train_step))
    x.append(total_train_step)
    total_train_step = total_train_step + 1
    for data in dataloader:
        spec,shape=data
        spec=spec.cuda()
        shape=shape.cuda()
        output=wr(spec)

        loss=loss_fn(output,shape)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if total_train_step/100==0:
    #if epoch %100==0:
    print("损失函数是{}".format(loss))
    y.append(loss.tolist())
    #print(spec)
    #print(shape)

torch.save(wr,"wr20-20-200000epoch.pth")

end_time=time.time()-start_time
print(end_time)
plt.plot(x,y)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()