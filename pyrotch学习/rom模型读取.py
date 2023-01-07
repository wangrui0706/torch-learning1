import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Sequential

spec_dir="20-20spec.xlsx"

spec=pd.read_excel(spec_dir)

spec=np.array(spec,dtype=np.float32)

shape_dir="20-20shape.xlsx"

class Meta(nn.Module):
    def __init__(self):
       super(Meta,self).__init__()
       self.module=Sequential(
        #nn.Conv2d(1,10,3),

        nn.Linear(100,200),

        nn.Linear(200, 400),

        nn.Linear(400, 128),

        nn.Linear(128,400),

       )
    def forward(self,x):
        x=self.module(x)
        return x

model=torch.load("wr20-20-200000epoch.pth",map_location=torch.device('cuda'))
#print(model)

i=0
spec=spec[i]
#print(spec)
spec=torch.from_numpy(spec).cuda()
output=model(spec).cuda()
#output=output.cuda()

x=[]
t=1
a='\n'

yes='1'
no='0'

f=open("wr模型预测形状.txt",'w')
for i in output:
    # x.append(a)
    x.append(i.tolist())
    # f.write(str(i.tolist()))
    if i > 0:
        f.write(yes)
    else:
        f.write(no)
    t = t + 1
    if t % 20 ==1:
        f.write(a)
f.close()

print(x)
