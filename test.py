import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math
import random
from torch.autograd import Variable
import mna
from copy import deepcopy

class MyNet(nn.Module):
    
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


dtrain=[random.random()*20 for cnt in range(1000)]
ddes = [math.sin(x) for x in dtrain]
ddes=[[x] for x in ddes]
dtrain=[[x] for x in dtrain]
dtest=[random.random()*20 for cnt in range(100)]
dtdes = [math.sin(x) for x in dtest]
dtest=[[x] for x in dtest]
dtdes=[[x] for x in dtdes]

#  model
net = MyNet(1, 5, 10, 1)
# loss function
criterion = nn.MSELoss()
# optimizer
optimizer = mna.MNA(net.parameters(), initTemp=1, schedule=0.99,window=2,terminateTemp=0.00001)


num_epochs =500
train_loss = []
test_loss = []
train_accuracy = []
terr = []
avterr = []
loss=torch.Tensor(1)

for epoch in range(num_epochs):
    train_correct = 0
    items=    Variable(torch.from_numpy(np.asarray(dtrain,dtype=np.float32 )))
    classes = Variable(torch.from_numpy(np.asarray(ddes ,dtype=np.float32  )))
    train_total = classes.size(0)
    #
    net.train()
    optimizer.zero_grad()
    outputs = net(items)
   
    loss = criterion(outputs, classes)
    loss.backward()
    def closure():
        optimizer.zero_grad()
        outputs = net(items)
        loss = criterion(outputs, classes)
        loss.backward()
        return loss
    for group in optimizer.param_groups:
        for p in group['params']:
            print('before= ', p.grad)

    optimizer.step(closure)

    #
    train_correct = (torch.abs(outputs.data - classes.data)).sum()
    #*******************************************************************************************
    net.eval()                  
    train_loss.append(loss.item())
    train_accuracy.append(( train_correct / train_total))

    #
    test_items = Variable(torch.from_numpy(np.asarray(dtest,dtype=np.float32 )))
    test_classes = Variable(torch.from_numpy(np.asarray(dtdes,dtype=np.float32)))
    total = test_classes.size(0)
    outputs = net(test_items)
    loss = criterion(outputs, test_classes)
    test_loss.append(loss.data.item())
        
    terr = (torch.abs(outputs.data - test_classes)).sum()
    avterr.append(( terr / total))
    if epoch == num_epochs-1:
        print('finished')
