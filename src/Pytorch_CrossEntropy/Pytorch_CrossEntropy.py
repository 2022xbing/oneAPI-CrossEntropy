#!/usr/bin/env python
# encoding: utf-8

'''
==============================================================
 Copyright © 2019 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data

# 训练集
minst_train = datasets.MNIST(
    root='../data_mnist', train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor()]))
# 测试集
minst_test = datasets.MNIST(
    root='../data_mnist', train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor()]))


class Net(nn.Module):
    def __init__(self, in_dim, n_hidden1, n_hidden2, out_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


model = Net(784, 256, 256, 10)
batch_size = 256
train_loader = data.DataLoader(minst_train, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(minst_test, batch_size=batch_size, shuffle=False)

lr = 0.1
num_epoches = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

eval_accs = []
eval_losses = []


def train():
    eval_loss = 0
    eval_acc = 0
    for data in train_loader:
        img, label = data
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        eval_acc += pred.eq(label.view_as(pred)).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'loss: {eval_loss / len(minst_train) : f}, acc: {eval_acc / len(minst_train) : f}')
    eval_losses.append(eval_loss / len(minst_train))
    eval_accs.append(eval_acc / len(minst_train))


def test():
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        eval_acc += pred.eq(label.view_as(pred)).sum().item()
    print(f'在测试集上的loss: {eval_loss / len(minst_test) : .4f}, acc: {eval_acc / len(minst_test) : .4f}')


for epoch in range(num_epoches):
    print(f'epoch: {epoch + 1}')
    train()

test()
