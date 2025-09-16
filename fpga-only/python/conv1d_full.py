import numpy as np
import torch
import math
import scipy.io
import glob
import os
import sys
import random
from datetime import datetime
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from dataset import get_wine_spoilage, set_seeds

def init_weights(shape):
    _weight = torch.empty(shape)
    torch.nn.init.xavier_normal_(_weight)
    _weight = _weight.detach().numpy().astype(np.double)
    return _weight

dataset = get_wine_spoilage()

np.random.seed(seed=0)
torch.manual_seed(0)

signal_length = 100
channel_num = 6

class DummyNet(torch.nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()

        signal_length = 100
        channel_num = 6

        self.conv_1d = torch.nn.Conv1d(in_channels=4, out_channels=32, kernel_size=2, bias=True, stride=1,
                                       padding_mode='zeros', padding='same')
        self.conv_1d_w = init_weights((32, channel_num, 2))
        self.conv_1d_b = np.zeros((32)).astype(np.double)

        self.conv_1d.weight = torch.nn.Parameter(torch.from_numpy(self.conv_1d_w))
        self.conv_1d.bias = torch.nn.Parameter(torch.from_numpy(self.conv_1d_b))
        self.conv_1d.bias.retain_grad()
        self.relu1 = torch.nn.ReLU()

        self.maxpool_1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        ##conv 2
        self.conv_1d_2 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, bias=True, stride=1, padding_mode='zeros', padding='same')
        self.conv_1d_2_w = init_weights((16, 32, 2))
        self.conv_1d_2_b = np.zeros((16)).astype(np.double)
        self.relu2 = torch.nn.ReLU()

        self.conv_1d_2.weight = torch.nn.Parameter(torch.from_numpy(self.conv_1d_2_w))
        self.conv_1d_2.bias = torch.nn.Parameter(torch.from_numpy(self.conv_1d_2_b))

        self.maxpool_2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        ##conv 3
        self.conv_1d_3 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, bias=True, stride=1, padding_mode='zeros', padding='same')
        self.conv_1d_3_w = init_weights((16, 16, 2))
        self.conv_1d_3_b = np.zeros((16)).astype(np.double)
        self.relu3 = torch.nn.ReLU()

        self.conv_1d_3.weight = torch.nn.Parameter(torch.from_numpy(self.conv_1d_3_w))
        self.conv_1d_3.bias = torch.nn.Parameter(torch.from_numpy(self.conv_1d_3_b))

        ##conv 4
        self.conv_1d_4 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, bias=True, stride=1, padding_mode='zeros', padding='same')
        self.conv_1d_4_w = init_weights((16, 16, 2))
        self.conv_1d_4_b = np.zeros((16)).astype(np.double)
        self.relu4 = torch.nn.ReLU()

        self.conv_1d_4.weight = torch.nn.Parameter(torch.from_numpy(self.conv_1d_4_w))
        self.conv_1d_4.bias = torch.nn.Parameter(torch.from_numpy(self.conv_1d_4_b))

        ##conv 5
        self.conv_1d_5 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, bias=True, stride=1, padding_mode='zeros', padding='same')
        self.conv_1d_5_w = init_weights((16, 16, 2))
        self.conv_1d_5_b = np.zeros((16)).astype(np.double)
        self.relu5 = torch.nn.ReLU()

        self.conv_1d_5.weight = torch.nn.Parameter(torch.from_numpy(self.conv_1d_5_w))
        self.conv_1d_5.bias = torch.nn.Parameter(torch.from_numpy(self.conv_1d_5_b))

        #######

        self.dense = torch.nn.Linear(in_features=400, out_features=3, bias=True)
        self.d1_w = init_weights((3, 400))
        self.d1_b = np.zeros((3)).astype(np.double)

        self.dense.weight = torch.nn.Parameter(torch.from_numpy(self.d1_w))
        self.dense.bias = torch.nn.Parameter(torch.from_numpy(self.d1_b))

    def forward(self, x):
        self.l1 = self.conv_1d(x)
        self.l1.retain_grad()
        self.l1_relu = self.relu1(self.l1)
        self.l1_relu.retain_grad()

        self.mp_1 = self.maxpool_1(self.l1_relu)
        self.mp_1.retain_grad()

        self.l2 = self.conv_1d_2(self.mp_1)
        self.l2.retain_grad()
        self.l2_relu = self.relu2(self.l2)
        self.l2_relu.retain_grad()

        self.mp_2 = self.maxpool_2(self.l2_relu)
        self.mp_2.retain_grad()

        self.l3 = self.conv_1d_3(self.mp_2)
        self.l3_relu = self.relu3(self.l3)
        self.l3.retain_grad()
        self.l3_relu.retain_grad()

        self.l4 = self.conv_1d_4(self.l3_relu)
        self.l4_relu = self.relu4(self.l4)
        self.l4.retain_grad()
        self.l4_relu.retain_grad()

        self.l5 = self.conv_1d_5(self.l4_relu)
        self.l5_relu = self.relu5(self.l5)
        self.l5.retain_grad()
        self.l5_relu.retain_grad()

        self.mp_flatten = torch.flatten(self.l5_relu)

        self.out = self.dense(self.mp_flatten)
        self.out.retain_grad()

        return self.out


fd = open("train_samples_wine.txt", 'r')
lines = fd.readlines()

X_train_gas = list()
Y_train_gas = list()
for l in range(0, len(lines)-1, 2):
    label = np.asarray(lines[l].split()).astype(float)
    data_record = np.asarray(lines[l+1].split()).astype(np.double).reshape((signal_length, channel_num))
    X_train_gas.append(data_record)
    Y_train_gas.append(label)

X_train_gas = np.asarray(X_train_gas)
Y_train_gas = np.asarray(Y_train_gas)

print(X_train_gas.shape, Y_train_gas.shape)

net = DummyNet()
loss = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0, weight_decay=0, dampening=0, nesterov=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

softm = torch.nn.Softmax(dim=0)

np.set_printoptions(precision=8)

for i in range(0, X_train_gas.shape[0]):

    x = np.swapaxes(X_train_gas[i], 1, 0)
    y = np.reshape(Y_train_gas[i], (Y_train_gas.shape[1]))

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    x_tensor.requires_grad = True

    optimizer.zero_grad()

    out_net = net(x_tensor)

    output = loss(out_net, y_tensor)

    output.retain_grad()

    output.backward()

    np.set_printoptions(precision=7)
    print("GT loss", output)

    optimizer.step()


