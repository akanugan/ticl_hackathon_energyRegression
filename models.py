import pdb
import os
import argparse

import sklearn
import numpy as np
from random import randrange
import subprocess

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import EdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataLoader
import torch_geometric

from DataLoader import TracksterLoader

BATCHSIZE = 8

if os.uname()[1] == 'patatrack02.cern.ch':
    root = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/hackathon/samples/close_by_single_kaon/production/7264977/"
    regex = "ntuples_7*"
    N_events = 10000
elif os.uname()[1] == 'x360':
    print("Running on x360")
    root = "/home/philipp/Code/ticl_hackathon_energy_regression/testdata/"
    regex = 'testdata*'
    N_events = 40
else:
    print("Please specify root path")


class TestNet(nn.Module):
    def __init__(self, n_features=4):
        super(TestNet, self).__init__()

        self.conv1 = EdgeConv(
            nn=torch.nn.Linear(2 * n_features, n_features),
            aggr='add')

        self.conv2 = EdgeConv(
            nn=torch.nn.Linear(2 * n_features, 10),
            aggr='add', )
        # self.conv1 = torch_geometric.nn.GCNConv(4, 16)
        # self.conv2 = torch_geometric.nn.GCNConv(16, 10)

        self.output = torch.nn.Linear(10, 1)
    
    def forward(self, data, batch_x):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.output(x)
        
        # energy, batch = torch_geometric.nn.avg_pool_x(batch_x, x, batch_x)
        energy = torch_geometric.graphgym.models.pooling.global_add_pool(x, batch_x)
        # print(x.shape[0])
        # energy = x.shape[0] * energy
        return energy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = TestNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


train_loader = DataLoader(
    TracksterLoader(root, regex=regex, N_events=N_events),
    follow_batch=['x'],
    batch_size=BATCHSIZE, shuffle=False)


losses = []
def train():
    print("Start Training")
    model.train()
    optimizer.zero_grad()
    for epoch in range(5):
        epoch_loss = []
        for data in train_loader:
            # pdb.set_trace()
            data = data.to(device)
            energy = model(data, data.x_batch)
            loss_value = loss(energy, data.y)
            loss_value.backward()
            epoch_loss.append(loss_value.item())
            optimizer.step()
        print(f"Loss after {epoch+1} epochs", np.mean(epoch_loss))


train()
print("DONE")