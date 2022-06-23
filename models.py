import pdb
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="3"

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
import csv

BATCHSIZE = 512
EPOCH = 2

if os.uname()[1] == 'patatrack02.cern.ch':
    root = "/data2/user/phzehetn/Hackathon/Data/"
    save_dir = '/afs/cern.ch/work/p/phzehetn/public/TICL/Hackathon/'
    regex = "ntuples_7*"
    N_events = 1024
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


CUDA_VISIBLE_DEVICES=2,4

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




ls = []
losses = []
def train():
    print("Start Training")
    model.train()
    optimizer.zero_grad()
    for epoch in range(EPOCH):
        epoch_loss = []
        for data in train_loader:
            # pdb.set_trace()
            data = data.to(device)
            energy = model(data, data.x_batch)
            loss_value = loss(energy, data.y)
            loss_value.backward()
            epoch_loss.append(loss_value.item())
            optimizer.step()
        #print(f"Loss after {epoch+1} epochs", np.mean(epoch_loss))
        sdir = save_dir + f'{epoch}/'
        try:
            os.mkdir(sdir)
        except:
            print(f"{sdir} already exists")
        torch.save(model.state_dict(), sdir + 'model.pb')
        print("Epoch", epoch, "Loss", np.mean(epoch_loss), "Pred", energy.detach(), "GT", data.y)
        ls.append([epoch, np.mean(epoch_loss), energy.detach().cpu().numpy()[0], data.y.detach().cpu().numpy()[0]])


train()
print("DONE")

filename = 'b1_ev100_ep100.csv'
header = ["Epoch", "Mean Loss", "Pred", "Truth"]
with open(filename, 'w', newline="") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)
    for row in ls:
        csvwriter.writerow(row)


torch.save(model.state_dict(), save_dir+'model.pb')
