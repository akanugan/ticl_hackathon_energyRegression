import pdb
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
import os.path as osp
from torch_geometric.data import Dataset, download_url, DataLoader, Data
import torch_geometric

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import h5py

import glob
import os.path as osp

import time
from ordered_set import OrderedSet


class TracksterLoader(Dataset):


    def __init__(self, root, regex='*.root', N_events=0, transform=None):
        super(TracksterLoader, self).__init__(root, transform)
        self.strides = [0]
        print("strides: ", self.strides[-1])
        self.regex = regex
        self.N_events = N_events
        self.calc_offset()
        self.root = root
        self.file_count = -1
        
        

    # Maybe should be called __len__
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    def calc_offset(self):
        events = 0
        for file in self.raw_file_names:
            with uproot.open(file) as f:
                events+=f["ntuplizer/tracksters"].num_entries
                if (self.N_events !=0 and events < self.N_events):
                    self.strides.append(events)
                else:
                    self.strides.append(self.N_events)
                    break
        print(self.strides)

    def len(self):
        return self.strides[-1]

    @property
    def raw_file_names(self):
        raw_files = glob.glob(osp.join(self.root, self.regex))
        return raw_files

    @property
    def processed_file_names(self):
        return []

    def calculate_edges(self, list_z):

        edges = []
        starts = []
        stops = [] 

        N_clusters = []
        for i, trackster in enumerate(list_z):
            #print(f"trackster {i}")
            z_values = OrderedSet(trackster)
            nz = len(z_values)
            for i,z in enumerate(z_values):
                layer = [x for x, e in enumerate(trackster) if e == z_values[i]]
                pre_layer = [x for x, e in enumerate(trackster) if e == z_values[i-1]]
                    
                for l in layer:
                    for k in layer:
                        if l < k:
                        # Connect nodes from the same layer (l < k makes sure connections aren't added twice)
                        # Adding the sum of N_clusters assures that our graph is not connecting nodes belonging to different tracksters
                            starts.append(int(np.sum(N_clusters)) + l)
                            stops.append(int(np.sum(N_clusters)) + k)
                    for k in pre_layer:
                        # Connect nodes from the previous layer
                        # Adding the sum of N_clusters assures that our graph is not connecting nodes belonging to different tracksters
                        starts.append(int(np.sum(N_clusters)) + l)
                        stops.append(int(np.sum(N_clusters)) + k)
            # It's important to only append this AFTER creating the edges for this trackster otherwise it will screw up the indices of the edges
            N_clusters.append(len(trackster))
        return starts, stops


    def turn_df_to_graph(self, x, y):
        """
        df has to be a graph including a single event, it should have two indices, 
        the first one belonging to the trackster and the second one to the cluster
        """

        ls = []

        # in x we store [list_x, list_y, list_z, list_e] with list_x = [[lc_vertices_x_0, lc_vertices_x_1, ...],[...],...]
        for i in range(len(x[0])):
            for j in range(len(x[0][i])):
                ls.append([x[0][i][j], x[1][i][j], x[2][i][j], x[3][i][j]])
                
        vertices = torch.tensor(ls, dtype=torch.float)
        stsReg = torch.tensor(y, dtype=torch.float)

        # Calculate edges using vertex_z positions
        event_edges = torch.tensor(self.calculate_edges(x[2]), dtype=torch.long)

        graph = torch_geometric.data.Data(x=vertices, edge_index=event_edges, y=stsReg)
        return graph


    def get(self, idx):
        file_idx = np.searchsorted(self.strides, idx,side='right')-1
        evt_idx = idx - self.strides[max(0,file_idx)]

        #print(file_idx, evt_idx)

        if (self.file_count != file_idx):
            with uproot.open(self.raw_paths[file_idx]) as f:
                trackster = f['ntuplizer/tracksters;4']
                sim_tracksters_CP = f["ntuplizer/simtrackstersCP"]

                sim_tracksters_CP_regE_df = sim_tracksters_CP.arrays('stsCP_regressed_energy', library='np')
                self.x = trackster.arrays(["event","vertices_x", "vertices_y","vertices_z","vertices_energy"],library='np')
                self.y = (sim_tracksters_CP_regE_df)
            self.file_count = file_idx

        
        #idx = self.x['event'][evt_idx]
        list_x = self.x['vertices_x'][evt_idx].tolist()
        list_y = self.x['vertices_y'][evt_idx].tolist()
        list_z = self.x['vertices_z'][evt_idx].tolist()
        list_e = self.x['vertices_energy'][evt_idx].tolist()
        y = self.y['stsCP_regressed_energy'][evt_idx]

        x = [list_x, list_y, list_z, list_e]
        if list_x ==[]:
            y = []
        data = self.turn_df_to_graph(x, y)
        return data

        '''

        x_jet = np.squeeze(f['x0'][idx_in_file])
        Ntrack = int(x_jet[0])
        if Ntrack > 0:
            x_track = f['x1'][idx_in_file,:Ntrack,:]
        else:
            Ntrack = 1
            x_track = np.zeros((1,8), dtype=np.float32)
        
        Nsv = int(x_jet[1])
        if Nsv > 0:
            x_sv = f['x2'][idx_in_file,:Nsv,:]
        else:
            Nsv = 1
            x_sv = np.zeros((1,2), dtype=np.float32)

        # convert to torch
        x_jet = torch.from_numpy(x_jet[2:])[None]
        x_track = torch.from_numpy(x_track)
        x_sv = torch.from_numpy(x_sv)
        
        
        # convert to non-onehot categories
        y = torch.from_numpy(f['y0'][idx_in_file])
        y = torch.argmax(y)
        
        # "z0" is the basic jet observables pt, eta, phi
        # store this as the usual x
        x = torch.from_numpy(f['z0'][idx_in_file])
        x_jet_raw = torch.from_numpy(f['z1'][idx_in_file])

        return Data(x=x, edge_index=edge_index, y=y,
                    x_jet=x_jet, x_track=x_track, x_sv=x_sv,
                    x_jet_raw=x_jet_raw)

        '''


if __name__ == '__main__':
    import os


    if os.uname()[1] == 'patatrack02.cern.ch':
        root = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/hackathon/samples/close_by_single_kaon/production/7264977/"
        #regex = "ntuples_7*"
        regex = "ntuples_7087242_2*"
        N_events = 10000
    elif os.uname()[1] == 'x360':
        print("Running on x360")
        root = "/home/philipp/Code/ticl_hackathon_energy_regression/testdata/"
        regex = 'testdata*'
        N_events = 3
    else:
        #root = "/Users/ankush/Documents/TICL-hack/ticl_hackathon_energy_regression/"
        root = "/cms/data/akanugan/ana_cms/TICL_hackathon/ticl_hackathon_energyRegression/"
        regex = 'testdata*'
        N_events = 3000
        print("Please specify root path")

    dataset = TracksterLoader(root, regex=regex, N_events=N_events)
    print(dataset.len())

    train_loader = DataLoader(dataset)
    for i, data in enumerate(train_loader):
        if i > 11: break
        print(f"Event {i}")
        print(f"Data: {data}")
        print()
        #print(f"Data[0].x: {data[0].x}")
        #print(f"Data[0].edge_index: {data[0].edge_index}")
        #print(f"Truth: ", data[0].y)
