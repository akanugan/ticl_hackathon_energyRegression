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

class TracksterLoader(Dataset):


    def __init__(self, root, regex='*.root', N_events=0, transform=None):
        super(TracksterLoader, self).__init__(root, transform)
        self.strides = [0]
        print("strides: ", self.strides[-1])
        self.regex = regex
        self.N_events = N_events
        self.calc_offset()
        self.root = root
        
        

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

    def calculate_edges(self, df):
        cluster_indices = ['vertices_x', 'vertices_y', 'vertices_z', 'vertices_energy']
        starts = []
        stops = []
        N_tracksters = df.NTracksters.median().astype(int)
        N_clusters = []
        for t in range(N_tracksters):
            trackster = df.loc[t].reset_index()
            df = df[cluster_indices]
            z_values = df.loc[t]['vertices_z'].unique()
            for i, z in enumerate(z_values):
                layer = trackster.where(trackster.vertices_z == z_values[i]).dropna()['subsubentry'].values
                pre_layer = trackster.where(trackster.vertices_z == z_values[i-1]).dropna()['subsubentry'].values
                # print(f"layer {i}: layer: {layer}")
                # print(f"layer {i}: pre_layer: {pre_layer}")
                for l in layer:
                    for k in layer:
                        if l < k:
                        # Connect nodes from the same layer (l < k makes sure connections aren't added twice)
                        # Adding the sum of N_clusters assures that our graph is not connecting nodes belonging to different tracksters
                            starts.append(int(np.sum(N_clusters)) + l.astype(int))
                            stops.append(int(np.sum(N_clusters)) + k.astype(int))
                    for k in pre_layer:
                        # Connect nodes from the previous layer
                        # Adding the sum of N_clusters assures that our graph is not connecting nodes belonging to different tracksters
                        starts.append(int(np.sum(N_clusters)) + l.astype(int))
                        stops.append(int(np.sum(N_clusters)) + k.astype(int))
            # It's important to only append this AFTER creating the edges for this trackster otherwise it will screw up the indices of the edges
            N_clusters.append(len(trackster))
        return starts, stops


    def turn_df_to_graph(self, df):
        """
        df has to be a graph including a single event, it should have two indices, 
        the first one belonging to the trackster and the second one to the cluster
        """
        cluster_indices = ['vertices_x', 'vertices_y', 'vertices_z', 'vertices_energy']
        N_tracksters = df.NTracksters.median().astype(int)
        event_edges = torch.tensor(self.calculate_edges(df))
        X_ID = []
        # y = calo energy
        vertices = torch.tensor(df[cluster_indices].values, dtype=torch.float)
        stsReg =  torch.tensor(df['sim_tracksters_CP_regE'][0].values[0], dtype=torch.float)
        for i in range(N_tracksters):
            # Loop over tracksters to get the number of layer clusters
            n_layer_clusters = len(df.loc[i])
            X_ID.append(i * np.ones(n_layer_clusters))
        X_ID = torch.from_numpy(np.concatenate(X_ID))
        graph = torch_geometric.data.Data(x=vertices, edge_index=event_edges, x_id=X_ID, y=stsReg)
        return graph


    def get(self, idx):
        file_idx = np.searchsorted(self.strides, idx,side='right')-1
        evt_idx = idx - self.strides[max(0,file_idx)]

        print(file_idx, evt_idx)

        counter = 0
        with uproot.open(self.raw_paths[file_idx]) as f:
            tracksters = f["ntuplizer/tracksters"]
            sim_tracksters_CP = f["ntuplizer/simtrackstersCP"]
            sim_tracksters_CP_regE_df = ak.to_pandas(sim_tracksters_CP.arrays('stsCP_regressed_energy', entry_start = evt_idx, entry_stop = evt_idx + 1))
            event = ak.to_pandas(tracksters.arrays(entry_start = evt_idx, entry_stop = evt_idx + 1)) # Update me!!!! Takes half a second
            event['sim_tracksters_CP_regE'] = sim_tracksters_CP_regE_df
            event = event.loc[0] # Now only two indices are left
            data = self.turn_df_to_graph(event)
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
        regex = "ntuples_7*"
        N_events = 10000
    elif os.uname()[1] == 'x360':
        print("Running on x360")
        root = "/home/philipp/Code/ticl_hackathon_energy_regression/testdata/"
        regex = 'testdata*'
        N_events = 3
    else:
        print("Please specify root path")

    dataset = TracksterLoader(root, regex=regex, N_events=N_events)
    print(dataset.len())

    train_loader = DataLoader(dataset)
    for i, data in enumerate(train_loader):
        #if i > 11: break
        '''
        print(f"Event {i}")
        print(f"Data: {data}")
        print()
        print(f"Data[0].x: {data[0].x}")
        print(f"Data[0].edge_index: {data[0].edge_index}")
        print(f"Truth: ", data[0].y)
        '''