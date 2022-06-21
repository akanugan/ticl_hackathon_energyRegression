import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
import os.path as osp
from torch_geometric.data import Dataset, download_url, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

import glob
import os.path as osp

class GraphNet(Dataset):
    r'''
        input z0: (FatJat basic stats, 1)
            'fj_pt',
            'fj_eta',
            'fj_sdmass',
            'fj_n_sdsubjets',
            'fj_doubleb',
            'fj_tau21',
            'fj_tau32',
            'npv',
            'npfcands',
            'ntracks',
            'nsv'
        input x0: (FatJet info, 1) (also input z1???)
        'fj_jetNTracks',
        'fj_nSV',
        'fj_tau0_trackEtaRel_0',
        'fj_tau0_trackEtaRel_1',
        'fj_tau0_trackEtaRel_2',
        'fj_tau1_trackEtaRel_0',
        'fj_tau1_trackEtaRel_1',
        'fj_tau1_trackEtaRel_2',
        'fj_tau_flightDistance2dSig_0',
        'fj_tau_flightDistance2dSig_1',
        'fj_tau_vertexDeltaR_0',
        'fj_tau_vertexEnergyRatio_0',
        'fj_tau_vertexEnergyRatio_1',
        'fj_tau_vertexMass_0',
        'fj_tau_vertexMass_1',
        'fj_trackSip2dSigAboveBottom_0',
        'fj_trackSip2dSigAboveBottom_1',
        'fj_trackSip2dSigAboveCharm_0',
        'fj_trackSipdSig_0',
        'fj_trackSipdSig_0_0',
        'fj_trackSipdSig_0_1',
        'fj_trackSipdSig_1',
        'fj_trackSipdSig_1_0',
        'fj_trackSipdSig_1_1',
        'fj_trackSipdSig_2',
        'fj_trackSipdSig_3',
        'fj_z_ratio'
        
        input x1: (Tracks, max 60)
        'trackBTag_EtaRel',
        'trackBTag_PtRatio',
        'trackBTag_PParRatio',
        'trackBTag_Sip2dVal',
        'trackBTag_Sip2dSig',
        'trackBTag_Sip3dVal',
        'trackBTag_Sip3dSig',
        'trackBTag_JetDistVal'
        
        input x2: (SVs, max 5)
        'sv_d3d',
        'sv_d3dsig',
        
        # truth categories are QCD=0 / Hbb=1
    '''

    url = '/eos/cms/store/group/dpg_hgcal/comm_hgcal/hackathon/samples/close_by_single_kaon/production/7264977/'

    def __init__(self, root, transform=None):
        super(GraphNet, self).__init__(root, transform)
        
        self.strides = [0]
        # self.calculate_offsets()
        self.root = root

    def calculate_offsets(self):
        for path in self.raw_paths:
            with h5py.File(path, 'r') as f:
                self.strides.append(f['n'][()][0])
        self.strides = np.cumsum(self.strides)

    def len(self):
        return 5000 * len(root) # is this uniform over all files?

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*ntuple')))
        return raw_files

    @property
    def processed_file_names(self):
        return []


    def get(self, idx):
        file_idx = int((idx % 5000)/5000)
        idx_in_file = idx % 5000
        #if file_idx >= self.strides.size:
        #    raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        edge_index = torch.empty((2,0), dtype=torch.long)
        print(self.raw_files)

        with uproot.open(self.raw_paths[file_idx]) as f:
            tracksters = f["ntuplizer/tracksters"]
            tmpdf = ak.to_pandas(tracksters.arrays(entry_start = idx_in_file, entry_stop = idx_in_file + 1))


            print(self.raw_paths[file_idx])
            return tmpdf

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



import os

root = "/" # close_by_single_kaon/production/7264977/

print(os.listdir(root))

dataset = GraphNet(root)

print(dataset.len())

train_loader = DataLoader(dataset)
for data in train_loader:
    print(data)


