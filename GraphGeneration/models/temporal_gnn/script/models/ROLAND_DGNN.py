"""
PyTorch implementation of ROLAND

reference: https://github.com/manuel-dileo/dynamic-gnn
"""


import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score,average_precision_score

import copy


from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit,NormalizeFeatures, Constant
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, Linear

import torch
import networkx as nx
import numpy as np

import json



class ROLANDGNN(torch.nn.Module):
    def __init__(self, input_channel,nhid, 
                 num_nodes, dropout=0.0, update='learnable'):
        
        super(ROLANDGNN, self).__init__()
        
        self.conv1 = GCNConv(input_channel, nhid)
        self.conv2 = GCNConv(nhid, nhid)  

        self.dropout = dropout
        self.update = update
        if update=='moving':
            self.tau = torch.Tensor([0])
        elif update=='learnable':
            self.tau = torch.nn.Parameter(torch.Tensor([0]))
        elif update=='gru':
            self.gru1 = GRUCell(nhid, nhid)
            self.gru2 = GRUCell(nhid, nhid)
        elif update=='mlp':
            self.mlp1 = Linear(nhid * 2, nhid)
            self.mlp2 = Linear(nhid * 2, nhid)
        else:
            assert(update>=0 and update <=1)
            self.tau = torch.Tensor([update])
        self.previous_embeddings = [torch.Tensor([[0 for i in range(nhid)] for j in range(num_nodes)]),\
                                    torch.Tensor([[0 for i in range(nhid)] for j in range(num_nodes)])]
                                    
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, previous_embeddings=None, num_current_edges=None, num_previous_edges=None):
        
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None: #None if test
            self.previous_embeddings = [previous_embeddings[0].clone(), previous_embeddings[1].clone()]
        if self.update=='moving' and num_current_edges is not None and num_previous_edges is not None: #None if test
            #compute moving average parameter
            self.tau = torch.Tensor([num_previous_edges / (num_previous_edges + num_current_edges)]).clone() # tau -- past weight
            
        
        current_embeddings = [torch.Tensor([]), torch.Tensor([])]
        
        #GRAPHCONV
        #GraphConv1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        #Embedding Update after first layer
        if self.update=='gru':
            h = torch.Tensor(self.gru1(h, self.previous_embeddings[0].clone().to(h.device)).detach())  # .numpy()
        elif self.update=='mlp':
            hin = torch.cat((h, self.previous_embeddings[0].clone().to(h.device)), dim=1)
            h = torch.Tensor(self.mlp1(hin).detach()) # .numpy()
        else:
            self.tau.to(x.device)
            # print(self.tau.device)
            h = torch.Tensor((self.tau * self.previous_embeddings[0].clone() + (1-self.tau) * h.clone()).detach())  # .numpy()
       
        current_embeddings[0] = h.clone()
        
        #GraphConv2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        #Embedding Update after second layer
        if self.update=='gru':
            h = torch.Tensor(self.gru2(h, self.previous_embeddings[1].clone().to(h.device)).detach()) # .numpy()
        elif self.update=='mlp':
            hin = torch.cat((h,self.previous_embeddings[1].clone().to(h.device)),dim=1)
            h = torch.Tensor(self.mlp2(hin).detach())  # .numpy()
        else:
            h = torch.Tensor((self.tau * self.previous_embeddings[1].clone() + (1-self.tau) * h.clone()).detach())  # .numpy()
        current_embeddings[1] = h.clone()
    
        # NOTE: last GCNConv layer is considered as the embeddings
        return current_embeddings