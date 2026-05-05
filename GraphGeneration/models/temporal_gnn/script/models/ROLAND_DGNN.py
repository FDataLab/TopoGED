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


# The `ROLANDGNN` class defines a graph neural network model with different update mechanisms (moving
# average, learnable, GRU, MLP) for node embeddings.

class ROLANDGNN(torch.nn.Module):
    def __init__(self, input_channel, nhid, num_nodes, dropout=0.0, update='mlp'):
        super(ROLANDGNN, self).__init__()
        self.conv1 = GCNConv(input_channel, nhid)
        self.conv2 = GCNConv(nhid, nhid)  
        self.dropout = dropout
        self.update = update
        
        if update == 'moving':
            # Register as buffer so it moves to GPU with the model
            self.register_buffer('tau', torch.tensor([0.5]))
        elif update == 'learnable':
            self.tau = torch.nn.Parameter(torch.tensor([0.5]))
        elif update == 'gru':
            self.gru1 = GRUCell(nhid, nhid)
            self.gru2 = GRUCell(nhid, nhid)
        elif update == 'mlp':
            self.mlp1 = Linear(nhid * 2, nhid)
            self.mlp2 = Linear(nhid * 2, nhid)
        
        # Initial dummy state
        self.previous_embeddings = [
            torch.zeros(num_nodes, nhid),
            torch.zeros(num_nodes, nhid)
        ]

    def forward(self, x, edge_index, previous_embeddings=None, num_current_edges=None, num_previous_edges=None):
        # 1. Update state reference (don't detach here!)
        if previous_embeddings is not None:
            self.previous_embeddings = previous_embeddings
        
        # 2. Moving average logic (if applicable)
        if self.update == 'moving' and num_current_edges is not None and num_previous_edges is not None:
            self.tau = torch.tensor([num_previous_edges / (num_previous_edges + num_current_edges)], device=x.device)
            
        # --- LAYER 1 ---
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        prev0 = self.previous_embeddings[0].to(h.device)
        if self.update == 'gru':
            h = self.gru1(h, prev0) # REMOVED .detach() and torch.Tensor()
        elif self.update == 'mlp':
            h = self.mlp1(torch.cat((h, prev0), dim=1))
        else:
            h = self.tau.to(h.device) * prev0 + (1 - self.tau.to(h.device)) * h
       
        current_emb0 = h
        
        # --- LAYER 2 ---
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        prev1 = self.previous_embeddings[1].to(h.device)
        if self.update == 'gru':
            h = self.gru2(h, prev1) # REMOVED .detach() and torch.Tensor()
        elif self.update == 'mlp':
            h = self.mlp2(torch.cat((h, prev1), dim=1))
        else:
            h = self.tau.to(h.device) * prev1 + (1 - self.tau.to(h.device)) * h
            
        return [current_emb0, h]



# # NEW ROLAND TESTING FROM 
# # https://github.com/manuel-dileo/dynamic-gnn/blob/main/BitcoinOTC-Example.ipynb
# class ROLANDGNN(torch.nn.Module):
#     def __init__(self, input_channel, nhid, num_nodes, dropout=0.0, update='moving', loss=BCEWithLogitsLoss):
        
#         super(ROLANDGNN, self).__init__()
#         #Architecture: 
#             #2 MLP layers to preprocess BERT repr, 
#             #2 GCN layer to aggregate node embeddings
#             #HadamardMLP as link prediction decoder
        
#         #You can change the layer dimensions but 
#         #if you change the architecture you need to change the forward method too
#         #TODO: make the architecture parameterizable
        
#         hidden_conv_1 = nhid
#         hidden_conv_2 = nhid // 2
#         self.preprocess1 = Linear(input_channel, 256)
#         self.preprocess2 = Linear(256, 128)
#         self.conv1 = GCNConv(128, hidden_conv_1)
#         self.conv2 = GCNConv(hidden_conv_1, hidden_conv_2)
#         self.postprocess1 = Linear(hidden_conv_2, 2)
        
#         #Initialize the loss function to BCEWithLogitsLoss
#         self.loss_fn = loss()

#         self.dropout = dropout
#         self.update = update
#         if update=='moving':
#             self.tau = torch.Tensor([0])
#         elif update=='learnable':
#             self.tau = torch.nn.Parameter(torch.Tensor([0]))
#         elif update=='gru':
#             self.gru1 = GRUCell(hidden_conv_1, hidden_conv_1)
#             self.gru2 = GRUCell(hidden_conv_2, hidden_conv_2)
#         elif update=='mlp':
#             self.mlp1 = Linear(hidden_conv_1*2, hidden_conv_1)
#             self.mlp2 = Linear(hidden_conv_2*2, hidden_conv_2)
#         else:
#             assert(update>=0 and update <=1)
#             self.tau = torch.Tensor([update])
#         self.previous_embeddings = [torch.Tensor([[0 for i in range(hidden_conv_1)] for j in range(num_nodes)]),\
#                                     torch.Tensor([[0 for i in range(hidden_conv_2)] for j in range(num_nodes)])]
                                    
        
#     def reset_loss(self,loss=BCEWithLogitsLoss):
#         self.loss_fn = loss()
        
#     def reset_parameters(self):
#         self.preprocess1.reset_parameters()
#         self.preprocess2.reset_parameters()
#         self.conv1.reset_parameters()
#         self.conv2.reset_parameters()
#         self.postprocess1.reset_parameters()

#     def forward(self, x, edge_index, edge_label_index, previous_embeddings=None, num_current_edges=None, num_previous_edges=None):
        
#         #You do not need all the parameters to be different to None in test phase
#         #You can just use the saved previous embeddings and tau
#         if previous_embeddings is not None: #None if test
#             self.previous_embeddings = [previous_embeddings[0].clone(),previous_embeddings[1].clone()]
#         if self.update=='moving' and num_current_edges is not None and num_previous_edges is not None: #None if test
#             #compute moving average parameter
#             self.tau = torch.Tensor([num_previous_edges / (num_previous_edges + num_current_edges)]).clone() # tau -- past weight
        
#         current_embeddings = [torch.Tensor([]),torch.Tensor([])]
        
#         #Preprocess text
#         h = self.preprocess1(x)
#         h = F.leaky_relu(h,inplace=False)
#         h = F.dropout(h, p=self.dropout,inplace=False)
#         h = self.preprocess2(h)
#         h = F.leaky_relu(h,inplace=False)
#         h = F.dropout(h, p=self.dropout, inplace=False)
        
#         #GRAPHCONV
#         #GraphConv1
#         h = self.conv1(h, edge_index)
#         h = F.leaky_relu(h,inplace=False)
#         h = F.dropout(h, p=self.dropout,inplace=False)
#         #Embedding Update after first layer
#         if self.update == 'gru':
#             h = self.gru1(h, self.previous_embeddings[0].clone())
#         elif self.update == 'mlp':
#             hin = torch.cat((h, self.previous_embeddings[0].clone()), dim=1)
#             h = self.mlp1(hin)
#         else:
#             h = self.tau * self.previous_embeddings[0].clone() + (1 - self.tau) * h
       
#         current_embeddings[0] = h.clone()
#         #GraphConv2
#         h = self.conv2(h, edge_index)
#         h = F.leaky_relu(h,inplace=False)
#         h = F.dropout(h, p=self.dropout,inplace=False)
#         #Embedding Update after second layer
#         if self.update == 'gru':
#             h = self.gru1(h, self.previous_embeddings[1].clone())
#         elif self.update == 'mlp':
#             hin = torch.cat((h, self.previous_embeddings[1].clone()), dim=1)
#             h = self.mlp1(hin)
#         else:
#             h = self.tau * self.previous_embeddings[1].clone() + (1 - self.tau) * h
    
#         current_embeddings[1] = h.clone()
#         #HADAMARD MLP
#         h_src = h[edge_label_index[0]]
#         h_dst = h[edge_label_index[1]]
#         h_hadamard = torch.mul(h_src, h_dst) #hadamard product
#         h = self.postprocess1(h_hadamard)
#         h = torch.sum(h.clone(), dim=-1).clone()
        
#         #return both 
#         #i)the predictions for the current snapshot 
#         #ii) the embeddings of current snapshot

#         return h, current_embeddings
    
#     def loss(self, pred, link_label):
#         return self.loss_fn(pred, link_label)