#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch_scatter
import inspect
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_mean, scatter_max, scatter_add
import copy
import pickle
import argparse
import os
import sys
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from benchmarkers.utils.dataset_setup import load_data

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
random.seed(seed)
print(f"Seed set to: {seed}")

# --- UTILITY FUNCTIONS (Kept Exact) ---

def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def scatter_(name, src, index, dim_size=None):
    assert name in ['add', 'mean', 'max']
    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name is 'max' else 0
    out = op(src, index, 0, None, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]
    if name is 'max':
        out[out == fill_value] = 0
    return out

# --- LAYERS (Kept Exact) ---

class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()
        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index
        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])
        update_args = [kwargs[arg] for arg in self.update_args]
        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)
        return out

    def message(self, x_j): 
        return x_j

    def update(self, aggr_out): 
        return aggr_out

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, act=F.relu, improved=True, bias=False):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.act = act
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=x.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        edge_index, _ = torch.utils.data.dataloader.default_collate([edge_index]) if isinstance(edge_index, list) else (edge_index, None)
        # Note: Original code used custom add_self_loops, assuming PyG utils here
        # For compatibility with standard PyG edge_index (2, E)
        
        # Self-loop logic simplified for standard PyG input
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]

        x = torch.matmul(x, self.weight)
        out = self.propagate('add', edge_index, x=x, norm=norm)
        return self.act(out)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pool='mean', act=F.relu, normalize=False, bias=False):
        super(SAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.act = act
        self.pool = pool
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        row, col = edge_index
        if self.pool == 'mean':
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out = scatter_mean(out[col], row, dim=0, dim_size=out.size(0))
        elif self.pool == 'max':
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out, _ = scatter_max(out[col], row, dim=0, dim_size=out.size(0))
        elif self.pool == 'add':
            x = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

class GINConv(torch.nn.Module):
    def __init__(self, nn, eps=0, train_eps=False):
        super(GINConv, self).__init__()
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = (1 + self.eps) * x + out
        out = self.nn(out)
        return out

class graph_gru_sage(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_sage, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        for i in range(self.n_layer):
            if i==0:
                self.weight_xz.append(SAGEConv(input_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hz.append(SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xr.append(SAGEConv(input_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hr.append(SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xh.append(SAGEConv(input_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hh.append(SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
            else:
                self.weight_xz.append(SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hz.append(SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xr.append(SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hr.append(SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xh.append(SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hh.append(SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
    
    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size()).to(inp.device)
        for i in range(self.n_layer):
            if i==0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i-1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i-1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i-1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        return h_out, h_out

class graph_gru_gcn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_gcn, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        for i in range(self.n_layer):
            if i==0:
                self.weight_xz.append(GCNConv(input_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xr.append(GCNConv(input_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xh.append(GCNConv(input_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias))
        
        # Register modules to ensure they are on the correct device
        self.modules_list = nn.ModuleList()
        for i in range(self.n_layer):
            self.modules_list.extend([self.weight_xz[i], self.weight_hz[i], self.weight_xr[i],
                                      self.weight_hr[i], self.weight_xh[i], self.weight_hh[i]])

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size()).to(inp.device)
        for i in range(self.n_layer):
            if i==0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i-1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i-1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i-1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        return h_out, h_out

class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()
        self.act = act
        self.dropout = dropout

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)

# --- VGRNN MODEL ---

class VGRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, eps, conv='GCN', bias=False):
        super(VGRNN, self).__init__()
        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        
        if conv == 'GCN':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
            self.enc = GCNConv(h_dim + h_dim, h_dim)            
            self.enc_mean = GCNConv(h_dim, z_dim, act=lambda x:x)
            self.enc_std = GCNConv(h_dim, z_dim, act=F.softplus)
            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)
        
        elif conv == 'SAGE':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
            self.enc = SAGEConv(h_dim + h_dim, h_dim)
            self.enc_mean = SAGEConv(h_dim, z_dim, act=lambda x:x)
            self.enc_std = SAGEConv(h_dim, z_dim, act=F.softplus)
            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
            self.rnn = graph_gru_sage(h_dim + h_dim, h_dim, n_layers, bias)
        
        elif conv == 'GIN':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
            self.enc = GINConv(nn.Sequential(nn.Linear(h_dim + h_dim, h_dim), nn.ReLU()))            
            self.enc_mean = GINConv(nn.Sequential(nn.Linear(h_dim, z_dim)))
            self.enc_std = GINConv(nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus()))
            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)  
    
    def forward(self, x, edge_idx_list, adj_orig_dense_list, hidden_in=None):
        kld_loss = 0
        nll_loss = 0
        all_enc_mean, all_enc_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_dec_t, all_z_t = [], []
        
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(x.device)
        else:
            h = Variable(hidden_in).to(x.device)
        
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])
            
            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1), edge_idx_list[t])
            enc_mean_t = self.enc_mean(enc_t, edge_idx_list[t])
            enc_std_t = self.enc_std(enc_t, edge_idx_list[t])
            
            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)
            
            #decoder
            dec_t = self.dec(z_t)
            
            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), edge_idx_list[t], h)
            
            nnodes = adj_orig_dense_list[t].size()[0]
            enc_mean_t_sl = enc_mean_t[0:nnodes, :]
            enc_std_t_sl = enc_std_t[0:nnodes, :]
            prior_mean_t_sl = prior_mean_t[0:nnodes, :]
            prior_std_t_sl = prior_std_t[0:nnodes, :]
            dec_t_sl = dec_t[0:nnodes, 0:nnodes]
            
            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t_sl, enc_std_t_sl, prior_mean_t_sl, prior_std_t_sl)
            nll_loss += self._nll_bernoulli(dec_t_sl, adj_orig_dense_list[t])
            
            all_enc_std.append(enc_std_t_sl)
            all_enc_mean.append(enc_mean_t_sl)
            all_prior_mean.append(prior_mean_t_sl)
            all_prior_std.append(prior_std_t_sl)
            all_dec_t.append(dec_t_sl)
            all_z_t.append(z_t)
        
        return kld_loss, nll_loss, all_enc_mean, all_prior_mean, h
    
    def dec(self, z):
        outputs = InnerProductDecoder(act=lambda x:x)(z)
        return outputs
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)
     
    def _init_weights(self, stdv):
        pass
    
    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_().to(mean.device)
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)
    
    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element =  torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                            torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element
    
    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits
                                                          , target=target_adj_dense
                                                          , pos_weight=torch.tensor(posw).to(logits.device)
                                                          , reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0,1])
        return - nll_loss

# --- BENCHMARKING LOGIC ---

def compute_vgrnn_probs(means):
    """Replicates InnerProduct + Sigmoid logic from author evaluation."""
    emb = means.detach().cpu().numpy()
    adj_rec = np.dot(emb, emb.T)
    return 1 / (1 + np.exp(-adj_rec))

def optimize_threshold(model, x_in, edge_list, adj_labels, h_in, val_idx):
    model.eval()
    with torch.no_grad():
        _, _, _, pri_means, _ = model(
            x_in[val_idx], 
            edge_list[val_idx], 
            adj_labels[val_idx], 
            h_in
        )
        
        all_probs, all_targets = [], []
        for i, t in enumerate(val_idx):
            all_probs.append(compute_vgrnn_probs(pri_means[i]).flatten())
            all_targets.append(adj_labels[t].cpu().numpy().flatten())
        
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        
        best_tau, best_f1 = 0.5, 0
        for t in np.arange(0.01, 1.0, 0.01):
            preds = (all_probs > t).astype(int)
            score = f1_score(all_targets, preds, zero_division=0)
            if score > best_f1:
                best_f1, best_tau = score, t
        
        print(f"Optimal Threshold: {best_tau:.2f} (Val F1: {best_f1:.4f})")
        return best_tau

def construct_graphs(model, x_in, edge_list, adj_labels, h_in, test_idx, threshold, dataset_name, node_count):
    model.eval()
    predicted_graphs = []
    with torch.no_grad():
        _, _, _, pri_means, _ = model(
            x_in[test_idx], 
            edge_list[test_idx], 
            adj_labels[test_idx], 
            h_in
        )
        
        for i in range(len(test_idx)):
            probs = compute_vgrnn_probs(pri_means[i])
            adj = (probs > threshold).astype(int)
            predicted_graphs.append(nx.from_numpy_array(adj))

    save_path = f"data/output/predicted/{dataset_name}_vgrnn_predicted.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({'graphs': predicted_graphs, 'node_count': node_count}, f)
    print(f"Saved {len(predicted_graphs)} prediction graphs to {save_path}")

def run_benchmark(dataset, device):
    data_dict = load_data('vgrnn', args.dataset)
    
    snapshots = data_dict['snapshots']
    node_count = data_dict['node_count']
    feat_dim = data_dict['feature_dim']
    
    # 1. 70/15/15 Split
    n = len(snapshots)
    train_idx = list(range(0, int(n * 0.7)))
    val_idx = list(range(int(n * 0.7), int(n * 0.85)))
    test_idx = list(range(int(n * 0.85), n))
    
    # 2. Prepare Data
    # Using snapshot features if available, otherwise identity is handled by creation below if feat_dim matches
    # Consistent with original code which creates Identity if needed, but we use cached data
    x_in = torch.stack([s.x for s in snapshots]).to(device)
    edge_idx_list = [s.edge_index.to(device) for s in snapshots]
    adj_label_list = [to_dense_adj(s.edge_index, max_num_nodes=node_count).squeeze().to(device) 
                      for s in snapshots]

    # 3. Model Setup
    h_dim, z_dim, n_layers, eps = 32, 16, 1, 1e-10
    model = VGRNN(feat_dim, h_dim, z_dim, n_layers, eps, conv='GCN', bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    best_train_auc = 0
    best_model_wts = None
    patience = 0
    
    # 4. Training Loop
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        
        h = Variable(torch.zeros(n_layers, node_count, h_dim)).to(device)
        
        kld, nll, _, pri_means, hidden_st = model(
            x_in[train_idx], 
            edge_idx_list[train_idx], 
            adj_label_list[train_idx], 
            h
        )
        
        loss = kld + nll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
        # Monitor Train AUC for Early Stopping (Faithful to original methodology)
        train_auc_scores = []
        for i, t in enumerate(train_idx):
            probs = compute_vgrnn_probs(pri_means[i])
            true = adj_label_list[t].cpu().numpy().flatten()
            train_auc_scores.append(roc_auc_score(true, probs.flatten()))
        
        curr_auc = np.mean(train_auc_scores)
        if curr_auc > best_train_auc:
            best_train_auc = curr_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Train AUC: {curr_auc:.4f}")
        if patience > 20:
            break
            
    # 5. Optimization & Construction
    model.load_state_dict(best_model_wts)
    
    # Bridge Hidden State
    # We must re-run training sequence with best weights to get the exact hidden state for validation
    with torch.no_grad():
        h_init = Variable(torch.zeros(n_layers, node_count, h_dim)).to(device)
        _, _, _, _, hidden_st_train = model(x_in[train_idx], edge_idx_list[train_idx], adj_label_list[train_idx], h_init)
        
    opt_threshold = optimize_threshold(model, x_in, edge_idx_list, adj_label_list, hidden_st_train, val_idx)
    
    # Bridge Hidden State for Test
    # Run validation sequence to get hidden state for test
    with torch.no_grad():
        _, _, _, _, hidden_st_val = model(x_in[val_idx], edge_idx_list[val_idx], adj_label_list[val_idx], hidden_st_train)
        
    construct_graphs(model, x_in, edge_idx_list, adj_label_list, hidden_st_val, test_idx, opt_threshold, dataset, node_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    run_benchmark(args.dataset, device)