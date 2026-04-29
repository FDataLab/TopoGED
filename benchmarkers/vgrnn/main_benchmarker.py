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
import gc
import time
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from benchmarkers.benchmarker_utils.dataset_setup import load_data

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

def get_gpu_memory(device_id=0):
    if torch.cuda.is_available():
        # Handle if a torch.device object is passed instead of an int
        if isinstance(device_id, torch.device):
            idx = device_id.index if device_id.index is not None else 0
        else:
            idx = device_id
        return torch.cuda.memory_reserved(idx) / 1024**2
    return 0

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2 # M

# --- UTILITY FUNCTIONS (Kept Exact) ---
def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
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
    fill_value = -1e38 if name == 'max' else 0
    out = op(src, index, dim=0, dim_size=dim_size)
    if isinstance(out, tuple):
        out = out[0]
    if name == 'max':
        out[out == fill_value] = 0
    return out

# --- LAYERS (Fixed for Inplace and Device Consistency) ---
class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()
        # FIX: Modern Python compatibility
        self.message_args = inspect.getfullargspec(self.message)[0][1:]
        self.update_args = inspect.getfullargspec(self.update)[0][1:]

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
        update_args = [kwargs[arg] for arg in self.update_args if arg in kwargs]
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
            # FIX: Explicit Device consistency
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=x.device)
        edge_weight = edge_weight.view(-1)
        
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
        if self.bias is not None:
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
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out = scatter_add(out[col], row, dim=0, dim_size=out.size(0))
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
        # FIX: Use ModuleList for registration
        self.weight_xz = nn.ModuleList([SAGEConv(input_size if i==0 else hidden_size, hidden_size, act=lambda x:x, bias=bias) for i in range(n_layer)])
        self.weight_hz = nn.ModuleList([SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias) for _ in range(n_layer)])
        self.weight_xr = nn.ModuleList([SAGEConv(input_size if i==0 else hidden_size, hidden_size, act=lambda x:x, bias=bias) for i in range(n_layer)])
        self.weight_hr = nn.ModuleList([SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias) for _ in range(n_layer)])
        self.weight_xh = nn.ModuleList([SAGEConv(input_size if i==0 else hidden_size, hidden_size, act=lambda x:x, bias=bias) for i in range(n_layer)])
        self.weight_hh = nn.ModuleList([SAGEConv(hidden_size, hidden_size, act=lambda x:x, bias=bias) for _ in range(n_layer)])
    
    def forward(self, inp, edgidx, h):
        h_out_list = [] # Use a list instead of a pre-allocated tensor
        for i in range(self.n_layer):
            curr_inp = inp if i == 0 else h_out_list[i-1]
            z_g = torch.sigmoid(self.weight_xz[i](curr_inp, edgidx) + self.weight_hz[i](h[i], edgidx))
            r_g = torch.sigmoid(self.weight_xr[i](curr_inp, edgidx) + self.weight_hr[i](h[i], edgidx))
            h_tilde_g = torch.tanh(self.weight_xh[i](curr_inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
            
            current_h = z_g * h[i] + (1 - z_g) * h_tilde_g
            h_out_list.append(current_h)
        
        h_stack = torch.stack(h_out_list)
        return h_stack, h_stack

class graph_gru_gcn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_gcn, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.weight_xz = nn.ModuleList([GCNConv(input_size if i==0 else hidden_size, hidden_size, act=lambda x:x, bias=bias) for i in range(n_layer)])
        self.weight_hz = nn.ModuleList([GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias) for _ in range(n_layer)])
        self.weight_xr = nn.ModuleList([GCNConv(input_size if i==0 else hidden_size, hidden_size, act=lambda x:x, bias=bias) for i in range(n_layer)])
        self.weight_hr = nn.ModuleList([GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias) for _ in range(n_layer)])
        self.weight_xh = nn.ModuleList([GCNConv(input_size if i==0 else hidden_size, hidden_size, act=lambda x:x, bias=bias) for i in range(n_layer)])
        self.weight_hh = nn.ModuleList([GCNConv(hidden_size, hidden_size, act=lambda x:x, bias=bias) for _ in range(n_layer)])

    def forward(self, inp, edgidx, h):
        h_out_list = [] # Use a list instead of a pre-allocated tensor
        for i in range(self.n_layer):
            curr_inp = inp if i == 0 else h_out_list[i-1]
            z_g = torch.sigmoid(self.weight_xz[i](curr_inp, edgidx) + self.weight_hz[i](h[i], edgidx))
            r_g = torch.sigmoid(self.weight_xr[i](curr_inp, edgidx) + self.weight_hr[i](h[i], edgidx))
            h_tilde_g = torch.tanh(self.weight_xh[i](curr_inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
            
            current_h = z_g * h[i] + (1 - z_g) * h_tilde_g
            h_out_list.append(current_h)
        
        h_stack = torch.stack(h_out_list)
        return h_stack, h_stack

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
    
    def forward(self, x, edge_idx_list, target_edge_indices, hidden_in=None):
        """
        target_edge_indices: List of edge_index tensors for G_t+1 (The Future)
        """
        kld_loss = 0
        nll_loss = 0
        all_enc_mean, all_prior_mean = [], []
        
        dev = next(self.parameters()).device
        
        if hidden_in is None:
            h = torch.zeros(self.n_layers, x[0].size(0), self.h_dim).to(dev)
        else:
            h = hidden_in.to(dev)
        
        for t in range(len(x)):
            x_t = x[t].to(dev)
            e_t = edge_idx_list[t].to(dev)
            
            target_e_t = target_edge_indices[t].to(dev)
            
            phi_x_t = self.phi_x(x_t)
            
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1), e_t)
            enc_mean_t = self.enc_mean(enc_t, e_t)
            enc_std_t = self.enc_std(enc_t, e_t)
            
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)
            
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), e_t, h)
            
            num_pos = target_e_t.size(1)
            neg_e_t = torch.randint(0, x_t.size(0), (2, num_pos), device=dev)
            
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, 
                                        prior_mean_t, prior_std_t)
            
            nll_loss += self._nll_bernoulli(z_t, target_e_t, neg_e_t)
            
            all_enc_mean.append(enc_mean_t)
            all_prior_mean.append(prior_mean_t)
        
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
        # FIX: Avoid Inplace operation
        return eps1.mul(std) + mean
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        # Adding eps inside the log prevents log(0) = -inf
        kld_element = (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        # Adding eps here protects the log calculation
        std_log = torch.log(std_in + self.eps)
        kld_element = torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                            torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element
    
    def _nll_bernoulli(self, z, pos_edge_index, neg_edge_index):
        # Modified to be sparse calculations
        """
        z: Latent embeddings (N x Z_dim)
        pos_edge_index: The actual edges in the target snapshot (2 x E_pos)
        neg_edge_index: Sampled non-edges for the target snapshot (2 x E_neg)
        """
        pos_u = z[pos_edge_index[0]]
        pos_v = z[pos_edge_index[1]]
        pos_logits = torch.sum(pos_u * pos_v, dim=1)

        neg_u = z[neg_edge_index[0]]
        neg_v = z[neg_edge_index[1]]
        neg_logits = torch.sum(neg_u * neg_v, dim=1)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), reduction='mean'
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction='mean'
        )

        return pos_loss + neg_loss

@torch.no_grad()
def compute_vgrnn_probs_gpu(means, is_directed=True):
    # MM on H200 Tensor Cores
    adj_rec = torch.mm(means, means.t())
    probs = torch.sigmoid(adj_rec)
    if not is_directed:
        probs = (probs + probs.t()) / 2.0
    return probs

@torch.no_grad()
def optimize_threshold(model, x_in, edge_idx_list, h_in, val_idx, is_directed=True):
    model.eval()
    dev = next(model.parameters()).device
    
    # Forecast alignment
    input_indices = [val_idx[0] - 1] + list(val_idx[:-1])
    target_indices = list(val_idx)
    
    _, _, _, pri_means, _ = model(
        [x_in[i].to(dev) for i in input_indices], 
        [edge_idx_list[i].to(dev) for i in input_indices], 
        [edge_idx_list[i].to(dev) for i in target_indices], 
        h_in
    )
    
    all_probs, all_targets = [], []
    for i, t_idx in enumerate(target_indices):
        z = pri_means[i]
        pos_edges = edge_idx_list[t_idx].to(dev)
        neg_edges = torch.randint(0, z.size(0), (2, pos_edges.size(1)), device=dev)
        
        # Sparse Dot-Product scores
        p_scores = torch.sigmoid(torch.sum(z[pos_edges[0]] * z[pos_edges[1]], dim=1))
        n_scores = torch.sigmoid(torch.sum(z[neg_edges[0]] * z[neg_edges[1]], dim=1))
        
        all_probs.append(torch.cat([p_scores, n_scores]))
        all_targets.append(torch.cat([torch.ones(p_scores.size(0), device=dev), 
                                      torch.zeros(n_scores.size(0), device=dev)]))
    
    # Convert to NumPy for easier analysis
    y_scores = torch.cat(all_probs).detach().cpu().numpy()
    y_true = torch.cat(all_targets).detach().cpu().numpy()
    
    # Define Pos/Neg groups for the Distribution Check
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]

    print(f"\n--- VGRNN Probability Distribution Check ---")
    print(f"Positives | Mean: {pos_scores.mean():.6f} | Std: {pos_scores.std():.6f} | Max: {pos_scores.max():.6f} | Min: {pos_scores.min():.6f}")
    print(f"Negatives | Mean: {neg_scores.mean():.6f} | Std: {neg_scores.std():.6f} | Max: {neg_scores.max():.6f} | Min: {neg_scores.min():.6f}")

    # Percentile-based Grid Search
    thresholds = np.linspace(0.05, 0.99, 95)
    best_f1, best_tau = 0, 0.5
    
    for tau in thresholds:
        preds = (y_scores > tau).astype(float)
        # Calculate F1 manually for speed on CPU
        tp = np.sum(preds * y_true)
        fp = np.sum(preds * (1 - y_true))
        fn = np.sum((1 - preds) * y_true)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        if f1 > best_f1: 
            best_f1, best_tau = f1, tau

    # --- CRASH PREVENTION ---
    # If the gap is too small, a 0.00 threshold will predict N^2 edges and crash your RAM.
    if best_tau < 0.01:
        print(f"WARNING: Optimal Threshold {best_tau:.4f} is too low (RAM Risk).")
        # Force it to at least the 90th percentile if it tries to go to 0.0
        best_tau = np.percentile(y_scores, 90)
        print(f"Forcing Safety Threshold to 90th Percentile: {best_tau:.4f}")

    print(f"Optimal Threshold: {best_tau:.4f} (Sampled Val F1: {best_f1:.4f})\n")
    return best_tau

def construct_graphs(model, x_in, edge_list, h_in, test_idx, threshold, dataset_name, node_count, file_path, is_directed=True):
    import scipy.sparse as sp
    import gc
    import pickle
    import torch
    import numpy as np
    import os
    import networkx as nx

    model.eval()
    dev = next(model.parameters()).device

    # 1. Load Ground Truth for Metrics and Dynamic Capping
    from GraphGeneration.scripts.load_data import load_data
    _, _, _, target_graphs = load_data(
        dataset_name, '', '', '', 'all', 
        use_predicted=False, num_buckets=10, use_test_style=None
    )
    
    # Flatten buckets and calculate edge counts
    target_graphs_flat = [bucket[-1] for bucket in target_graphs]
    num_edges_in_targets = [g.number_of_edges() for g in target_graphs_flat]

    # Input indices for VGRNN sequential logic
    input_indices = [test_idx[0] - 1] + list(test_idx[:-1])
    target_indices = list(test_idx)

    # Generate embeddings
    with torch.no_grad():
        _, _, _, pri_means, _ = model(
            [x_in[i].to(dev) for i in input_indices], 
            [edge_list[i].to(dev) for i in input_indices], 
            [edge_list[i].to(dev) for i in target_indices], 
            h_in
        )

    predicted_networks = []

    print(f"--- Starting VGRNN Sparse Construction (5x Dynamic Cap) ---")
    
    # Iterate through pri_means. 
    # 't_local' is the loop index (0, 1, 2...)
    # 't_global' is the actual snapshot index (e.g., 255, 256...)
    for t_local, z in enumerate(pri_means):
        t_global = target_indices[t_local]
        
        all_rows, all_cols, all_scores = [], [], []
        chunk_size = 512
        
        # CHUNKED CANDIDATE COLLECTION
        for i in range(0, node_count, chunk_size):
            row_end = min(i + chunk_size, node_count)
            
            logits_chunk = torch.mm(z[i:row_end], z.t())
            probs_chunk = torch.sigmoid(logits_chunk)
            
            # Zero out self-loops
            diag_idx = torch.arange(i, row_end, device=dev)
            probs_chunk[torch.arange(row_end - i), diag_idx] = 0
            
            mask = probs_chunk >= threshold
            rows, cols = torch.where(mask)
            scores = probs_chunk[mask]
            
            all_rows.append((rows + i).cpu())
            all_cols.append(cols.cpu())
            all_scores.append(scores.cpu())
            del logits_chunk, probs_chunk, mask

        # Consolidate candidates
        full_rows_tensor = torch.cat(all_rows)
        full_cols_tensor = torch.cat(all_cols)
        full_scores_tensor = torch.cat(all_scores)
        num_threshold_passed = full_scores_tensor.numel()

        # 2. STANDARDIZED DYNAMIC CAPPING (5x edges of T-1)
        max_num_edges = num_edges_in_targets[t_global - 1] * 5
        
        # Capture raw/uncapped arrays BEFORE capping
        raw_rows = full_rows_tensor.numpy()
        raw_cols = full_cols_tensor.numpy()
        
        if num_threshold_passed > max_num_edges:
            _, top_k_idx = torch.topk(full_scores_tensor, max_num_edges)
            final_rows = full_rows_tensor[top_k_idx].numpy()
            final_cols = full_cols_tensor[top_k_idx].numpy()
            status = "CAPPED"
        else:
            final_rows = raw_rows
            final_cols = raw_cols
            status = "ACCEPTED"

        # 3. PREPARE GROUND TRUTH (Keep it sparse!)
        true_graph = target_graphs_flat[t_global].copy()
        true_graph.add_nodes_from(range(node_count))
        
        true_adj_sp = nx.to_scipy_sparse_array(true_graph, nodelist=range(node_count), format='csr')
        num_true_edges = true_adj_sp.nnz

        # --- HELPER FUNCTION FOR METRICS ---
        def get_metrics(pred_rows, pred_cols, N):
            if len(pred_rows) > 0:
                matched = np.array(true_adj_sp[pred_rows, pred_cols]).flatten()
                tp = np.sum(matched > 0)
                fp = len(pred_rows) - tp
                fn = num_true_edges - tp
                tn = (N * (N - 1)) - (tp + fp + fn)
                return tp, fp, tn, fn
            else:
                return 0, 0, (N * (N - 1)) - num_true_edges, num_true_edges

        # Calculate both sets of metrics
        tp_raw, fp_raw, tn_raw, fn_raw = get_metrics(raw_rows, raw_cols, node_count)
        tp_cap, fp_cap, tn_cap, fn_cap = get_metrics(final_rows, final_cols, node_count)

        # 4. BUILD FINAL SPARSE MATRIX (We still only save the capped version)
        adj_final = sp.csr_matrix(
            (np.ones(len(final_rows), dtype=np.int8), (final_rows, final_cols)),
            shape=(node_count, node_count)
        )

        # 5. PRINT DUAL METRICS
        print(f"\nSnap {t_global} | {status} | True Edges: {num_true_edges} | Cap Limit: {max_num_edges}")
        print(f"  [UNCAPPED] Pred: {len(raw_rows):<7} | TP: {tp_raw:<5} | FP: {fp_raw:<7} | TN: {tn_raw:<7} | FN: {fn_raw:<5}")
        print(f"  [CAPPED]   Pred: {adj_final.nnz:<7} | TP: {tp_cap:<5} | FP: {fp_cap:<7} | TN: {tn_cap:<7} | FN: {fn_cap:<5}")

        if not is_directed:
            adj_final = adj_final + adj_final.T
            adj_final.data[:] = 1

        predicted_networks.append(adj_final)
        
        # Cleanup
        del all_rows, all_cols, all_scores, full_rows_tensor, full_cols_tensor, full_scores_tensor, raw_rows, raw_cols, final_rows, final_cols
        gc.collect()

    # 6. Save Logic
    strategy = 'threshold_5xCap'
    save_path = f"data/output/predicted/VGRNN/{file_path}_{strategy}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump({'graphs': predicted_networks, 'node_count': node_count}, f)
    
    print(f"Saved memory-safe VGRNN sparse graphs to {save_path}")
    return predicted_networks


def run_benchmark(dataset, device, h_dim, z_dim, n_layers, lr, conv, eps, is_directed=True):
    data_dict = load_data('vgrnn', dataset)
    snapshots, node_count, feat_dim = data_dict['snapshots'], data_dict['node_count'], data_dict['feature_dim']
    n = len(snapshots)
    train_idx, val_idx, test_idx = range(int(n*0.7)), range(int(n*0.7), int(n*0.85)), range(int(n*0.85), n)
    
    x_in = [s.x for s in snapshots]
    edge_idx_list = [s.edge_index for s in snapshots] 

    model = VGRNN(feat_dim, h_dim, z_dim, n_layers, eps=eps, conv=conv, bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Early stopping setup
    best_loss = float('inf')
    patience = 15
    no_improve = 0
    best_model_wts = None
    
    # === PHASE 1: TRAINING ===
    start_train = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    print(f"--- Training VGRNN ({dataset}) | Patience: {patience} ---")
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        h_init = torch.zeros(n_layers, node_count, h_dim, device=device)
        
        input_idx_list = list(train_idx[:-1])
        target_idx_list = list(train_idx[1:])
        
        kld, nll, mu_list, _, _ = model(
            [x_in[i] for i in input_idx_list], 
            [edge_idx_list[i] for i in input_idx_list], 
            [edge_idx_list[i] for i in target_idx_list], 
            h_init
        )
        
        # Loss calculation (Optional: add KLD annealing if it stays stuck at 0.65 AUC)
        loss = kld + nll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
        current_loss = loss.item()

        # --- EARLY STOPPING LOGIC ---
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                all_p, all_t = [], []
                for i in range(len(mu_list)):
                    t_idx = target_idx_list[i]
                    pos_edges = edge_idx_list[t_idx].to(device)
                    neg_edges = torch.randint(0, node_count, (2, pos_edges.size(1)), device=device)
                    
                    z = mu_list[i]
                    pos_scores = torch.sigmoid(torch.sum(z[pos_edges[0]] * z[pos_edges[1]], dim=1))
                    neg_scores = torch.sigmoid(torch.sum(z[neg_edges[0]] * z[neg_edges[1]], dim=1))
                    
                    all_p.append(torch.cat([pos_scores, neg_scores]).cpu())
                    all_t.append(torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]))
                
                train_auc = roc_auc_score(torch.cat(all_t), torch.cat(all_p))
                print(f"Epoch {epoch:03d} | Loss: {current_loss:.4f} | Sampled AUC: {train_auc:.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Loss: {best_loss:.4f}")
            break

    t1, g1, r1 = time.time() - start_train, get_gpu_memory(device), get_ram_usage()
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # === PHASE 2: THRESHOLD OPTIMIZATION ===
    start_opt = time.time()
    with torch.no_grad():
        h_init = torch.zeros(n_layers, node_count, h_dim, device=device)
        _, _, _, _, hidden_st_train = model(
            [x_in[i] for i in train_idx], 
            [edge_idx_list[i] for i in train_idx], 
            [edge_idx_list[i] for i in train_idx], 
            h_init
        )
        opt_threshold = optimize_threshold(model, x_in, edge_idx_list, hidden_st_train.detach(), val_idx, is_directed)
    
    t2, g2, r2 = time.time() - start_opt, get_gpu_memory(device), get_ram_usage()

    # === PHASE 3: CONSTRUCTION ===
    start_cons = time.time()
    with torch.no_grad():
        _, _, _, _, hidden_st_val = model(
            [x_in[i] for i in val_idx], 
            [edge_idx_list[i] for i in val_idx], 
            [edge_idx_list[i] for i in val_idx], 
            hidden_st_train.detach()
        )
        file_path = f"{dataset}_{h_dim}_{z_dim}_{n_layers}_{lr}_{conv}_{'directed' if is_directed else 'undirected'}"
        construct_graphs(model, x_in, edge_idx_list, hidden_st_val.detach(), test_idx, opt_threshold, dataset, node_count, file_path, is_directed)

    t3, g3, r3 = time.time() - start_cons, get_gpu_memory(device), get_ram_usage()

    print(f"\n--- DATASET: {dataset} (VGRNN) METRICS ---")
    print(f"TRAIN:  Time={t1:.2f}s, GPU={g1:.2f}MB, RAM={r1:.2f}MB")
    print(f"THRESH: Time={t2:.2f}s, GPU={g2:.2f}MB, RAM={r2:.2f}MB")
    print(f"CONST:  Time={t3:.2f}s, GPU={g3:.2f}MB, RAM={r3:.2f}MB\n")
    gc.collect() 
    if device.type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--conv", type=str, default='GCN', choices=['GCN', 'GIN', 'SAGE'])
    parser.add_argument("--h_dim", type=int, default=32)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=1e-10)
    args = parser.parse_args()
    run_benchmark(args.dataset, device, h_dim=args.h_dim, z_dim=args.z_dim, n_layers=args.n_layers, lr=args.lr, conv=args.conv, eps=args.eps, is_directed=not args.undirected)
