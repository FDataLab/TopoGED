import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM
from torch_geometric.utils import to_dense_adj

class GCLSTMModel(torch.nn.Module):
    def __init__(self, node_count, node_features, hidden_dim, K=3):
        super(GCLSTMModel, self).__init__()
        self.recurrent = GCLSTM(node_features, hidden_dim, K=K)
        
    def forward(self, x, edge_index, h, c, edge_weight=None):
        h_next, c_next = self.recurrent(x, edge_index, edge_weight, h, c)
        return h_next, h_next, c_next