import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn import TGCN
import torch

# Adjusted to use sparse matrices
class TGCNModel(nn.Module):
    def __init__(self, node_count, node_features, hidden_dim):
        super(TGCNModel, self).__init__()
        # The core TGCN cell handles sparse edge_index internally
        self.tgcn_cell = TGCN(node_features, hidden_dim) 

    def forward(self, x, edge_index, h=None):
        # h_next is [node_count, hidden_dim]
        h_next = self.tgcn_cell(x, edge_index, H=h)
        
        # We return the embeddings; the "decoding" happens sparsely in the trainer
        return h_next, h_next