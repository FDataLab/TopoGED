import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn import TGCN
import torch

# 1. ARCHITECTURE: T-GCN (GCN + GRU)
class TGCNModel(nn.Module):
    def __init__(self, node_count, node_features, hidden_dim):
        super(TGCNModel, self).__init__()
        # PyG's TGCN class wraps the GCN and GRU logic into a single temporal layer
        # This is faithful to the paper's TGCN cell structure [cite: 233, 234]
        self.tgcn_cell = TGCN(node_features, hidden_dim)
        
        # Author script uses a linear output layer + bias for final prediction
        # output = tf.matmul(last_output, weights['out']) + biases['out']
        self.out_layer = nn.Linear(hidden_dim, node_count)

    def forward(self, x, edge_index, h=None):
        """
        x: Node features [node_count, node_features]
        edge_index: Graph connectivity [2, num_edges]
        h: Hidden state from previous step [node_count, hidden_dim]
        """
        # The TGCN cell handles spatial (GCN) and temporal (GRU) features [cite: 6, 234]
        # Calculation follows Eq. 3-6 in the paper [cite: 236, 237, 238]
        h_next = self.tgcn_cell(x, edge_index, h)
        
        # Final output layer followed by sigmoid for link probability [0, 1]
        # Results are obtained through a fully connected layer [cite: 186]
        predictions = torch.sigmoid(self.out_layer(h_next))
        
        return predictions, h_next