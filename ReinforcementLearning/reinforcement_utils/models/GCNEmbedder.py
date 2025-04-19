import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    
    def forward(self, x, edge_index, batch):
        if x.size(0) == 0:  # No nodes in the graph
            return torch.zeros(self.conv2.out_channels)  # Return a zero vector of size [1, 128]
        
        if edge_index.size(0) == 0:  # No edges in the graph
            return torch.zeros(self.conv2.out_channels)  # Return zero embeddings for nodes
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Graph-level embedding via pooling
        x = global_mean_pool(x, batch)  # shape: [batch_size, out_dim]
        
        # Ensure the output is always of shape [1, 128]
        if x.size(0) == 0:  # If no graph data exists, return a zero vector
            return torch.zeros(1, self.conv2.out_channels)  # Shape: [1, 128]

        return x