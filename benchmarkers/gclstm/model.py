import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM
from torch_geometric.utils import to_dense_adj

class GCLSTMModel(torch.nn.Module):
    def __init__(self, node_count, node_features, hidden_dim):
        super(GCLSTMModel, self).__init__()
        # Encoder: GCN embedded into LSTM cells [cite: 7, 180]
        # K=3 is the paper's specified order [cite: 278, 319]
        self.recurrent = GCLSTM(node_features, hidden_dim, K=3)
        
        # Decoder: Fully connected layer network [cite: 185, 288]
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, node_count),
            torch.nn.Sigmoid() # Predicted link probability P_t(i,j) [cite: 292]
        )

    def forward(self, x, edge_index, h, c):
        # Learn spatio-temporal features [cite: 184, 201]
        h_next, c_next = self.recurrent(x, edge_index, None, h, c)
        
        # Project hidden state back to node space [cite: 185, 288]
        predictions = self.decoder(h_next) 
        return predictions, h_next, c_next