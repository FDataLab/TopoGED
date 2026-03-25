import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class RolandUpdate(nn.Module):
    """
    Update Module: Updates hierarchical node state H_t using a GRU cell.
   
    """
    def __init__(self, input_dim, hidden_dim):
        super(RolandUpdate, self).__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, h_prev, h_new):
        # h_prev: State from t-1 (History)
        # h_new:  State from t (New Observation from Static GNN)
        return self.gru(h_new, h_prev)

class StaticGNNLayer(MessagePassing):
    """
    Static GNN Layer with Skip-Connection & BatchNorm.
   
    """
    def __init__(self, in_dim, out_dim, edge_dim=0):
        super(StaticGNNLayer, self).__init__(aggr='add')
        
        # Input to linear is Node Feats + Neighbor Feats + Edge Feats
        input_size = in_dim * 2 + edge_dim
        self.linear = nn.Linear(input_size, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.PReLU()
        
        # Skip connection handling
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.edge_dim = edge_dim

    def forward(self, x, edge_index, edge_attr=None):
        # 1. Message Passing
        if self.edge_dim > 0 and edge_attr is None:
            # Handle case where edge_dim is expected but not provided
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device)
            
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 2. Skip Connection
        out = out + self.skip(x)
        
        # 3. BatchNorm + Act
        out = self.norm(out)
        return self.act(out)

    def message(self, x_i, x_j, edge_attr):
        # Concat source, target, and edge features
        if edge_attr is not None:
            cat_msg = torch.cat([x_i, x_j, edge_attr], dim=1)
        else:
            cat_msg = torch.cat([x_i, x_j], dim=1)
            
        return self.linear(cat_msg)

class ROLAND(nn.Module):
    """
    The ROLAND Framework.
    Treats embeddings as hierarchical states updated over time.
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, num_layers=2, edge_dim=0, dropout=0.0):
        super(ROLAND, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Encoder: Static GNN Layers
        self.gnn_layers = nn.ModuleList()
        # First layer transforms Input -> Hidden
        self.gnn_layers.append(StaticGNNLayer(input_dim, hidden_dim, edge_dim))
        # Subsequent layers transform Hidden -> Hidden
        for _ in range(num_layers - 1):
            self.gnn_layers.append(StaticGNNLayer(hidden_dim, hidden_dim, edge_dim))
            
        # Recurrence: Update Modules (GRUs)
        # One recurrent unit per GNN layer depth
        self.update_modules = nn.ModuleList()
        for _ in range(num_layers):
            self.update_modules.append(RolandUpdate(hidden_dim, hidden_dim))
            
        # Decoder: Link Prediction Head (MLP)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def init_states(self, num_nodes, device):
        """Initialize H_0 (History) as zeros."""
        states = []
        for _ in range(self.num_layers):
            states.append(torch.zeros(num_nodes, self.hidden_dim).to(device))
        return states

    def forward(self, x, edge_index, edge_attr, previous_states):
        """
        Forward pass for ONE snapshot t.
        Returns: Updated States H_t (list of tensors)
        """
        current_states = []
        h = x
        
        for l in range(self.num_layers):
            # A. Static Observation (\tilde{H})
            # Compute what the embedding would be if this were just a static graph
            h_tilde = self.gnn_layers[l](h, edge_index, edge_attr)
            h_tilde = F.dropout(h_tilde, p=self.dropout, training=self.training)
            
            # B. Recurrent Update (H)
            # Update the "state" of this layer using history + new static observation
            h_updated = self.update_modules[l](previous_states[l], h_tilde)
            
            current_states.append(h_updated)
            h = h_updated # Pass the UPDATED state to the next layer depth
            
        return current_states

    def predict_links(self, z, edge_index):
        """
        Predict probability of edges existing between pairs in edge_index.
        z: Final layer node embeddings
        """
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        # Concat embeddings for link prediction
        return self.pred_head(torch.cat([z_src, z_dst], dim=1))