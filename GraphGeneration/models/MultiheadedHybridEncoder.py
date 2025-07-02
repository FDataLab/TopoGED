import torch
import torch.nn as nn

class HybridEdgeTypeEncoder(nn.Module):
    def __init__(self, edge_encoders: dict):
        """
        edge_encoders: dict mapping edge_type string to nn.Module (HTGNEncoder, LSTMEncoder, etc.)
        """
        super().__init__()
        self.encoders = nn.ModuleDict(edge_encoders)

    def forward(self, node_seq, edge_type, **kwargs):
        encoder = self.encoders[edge_type]
        return encoder(node_seq, **kwargs)