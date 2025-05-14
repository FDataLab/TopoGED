import torch
import torch.nn as nn

class MultiHeadedEdgePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, edge_types=None, input_type='Concat'):
        super().__init__()
        if edge_types is None:
            edge_types = ['o-o-nobank', 'o-o-bank', 'o-n', 'n-n']  # default 4 edge types
        self.input_type = input_type
        self.heads = nn.ModuleDict({
            etype: nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )
            for etype in edge_types
        })

    def forward(self, src_embed, dst_embed, edge_type):
        if self.input_type == 'Concat':
            edge_input = torch.cat([src_embed, dst_embed], dim=1)
        elif self.input_type == 'Addition':
            edge_input = src_embed + dst_embed
        elif self.input_type == 'Subtraction':
            edge_input = src_embed - dst_embed
        elif self.input_type == 'Product':
            edge_input = src_embed * dst_embed
            
        return self.heads[edge_type](edge_input).squeeze()