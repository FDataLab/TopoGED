import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgePredictorMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, input_type='Concat'):
        super().__init__()
        self.input_type = input_type  # This is the mode for processing node embeddings
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # This ensures output is a probability (0 to 1)
        )

    def forward(self, src_embed, dst_embed):
        if self.input_type == 'Concat':
            edge_input = torch.cat([src_embed, dst_embed], dim=1)
        elif self.input_type == 'Addition':
            edge_input = src_embed + dst_embed
        elif self.input_type == 'Subtraction':
            edge_input = src_embed - dst_embed
        elif self.input_type == 'Product':
            edge_input = src_embed * dst_embed
            
        return self.model(edge_input).squeeze()