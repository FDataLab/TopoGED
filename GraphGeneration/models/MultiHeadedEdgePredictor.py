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
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1), 
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
    
    
class ExistMultiHeadedEdgePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, edge_types=None, input_type='Concat'):
        super().__init__()
        if edge_types is None:
            edge_types = ['o-o-nobank', 'o-o-bank', 'o-n', 'n-n']

        self.input_type = input_type

        # -------------------------
        # (1) EDGE-TYPE HEADS
        # -------------------------
        self.heads = nn.ModuleDict({
            etype: nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1),
                nn.Sigmoid()
            )
            for etype in edge_types
        })

        # ---------------------------------------------------
        # (2) NODE-EXISTENCE HEAD (added new prediction head)
        # ---------------------------------------------------
        # Input = node embedding (same dimension as src_embed/dst_embed)
        self.node_head = nn.Sequential(
            nn.Linear(in_channels // 2 if input_type == "Concat" else in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

    # --------------------------------------
    # EDGE PREDICTION (original functionality)
    # --------------------------------------
    def forward(self, src_embed, dst_embed, edge_type):
        # Decide embedding combination
        if self.input_type == 'Concat':
            edge_input = torch.cat([src_embed, dst_embed], dim=1)
        elif self.input_type == 'Addition':
            edge_input = src_embed + dst_embed
        elif self.input_type == 'Subtraction':
            edge_input = src_embed - dst_embed
        elif self.input_type == 'Product':
            edge_input = src_embed * dst_embed

        return self.heads[edge_type](edge_input).squeeze()

    # -----------------------------------------------------
    # NODE EXISTENCE PREDICTION (NEW METHOD YOU WILL CALL)
    # -----------------------------------------------------
    def predict_node_exists(self, node_embed):
        return self.node_head(node_embed).squeeze()