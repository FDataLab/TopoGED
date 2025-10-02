import torch.nn as nn
import torch
import torch.nn as nn
import torch

class SimpleNodeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        """
        Initializes the LSTM model and moves it to the specified device.
        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim (int): The dimensionality of the hidden state.
            device (torch.device): The device (e.g., 'cuda' or 'cpu') for computation.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True).to(device)
        self.device = device

    def forward(self, node_feature_history):
        """
        Processes node feature histories in a batched manner on the specified device.
        Args:
            nodes (list): A list of node IDs.
            node_feature_history (dict): A dictionary mapping node IDs to their
                                         history of feature embeddings.
        Returns:
            dict: A dictionary of {node_id: final temporal embedding}.
        """
        # 1. Prepare data for batching and send to GPU
        # We ensure each tensor is on the correct device before stacking.
        stacked_features = torch.stack([
            torch.stack(node_feature_history[node]).float().to(self.device) 
            for node in node_feature_history.keys()
        ])
        
        # 2. Process all sequences in a single batched operation on the GPU
        _, (h_n, _) = self.lstm(stacked_features)
        
        # 3. Restructure the results into a dictionary
        final_node_embeddings = {}
        # The h_n tensor is already on the GPU. We move each embedding back to the CPU
        # for a clean Python dictionary, then detach it from the computation graph.
        for i, node in enumerate(node_feature_history.keys()):
            final_node_embeddings[node] = h_n[-1][i].detach().cpu()
            
        return final_node_embeddings
