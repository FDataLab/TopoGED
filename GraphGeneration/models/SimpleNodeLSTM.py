import torch.nn as nn
import torch
class SimpleNodeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, node_feature_history):
        final_node_embeddings = {}
        for v, feat_seq in node_feature_history.items():
            x_seq = torch.stack(feat_seq).unsqueeze(0).float()  # shape [1, T, input_dim]
            _, (h_n, _) = self.lstm(x_seq)
            final_node_embeddings[v] = h_n[-1].squeeze(0)  # shape [hidden_dim]
        return final_node_embeddings