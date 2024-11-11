import torch
import torch.nn as nn

# Need help filling out the attention method and using it

# Tmp
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.Wa = nn.Linear(hidden_dim, hidden_dim)  # Attention weight matrix
        self.Ua = nn.Linear(hidden_dim, hidden_dim)  # Context weight matrix

    def forward(self, x):
        # Compute attention scores
        scores = torch.tanh(self.Wa(x))  # Shape: (batch_size, seq_length, hidden_dim)
        weights = torch.softmax(scores, dim=1)  # Shape: (batch_size, seq_length, hidden_dim)

        # Compute context vector as the weighted sum of the input
        context = torch.bmm(weights.transpose(1, 2), x)  # Shape: (batch_size, hidden_dim, hidden_dim)
        return context, weights