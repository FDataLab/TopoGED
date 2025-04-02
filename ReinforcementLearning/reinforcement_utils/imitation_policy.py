import torch
import torch.nn as nn

# Can also use a simple MLP
class ImitationPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, num_layers=2):
        super(ImitationPolicy, self).__init__()
        
        self.lstm = nn.LSTM(state_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_dim)

    def forward(self, state_seq):
        # Ensure state_seq has three dimensions (batch_size, seq_len, state_dim)
        if state_seq.dim() == 2:  
            state_seq = state_seq.unsqueeze(1)  # Add sequence length dimension (batch, 1, state_dim)

        lstm_out, _ = self.lstm(state_seq)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Handle both multi-step and single-step sequences
        last_timestep_output = lstm_out[:, -1]  # If seq_len > 1, take last time step
        
        return self.fc(last_timestep_output)  # Shape: (batch_size, action_dim)