import torch
import torch.nn as nn


class LSTMGRU_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_1=64, hidden_dim_2=32, mlp_dim=32, num_layers_LSTM=1, num_layers_GRU=1):
        super(LSTMGRU_MLP, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        
         # Define LSTM layers
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim_1, num_layers=num_layers_LSTM)

        # Define GRU layers
        self.gru1 = nn.GRU(input_size=self.hidden_dim_1, hidden_size=self.hidden_dim_2, num_layers=num_layers_GRU)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim_2, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, output_dim),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through the LSTM layers
        x, _ = self.lstm1(x)

        # Forward pass through the GRU layers
        x, _ = self.gru1(x)

        # Pass through the fully connected layers
        if self.output_dim != 1:
            x = x[:, -1, :]  # Take the output from the last time step

        x = self.mlp(x)  # Go through the MLP

        # In case we are doing binary classification, do a sigmoid activation
        if self.output_dim == 1:
            x = self.sigmoid(x.squeeze())
        
        return x