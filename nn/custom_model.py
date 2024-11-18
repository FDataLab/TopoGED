import torch 
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_prob=0.0, layers=["LSTM", "GRU", "Dense", "Dropout", "Dense"], activation = "ReLU", num_layers_LSTM=3, num_layers_GRU=3):
        super(Model, self).__init__()
        
        # Assertions:
        # Assert Dropout isnt first layer
        # Assert last layer is fully connected or mlp
        # MLP must be followed by an activation function
        
        
        hidden_sizes = self._init_hidden(input_dim, output_dim, layers)
        
        self.decoder = nn.Sequential()
        
        self._init_model(layers, hidden_sizes, dropout_prob, num_layers_LSTM, num_layers_GRU)
            
            
    def _init_model(self, layers, hidden_sizes, dropout_prop, num_layers_LSTM, num_layers_GRU):
        i = 0  # For indexing the hidden sizes
        for layer in layers:
            if layer == "LSTM":
                self.decoder.add_module(nn.LSTM(input_size=hidden_sizes[i], hidden_size=hidden_sizes[i + 1], num_layers=num_layers_LSTM))
            elif layer == "GRU":
                self.decoder.add_module(nn.GRU(input_size=hidden_sizes[i], hidden_size=hidden_sizes[i + 1], num_layers=num_layers_GRU))
            elif layer == "FC":
                self.decoder.add_module(nn.Linear(in_features=hidden_sizes[i], out_features=hidden_sizes[i + 1]))
            elif layer == "MLP":
                self.decoder.add_module(nn.Linear(in_features=hidden_sizes[i], out_features=hidden_sizes[i + 1]))
                self.decoder.add_module()
                i += 2
            elif layer == "Dropout":
                self.decoder.add_module(nn.Dropout(dropout_prop))
    
    
    def _init_hidden(self, input_dim, output_dim, hidden_dim, layers):
        hidden_sizes = []
        hidden_sizes.append(input_dim)
        
        # Calculate how many hidden sizes are needed
        for layer in layers:
            # Since dropout doesnt have hidden size param
            if layer != "Dropout":
                for i in range(2):
                    hidden_sizes.append(hidden_dim)
        
        # Remove two to account for input and output
        for i in range(2):
            hidden_sizes.pop(-1)
        
        hidden_sizes.append(output_dim)
        
        return hidden_sizes
    
    
    def forward(self, x):
        pass 
    
    
    def train_model(self):
        pass 