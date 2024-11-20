import torch 
import torch.nn as nn


class Decoder(nn.Module):
    """
    A flexible model that can take in a combination of the following layers:
        - LSTM
        - GRU
        - RNN
        - MLP
        - Attention
        - Fully Connected
        - ReLU
        - Sigmoid
        - Dropout
    
    Args:
        in_channels(int) : dimension of input
        out_channels:(int) : dimension of output
        hids_size (List[int]) : hidden dimension of each models
        nlayers (List[int]) : number of layers of each models
        models_seq (List[str]) : sequence of models used in decoder (order matter)
        bias (List[bool]) : list of boolean to define whether to use bias for each model
        drop_out (List[float]) : list of float to define drop out value for each model

    *Last layer must be MLP or Fully connected module*
    *First layer cannot be an activation function*
    
    Example Usage:
        decoder = Decoder(
            in_channels=8,
            out_channels=1,
            hids_size=[32, 64, 128],
            num_layers=[2, 3, 1],
            layers=["LSTM", "Attention", "MLP"],
            bias=[True, True, True],
            dropout=[0.3, 0.4, 0.0]
        )
    """
    
    def __init__(self,
        in_channels: int,
        out_channels: int,
        hids_size: list[int] = [32],
        num_layers: list[int] = [2],
        layers: list[str] = ['GRU','FC'],
        bias: list[bool] = [True],
        dropout : list[float]= [0.5]):
        
        super(Decoder, self).__init__()
        
        self.layer_map = {
            'LSTM': nn.LSTM,
            'GRU': nn.GRU,
            'RNN': nn.RNN,
            'ReLU': nn.ReLU,
            'Sigmoid': nn.Sigmoid,
            'FC': nn.Linear,
            'Dropout': nn.Dropout,
            'Attention': nn.MultiheadAttention
        }
        
        num_modules = len(layers)
        assert num_modules >= 1, "Decoder must have at least one model"
        assert len(dropout) == num_modules or len(dropout) == 1, "Drop out should be defined for each models or defined once"
        assert len(num_layers) == num_modules or len(num_layers) == 1, "Number of layers should be defined for each models or defined once"
        assert len(hids_size) == num_modules or len(hids_size) == 1, "Number of hiddens should be defined for each models or defined once"
        assert len(bias) == num_modules or len(bias) == 1, "Number of bias should be defined for each models or defined once"
        assert len(dropout) == num_modules or len(dropout) == 1, "Number of drop out values should be defined for each models or defined once"
        assert layers[0] != 'Dropout', "First layer of the model must not be \'Dropout\' layer"
        assert layers[-1] in ['MLP', 'FC'], "Layer layer must be MLP or Fully Connected layer"
        
        # Set up parameter lists if they are not appropriate size
        hids_size = self._repeat(hids_size,num_modules)
        num_layers = self._repeat(num_layers,num_modules)
        dropout = self._repeat(dropout,num_modules)
        bias = self._repeat(bias,num_modules)
        
        self.decoder = nn.ModuleList()  # Initialize our model
        
        # A single layer is present
        current_in_channels = in_channels

        for i, layer_name in enumerate(layers):
            if layer_name == 'Attention':
                self.decoder.append(
                    self.layer_map['Attention'](
                        embed_dim=current_in_channels,
                        num_heads=4,  # Example number of attention heads
                        dropout=dropout[i]
                    )
                )
            elif layer_name == 'MLP':
                self.decoder.append(self._build_mlp(current_in_channels, hids_size[i], out_channels if i == num_modules - 1 else None, bias[i]))
                current_in_channels = hids_size[i]
            else:
                self.decoder.append(
                    self._init_layer(layer_name, current_in_channels, hids_size[i], num_layers[i], dropout[i], bias[i], out_channels if i == num_modules - 1 else None)
                )
                if layer_name != 'Dropout':  # Only update channels if not dropout
                    current_in_channels = hids_size[i]
    
    def _init_layer(self,
                    layer :str, 
                    in_channel:int, 
                    hidden_size:int,
                    num_layers:int,
                    dropout_prob :float,
                    bias: bool,
                    out_channel:int = None):
        
        if layer in ['LSTM', 'GRU', 'RNN']:
            return self.layer_map[layer](input_size=in_channel, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout_prob)
        elif layer in ['ReLU', 'Sigmoid']:
            return self.layer_map[layer]()
        elif layer == 'Dropout':
            return self.layer_map['Dropout'](dropout_prob)
        elif layer == 'FC':
            return self.layer_map['FC'](in_features=in_channel, out_features=out_channel or hidden_size, bias=bias)
        else:
            raise Exception(f"Unsupport decoder {layer}")
    
    
    def _repeat(self, list: list, times:int):
        if len(list) != times:
            list = list * times 
        return list
    
    
    def _build_mlp(self, in_features, hidden_size, out_features: None, bias):
        """
        Helper function to build MLP
        """
        if out_features is None:
            out_features = hidden_size
            
        return nn.Sequential(
            nn.Linear(in_features, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features, bias=bias)
        )
    
    def forward(self, x):
        for layer in self.layers:
            # Different logic for handling attention
            if isinstance(layer, nn.MultiheadAttention):
                x, _ = layer(x, x, x)  # query, key, value
            else:
                x = layer(x)
        return x