import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score

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
        hids_size_rnn: list[int] = [32],
        hids_size_other: list[int] = [32],
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
        assert num_modules >= 1, "Decoder must have at least one module"
        assert len(dropout) == num_modules or len(dropout) == 1, "Drop out should be defined for each models or defined once"
        assert len(num_layers) == num_modules or len(num_layers) == 1, "Number of layers should be defined for each models or defined once"
        assert len(hids_size_rnn) == num_modules or len(hids_size_rnn) == 1, "Number of hiddens should be defined for each models or defined once"
        assert len(hids_size_other) == num_modules or len(hids_size_other) == 1, "Number of hiddens should be defined for each models or defined once"
        assert len(bias) == num_modules or len(bias) == 1, "Number of bias should be defined for each models or defined once"
        assert len(dropout) == num_modules or len(dropout) == 1, "Number of drop out values should be defined for each models or defined once"
        assert layers[0] not in ['ReLU', 'Sigmoid', 'Dropout'], "First layer of the model must not be an activation or dropout layer"
        assert layers[-1] in ['MLP', 'FC', 'Sigmoid'], "Last layer must be MLP Fully Connected, or Sigmoid for output"
        
        # Set up parameter lists if they are not appropriate size
        hids_size_rnn = self._repeat(hids_size_rnn, num_modules)
        hids_size_other = self._repeat(hids_size_other, num_modules)
        num_layers = self._repeat(num_layers, num_modules)
        dropout = self._repeat(dropout, num_modules)
        bias = self._repeat(bias, num_modules)
        
        self.decoder = nn.ModuleList()  # Initialize our model
        
        # A single layer is present
        current_in_channels = in_channels

        for i, layer_name in enumerate(layers):
            if layer_name == 'Attention':
                #print(f'Making attention with embed_dim: {current_in_channels}')
                self.decoder.append(
                    self.layer_map['Attention'](
                        embed_dim=current_in_channels,
                        num_heads=4,  # Can change  
                        dropout=dropout[i],
                        batch_first=True
                    )
                )
            elif layer_name == 'MLP':
                self.decoder.append(self._build_mlp(current_in_channels, hids_size_other[i], out_channels, bias[i]))
                current_in_channels = hids_size_other[i]
            
            # Since we want non recurrent layers to have a different size sometimes
            elif layer_name == 'FC':
                self.decoder.append(
                    self._init_layer(layer_name, current_in_channels, out_channels if i >= num_modules - 2 else hids_size_other[i], num_layers[i], dropout[i], bias[i])
                )
                current_in_channels = out_channels if i >= num_modules - 2 else hids_size_other[i]
            else:
                self.decoder.append(
                    self._init_layer(layer_name, current_in_channels, out_channels if i == num_modules - 2 else hids_size_rnn[i], num_layers[i], dropout[i], bias[i])
                )
                if layer_name != 'Dropout':  # Only update channels if not dropout
                    current_in_channels = out_channels if i >= num_modules - 2 else hids_size_rnn[i]
    
    
    def _init_layer(self,
                    layer :str, 
                    in_channel:int, 
                    hidden_size:int,
                    num_layers:int,
                    dropout_prob :float,
                    bias: bool):
        
        #print(f'Making {layer} with input_size: {in_channel} and output_size: {hidden_size}')
        
        if layer in ['LSTM', 'GRU', 'RNN']:
            return self.layer_map[layer](input_size=in_channel, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout_prob)
        elif layer in ['ReLU', 'Sigmoid']:
            return self.layer_map[layer]()
        elif layer == 'Dropout':
            return self.layer_map['Dropout'](dropout_prob)
        elif layer == 'FC':
            return self.layer_map['FC'](in_features=in_channel, out_features=hidden_size, bias=bias)
        else:
            raise Exception(f"Unsupport decoder {layer}")
    
    
    def _repeat(self, values: list, num_modules:int):
        if len(values) == 1:
            return values * num_modules 
        return values[:num_modules]  # Truncate if needed
    
    
    def _build_mlp(self, in_features, hidden_size, out_features: None, bias):
        """
        Helper function to build MLP
        """
        if out_features is None:
            out_features = hidden_size
        
        #print(f'Making MLP with input_size: {in_features}, hidden_size: {hidden_size} and output_size: {out_features}')
        
        return nn.Sequential(
            nn.Linear(in_features, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features, bias=bias)
        )
    
    
    def forward(self, x):
        for layer in self.decoder:
            # Since logic is different
            if isinstance(layer, nn.MultiheadAttention):
                x, _ = layer(x, x, x)  # query, key, value (k can equal v, q should be changed)
            
            elif isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                x, _ = layer(x)
                
            else:
                x = layer(x)
                            
        return x
    
    
    def train_model_binary(self, model, train_loader, optimizer, criterion):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        model.train()
        epoch_loss = 0
        predictions = []
        labels = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            output = output.squeeze()
            y = y.squeeze().float()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            predictions.append(output.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        # Compute metrics
        train_aucroc = roc_auc_score(labels, predictions)
        train_aucpr = average_precision_score(labels, predictions)
        train_pred_labels = [1 if prob >= 0.5 else 0 for prob in predictions]  # Since accuracy needs exact labels
        train_accuracy = accuracy_score(labels, train_pred_labels)
        
        return (epoch_loss / len(train_loader)), train_aucroc, train_aucpr, train_accuracy
    
    
    def test_model_binary(self, model, test_loader, criterion, y_test, display_confusion=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        model.eval()
        test_loss = 0
        test_preds = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)  # Maintain hidden state across time steps
                output = output.squeeze()
                test_preds.append(output.detach().numpy())
                y = y.squeeze().float()
                loss = criterion(output, y)
                test_loss += loss.item()

        test_preds = np.concatenate(test_preds, axis=0)  # Ensure val_preds is a flat array  # Flatten if it's a list
        test_preds = np.array(test_preds)
        test_loss /= len(test_loader)
        
        # Compute metrics
        test_aucroc = roc_auc_score(y_test, test_preds)
        test_aucpr = average_precision_score(y_test, test_preds)
        test_pred_labels = [1 if prob >= 0.5 else 0 for prob in test_preds]
        test_accuracy = accuracy_score(y_test, test_pred_labels)

        if display_confusion:
            cm = confusion_matrix(y_test, test_pred_labels)

            # Display confusion matrix
            print("Confusion Matrix:")
            # Optionally, plot the confusion matrix using seaborn for better visualization
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Shrink', 'Growth'], yticklabels=['Shrink', 'Growth'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix')
            plt.show()
            plt.clf()

        return test_loss, test_aucroc, test_aucpr, test_accuracy