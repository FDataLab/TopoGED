import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import argparse

from utils.loader import Loader
from utils.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from nn.lstmgru_mlp import LSTMGRU_MLP
from utils.utils import Utils
from utils.embedding import process_graphs_for_embeddings
from sklearn.model_selection import train_test_split


# Default training function
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


parser = argparse.ArgumentParser()
#parser.add_argument("--model", type=str, required=True, help="The name of the model file to test")
#parser.add_argument("--output_name", type=str, required=True, help="The output file name to which results will be posted")
#parser.add_argument("--task", type=str, required=True, choices=["regression", "binary_classification"], help="The type of task to perform")
#parser.add_argument("--num_hidden", type=int, required=True, choices=[2, 3], help="Number of hidden dimensions to test on")
parser.add_argument("--dataset", type=str, required=True, 
                    choices=['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 
                             'networkcentra', 'networkindicator', 'networkcoindash', 'networkdgd', 'networkiconomi', 'Reddit_B'],
                    help="The dataset to perform training on")
args = parser.parse_args()


csv_file_path = os.path.abspath('data/output/results/RegressionTesting/LSTMGRU_MLP_individualdata.csv')
model_dir = os.path.abspath('data/output/cached_model/RegressionTesting/')

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['dataset', 'hidden_dim_1', 'hidden_dim_2', 'mlp_dim', 'learning_rate', 'num_layers_LSTM', 'num_layers_GRU', 'trained_epochs', 'valid_loss']).to_csv(csv_file_path, index=False)


# Prep data
my_loader = Loader()
data, labels = my_loader.load_data(args.dataset)
embeddings = process_graphs_for_embeddings(data)

# Split data 70/15/15
train_embeddings, tmp_embeddings = train_test_split(embeddings, test_size=0.3, shuffle=False)
val_embeddings, test_embeddings = train_test_split(tmp_embeddings, test_size=0.5, shuffle=False)

train_dataset = EmbeddingDataset(train_embeddings)
valid_dataset = EmbeddingDataset(val_embeddings)
test_dataset = EmbeddingDataset(test_embeddings)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)  
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Constants
output_dim = 20  # Regression
input_dim = 20  # 20-dimensional embeddings
patience = 15  # Early stopping patience

# Grid search params
num_layers = [3, 2]
hidden_dims = [64, 128, 256, 512]
dropouts = [0.2, 0.35, 0.5]
hidden_dim_1 = [64, 128, 256, 512]
hidden_dim_2 = [32, 64, 128, 256]
mlp_dims = [32, 64]
learning_rates = [0.0001, 0.001]
num_epochs = 500
best_aucroc = float('-inf')  # Init

print(f'Beginning grid search on dataset: {args.dataset}')

for num_layer in num_layers:
    for hidden_1 in hidden_dim_1:
        for hidden_2 in hidden_dim_2:
            for mlp_dim in mlp_dims:
                for lr_val in learning_rates:
                    best_valid_loss = float('inf')
                    old_valid_loss = float('inf')
                    no_improvement_counter = 0
                    input_dim = 20  # 20-dimensional embeddings
                    num_layers = num_layer
                    learning_rate = lr_val

                    # Define the model (assuming LSTMGRUPredictor is defined elsewhere)
                    model = LSTMGRU_MLP(input_dim, output_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2, num_layers_LSTM=num_layers, num_layers_GRU=num_layers)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    criterion = nn.MSELoss()    

                    for epoch in range(num_epochs):
                        model.train()
                        train_loss = train_model(model, train_loader, optimizer, criterion)
                        valid_loss = 0

                        with torch.no_grad():
                            model.eval()
                            hidden = None  # Initialize hidden state
                            for x, y in valid_loader:
                                output = model(x)  # Maintain hidden state across time steps
                                y = y.float()
                                loss = criterion(output, y)
                                valid_loss += loss.item()

                        valid_loss /= len(valid_loader)

                        # Early stopping
                        if epoch >= 100:
                            # After 100 epochs, look at early stopping
                            if valid_loss < old_valid_loss:
                                no_improvement_counter = 0
                                old_valid_loss = valid_loss
                            elif  valid_loss >= old_valid_loss:
                                no_improvement_counter += 1
                                
                            if no_improvement_counter == patience:
                                print(f'Training ending at epoch number: {epoch + 1}')
                                break

                        if epoch % 5 - 4 == 0:
                            # Validation
                            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {valid_loss}")

                    if valid_loss < best_valid_loss:
                        best_model = model
                        best_valid_loss = valid_loss
                    
                    # Write to dataframe
                    new_row = {
                        'dataset': args.dataset,
                        'hidden_dim_1': hidden_1,
                        'hidden_dim_2': hidden_2,
                        'mlp_dim': mlp_dim,
                        'learning_rate': lr_val,
                        #'dropout': dropout,
                        'num_layers_LSTM': num_layer,
                        'num_layers_GRU': num_layer,
                        'trained_epochs': epoch + 1,
                        'valid_loss': loss
                    }

                    pd.DataFrame([new_row]).to_csv(csv_file_path, mode='a', header=False, index=False)


# Test on the best model that we found during training
best_model.eval()
test_loss = 0
split_index = len(train_loader) + len(valid_loader)  # The start of the test set time index
time_index = split_index  # Start time index at the beginning of the test set
predicted_embeddings = []
my_utils = Utils()

print('USING LINEAR REGRESSION MODEL PRIOR TO DISPLAYING EMBEDDINGS')

with torch.no_grad():
    hidden = None  # Initialize hidden state
    for x, y in test_loader:
        output = best_model(x)  # Maintain hidden state across time steps
        predicted_embeddings.append(output.numpy())
        y = y.float()
        loss = criterion(output, y)
        test_loss += loss.item()
        
        # Print time index, predicted embedding, and real embedding
        for i in range(len(x)):
            predicted_embedding = output[i].numpy()
            real_embedding = y[i].numpy()

            predicted_embeddings = my_utils.linear_fit(predicted_embeddings)  # Fit a LinearRegression model for monotonically increasing behavior
            
            # Visualize 20-dim embeddings
            predicted_str = '\t'.join(map(str, predicted_embedding))
            real_str = '\t'.join(map(str, real_embedding))
            print(f"Time Index:\t{time_index}\tPredicted Embedding:\t{predicted_str}\tReal Embedding:\t{real_str}")
            print("-" * 50)

            time_index += 1
        
        # Print time index, predicted embedding, and real embedding
        for i in range(len(x)):
            predicted_embedding = output[i].numpy()
            real_embedding = y[i].numpy()                            
            predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
            time_index += 1

test_loss /= len(test_loader)

print(f'On the best model for the {args.dataset} dataset, the test loss is {test_loss}')

# Save model
model_path = os.path.join(model_dir, f'LSTMGRU_MLP_{args.dataset}.pkl')
torch.save(best_model.state_dict(), model_path)