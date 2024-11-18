import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import argparse

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.dataset import BinaryDataset
from torch.utils.data import DataLoader
from nn.lstmgru_mlp import LSTMGRU_MLP
from utils.embedding import process_graphs_for_embeddings
from utils.visualizers import Visualizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 


# Default training function
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    predictions = []
    labels = []
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        y = y.squeeze().float()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        predictions.append(output.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    auc_roc = roc_auc_score(labels, predictions)

    return epoch_loss / len(train_loader), auc_roc


# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str, required=True, 
#                     choices=['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 
#                              'networkcentra', 'networkindicator', 'networkcoindash', 'networkdgd', 'networkiconomi', 'Reddit_B'],
#                     help="The dataset to perform training on")
# args = parser.parse_args()


csv_file_path = os.path.abspath('data/output/results/BinaryTesting/data/LSTMGRU_MLP_individualdata.csv')
model_dir = os.path.abspath('data/output/cached_model/BinaryTesting/')

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['dataset', 'seed', 'hidden_dim_1', 'hidden_dim_2', 'mlp_dim', 'learning_rate', 'num_layers_LSTM', 'num_layers_GRU', 'trained_epochs', 'train_aucroc', 'valid_aucroc', 'train_loss', 'valid_loss']).to_csv(csv_file_path, index=False)

# Constants
output_dim = 1  # Binary classification
input_dim = 20  # 20-dimensional embeddings
patience = 15  # Early stopping patience

# Grid search params
datasets = [ 'networkcoindash', 'networkdgd', 'networkiconomi', 'Reddit_B', 'networkcindicator',
            'networkaion', 'networkadex', 'CollegeMsg', 'networkaeternity', 'mathoverflow', 'networkaragon', 'networkbancor', 'networkcentra', ]
num_layers = [3, 2]
dropouts = [0, 0.2, 0.35, 0.5]
hidden_dim_1 = [64, 128, 256, 512]
hidden_dim_2 = [32, 64, 128, 256]
mlp_dims = [32, 64]
learning_rates = [0.0001, 0.001]
#seeds = [42, 7, 9999]
seed = 42
num_epochs = 1000

# Loop over different seeds
for dataset in datasets:
    
    # Prep data
    my_loader = Loader()
    data, labels = my_loader.load_data(dataset)
    embeddings = process_graphs_for_embeddings(data)

    # Split data 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(embeddings, labels, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)

    train_dataset = BinaryDataset(X_train, y_train)
    valid_dataset = BinaryDataset(X_val, y_val)
    test_dataset = BinaryDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False)  # On specific datasets the shape mismatch will crash the code on some batches
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, drop_last=False)  # drop_last fixes this issue
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)
    print(f'Beginning grid search on dataset: {dataset}')
    best_aucroc = float('-inf')  # Init
        
    #for seed in seeds:    
    np.random.seed(seed)  # Set the seed
    for num_layer in num_layers:
        for dropout in dropouts:
            for hidden_1 in hidden_dim_1:
                for hidden_2 in hidden_dim_2:
                    for mlp_dim in mlp_dims:
                        for lr_val in learning_rates:
                            learning_rate = lr_val
                            no_improvement_counter = 0  # Number of epochs that we haven't seen an improvement in the validation AUCROC
                            old_valid_aucroc = float('-inf')
                            valid_losses = []
                            train_losses = []
                            valid_aucrocs = []
                            train_aucrocs = []

                            model = LSTMGRU_MLP(input_dim, output_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2, mlp_dim=mlp_dim, dropout=dropout, num_layers_LSTM=num_layer, num_layers_GRU=num_layer)
                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                            criterion = nn.BCELoss()    
                            
                            for epoch in range(num_epochs):
                                model.train()
                                train_loss, train_auc_roc = train_model(model, train_loader, optimizer, criterion)

                                with torch.no_grad():
                                    model.eval()
                                    hidden = None  # Initialize hidden state
                                    val_preds = []
                                    valid_loss = 0
                                    for x, y in valid_loader:
                                        output = model(x)  # Maintain hidden state across time steps
                                        y = y.squeeze().float()
                                        loss = criterion(output, y)
                                        valid_loss += loss.item()
                                        val_preds.append(output.detach().numpy())

                                    valid_loss /= len(valid_loader)
                                    val_preds = np.concatenate(val_preds, axis=0)  # Ensure val_preds is a flat array
                                    val_preds = np.array(val_preds)
                                    valid_aucroc = roc_auc_score(y_val, val_preds)
                                    
                                # Append to list for graphing later
                                valid_losses.append(valid_loss)
                                train_losses.append(train_loss)
                                valid_aucrocs.append(valid_aucroc)
                                train_aucrocs.append(train_auc_roc)
                                
                                # Early stopping only after 100 epochs
                                if epoch >= 100:
                                    if valid_aucroc >= old_valid_aucroc:
                                        no_improvement_counter = 0
                                        best_model = model
                                        old_valid_aucroc = valid_aucroc
                                        best_valid_losses = valid_losses
                                        best_train_losses = train_losses
                                        best_valid_aucrocs = valid_aucrocs
                                        best_train_aucrocs = train_aucrocs
                                        best_params = {
                                            'dataset': dataset,
                                            'seed': seed,
                                            'hidden_dim_1': hidden_1,
                                            'hidden_dim_2': hidden_2,
                                            'mlp_dim': mlp_dim,
                                            'learning_rate': lr_val,
                                            'dropout': dropout,
                                            'num_layers_LSTM': num_layer,
                                            'num_layers_GRU': num_layer,
                                        }
                                    else:
                                        no_improvement_counter += 1
                                        
                                    if no_improvement_counter == patience:
                                        print(f'Training ending at epoch number: {epoch + 1}')
                                        break
                                    
                                # Display current results
                                if epoch % 5 - 4 == 0:
                                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {valid_loss}, Train AUCROC: {train_auc_roc}, Validation AUCROC: {valid_aucroc}")
                            
                            # Write to dataframe
                            new_row = {
                                'dataset': dataset,
                                'seed': seed,
                                'hidden_dim_1': hidden_1,
                                'hidden_dim_2': hidden_2,
                                'mlp_dim': mlp_dim,
                                'learning_rate': lr_val,
                                'dropout': dropout,
                                'num_layers_LSTM': num_layer,
                                'num_layers_GRU': num_layer,
                                'trained_epochs': epoch + 1,
                                'train_aucroc': train_auc_roc,
                                'valid_aucroc': valid_aucroc,
                                'train_loss': train_loss,
                                'valid_loss': valid_loss
                            }

                            pd.DataFrame([new_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
    print("testing on dataset " + dataset)                        
    # Visualize our results
    my_visualizer = Visualizer(dataset=dataset, task='binary')
    my_visualizer.display_loss(best_train_losses, best_valid_losses, epoch)
    my_visualizer.display_aucroc(best_train_aucrocs, best_valid_aucrocs, epoch)

    # Test on the best model that we found during training
    best_model.eval()
    test_loss = 0
    split_index = len(train_loader) + len(valid_loader)  # The start of the test set time index
    time_index = split_index  # Start time index at the beginning of the test set
    test_preds = []

    with torch.no_grad():
        hidden = None  # Initialize hidden state
        for x, y in test_loader:
            output = best_model(x)  # Maintain hidden state across time steps
            test_preds.append(output.detach().numpy())
            y = y.squeeze().float()
            loss = criterion(output, y)
            test_loss += loss.item()

    test_preds = np.concatenate(test_preds, axis=0)  # Ensure val_preds is a flat array  # Flatten if it's a list
    test_preds = np.array(test_preds)

    print(f'Testing on dataset: {dataset}')
    print(f'Preds: {test_preds}')
    print(f'True: {y_test}')
    auc_roc = roc_auc_score(y_test, test_preds)
    test_loss /= len(test_loader)

    # Write results to txt
    file_path = 'data/output/results/BinaryTesting/results.txt'
    with open(file_path, 'a') as file:
        file.write(f'On the best model for the {dataset} dataset, the test AUCROC is {auc_roc} with loss {test_loss}')
        file.write(f'The best parameters found were: {best_params}')

    # Save model
    model_path = os.path.join(model_dir, f'LSTMGRU_MLP_{dataset}.pkl')
    torch.save(best_model.state_dict(), model_path)



# Run with different random seeds
# Make the validation plot (in the documentation somewhere)
# Distance between real and predicted vectors