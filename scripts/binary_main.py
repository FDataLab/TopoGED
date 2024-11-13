import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import argparse
from utils.loader import Loader
from utils.dataset import BinaryDataset
from torch.utils.data import DataLoader
from nn.lstmgru_mlp import LSTMGRU_MLP
from utils.embedding import process_graphs_for_embeddings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 


# Default training function
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        y = y.squeeze().float()
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


csv_file_path = os.path.abspath('data/output/results/BinaryTesting/LSTMGRU_MLP_individualdata.csv')
model_dir = os.path.abspath('data/output/cached_model/BinaryTesting/')

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['dataset', 'hidden_dim_1', 'hidden_dim_2', 'mlp_dim', 'learning_rate', 'num_layers_LSTM', 'num_layers_GRU', 'trained_epochs',  'valid_aucroc', 'valid_loss']).to_csv(csv_file_path, index=False)


# Prep data
my_loader = Loader()
data, labels = my_loader.load_data(args.dataset)
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

# Constants
output_dim = 1  # Binary classification
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
old_valid_aucroc = float('-inf')

print(f'Beginning grid search on dataset: {args.dataset}')

for num_layer in num_layers:
    for hidden_1 in hidden_dim_1:
        for hidden_2 in hidden_dim_2:
            for mlp_dim in mlp_dims:
                for lr_val in learning_rates:
                    num_layers = num_layer
                    learning_rate = lr_val
                    no_improvement_counter = 0  # Number of epochs that we haven't seen an improvement in the validation AUCROC

                    model = LSTMGRU_MLP(input_dim, output_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2, mlp_dim=mlp_dim, num_layers_LSTM=num_layer, num_layers_GRU=num_layer)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    criterion = nn.BCELoss()    
                    
                    for epoch in range(num_epochs):
                        model.train()
                        train_loss = train_model(model, train_loader, optimizer, criterion)

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

                        # Early stopping only after 100 epochs
                        if epoch >= 100:
                            if valid_aucroc < old_valid_aucroc:
                                no_improvement_counter = 0
                                old_valid_aucroc = valid_aucroc
                            elif  valid_aucroc >= old_valid_aucroc:
                                no_improvement_counter += 1
                                
                            if no_improvement_counter == patience:
                                print(f'Training ending at epoch number: {epoch + 1}')
                                break
                            
                        # Display current results
                        if epoch % 5 - 4 == 0:
                            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {valid_loss}, Validation AUCROC: {valid_aucroc}")

                    # Save the best model at the end of each training
                    if valid_aucroc > best_aucroc:
                        best_model = model
                        best_aucroc = valid_aucroc
                    
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
                        'valid_aucroc': valid_aucroc,
                        'valid_loss': valid_loss
                    }

                    pd.DataFrame([new_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
                            

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

auc_roc = roc_auc_score(y_test, test_preds)
test_loss /= len(test_loader)

print(f'On the best model for the {args.dataset} dataset, the test AUCROC is {auc_roc} with loss {test_loss}')

# Save model
model_path = os.path.join(model_dir, f'LSTMGRU_MLP_{args.dataset}.pkl')
torch.save(best_model.state_dict(), model_path)
