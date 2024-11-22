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
from utils.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from nn.lstmgru_mlp import LSTMGRU_MLP
from utils.utils import Utils
from utils.embedding import process_graphs_for_embeddings
from utils.visualizers import Visualizer
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


# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str, required=True, 
#                     choices=['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 
#                              'networkcentra', 'networkindicator', 'networkcoindash', 'networkdgd', 'networkiconomi', 'Reddit_B'],
#                     help="The dataset to perform training on")
# args = parser.parse_args()


csv_file_path = os.path.abspath('data/output/results/RegressionTesting/data/LSTMGRU_MLP_individualdata.csv')
model_dir = os.path.abspath('data/output/cached_model/RegressionTesting/')

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['dataset', 'seed', 'hidden_dim_1', 'hidden_dim_2', 'mlp_dim', 'dropout', 'learning_rate', 'num_layers_LSTM', 'num_layers_GRU', 'trained_epochs', 'valid_loss', 'train_loss']).to_csv(csv_file_path, index=False)
    
# Constants
output_dim = 20  # Regression
input_dim = 20  # 20-dimensional embeddings
patience = 15  # Early stopping patience

# Grid search params
datasets = ['networkbancor', 'networkcentra', 'networkindicator', 'networkcoindash', 'networkdgd', 'networkiconomi', 'mathoverflow', 'Reddit_B', 'networkadex', 
            'CollegeMsg', 'networkaeternity', 'networkaion', 'networkaragon', ]
num_layers = [3, 2]
hidden_dims = [64, 128, 256, 512]
dropouts = [0, 0.2, 0.35, 0.5]
hidden_dim_1 = [64, 128, 256, 512]
hidden_dim_2 = [32, 64, 128, 256]
mlp_dims = [32, 64]
learning_rates = [0.0001, 0.001]
seeds = [42, 7, 9999]
num_epochs = 1000

found = False

for dataset in datasets:
    if dataset == "CollegeMsg" or dataset == "networkaeternity":
        continue
    # Prep data
    my_loader = Loader()
    data, labels = my_loader.load_data(dataset)
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
    print(f'Beginning grid search on dataset: {dataset}')
    best_valid_loss = float('inf')  # Init
    seed = 42
    #for seed in seeds:
    np.random.seed(seed)
    for num_layer in num_layers:
        for dropout in dropouts:
            for hidden_1 in hidden_dim_1:
                for hidden_2 in hidden_dim_2:
                    for mlp_dim in mlp_dims:
                        for lr_val in learning_rates:
                            valid_losses = []
                            train_losses = []
                            no_improvement_counter = 0
                            input_dim = 20  # 20-dimensional embeddings
                            learning_rate = lr_val

                            # Define the model (assuming LSTMGRUPredictor is defined elsewhere)
                            model = LSTMGRU_MLP(input_dim, output_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2, mlp_dim=mlp_dim, dropout=dropout, num_layers_LSTM=num_layer, num_layers_GRU=num_layer)
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
                                
                                # Append to list for graphing later
                                train_losses.append(train_loss)
                                valid_losses.append(valid_loss)

                                # Save the best model and parameters
                                if valid_loss < best_valid_loss:
                                    no_improvement_counter = 0
                                    best_model = model
                                    best_valid_loss = valid_loss
                                    best_valid_losses = valid_losses
                                    best_train_losses = train_losses
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

                                # Early stopping
                                if epoch >=50:
                                    # After 100 epochs, look at early stopping
                                    if valid_loss < best_valid_loss:
                                        no_improvement_counter = 0
                                    else:
                                        no_improvement_counter += 1
                                        
                                    if no_improvement_counter == patience:
                                        print(f'Training ending at epoch number: {epoch + 1}')
                                        break

                                if (epoch  + 1)% 5 == 0:
                                    # Validation
                                    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {valid_loss}")
                            
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
                                'train_loss': train_loss,
                                'valid_loss': valid_loss
                            }

                            pd.DataFrame([new_row]).to_csv(csv_file_path, mode='a', header=False, index=False)

    # Visualize our results
    my_visualizer = Visualizer(dataset=dataset, task='regression')
    my_visualizer.display_loss(best_train_losses, best_valid_losses, epoch)

    # Test on the best model that we found during training
    best_model.eval()
    test_loss = 0
    split_index = len(train_loader) + len(valid_loader)  # The start of the test set time index
    time_index = split_index  # Start time index at the beginning of the test set
    predicted_embeddings = []
    predicted_embeddings_linfit = []
    real_embeddings = []
    my_utils = Utils()

    print('USING LINEAR REGRESSION MODEL PRIOR TO DISPLAYING EMBEDDINGS')

    # For storing overshoots for later reference
    overshoot_file_path = 'data/output/results/RegressionTesting/data/' + dataset + '_overshoots.csv'
    if not os.path.isfile(overshoot_file_path):
        columns = [f'overshoot{i}' for i in range(20)]
        pd.DataFrame(columns=columns).to_csv(overshoot_file_path, index=False)


    with torch.no_grad():
        hidden = None  # Initialize hidden state
        for x, y in test_loader:
            output = best_model(x)  # Maintain hidden state across time steps
            y = y.float()
            loss = criterion(output, y)
            test_loss += loss.item()
            
            # Print time index, predicted embedding, and real embedding
            for i in range(len(x)):
                predicted_embedding = output[i].numpy()
                real_embedding = y[i].numpy()
                predicted_embedding_linfit = my_utils.linear_fit(predicted_embedding)

                predicted_embeddings_linfit.append(predicted_embedding_linfit)  # Fit a LinearRegression model for monotonically increasing behavior
                real_embeddings.append(real_embedding)
                predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                
                # Visualize 20-dim embeddings
                predicted_linfit_str = '\t'.join(map(str, predicted_embedding_linfit))
                predicted_str = '\t'.join(map(str, predicted_embedding))
                real_str = '\t'.join(map(str, real_embedding))
                print(f"Time Index:\t{time_index}\nPredicted Embedding:\t{predicted_str}\nLinear Fit Embedding:\t{predicted_embedding_linfit}\nReal Embedding:\t{real_str}")
                print("-" * 50)
                
                # Save overshoots
                overshoots = my_utils.compute_overshoots(predicted_embedding, real_embedding)
                new_row = {f'overshoot_{i}': overshoots[i] for i in range(len(overshoots))}
                pd.DataFrame([new_row]).to_csv(overshoot_file_path, mode='a', header=False, index=False)

                time_index += 1
        
        
    test_loss /= len(test_loader)

    # Write to file for later reference
    file_path = 'data/output/results/RegressionTesting/results.txt'
    with open(file_path, 'a') as file:
        file.write(f'On the best model for the {dataset} dataset, the test loss is {test_loss}')
        file.write(f'The best parameters found were: {best_params}')

    # Save model
    model_path = os.path.join(model_dir, f'LSTMGRU_MLP_{dataset}.pkl')
    torch.save(best_model.state_dict(), model_path)

    # Plot the embeddings to see distances
    my_visualizer.display_embeddings(predicted_embeddings, real_embeddings, predicted_embeddings_linfit)

    # Display differences between the predicted and real embeddings as a heatmap
    #my_visualizer.display_differences(predicted_embeddings, real_embeddings, predicted_embeddings_linfit)