import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.dataset import BinaryDataset
from torch.utils.data import DataLoader
from nn.lstmgru_mlp import LSTMGRU_MLP
from utils.utils import Utils
from sklearn.model_selection import train_test_split

# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

import wandb

os.environ["WANDB_API_KEY"] = "6a5ccf040a6c90944032e58878e46c19d673cdb0"

wandb.init(
    project="Binary", name="newton_testing"
)

csv_file_path = os.path.abspath('data/output/results/BinaryTesting/data/embedding_testing_tmp.csv')
model_dir = os.path.abspath('data/output/cached_model/BinaryTesting/EmbeddingTesting')

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['run_id', 'dataset', 'activation', 'seed', 'normalization', 'hidden_dim_1', 'hidden_dim_2', 'mlp_dim', 'learning_rate', 'dropout', 'l2_regularization', 'num_layers_LSTM', 'num_layers_GRU', 'trained_epochs', 'train_loss', 'valid_loss', 'train_aucroc', 'valid_aucroc', 'train_aucpr', 'valid_aucpr', 'train_accuracy', 'valid_accuracy', 'test_aucroc', 'test_aucpr', 'test_accuracy', 'test_loss']).to_csv(csv_file_path, index=False)

# Constants
output_dim = 1  # Binary classification
input_dim = 30  # 30-dimensional embeddings
patience = 25  # Early stopping patience
num_epochs = 500  # Max epochs to train
num_buckets = 10  # Number of buckets for embeddings


# Grid search params
activation_combos = [[EmbedDegree], [EmbedForman], [EmbedWeight], [EmbedDegree, EmbedForman], [EmbedDegree, EmbedWeight], [EmbedForman, EmbedWeight], [EmbedDegree, EmbedForman, EmbedWeight]]
activation_combos_names = ['Degree', 'Forman', 'Weight', 'Degree_Forman', 'Degree_Weight', 'Forman_Weight', 'Degree_Forman_Weight']
model_combos = [['LSTM', 'MLP', 'Sigmoid'], ['GRU', 'MLP', 'Sigmoid'], ['LSTM', 'Sigmoid'], ['GRU', 'Sigmoid']]
datasets = ['networkadex', 'networkcindicator', 'CollegeMsg', 'networkaion', 'networkaeternity']
num_layers = [3, 2]
dropouts = [0, 0.2, 0.35]
hidden_dim_1 = [64, 128, 256]
hidden_dim_2 = [32, 64, 128, 256]
mlp_dims = [32, 64]
learning_rates = [0.0001, 0.001]
l2_regularizations = [0.00001, 0.0001, 0.001]
norm_status = [False, True]

# Prep objects and variables
my_loader = Loader()
my_utils = Utils()
seed = 42
my_utils.set_seeds(seed)

for dataset in datasets:
    top_runs = {}  # For storing the best models through training
    
    for activations, activation_name in zip(activation_combos, activation_combos_names):
        embeddings = None  # Initialize for storing final embeddings
        for activation in activations:
            tmp_activation_name = my_utils.get_activation_name(activation)
            data, labels = my_loader.load_data(dataset, tmp_activation_name)  # Load embeddings and labels
            embeddings = my_utils.concat_embeddings(embeddings, data)  # Add the new data
            
        # Since forman ricci does not work on Reddit_B dataset:
        if dataset=='Reddit_B' and activation_name == 'Forman':
            continue
        data, labels = my_loader.load_data(dataset, activation_name)  # Load embeddings and labels
        
        # Split data 70/15/15
        X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)
            
        print(f'Running on {dataset} with activation {activation_name}')
        
        counter = -1  # Initialize for logging purposes
        best_valid_aucroc = float('-inf')  # Init
        
        for norm in norm_status:
            if norm:
                max_weight = float('-inf')
                max_edges = float('-inf')
                max_nodes = float('-inf')
                
                for embedding in X_train:
                    weight = embedding[-1]
                    edges = embedding[-2]
                    nodes = embedding[-3]
                    
                    if weight > max_weight:
                        max_weight = weight
                    if edges > max_edges:
                        max_edges = edges
                    if nodes > max_nodes:
                        max_nodes = nodes
                    
                # So that we don't overwrite the original embeddings
                X_train_scaled = []
                X_val_scaled = []
                X_test_scaled = []
                    
                for embedding in X_train:
                    tmp_embedding = []
                    for i in range(0, len(embedding), 3):
                        tmp_embedding.append(embedding[i] / max_nodes)
                        tmp_embedding.append(embedding[i + 1] / max_edges)
                        tmp_embedding.append(embedding[i + 2] / max_weight)
                    
                    X_train_scaled.append(tmp_embedding)
                    
                for embedding in X_val:
                    tmp_embedding = []
                    for i in range(0, len(embedding), 3):
                        tmp_embedding.append(embedding[i] / max_nodes)
                        tmp_embedding.append(embedding[i + 1] / max_edges)
                        tmp_embedding.append(embedding[i + 2] / max_weight)
                    
                    X_val_scaled.append(tmp_embedding)
                        
                for embedding in X_test:
                    tmp_embedding = []
                    for i in range(0, len(embedding), 3):
                        tmp_embedding.append(embedding[i] / max_nodes)
                        tmp_embedding.append(embedding[i + 1] / max_edges)
                        tmp_embedding.append(embedding[i + 2] / max_weight)
                    
                    X_test_scaled.append(tmp_embedding)
                
                X_train_scaled = np.array(X_train_scaled)
                X_val_scaled = np.array(X_val_scaled)
                X_test_scaled = np.array(X_test_scaled)
                
            else:
                X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test


            train_dataset = BinaryDataset(X_train_scaled, y_train)
            valid_dataset = BinaryDataset(X_val_scaled, y_val)
            test_dataset = BinaryDataset(X_test_scaled, y_test)

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False)
            valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, drop_last=False)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)
            
            for num_layer in num_layers:
                for dropout in dropouts:
                    for hidden_1 in hidden_dim_1:
                        for hidden_2 in hidden_dim_2:
                            for mlp_dim in mlp_dims:
                                for lr_val in learning_rates:
                                    for l2_val in l2_regularizations:
                                        counter += 1  # Increment for run identifier
                                        
                                        run_name = dataset + '_' + activation_name + '_' + str(counter)
                                        
                                        curr_batch_best_aucroc = float('-inf')
                                        
                                        # Initialize wandb
                                        run = wandb.init(
                                            project="embedding_activations_gs", 
                                            name = run_name, 
                                            config={
                                            'dataset': dataset,
                                            'activation': activation_name,
                                            'num_layers_LSTM': num_layer,
                                            'num_layers_GRU': num_layer,
                                            'dropout': dropout,
                                            'l2_regularization': l2_val,
                                            'hidden_dim_1': hidden_1,
                                            'hidden_dim_2': hidden_2,
                                            'mlp_dim': mlp_dim,
                                            'learning_rate': lr_val,
                                            'seed': seed,
                                            'normalization': norm
                                            },
                                            reinit=True)
                                        
                                        no_improvement_counter = 0  # Number of epochs that we haven't seen an improvement in the validation AUCROC
                                        model = LSTMGRU_MLP(input_dim, output_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2, mlp_dim=mlp_dim, dropout=dropout, num_layers_LSTM=num_layer, num_layers_GRU=num_layer)
                                        optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
                                        criterion = nn.BCELoss()  
                                        
                                        for epoch in range(num_epochs):
                                            model.train()
                                            train_loss, train_aucroc, train_aucpr, train_accuracy = model.train_model_binary(model, train_loader, optimizer, criterion)

                                            valid_loss, valid_aucroc, valid_aucpr, valid_accuracy = model.test_model_binary(model, valid_loader, criterion, y_val)
                                                
                                            # Log each epoch results
                                            wandb.log({
                                                'epoch': epoch,
                                                'train_loss': train_loss,
                                                'valid_loss': valid_loss,
                                                'train_aucroc': train_aucroc,
                                                'valid_aucroc': valid_aucroc,
                                                'train_aucpr': train_aucpr,
                                                'valid_aucpr': valid_aucpr,
                                                'train_accuracy': train_accuracy,
                                                'valid_accuracy': valid_accuracy
                                            })

                                            # Optimize for the best aucroc
                                            if valid_aucroc >= curr_batch_best_aucroc:
                                                best_model_run = model  # For testing as we go
                                                
                                                # Save for dataframe
                                                best_moment_row = {
                                                    'run_id': run.name,  # For checking Wandb Logs
                                                    'dataset': dataset,
                                                    'activation': activation_name,
                                                    'seed': seed,
                                                    'normalization': norm,
                                                    'hidden_dim_1': hidden_1,
                                                    'hidden_dim_2': hidden_2,
                                                    'mlp_dim': mlp_dim,
                                                    'learning_rate': lr_val,
                                                    'dropout': dropout,
                                                    'l2_regularization': l2_val,
                                                    'num_layers_LSTM': num_layer,
                                                    'num_layers_GRU': num_layer,
                                                    'trained_epochs': epoch + 1,
                                                    'train_loss': train_loss,
                                                    'valid_loss': valid_loss,
                                                    'train_aucroc': train_aucroc,
                                                    'valid_aucroc': valid_aucroc,
                                                    'train_aucpr': train_aucpr,
                                                    'valid_aucpr': valid_aucpr,
                                                    'train_accuracy': train_accuracy,
                                                    'valid_accuracy': valid_accuracy
                                                }
                                                
                                                # Save the model
                                                top_runs = my_utils.update_top_models(top_runs, activation_name, run.name, model, valid_aucroc, dataset, top_x=3)
                                                curr_batch_best_aucroc = valid_aucroc
                                                
                                                # If we have a new best model for this activation
                                                if valid_aucroc > best_valid_aucroc:
                                                    best_valid_aucroc = valid_aucroc
                                                    print(f'We have a new best model with a Validation AUCROC: {valid_aucroc}')
                                                    test_loss, test_aucroc, test_aucpr, test_accuracy = model.test_model_binary(model, test_loader, criterion, y_test)
                                                    print(f"""\tTest Loss: {valid_loss}\n\tTest Validation AUCROC: {test_aucroc}\n\tTest AUCPR: {test_aucpr}\n\tTest Accuracy: {test_accuracy}\n
                                                    """)
                                            
                                            # See how long training takes before using
                                            '''
                                            # Early stopping only after 50 epochs
                                            if epoch >= 50:
                                                if valid_aucroc >= curr_batch_best_aucroc:
                                                    no_improvement_counter = 0
                                                    curr_batch_best_aucroc = valid_aucroc
                                                else:
                                                    no_improvement_counter += 1
                                                    
                                                if no_improvement_counter == patience:
                                                    print(f'Training ending at epoch number: {epoch + 1}')
                                                    break'''
                                            '''    
                                            # Display current results
                                            if epoch % 10 - 9 == 0:
                                                print(f"""
                                                    Epoch {epoch+1}/{num_epochs}:\n\tTrain Loss: {train_loss}, Validation Loss: {valid_loss}\n\tTrain AUCROC: {train_aucroc}, Validation AUCROC: {valid_aucroc}\n\tTrain AUCPR: {train_aucpr}, Validation AUCPR: {valid_aucpr}\n\tTrain Accuracy: {train_accuracy}, Validation Accuracy: {valid_accuracy}\n
                                                """)
                                        '''
                                        
                                        test_loss, test_aucroc, test_aucpr, test_accuracy = best_model_run.test_model_binary(best_model_run, test_loader, criterion, y_test, display_confusion=False)
                                        
                                        best_moment_row['test_loss'] = test_loss
                                        best_moment_row['test_aucroc'] = test_aucroc
                                        best_moment_row['test_aucpr'] = test_aucpr
                                        best_moment_row['test_accuracy'] = test_accuracy
                                        
                                        # Save the best moment from this training
                                        pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
    
        # Save after testing each activation
        my_utils.save_models(top_runs, model_dir)

        # Test models
        test_loss, test_aucroc, test_aucpr, test_accuracy = model.test_model_binary(model, test_loader=test_loader, criterion=criterion, y_test=y_test)
        print(f'\tTest Loss: {test_loss}')
        print(f'\tTest AUCROC: {test_aucroc}')
        print(f'\tTest AUCPR: {test_aucpr}')
        print(f'\tTest Accuracy: {test_accuracy}')
    
    wandb.finish()


