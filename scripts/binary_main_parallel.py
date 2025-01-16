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
from nn.custom_model import Decoder

from utils.utils import Utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score


# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

import wandb

import multiprocessing
import time
import logging

def parallel_train_wrapper(model_params):
    """Wrapper to handle parameters for multiprocessing."""
    try:
        print(f'Starting training for: {model_params}')
        dataset, activation_name, activations, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, combo, counter, seed, csv_file_path, model_dir = model_params
        train_and_eval(dataset, activation_name, activations, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, combo, counter, seed, csv_file_path, model_dir)
        print(f'Finished training for: {model_params}')
    except Exception as e:
        print(f'Encountered error during training model with params {model_params}: {e}')


def train_and_eval(dataset, activation_name, activations, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, combo, counter, seed, csv_file_path, model_dir):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 1  # Binary classification
    input_dim = 0  # Dynamic based on concatenations
    patience = 25  # Early stopping patience
    num_epochs = 500  # Max epochs to train
    
    # Set up embeddings
    embeddings = None  # Init
    for activation in activations:
        tmp_activation_name = my_utils.get_activation_name(activation)
        data, labels = my_loader.load_data(dataset, tmp_activation_name)  # Load embeddings and labels
        embeddings = my_utils.concat_embeddings(embeddings, data)  # Add the new data
        
        input_dim += 30  # To account for changing embeddings
        
    # Split data 70/15/15
    n = len(embeddings)

    # Calculate split indices
    train_end = int(0.7 * n)  # 70% for training
    val_end = int(0.85 * n)   # Next 15% for validation (70% + 15% = 85%)
    X_train, y_train = embeddings[:train_end], labels[:train_end]
    X_val, y_val = embeddings[train_end:val_end], labels[train_end:val_end]
    X_test, y_test = embeddings[val_end:], labels[val_end:]
        
    print(f'Running on {dataset} with activation {activation_name}')
            
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

    # ToDo: Maybe add a batch size param here (try with 1 and None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    run_name = dataset + '_' + activation_name + '_' + str(counter)
                                        
    # Initialize wandb
    run = wandb.init(
        project="parallel_testing_new2", 
        name = run_name, 
        config={
        'dataset': dataset,
        'activation': activation_name,
        'num_layers': num_layer,
        'dropout': dropout,
        'l2_regularization': l2_val,
        'hidden_size_rnn': hidden_1,
        'hidden_size_other': hidden_2,
        'learning_rate': lr_val,
        'seed': seed,
        'normalization': norm,
        'model': combo
        },
        reinit=True)
            
    no_improvement_counter = 0  # Number of epochs that we haven't seen an improvement in the validation AUCROC
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    criterion = nn.BCELoss()  
     
      
    # Initialize wandb
    run = wandb.init(
        project="parallel_testing_new2", 
        name = run_name, 
        config={
        'dataset': dataset,
        'activation': activation_name,
        'num_layers': num_layer,
        'dropout': dropout,
        'l2_regularization': l2_val,
        'hidden_size_rnn': hidden_1,
        'hidden_size_other': hidden_2,
        'learning_rate': lr_val,
        'seed': seed,
        'normalization': norm,
        'model': combo
        },
        reinit=True)
            
    no_improvement_counter = 0  # Number of epochs that we haven't seen an improvement in the validation AUCROC
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    criterion = nn.BCELoss()  
    
    curr_batch_best_aucroc = float('-inf')    
     
    for epoch in range(num_epochs):
        # Training
        #train_loss, train_aucroc, train_aucpr, train_accuracy = model.train_model_binary(model, train_loader, optimizer, criterion)
        model.train()
        epoch_loss = 0
        predictions = []
        labels = []

        for x, y in train_loader:
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
            
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Compute metrics
        train_aucroc = roc_auc_score(labels, predictions)
        train_aucpr = average_precision_score(labels, predictions)
        train_pred_labels = [1 if prob >= 0.5 else 0 for prob in predictions]  # Since accuracy needs exact labels
        train_accuracy = accuracy_score(labels, train_pred_labels)
        train_loss = (epoch_loss / len(train_loader))
                
        # Validation
        model.eval()
        valid_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x, y in valid_loader:
                output = model(x)  # Maintain hidden state across time steps
                output = output.squeeze()
                val_preds.append(output.detach().cpu().numpy())
                y = y.squeeze().float()
                loss = criterion(output, y)
                valid_loss += loss.item()
                val_labels.append(y.detach().cpu().numpy())

        #val_preds = np.concatenate(val_preds, axis=0)  # Ensure val_preds is a flat array  # Flatten if it's a list
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        valid_loss /= len(valid_loader)
        
        # Compute metrics
        valid_aucroc = roc_auc_score(val_labels, val_preds)
        valid_aucpr = average_precision_score(val_labels, val_preds)
        val_pred_labels = [1 if prob >= 0.5 else 0 for prob in val_preds]
        valid_accuracy = accuracy_score(val_labels, val_pred_labels)
                
        # Testing        
        model.eval()
        test_loss = 0
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                output = model(x)  # Maintain hidden state across time steps
                output = output.squeeze()
                test_preds.append(output.detach().cpu().numpy())
                y = y.squeeze().float()
                loss = criterion(output, y)
                test_loss += loss.item()
                test_labels.append(y.detach().cpu().numpy())

        #test_preds = np.concatenate(test_preds, axis=0)  # Ensure val_preds is a flat array  # Flatten if it's a list
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        test_loss /= len(test_loader)
        
        # Compute metrics
        test_aucroc = roc_auc_score(test_labels, test_preds)
        test_aucpr = average_precision_score(test_labels, test_preds)
        test_pred_labels = [1 if prob >= 0.5 else 0 for prob in test_preds]
        test_accuracy = accuracy_score(test_labels, test_pred_labels)
                
        # Log each epoch results
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'test_loss': test_loss,
            'train_aucroc': train_aucroc,
            'valid_aucroc': valid_aucroc,
            'test_aucroc': test_aucroc,
            'train_aucpr': train_aucpr,
            'valid_aucpr': valid_aucpr,
            'test_aucpr': test_aucpr,
            'train_accuracy': train_accuracy,
            'valid_accuracy': valid_accuracy,
            'test_accuracy': test_accuracy
        })
        
        if ((epoch % 10 == 0)):
            wandb.log({
                f'Train Preds {epoch}': wandb.Histogram(np_histogram=np.histogram(predictions, bins=20)),
                f'Val Preds {epoch}': wandb.Histogram(np_histogram=np.histogram(val_preds, bins=20)),
                f'Test Preds {epoch}': wandb.Histogram(np_histogram=np.histogram(test_preds, bins=20)),
                } 
            )

        # Optimize for the best aucroc
        if valid_aucroc >= curr_batch_best_aucroc:
            #best_model_run = model  # For testing as we go  # UPDATE
            # ToDo: update this to be saving the param dict and then reload the dict
            
            # Save for dataframe
            best_moment_row = {
                'run_id': run.name,  # For checking Wandb Logs
                'dataset': dataset,
                'activation': activation_name,
                'seed': seed,
                'normalization': norm,
                'hidden_size_rnn': hidden_1,
                'hidden_size_other': hidden_2,
                'learning_rate': lr_val,
                'dropout': dropout,
                'l2_regularization': l2_val,
                'num_layers': num_layer,
                'model': combo,
                'trained_epochs': epoch + 1,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss,
                'train_aucroc': train_aucroc,
                'valid_aucroc': valid_aucroc,
                'test_aucroc': test_aucroc,
                'train_aucpr': train_aucpr,
                'valid_aucpr': valid_aucpr,
                'test_aucpr': test_aucpr,
                'train_accuracy': train_accuracy,
                'valid_accuracy': valid_accuracy,
                'test_accuracy': test_accuracy
            }
                    
                            
        # Early stopping only after 50 epochs
        if epoch >= 100:
            if valid_aucroc >= curr_batch_best_aucroc:
                no_improvement_counter = 0
                curr_batch_best_aucroc = valid_aucroc
            else:
                no_improvement_counter += 1
                
            if no_improvement_counter == patience:
                print(f'Training ending at epoch number: {epoch + 1}')
                break
    
    # Save the best moment from this training
    pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)


def main():
    # Setup
    os.environ["WANDB_API_KEY"] = "6a5ccf040a6c90944032e58878e46c19d673cdb0"
    wandb.init(project="Binary", name="parallel_testing_new2")

    # Constants
    csv_file_path = os.path.abspath('data/output/results/BinaryTesting/data/embedding_testing_parallel_new2.csv')
    model_dir = os.path.abspath('data/output/cached_model/BinaryTesting/EmbeddingTesting')
    num_cores = 1

    # Grid search params
    activation_combos = [[EmbedBetweenness, EmbedCloseness], [EmbedCloseness], [EmbedBetweenness],  [EmbedDegree, EmbedBetweenness], [EmbedForman, EmbedBetweenness], [EmbedDegree, EmbedCloseness], [EmbedForman, EmbedCloseness], [EmbedDegree, EmbedForman, EmbedWeight],[EmbedForman], [EmbedForman, EmbedWeight], [EmbedDegree],  [EmbedDegree, EmbedForman], [EmbedWeight], [EmbedDegree, EmbedWeight], [EmbedWeight], ]
    activation_combos_names = ['Betweenness_Closeness', 'Closeness', 'Betweenness',  'Degree_Betweenness', 'Forman_Betweenness', 'Degree_Closeness', 'Forman_Closeness', 'Degree_Forman_Weight', 'Forman', 'Forman_Weight', 'Degree',  'Degree_Forman', 'Weight', 'Degree_Weight' ]
    new_combos = []
    isolated_combo_names = []
    isolated_combos = []
    model_combos = [['LSTM', 'MLP', 'Sigmoid'], ['GRU', 'MLP', 'Sigmoid'], ['LSTM', 'FC', 'Sigmoid'], ['GRU', 'FC', 'Sigmoid'], 
                    ['GRU', 'Attention', 'FC', 'Sigmoid'], ['LSTM', 'Attention', 'FC', 'Sigmoid'], ['LSTM', 'GRU', 'FC', 'Sigmoid'], ['LSTM', 'GRU', 'MLP', 'Sigmoid']]
    datasets = ['networkbancor', 'networkadex', 'networkcentra', 'networkcoindash',  
                'mathoverflow', 'networkaeternity', 'Reddit_B',  'networkaragon', ]
    other_datasets = ['networkcindicator', 'networkiconomi', 'CollegeMsg', 'networkaoin', 'networkdgd']  # These dont work
    num_layers = [3, 2]
    dropouts = [0, 0.2, 0.35]
    hidden_dim_1 = [1000, 2500, 4000, 8000, 32, 128, 256, ]
    hidden_dim_2 = [1000, 2500, 4000, 8000, 32, 128, 256, ]
    learning_rates = [0.0001, 0.001]
    l2_regularizations = [0.00001, 0.0001, 0.001]
    norm_status = [False]  # Normalization doesnt help

    # Prep objects and variables
    seed = 42  # Can change

    # Write the header if the file doesn't already exist
    if not os.path.isfile(csv_file_path):
        pd.DataFrame(columns=['run_id', 'dataset', 'activation', 'seed', 'normalization', 'hidden_size_rnn', 'hidden_size_other', 'learning_rate', 'dropout', 'l2_regularization', 'num_layers', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss', 'train_aucroc', 'valid_aucroc', 'test_aucroc', 'train_aucpr', 'valid_aucpr', 'test_aucpr', 'train_accuracy', 'valid_accuracy', 'test_accuracy']).to_csv(csv_file_path, index=False)

    results_df = pd.read_csv(csv_file_path)
    completed_runs = results_df['run_id'].values

    # Prepare a list of all combinations of parameters
    models_params = []
    for dataset in datasets:
        for activations, activation_name in zip(activation_combos, activation_combos_names):
            if dataset == 'Reddit_B' and EmbedForman in activations:
                continue
            counter = -1  # Counter for run identifier
            for norm in norm_status:
                for num_layer in num_layers:
                    for dropout in dropouts:
                        for hidden_1 in hidden_dim_1:
                            for hidden_2 in hidden_dim_2:
                                for lr_val in learning_rates:
                                    for l2_val in l2_regularizations:
                                        for combo in model_combos:
                                            counter += 1  # Increment for run identifier
                                            run_ident = dataset + '_' + activation_name + '_' + str(counter)
                                            if run_ident in completed_runs:
                                                continue
                                            
                                            models_params.append((dataset, activation_name, activations, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, combo, counter, seed, csv_file_path, model_dir))

    # Limit processes to a lower number than the number of physical cores (e.g., 8)
    num_processes = min(num_cores, multiprocessing.cpu_count())
    print(f'There are {len(models_params)} models to train')
    # Use multiprocessing Pool to train models in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(parallel_train_wrapper, models_params, chunksize=8)

if __name__ == "__main__":
    main()