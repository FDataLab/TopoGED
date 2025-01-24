import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import optuna
import sqlite3

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.dataset import BinaryDataset
from torch.utils.data import DataLoader
from nn.custom_model import Decoder

from utils.utils import Utils
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score


# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

import wandb

# Constants
csv_file_path = os.path.abspath('data/output/results/BinaryTesting/data/embedding_testing_bayesian_individual_dataset.csv')
model_dir = os.path.abspath('data/output/cached_model/BinaryTesting/EmbeddingTesting')
seed = 42  # Can change

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['run_id', 'dataset', 'activation', 'seed', 'normalization', 'hidden_size_rnn', 'hidden_size_other', 'learning_rate', 'dropout', 'l2_regularization', 'batch_size', 'num_layers', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss', 'train_aucroc', 'valid_aucroc', 'test_aucroc', 'train_aucpr', 'valid_aucpr', 'test_aucpr', 'train_accuracy', 'valid_accuracy', 'test_accuracy']).to_csv(csv_file_path, index=False)


# Activation name map
activation_map = {
    "[Degree]": ['Degree'],
    "[Betweenness]": ['Betweenness'],
    "[Forman]": ['Forman'],
    "[Closeness]": ['Closeness'],
    "[Weight]": ['Weight'],
    "['Betweenness', 'Closeness']": ['Betweenness', 'Closeness'], 
    "['Degree', 'Forman', 'Weight']": ['Degree', 'Forman', 'Weight'], 
    "['Degree', 'Betweenness', 'Closeness']": ['Degree', 'Betweenness', 'Closeness'], 
    "['Betweenness', 'Forman']": ['Betweenness', 'Forman'], 
    "['Closeness', 'Forman']": ['Closeness', 'Forman'], 
    "['Degree', 'Betweenness', 'Closeness', 'Forman']": ['Degree', 'Betweenness', 'Closeness', 'Forman'], 
    "['Betweenness', 'Closeness', 'Degree', 'Forman', 'Weight']": ['Betweenness', 'Closeness', 'Degree', 'Forman', 'Weight'],
}

embedding_map = {
    'Betweenness': [EmbedBetweenness],
    'Closeness': [EmbedCloseness],
    'Betweenness_Closeness': [EmbedBetweenness, EmbedCloseness],
    'Degree': [EmbedDegree],
    'Degree_Forman_Weight': [EmbedDegree, EmbedForman, EmbedWeight],
    'Degree_Forman': [EmbedDegree, EmbedForman],
    'Forman': [EmbedForman],
    'Weight': [EmbedWeight],
    'Degree_Weight': [EmbedDegree, EmbedWeight],
    'Forman_Weight': [EmbedForman, EmbedWeight],
}

combo_map = {
    "['LSTM', 'MLP', 'Sigmoid']": ['LSTM', 'MLP', 'Sigmoid'], 
    "['GRU', 'MLP', 'Sigmoid']": ['GRU', 'MLP', 'Sigmoid'], 
    "['LSTM', 'FC', 'Sigmoid']": ['LSTM', 'FC', 'Sigmoid'], 
    "['GRU', 'FC', 'Sigmoid']": ['GRU', 'FC', 'Sigmoid'], 
    "['GRU', 'Attention', 'FC', 'Sigmoid']": ['GRU', 'Attention', 'FC', 'Sigmoid'], 
    "['LSTM', 'Attention', 'FC', 'Sigmoid']": ['LSTM', 'Attention', 'FC', 'Sigmoid'], 
    "['LSTM', 'GRU', 'FC', 'Sigmoid']": ['LSTM', 'GRU', 'FC', 'Sigmoid'], 
    "['LSTM', 'GRU', 'MLP', 'Sigmoid']": ['LSTM', 'GRU', 'MLP', 'Sigmoid']
}

def train_and_eval(dataset, activations, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed, csv_file_path):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 1  # Binary classification
    input_dim = 0  # Dynamic based on concatenations
    patience = 25  # Early stopping patience
    num_epochs = 500  # Max epochs to train
    
    run_name = dataset
    activation_name = ""
    
    # Set up embeddings
    embeddings = None  # Init
    for activation in activations:
        data, labels = my_loader.load_data(dataset, activation)  # Load embeddings and labels
        embeddings = my_utils.concat_embeddings(embeddings, data)  # Add the new data
        activation_name += activation + '_'
        
    input_dim = 30 * len(activations)
        
    run_name = run_name + '_'+ activation_name + str(counter)    
        
    # Split data 70/15/15
    n = len(embeddings)

    # Calculate split indices
    train_end = int(0.7 * n)  # 70% for training
    val_end = int(0.80 * n)   # Next 10% for validation (70% + 10% = 80%)
    X_train, y_train = embeddings[:train_end], labels[:train_end]
    X_val, y_val = embeddings[train_end:val_end], labels[train_end:val_end]
    X_test, y_test = embeddings[val_end:], labels[val_end:]
                    
    if norm:
        X_train_scaled, X_val_scaled, X_test_scaled = my_utils.normalize_embeddings(X_train, X_val, X_test)
        
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test


    train_dataset = BinaryDataset(X_train_scaled, y_train)
    valid_dataset = BinaryDataset(X_val_scaled, y_val)
    test_dataset = BinaryDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                                        
    # Initialize wandb
    run = wandb.init(
        project="bayesian_testing_individual_dataset", 
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
            if batch_size == None:
                x = x.unsqueeze(0)  # Shape: (1, seq_len) for batch_size=1
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
        
        # Behavior is different for batch sizes (avoids errors)
        if batch_size > 1:
            predictions = np.concatenate(predictions, axis=0)
            labels = np.concatenate(labels, axis=0)
        else:
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
                if batch_size == None:
                    x = x.unsqueeze(0)  # Shape: (1, seq_len) for batch_size=1
                output = model(x)  # Maintain hidden state across time steps
                output = output.squeeze()
                val_preds.append(output.detach().cpu().numpy())
                y = y.squeeze().float()
                loss = criterion(output, y)
                valid_loss += loss.item()
                val_labels.append(y.detach().cpu().numpy())

        valid_loss /= len(valid_loader)
        
        # Behavior is different for batch sizes (avoids errors)
        if batch_size > 1:
            val_preds = np.concatenate(val_preds, axis=0)
            val_labels = np.concatenate(val_labels, axis=0)
        else:
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
                
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
                if batch_size == None:
                    x = x.unsqueeze(0)  # Shape: (1, seq_len) for batch_size=1
                output = model(x)  # Maintain hidden state across time steps
                output = output.squeeze()
                test_preds.append(output.detach().cpu().numpy())
                y = y.squeeze().float()
                loss = criterion(output, y)
                test_loss += loss.item()
                test_labels.append(y.detach().cpu().numpy())

        test_loss /= len(test_loader)
        
        # Behavior is different for batch sizes (avoids errors)
        if batch_size > 1:
            test_preds = np.concatenate(test_preds, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)
        else:
            test_preds = np.array(test_preds)
            test_labels = np.array(test_labels)
            
        # Compute metrics
        test_aucroc = roc_auc_score(test_labels, test_preds)
        test_aucpr = average_precision_score(test_labels, test_preds)
        test_pred_labels = [1 if prob >= 0.5 else 0 for prob in test_preds]
        test_accuracy = accuracy_score(test_labels, test_pred_labels)
                
        # Stores our current epoch 'steps' results
        to_log = {
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
        }    
        # Add on the histograms if epoch is a multiple of 10
        if((epoch % 10 == 0)):
            to_log.update(
                {
                f'Train Preds {epoch}': wandb.Histogram(np_histogram=np.histogram(predictions, bins=20)),
                f'Val Preds {epoch}': wandb.Histogram(np_histogram=np.histogram(val_preds, bins=20)),
                f'Test Preds {epoch}': wandb.Histogram(np_histogram=np.histogram(test_preds, bins=20)),
                } 
            )   
                
        # Log each epoch results
        wandb.log(to_log)

        # Optimize for the best aucroc
        if valid_aucroc >= curr_batch_best_aucroc:
            curr_batch_best_aucroc = valid_aucroc

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
                'batch_size': batch_size,
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
    
    return best_moment_row['train_aucroc'], best_moment_row['valid_aucroc']
    

def objective(trial, dataset):
    # Suggest hyperparameters
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_1 = trial.suggest_categorical('hidden_1', [32, 64, 128, 256, 512, 1024, 2048])
    hidden_2 = trial.suggest_categorical('hidden_2', [32, 64, 128, 256, 512, 1024, 2048])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr_val = trial.suggest_float('lr_val', 1e-4, 1e-2, log=True)
    l2_val = trial.suggest_float('l2_val', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    activation = trial.suggest_categorical('activation', [
        "[Degree]",
        "[Betweenness]",
        "[Forman]",
        "[Closeness]",
        "[Weight]",
        "['Betweenness', 'Closeness']", 
        "['Degree', 'Forman', 'Weight']", 
        "['Degree', 'Betweenness', 'Closeness']", 
        "['Betweenness', 'Forman']", 
        "['Closeness', 'Forman']", 
        "['Degree', 'Betweenness', 'Closeness', 'Forman']", 
        "['Betweenness', 'Closeness', 'Degree', 'Forman', 'Weight']",
    ])
    activations = activation_map[activation]
    model = trial.suggest_categorical('combo', [
        "['LSTM', 'MLP', 'Sigmoid']", 
        "['GRU', 'MLP', 'Sigmoid']", 
        "['LSTM', 'FC', 'Sigmoid']", 
        "['GRU', 'Attention', 'FC', 'Sigmoid']", 
        "['LSTM', 'GRU', 'FC', 'Sigmoid']", 
        "['LSTM', 'GRU', 'MLP', 'Sigmoid']"
    ])   
    model = combo_map[model]
    norm = False

    # Evaluate on all datasets
    if dataset == 'Reddit_B' and 'Forman' in activations:
        raise optuna.TrialPruned("Forman-based activation not supported for Reddit_B.")
    
    # Call your train_and_eval function for each dataset
    train_auc, val_auc = train_and_eval(
        dataset=dataset,
        activations=activations,
        norm=norm,
        num_layer=num_layers,
        dropout=dropout,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        lr_val=lr_val,
        l2_val=l2_val,
        batch_size=batch_size,
        combo=model,
        counter=trial.number,
        seed=42,
        csv_file_path=csv_file_path,
    )
    auc_score = (train_auc * 0.3 + val_auc * 0.7)
    
    return auc_score
    
    
def main():
    os.environ["WANDB_API_KEY"] = "6a5ccf040a6c90944032e58878e46c19d673cdb0"
    wandb.init(project="Binary", name="bayesian_testing_individual_dataset")
    
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', ]
    other_datasets = ['networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']  # These dont work
    
    
    # Dictionary to store best results for each dataset
    best_results = {}

    for dataset in datasets:
        if dataset == 'networkaion':
            continue
        STORAGE = f"sqlite:///./output/cached_model/BinaryTesting/bayesianSave/model_selection_{dataset}.db"  # Where we save the study
        os.makedirs(os.path.dirname(f'output/cached_model/BinaryTesting/bayesianSave/model_selection_{dataset}.db'), exist_ok=True)
        print(f"Optimizing for dataset: {dataset}")
        
        study_name = f"model_selection_{dataset}"  # Unique study name for each dataset
        
        #optuna.delete_study(study_name=study_name, storage=STORAGE)
        
        study = optuna.create_study(
            study_name=study_name, 
            storage=STORAGE, 
            direction="maximize", 
            load_if_exists=True
        )
        
        # Pass dataset to the objective function
        study.optimize(lambda trial: objective(trial, dataset), n_trials=500)
        
        # Save the best trial for the current dataset
        best_results[dataset] = study.best_trial

        print(f"Best trial for {dataset}: {study.best_trial}")

    # Print all results
    print("\nBest results for each dataset:")
    for dataset, trial in best_results.items():
        print(f"{dataset}: {trial}")
    
    
if __name__ == "__main__":
    main()