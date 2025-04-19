import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import optuna
import sqlite3

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from utils.dataset import BinaryDataset
from torch.utils.data import DataLoader
from nn.custom_model import Decoder

from utils.utils import Utils
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score

import wandb

# Constants
probabilities_path = 'ReinforcmementLearning/output/probabilities/all_back/'
probabilities_end = '_from_start.csv'  # TODO Make a loader for these
csv_file_path = os.path.abspath('data/output/results/BinaryTesting/data/probabilities_gs/probabilites_gs_multidata_part2.csv')
model_dir = os.path.abspath('data/output/cached_model/BinaryTesting/probabilities_gs/MultiTesting')
STORAGE = "sqlite:///./output/cached_model/BinaryTesting/bayesianSave/probabilites_gs_multidata_part2.db"  # Where we save the study
os.makedirs(os.path.dirname('output/cached_model/BinaryTesting/bayesianSave/probabilites_gs_multidata_part2.db'), exist_ok=True)
seed = 42  # Can change

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['run_id', 'dataset', 'seed', 'hidden_size_rnn', 'hidden_size_other', 'learning_rate', 'dropout', 'l2_regularization', 'batch_size', 'num_layers', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss', 'train_aucroc', 'valid_aucroc', 'test_aucroc', 'train_aucpr', 'valid_aucpr', 'test_aucpr', 'train_accuracy', 'valid_accuracy', 'test_accuracy']).to_csv(csv_file_path, index=False)


combo_map = {
    "['LSTM', 'MLP', 'Sigmoid']": ['LSTM', 'MLP', 'Sigmoid'], 
    "['GRU', 'MLP', 'Sigmoid']": ['GRU', 'MLP', 'Sigmoid'], 
    "['LSTM', 'FC', 'Sigmoid']": ['LSTM', 'FC', 'Sigmoid'], 
    "['GRU', 'FC', 'Sigmoid']": ['GRU', 'FC', 'Sigmoid'], 
    "['GRU', 'Attention', 'FC', 'Sigmoid']": ['GRU', 'Attention', 'FC', 'Sigmoid'], 
    "['GRU', 'Attention', 'MLP', 'Sigmoid']": ['GRU', 'Attention', 'MLP', 'Sigmoid'],
    "['LSTM', 'Attention', 'FC', 'Sigmoid']": ['LSTM', 'Attention', 'FC', 'Sigmoid'], 
    "['LSTM', 'Attention', 'MLP', 'Sigmoid']": ['LSTM', 'Attention', 'MLP', 'Sigmoid'], 
    "['LSTM', 'GRU', 'FC', 'Sigmoid']": ['LSTM', 'GRU', 'FC', 'Sigmoid'], 
    "['LSTM', 'GRU', 'MLP', 'Sigmoid']": ['LSTM', 'GRU', 'MLP', 'Sigmoid']
}

def train_and_eval(dataset, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed, csv_file_path):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 1  # Binary classification
    input_dim = 6  # n_o, n_n, E_oo, E_nn, E_oon, E_on
    patience = 25  # Early stopping patience
    num_epochs = 500  # Max epochs to train
    
    run_name = dataset
                
    run_name = run_name + '_' + str(counter)    
        
    probabilities_df = pd.read_csv(f'ReinforcementLearning/output/probabilities/all_back/{dataset}_from_start.csv').iloc[:, 1:]  # Need to make a loader for this
    _, labels = my_loader.load_data(dataset, 'Degree')  # Load labels
    # Load the features and their subgraphs
    probabilities = probabilities_df.values.tolist()    
        
    # Split data 70/15/15
    n = len(probabilities)

    # Calculate split indices
    train_end = int(0.7 * n)  # 70% for training
    val_end = int(0.85 * n)   # Next 15% for validation (70% + 15% = 85%)
    X_train, y_train = probabilities[:train_end], labels[:train_end]
    X_val, y_val = probabilities[train_end:val_end], labels[train_end:val_end]
    X_test, y_test = probabilities[val_end:], labels[val_end:]


    train_dataset = BinaryDataset(X_train, y_train)
    valid_dataset = BinaryDataset(X_val, y_val)
    test_dataset = BinaryDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                                        
    # Initialize wandb
    run = wandb.init(
        project="probabilites_gs_multidata_part2", 
        name = run_name, 
        config={
        'dataset': dataset,
        'num_layers': num_layer,
        'dropout': dropout,
        'l2_regularization': l2_val,
        'hidden_size_rnn': hidden_1,
        'hidden_size_other': hidden_2,
        'learning_rate': lr_val,
        'seed': seed,
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
                'seed': seed,
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
    

model_combos = [['LSTM', 'MLP', 'Sigmoid'], ['GRU', 'MLP', 'Sigmoid'], ['LSTM', 'FC', 'Sigmoid'], ['GRU', 'FC', 'Sigmoid'], 
                ['GRU', 'Attention', 'FC', 'Sigmoid'], ['LSTM', 'GRU', 'FC', 'Sigmoid'], ['LSTM', 'GRU', 'MLP', 'Sigmoid']]
other_datasets = ['networkcindicator', 'networkiconomi', 'CollegeMsg', 'networkaion', 'networkdgd']  # These dont work

def objective(trial):
    # Suggest hyperparameters
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_1 = trial.suggest_categorical('hidden_1', [32, 64, 128, 256, 512, 1024, 2048])
    hidden_2 = trial.suggest_categorical('hidden_2', [32, 64, 128, 256, 512, 1024, 2048])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr_val = trial.suggest_float('lr_val', 1e-4, 1e-2, log=True)
    l2_val = trial.suggest_float('l2_val', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    model = trial.suggest_categorical('combo', [
        "['LSTM', 'MLP', 'Sigmoid']",
        "['GRU', 'MLP', 'Sigmoid']",
        "['LSTM', 'FC', 'Sigmoid']", 
        "['GRU', 'Attention', 'FC', 'Sigmoid']",
        "['GRU', 'Attention', 'MLP', 'Sigmoid']",
        "['LSTM', 'Attention', 'FC', 'Sigmoid']",
        "['LSTM', 'Attention', 'MLP', 'Sigmoid']",
        "['LSTM', 'GRU', 'FC', 'Sigmoid']",
        "['LSTM', 'GRU', 'MLP', 'Sigmoid']",
    ])   
    model = combo_map[model]
    
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity']
    other_datasets = ['networkiconomi','CollegeMsg',  'networkcindicator', 'networkdgd']  # These dont work
    
    results = []

    # Evaluate on all datasets
    for dataset in datasets:
        # Call your train_and_eval function for each dataset
        train_auc, val_auc = train_and_eval(
            dataset=dataset,
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
        auc_score = (train_auc * 0.3 + val_auc * 0.7)  # Play with these numbers a bit, (0.2, 0.8) and (0.4, 0.6)
        results.append(auc_score)

    # Return the mean AUC across datasets
    return sum(results) / len(results)
    

def main():
    os.environ["WANDB_API_KEY"] = "6a5ccf040a6c90944032e58878e46c19d673cdb0"
    wandb.init(project="Binary", name="probabilites_gs_multidata_part2")
    #optuna.delete_study(study_name="probabilites_gs_multidata", storage=STORAGE)
    study = optuna.create_study(study_name="probabilites_gs_multidata_part2", storage=STORAGE, direction="maximize", load_if_exists=True)
    study.optimize(objective, n_trials=2500)

    print(f"Best trial: {study.best_trial}")
    
    
if __name__ == "__main__":
    main()