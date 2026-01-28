import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from dotenv import load_dotenv

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from torch.utils.data import DataLoader
from nn.custom_model import Decoder
from torch.utils.data import DataLoader, Dataset
from utils.utils import Utils
from utils.dataset import DeltaEmbeddingDataset

import wandb
import optuna


# Constants
seed = 42
FLOAT_MAX = np.finfo(np.float32).max

# Device Selection logic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Activation name map
activation_map = {
    "Degree": ['Degree'],
    "Betweenness": ['Betweenness'],
    "Forman": ['Forman'],
    "Closeness": ['Closeness'],
    "Weight": ['Weight'],
    "Degree_Betweenness_Closeness": ['Degree', 'Betweenness', 'Closeness'],
    "Degree_Forman_Weight": ['Degree', 'Forman', 'Weight'],
    "Degree_Forman_Closeness": ['Degree', 'Forman', 'Closeness'],
    "Degree_Weight_Closeness": ['Degree', 'Weight', 'Closeness'],
    "Degree_Forman": ['Degree', 'Forman'],
    "Degree_Weight": ['Degree', 'Weight'],
    "Degree_Betweenness": ['Degree', 'Betweenness'],
    "Degree_Closeness": ['Degree', 'Closeness'],
}

combo_map = {
    "['LSTM', 'ReLU']": ['LSTM', 'ReLU'], 
    "['GRU', 'ReLU']": ['GRU', 'ReLU'], 
    "['LSTM', 'GRU', 'ReLU']": ['LSTM', 'GRU', 'ReLU'], 
    "['RNN']": ['RNN'],
    "['GRU']": ['GRU'],
    "['LSTM']": ['LSTM'],
    "['RNN', 'FC']": ['RNN', 'FC'],
    "['LSTM', 'FC']": ['LSTM', 'FC'],
    "['GRU', 'FC']": ['GRU', 'FC'],
    "['LSTM', 'GRU', 'FC']": ['LSTM', 'GRU', 'FC'],
    "['LSTM', 'FC', 'FC']": ['LSTM', 'FC', 'FC'],
    "['GRU', 'FC', 'FC']": ['GRU', 'FC', 'FC'],
    "['RNN', 'MLP']": ['RNN', 'MLP'],
    "['LSTM', 'MLP']": ['LSTM', 'MLP'], 
    "['GRU', 'MLP']": ['GRU', 'MLP'], 
    "['LSTM', 'GRU', 'MLP']": ['LSTM', 'GRU', 'MLP']
}

def train_and_eval_delta(dataset, activations, window_size, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed, results_csv, num_buckets):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 0
    input_dim = 0
    patience = 25
    num_epochs = 500
    
    # Set up embeddings
    embeddings = None
    using_weight = False 
    for activation in activations:
        data, labels = my_loader.load_data(dataset, activation, include_weights=using_weight, num_buckets=num_buckets)
        embeddings = my_utils.concat_embeddings(embeddings, data)
        
    if not using_weight:
        input_dim = 2 * num_buckets * len(activations)
        output_dim = input_dim
    else:
        input_dim = 3 * num_buckets * len(activations)
        output_dim = input_dim
        
    # Split data
    n = len(embeddings)
    train_end = int(0.7 * n)
    val_start = train_end - window_size
    val_end = int(0.85 * n)
    test_start = val_end - window_size

    embeddings = np.array([np.array(e, dtype=np.float32) for e in embeddings])

    X_train = embeddings[:train_end]
    X_val = embeddings[val_start:val_end]
    X_test = embeddings[test_start:]
    
    if norm:
        X_train_scaled, X_val_scaled, X_test_scaled = my_utils.normalize_embeddings(X_train, X_val, X_test)
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test

    train_dataset = DeltaEmbeddingDataset(X_train_scaled, k=window_size)
    valid_dataset = DeltaEmbeddingDataset(X_val_scaled, k=window_size)
    test_dataset = DeltaEmbeddingDataset(X_test_scaled, k=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            
    no_improvement_counter = 0
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    criterion = nn.MSELoss().to(device)
    
    curr_batch_best_loss = float('inf')    
    best_moment_row = {}
    
    for epoch in range(num_epochs):
        model.train()
        for i, (x, y_delta, x_last) in enumerate(train_loader):
            x, y_delta = x.to(device), y_delta.to(device)
            optimizer.zero_grad()
            predicted_delta = model(x)[:, -1, :].squeeze(1)
            loss = criterion(predicted_delta, y_delta.float())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        valid_loss = 0
        val_cosine_similarities = []
        val_norms = []

        with torch.no_grad():
            for i, (x, y_delta, x_last) in enumerate(valid_loader):
                x, y_delta, x_last = x.to(device), y_delta.to(device), x_last.to(device)
                
                predicted_delta = model(x)[:, -1, :].squeeze(1)
                valid_loss += criterion(predicted_delta, y_delta.float()).item()
                
                # POSITIVITY CONSTRAINT: Apply max(0, x_last + delta)
                # This ensures your metrics (Cosine/Dist) are calculated on realistic values
                pred_abs = torch.clamp(x_last + predicted_delta, min=0)
                real_abs = x_last + y_delta # Ground truth is always positive

                pred_abs_np = pred_abs.cpu().numpy()
                real_abs_np = real_abs.cpu().numpy()

                for j in range(len(x)):
                    try:
                        val_cosine_similarities.append(my_utils.compute_cosine_similarity(pred_abs_np[j], real_abs_np[j]))
                        val_norms.append(my_utils.compute_distances(pred_abs_np[j], real_abs_np[j]))
                    except:
                        val_cosine_similarities.append(float('nan'))
                        val_norms.append(float('nan'))
            
        valid_loss /= len(valid_loader)
        val_avg_norm = np.nanmean(val_norms)
        val_avg_cosine_similarity = np.nanmean(val_cosine_similarities)
        
        if valid_loss <= curr_batch_best_loss:
            curr_batch_best_loss = valid_loss
            no_improvement_counter = 0 
            

            best_moment_row = {
                'run_id': f'{dataset}_{counter}',
                'dataset': dataset,
                'activation': 'Degree',
                'window_size': window_size,
                'seed': seed,
                'normalization': norm,
                'hidden_1': hidden_1,
                'hidden_2': hidden_2,
                'learning_rate': lr_val,
                'dropout': dropout,
                'l2_regularization': l2_val,
                'batch_size': batch_size,
                'num_layers': num_layer,
                'model': combo,
                'trained_epochs': epoch + 1,
                'train_loss': np.nan,
                'valid_loss': valid_loss,
                'test_loss': np.nan,
            }
        else:
            if epoch >= 50:
                no_improvement_counter += 1
            
        if no_improvement_counter >= patience:
            break
    
    
    # Save best parameters to CSV
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    df_result = pd.DataFrame([best_moment_row])
    file_exists = os.path.isfile(results_csv)
    df_result.to_csv(results_csv, mode='a', index=False, header=not file_exists)
    
    return 0.0, valid_loss


def objective(trial, dataset, activations, results_csv, num_buckets):
    # Search Space
    batch_size = trial.suggest_categorical("batch_size", [16])
    dropout = trial.suggest_float("dropout", 0, 0.2, step=0.1)
    hidden_1 = trial.suggest_categorical("hidden_1", [32, 64, 128, 256])
    hidden_2 = trial.suggest_categorical("hidden_2", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    lr_val = trial.suggest_float("lr_val", 1e-5, 1e-3, log=True)
    l2_val = trial.suggest_float("l2_val", 1e-5, 1e-3, log=True)
    
    model_str = trial.suggest_categorical("model", [
        "['RNN', 'MLP']",
        "['LSTM', 'MLP']", "['GRU', 'MLP']", "['LSTM', 'GRU', 'MLP']"
    ])
    
    _, val_loss = train_and_eval_delta(
        dataset=dataset,
        window_size=5,
        activations=activations,
        norm=False,
        num_layer=num_layers,
        dropout=dropout,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        lr_val=lr_val,
        l2_val=l2_val,
        batch_size=batch_size,
        combo=combo_map[model_str],
        counter=trial.number,
        seed=42,
        results_csv=results_csv,
        num_buckets=num_buckets
    )
    
    return val_loss

def main():
    load_dotenv()
    
    for num_buckets in [10]:
        print(f'--- Generating embeddings for num_buckets={str(num_buckets)} ---')
            
        
        # Ensure results directory exists
        results_csv = f'data/output/TopERTesting/data/bayesian_training_results_{str(num_buckets)}buckets_delta.csv'
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        
        datasets = [
            'networkbancor', 'networkadex', 'networkcentra', 
            'networkcoindash', 'mathoverflow', 'Reddit_B', 'networkaragon', 
            'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd', 'tgbl-wiki'
        ]
        
        activations = activation_map['Degree']

        for ds in datasets:
            
            print(f"--- Optimizing: {ds} ---")
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
            study.optimize(lambda trial: objective(trial, ds, activations, results_csv, num_buckets), n_trials=30)
            print(f"Finished {ds}. Best Loss: {study.best_value}")

if __name__ == "__main__":
    main()