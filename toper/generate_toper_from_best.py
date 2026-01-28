import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import pickle
import ast

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from torch.utils.data import DataLoader
from nn.custom_model import Decoder
from torch.utils.data import DataLoader, Dataset
from utils.utils import Utils
from utils.dataset import DeltaEmbeddingDataset
from utils.visualizer import Visualizer

# Define the new results path
RESULTS_CSV = 'data/output/TopERTesting/data/bayesian_training_results.csv'


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


def train_and_eval_delta(dataset, activations, window_size, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed, num_buckets):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 0
    input_dim = 0
    patience = 25
    num_epochs = 500
    
    run_name = dataset
    activation_name = ""
    
    # Set up embeddings
    embeddings = None
    using_weight = False 
    for activation in activations:
        data, labels = my_loader.load_data(dataset, activation, include_weights=using_weight, num_buckets=num_buckets)
        embeddings = my_utils.concat_embeddings(embeddings, data)
        activation_name += activation + '_'
        
    if not using_weight:
        input_dim = 2 * num_buckets * len(activations)
        output_dim = input_dim
    else:
        input_dim = 3 * num_buckets * len(activations)
        output_dim = input_dim
        
    run_name = run_name + '_'+ activation_name + str(counter)
        
    # Split data 80/10/10
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
     
            
    no_improvement_counter = 0
    # Initialize model and move to device
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    # Move criterion to device
    criterion = nn.MSELoss().to(device)
    
    curr_batch_best_loss = float('inf')    
    best_vectors = None  # To store the winners
      
    for epoch in range(num_epochs):
        # 1. TRAIN
        model.train()
        epoch_loss = 0
        for i, (x, y_delta, x_last) in enumerate(train_loader):
            x, y_delta, x_last = x.to(device), y_delta.to(device), x_last.to(device)
            optimizer.zero_grad()
            predicted_delta = model(x)[:, -1, :].squeeze(1)
            loss = criterion(predicted_delta, y_delta.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 2. VALIDATE
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for i, (x, y_delta, x_last) in enumerate(valid_loader):
                x, y_delta, x_last = x.to(device), y_delta.to(device), x_last.to(device)
                predicted_delta = model(x)[:, -1, :].squeeze(1)
                valid_loss += criterion(predicted_delta, y_delta.float()).item()
        valid_loss /= len(valid_loader)

        # 3. CHECK IMPROVEMENT & GENERATE VECTORS
        if valid_loss <= curr_batch_best_loss:
            curr_batch_best_loss = valid_loss
            no_improvement_counter = 0

            # Add first few vectors to preserve length
            temp_vectors = []
            for k in range(window_size):
                temp_vectors.append(embeddings[k]) 
            
            # FIX: Always use the full sequence for inference
            if norm:
                # Normalize the full embeddings using the training stats to avoid data leakage
                inf_data, _, _ = my_utils.normalize_embeddings(embeddings, X_val, X_test)
            else:
                inf_data = embeddings
            
            # Create the loader with the full dataset
            full_loader = DataLoader(DeltaEmbeddingDataset(inf_data, k=window_size), batch_size=1, shuffle=False)
            
            with torch.no_grad():
                for x_inf, _, _ in full_loader:
                    x_inf = x_inf.to(device)
                    delta_pred = model(x_inf)[:, -1, :].squeeze(1).cpu().numpy()
                    temp_vectors.append(temp_vectors[-1] + delta_pred[0])
            
            best_vectors = np.array(temp_vectors)
        
        else:
            if epoch >= 100:
                no_improvement_counter += 1
            
        if no_improvement_counter >= patience:
            break

    # Generate figures
    # Real vectors in embeddings
    # Take the best predicted vectors
    pred_df_discrete = pd.DataFrame(best_vectors)
    real_df_discrete = pd.DataFrame(embeddings)
    pred_col_nodes = pred_df_discrete.iloc[:, -2]  # Nodes
    real_col_nodes = real_df_discrete.iloc[:, -2]  # Nodes
    pred_col_edges = pred_df_discrete.iloc[:, -1]  # Edges
    real_col_edges = real_df_discrete.iloc[:, -1]  # Edges
    figures_output_path = 'data/output/figures/PredvsReal/'
    os.makedirs(figures_output_path, exist_ok=True)
    Visualizer.plot_scatter(pred_col_nodes, real_col_nodes, figures_output_path + f'{dataset}_Nodes.png', mode="nodes")
    Visualizer.plot_scatter(pred_col_edges, real_col_edges, figures_output_path + f'{dataset}_Edges.png', mode="edges")
    

    # 4. SAVE BEST OUTPUTS
    # Save the winner vectors to PKL
    pickle_dir = f'data/input/cached/{dataset}/predValues/'
    os.makedirs(pickle_dir, exist_ok=True)
    pickle_path = os.path.join(pickle_dir, f"{dataset}_toper_noweight_{num_buckets}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(best_vectors, f)
        

def main():
    for num_buckets in [10]:
        results_csv = f'data/output/TopERTesting/data/bayesian_training_results_{str(num_buckets)}buckets.csv'
        if not os.path.exists(results_csv):
            print(f"Error: {results_csv} not found.")
            return

        # 1. Read the CSV and find the best trial for each dataset
        df = pd.read_csv(results_csv)
        
        # Sort by valid_loss (ascending) and take the first (best) for each dataset
        best_trials = df.sort_values('valid_loss').groupby('dataset').first().reset_index()
        
        print(f"Found best trials for {len(best_trials)} datasets.")

        # 2. Iterate through the best trials and re-run training/generation
        for _, row in best_trials.iterrows():
            dataset = row['dataset']
            print(f"\n>>> Re-running best trial for: {dataset} and {num_buckets} buckets (Best Val Loss: {row['valid_loss']:.6f})")

            # Map string combo back to list if necessary
            # Depending on how it was saved, it might be a string like "['LSTM', 'FC']"
            combo_val = row['model']
            if isinstance(combo_val, str):
                import ast
                combo_list = ast.literal_eval(combo_val)
            else:
                combo_list = combo_val

            # 3. Call your training function with the exact best parameters
            # This will trigger the epoch loop, find the best moment, and save the .pkl
            train_and_eval_delta(
                dataset=dataset,
                activations=activation_map['Degree'], # Adjust if you used multiple
                window_size=int(row['window_size']),
                norm=row['normalization'],
                num_layer=int(row['num_layers']),
                dropout=row['dropout'],
                hidden_1=int(row['hidden_1']),
                hidden_2=int(row['hidden_2']), 
                lr_val=row['learning_rate'],
                l2_val=row['l2_regularization'],
                batch_size=int(row['batch_size']),
                combo=combo_list,
                counter="FINAL_GEN", # Marker for run_name
                seed=42,
                num_buckets=num_buckets
            )
            
            print(f"Completed {dataset}")

if __name__ == "__main__":
    main()