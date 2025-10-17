import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from nn.custom_model import Decoder
from utils.loader import Loader
from utils.dataset import EmbeddingDataset
from utils.dataset import DeltaEmbeddingDataset
from torch.utils.data import DataLoader
from nn.custom_model import Decoder
from torch.utils.data import DataLoader, Dataset
from utils.utils import Utils
from utils.visualizer import Visualizer


# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

from utils.utils import Utils


RESULTS_PATH = 'data/input/cached/'
TRIALS_HISTORY_PATH = 'data/output/results/RegressionTesting/data/embedding_testing_bayesian_regression_20dim_deltas_trainlossonly.csv'


using_20dim = True

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


def train_and_eval_delta(dataset, activations, window_size, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed):
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
    using_20dim = True # Assuming you're using 20-dim vectors
    for activation in activations:
        data, labels = my_loader.load_data(dataset, activation, include_weights=(not using_20dim))
        embeddings = my_utils.concat_embeddings(embeddings, data)
        activation_name += activation + '_'
        
    if using_20dim:
        input_dim = 20 * len(activations)
        output_dim = input_dim
    else:
        input_dim = 30 * len(activations)
        output_dim = input_dim
        
    run_name = run_name + '_'+ activation_name + str(counter) 
        
    # Split data 80/10/10
    n = len(embeddings)
    train_end = int(0.8 * n)
    val_start = train_end - window_size
    val_end = int(0.9 * n)
    test_start = val_end - window_size

    embeddings = np.array([np.array(e, dtype=np.float32) for e in embeddings])

    X_train = embeddings[:train_end]
    X_val = embeddings[val_start:val_end]
    X_test = embeddings[test_start:]
    
    if norm:
        print('Normalizing embeddings')
        X_train_scaled, X_val_scaled, X_test_scaled = my_utils.normalize_embeddings(X_train, X_val, X_test)
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test

    # Use the DeltaEmbeddingDataset class for training and validation loaders
    train_dataset = DeltaEmbeddingDataset(X_train_scaled, k=window_size)
    valid_dataset = DeltaEmbeddingDataset(X_val_scaled, k=window_size)
    
    # We still need the raw test data to reconstruct the final vectors
    test_dataset = DeltaEmbeddingDataset(X_test_scaled, k=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                                    
    # Initialize wandb
    # run = wandb.init(
    #     project="bayesian_testing_regression_updated_datasets", 
    #     name = run_name, 
    #     config={
    #     'dataset': dataset,
    #     'activation': activation_name,
    #     'window_size': window_size,
    #     'num_layers': num_layer,
    #     'dropout': dropout,
    #     'l2_regularization': l2_val,
    #     'hidden_size_rnn': hidden_1,
    #     'learning_rate': lr_val,
    #     'seed': seed,
    #     'normalization': norm,
    #     'model': combo
    #     },
    #     reinit=True)
            
    no_improvement_counter = 0
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    criterion = nn.MSELoss()
    
    curr_batch_best_loss = float('inf')    
      
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        cosine_similarities = []
        train_cosine_similarities = []
        train_norms = []
        norms = []
        time_index = 0
        pred_embeddings = []
        real_embeddings = []

        for i, (x, y_delta, x_last) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_delta = model(x)
            predicted_delta = predicted_delta[:, -1, :]
            predicted_delta = predicted_delta.squeeze(1)
            loss = criterion(predicted_delta, y_delta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Reconstruct full vectors for metrics and logging
            predicted_embedding = x_last + predicted_delta
            real_embedding = x_last + y_delta
            
            pred_embeddings.extend(predicted_embedding.detach().cpu().numpy())
            real_embeddings.extend(real_embedding.detach().cpu().numpy())

            for j in range(len(x)):
                try:
                    train_cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                except:
                    train_cosine_similarities.append(float('nan'))
                try:
                    train_norms.append(my_utils.compute_distances(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                except:
                    train_norms.append(float('nan'))
        
        train_avg_norm = np.nanmean(train_norms)
        train_avg_cosine_similarity = np.nanmean(train_cosine_similarities)
        train_loss = (epoch_loss / len(train_loader))
                            
        # Validation
        model.eval()
        valid_loss = 0
        val_cosine_similarities = []
        val_norms = []
        # We will collect the actual vectors here

        with torch.no_grad():
            for i, (x, y_delta, x_last) in enumerate(valid_loader):
                predicted_delta = model(x)
                predicted_delta = predicted_delta[:, -1, :]
                predicted_delta = predicted_delta.squeeze(1)
                y_delta = y_delta.float()
                loss = criterion(predicted_delta, y_delta)
                valid_loss += loss.item()
                
                # Reconstruct full vectors for metrics and logging
                predicted_embedding = x_last + predicted_delta
                real_embedding = x_last + y_delta
                
                pred_embeddings.extend(predicted_embedding.detach().cpu().numpy())
                real_embeddings.extend(real_embedding.detach().cpu().numpy())

                for j in range(len(x)):
                    try:
                        val_cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                    except:
                        val_cosine_similarities.append(float('nan'))
                    try:
                        val_norms.append(my_utils.compute_distances(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                    except:
                        val_norms.append(float('nan'))
            
        valid_loss /= len(valid_loader)
        val_avg_norm = np.nanmean(val_norms)
        val_avg_cosine_similarity = np.nanmean(val_cosine_similarities)
        
        # Testing
        model.eval()
        test_loss = 0
        test_cosine_similarities = []
        test_norms = []

        with torch.no_grad():
            for i, (x, y_delta, x_last) in enumerate(test_loader):
                predicted_delta = model(x)
                predicted_delta = predicted_delta[:, -1, :]
                predicted_delta = predicted_delta.squeeze(1)
                y_delta = y_delta.float()
                loss = criterion(predicted_delta, y_delta)
                test_loss += loss.item()
                
                # Reconstruct full vectors for metrics and logging
                predicted_embedding = x_last + predicted_delta
                real_embedding = x_last + y_delta
                
                pred_embeddings.extend(predicted_embedding.detach().cpu().numpy())
                real_embeddings.extend(real_embedding.detach().cpu().numpy())
                
                for j in range(len(x)):
                    try:
                        test_cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                    except:
                        test_cosine_similarities.append(float('nan'))
                    try:
                        test_norms.append(my_utils.compute_distances(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                    except:
                        test_norms.append(float('nan'))
            
        test_loss /= len(test_loader)
        test_avg_norm = np.nanmean(test_norms)
        test_avg_cosine_similarity = np.nanmean(test_cosine_similarities)
                
        to_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'test_loss': test_loss,
            'train_avg_norm': train_avg_norm,
            'train_avg_cosine_similarity': train_avg_cosine_similarity,
            'val_avg_norm': val_avg_norm,
            'val_avg_cosine_similarity': val_avg_cosine_similarity,
            'test_avg_norm': test_avg_norm,
            'test_avg_cosine_similarity': test_avg_cosine_similarity,
        }
        
        # wandb.log(to_log)

        if valid_loss <= curr_batch_best_loss:
            curr_batch_best_loss = valid_loss
            
            best_moment_row = {
                'run_id': 999,
                'dataset': dataset,
                'activation': activation_name,
                'window_size': window_size,
                'seed': seed,
                'normalization': norm,
                'hidden_size_rnn': hidden_1,
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
            }
            # We save the reconstructed and real vectors here
            best_pred_embeddings = pred_embeddings
            best_real_embeddings = real_embeddings
            
        if epoch >= 100:
            if valid_loss <= curr_batch_best_loss:
                no_improvement_counter = 0
                curr_batch_best_loss = valid_loss
                best_train_loss = train_loss 
                best_val_loss = valid_loss
                best_pred_embeddings = pred_embeddings
                best_real_embeddings = real_embeddings
            else:
                no_improvement_counter += 1
                
            if no_improvement_counter == patience:
                print(f'Training ending at epoch number: {epoch + 1}')
                break
    
    # We return the best predicted and real embeddings (full vectors)
    return best_moment_row['train_loss'], best_moment_row['valid_loss'], best_pred_embeddings, best_real_embeddings


def main():
    my_visualizer = Visualizer()
    
    df = pd.read_csv(TRIALS_HISTORY_PATH)
    
    df["train_loss"] = df["train_loss"].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    df["valid_loss"] = df["valid_loss"].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    
    # Strip the _# suffix to create a trial group
    df["trial_group"] = df["run_id"].str.rsplit("_", n=1).str[1]
    
    # Count entries per group
    group_counts = df.groupby("trial_group").size().reset_index(name="count")
    
    # Keep only groups with at least 13 entries
    valid_groups = group_counts[group_counts["count"] >= 12]["trial_group"]
    
    # Compute the mean score for valid groups only
    group_means = df[df["trial_group"].isin(valid_groups)].groupby("trial_group")["train_loss"].mean().reset_index()
    
    # Find the trial group with the lowest average score
    best_group = group_means.loc[group_means["train_loss"].idxmin()]["trial_group"]
    
    print(f"Best trial group across all datasets (>=13 entries): {best_group}\n{'-'*40}")
    
    # Optionally, display the best trials for each dataset within that group
    # Find the trial group with the lowest average score
    best_group = group_means.loc[group_means["train_loss"].idxmin()]["trial_group"]
    
    print(f"Best trial group across all datasets (>=13 entries): {best_group}\n{'-'*40}")
    
    # Optionally, display the best trials for each dataset within that group
    best_trials_group = df[df["trial_group"] == best_group]

    best_trials = best_trials_group.loc[
        best_trials_group.groupby("dataset")["train_loss"].idxmin()
    ].reset_index(drop=True)
    
    best_trials = df.loc[
        df.groupby("dataset")["train_loss"].idxmin()
    ].reset_index(drop=True)
                
    # Loop and print results as variable assignments
    for _, row in best_trials.iterrows():
        print(f"\n🔧 Best Trial for Dataset: {row['run_id']}\n{'-'*40}")

        # Extract parameters
        window_size = int(row['seed']) if int(row['window_size']) == 42 else int(row['window_size'])
        dropout = float(row['dropout'])
        hidden_1 = int(row['hidden_size_rnn'])  
        hidden_2 = int(0)
        num_layers = int(row['num_layers'])
        lr_val = float(row['learning_rate'])
        l2_val = float(row['l2_regularization'])
        batch_size = int(row['batch_size'])
        model_str = row['combo']
        model = combo_map[model_str]
        dataset=row['dataset']
        activations = activation_map[row['activation'][:-1]]
        
        
        train_loss, val_loss, pred_embeddings, real_embeddings = train_and_eval_delta(
            dataset=dataset,
            window_size=window_size,
            activations=activations,
            norm=False,
            num_layer=num_layers,
            dropout=dropout,
            hidden_1=hidden_1,
            hidden_2=64,
            lr_val=lr_val,
            l2_val=l2_val,
            batch_size=batch_size,
            combo=model,
            counter=999,
            seed=42,
        )
    
        loss_score = train_loss
        print(f"The old score was {row['train_loss']} and the new one was {loss_score}")

        res_path = RESULTS_PATH + dataset + '/PredTopER/'

        os.makedirs(res_path, exist_ok=True)
            
        pred_df_discrete = pd.DataFrame(pred_embeddings).iloc[:, :20]
        real_df_discrete = pd.DataFrame(real_embeddings).iloc[:, :20]
        
        # Form dataframes of pred, real, and a train_test split for graph construction
        n = len(pred_df_discrete)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        splits = {
            "Train": (pred_df_discrete.iloc[:train_end], real_df_discrete.iloc[:train_end]),
            "Validation": (pred_df_discrete.iloc[train_end:val_end], real_df_discrete.iloc[train_end:val_end]),
            "Test": (pred_df_discrete.iloc[val_end:], real_df_discrete.iloc[val_end:])
        }

        # Function to compute metrics
        def compute_metrics(pred_df, real_df, split_name):
            pred_col = pred_df.iloc[:, -2]
            real_col = real_df.iloc[:, -2]

            rmse = np.sqrt(mean_squared_error(real_col, pred_col))
            mae = mean_absolute_error(real_col, pred_col)

            pred = pred_df.to_numpy()
            real = real_df.to_numpy()

            # RMSE per row, then average
            rmse_per_row = np.sqrt(np.mean((pred - real) ** 2, axis=1))
            avg_rmse = np.mean(rmse_per_row)

            # MAE per row, then average
            mae_per_row = np.mean(np.abs(pred - real), axis=1)
            avg_mae = np.mean(mae_per_row)

            pred_tensor = torch.tensor(pred, dtype=torch.float32)
            real_tensor = torch.tensor(real, dtype=torch.float32)
            criterion = nn.MSELoss(reduction='sum')
            total_loss = criterion(pred_tensor, real_tensor).item()

            print(f"\n📊 {split_name} Results")
            print(f"Final Edge RMSE: {rmse}")
            print(f"Final Edge MAE: {mae}")
            print(f"Final Edge Mean: {pred_col.mean()}")
            print(f"Final Edge STDDEV: {pred_col.std()}")
            print(f"Average row-wise RMSE: {avg_rmse}")
            print(f"Average row-wise MAE: {avg_mae}")
            print(f"Total Loss across all vectors: {total_loss}")

        # Run for each split
        for split_name, (pred_split, real_split) in splits.items():
            compute_metrics(pred_split, real_split, split_name)

                
        real_part_len = int(len(real_df_discrete) * 0.7)
        pred_part_len = len(real_df_discrete) - real_part_len 
        real_part = real_df_discrete.iloc[:real_part_len, :].to_numpy()
        pred_part = pred_df_discrete.iloc[-pred_part_len:, :].to_numpy()
        hybrid_array = np.vstack([real_part, pred_part])
        real_pred_df_discrete = pd.DataFrame(hybrid_array)

        # Save the embeddings
        pred_df_discrete.to_csv(os.path.join(res_path, f"{dataset}_pred.csv"), index=False)
        real_df_discrete.to_csv(os.path.join(res_path, f"{dataset}_real.csv"), index=False) 
        real_pred_df_discrete.to_csv(os.path.join(res_path, f"{dataset}_train_test.csv"), index=False) 


        def print_pred_stats(pred_df):
            """
            Print mean and std deviation for each column in a prediction dataframe.
            """
            print(f"\n--- Predicted Statistics ---")
            for col in pred_df.columns:
                mean_val = pred_df[col].mean()
                std_val = pred_df[col].std()
                print(f"{col}: mean = {mean_val:.6f}, std = {std_val:.6f}")


        # --- Add this after creating pred_df_discrete and pred_df ---
        print_pred_stats(pred_df_discrete)
        
        pred_col_nodes = pred_df_discrete.iloc[:, -2]  # Nodes
        real_col_nodes = real_df_discrete.iloc[:, -2]  # Nodes
        pred_col_edges = pred_df_discrete.iloc[:, -1]  # Edges
        real_col_edges = real_df_discrete.iloc[:, -1]  # Edges

        scatter_path = 'data/output/figures/TopERScatter/'
        os.makedirs(res_path, exist_ok=True)

        my_visualizer.plot_scatter(pred_col_nodes, real_col_nodes, scatter_path + f'{dataset}_Nodes.png', mode="nodes")
        my_visualizer.plot_scatter(pred_col_edges, real_col_edges, scatter_path + f'{dataset}_Edges.png', mode="edges")
    
main()