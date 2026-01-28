import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import optuna
import sqlite3
import re
import matplotlib.pyplot as plt 
import os
from dotenv import load_dotenv

    
# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from utils.dataset import EmbeddingDataset
from utils.dataset import DeltaEmbeddingDataset
from nn.custom_model import Decoder
from torch.utils.data import DataLoader, Dataset
from utils.utils import Utils

import wandb


csv_file_path = os.path.abspath('data/output/results/ProbabilityTesting/data/testing_probs_deltas.csv')

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['run_id', 'dataset', 'activation', 'seed', 'window_size', 'normalization', 'hidden_1', 'hidden_2', 'learning_rate', 'dropout', 'l2_regularization', 'batch_size', 'num_layers', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss', 'train_avg_norm', 'val_avg_norm', 'test_avg_norm', 'train_avg_cosine_similarity','val_avg_cosine_similarity', 'test_avg_cosine_similarity',]).to_csv(csv_file_path, index=False)

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


def train_and_eval_delta(dataset, window_size, num_back, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 6
    input_dim = 6
    patience = 25
    num_epochs = 500
    
    run_name = dataset
    
    # Load the probabilities for prediction
    probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')  
    if norm:
        embeddings = probabilities_df.apply(
            lambda row: my_utils.normalize_vector_by_groups(row.values),
            axis=1
        )
        embeddings = np.vstack(embeddings.values).astype(np.float32) 
    else:
        embeddings = probabilities_df.values.astype(np.float32)
        
        
    run_name = run_name + '_' + str(counter) 
        
    # Split data 80/10/10
    n = len(embeddings)
    train_end = int(0.8 * n)
    val_start = train_end - window_size
    val_end = int(0.9 * n)
    test_start = val_end - window_size

    X_train = embeddings[:train_end]
    X_val = embeddings[val_start:val_end]
    X_test = embeddings[test_start:]

    # Use the DeltaEmbeddingDataset class for training and validation loaders
    train_dataset = DeltaEmbeddingDataset(X_train, k=window_size)
    valid_dataset = DeltaEmbeddingDataset(X_val, k=window_size)
    
    # We still need the raw test data to reconstruct the final vectors
    test_dataset = DeltaEmbeddingDataset(X_test, k=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                                    
    # Initialize wandb
    run = wandb.init(
        project="probs_deltas_pred", 
        name = run_name, 
        config={
        'dataset': dataset,
        'hidden_1': hidden_1,
        'hidden_2': hidden_2,
        'window_size': window_size,
        'num_layers': num_layer,
        'dropout': dropout,
        'l2_regularization': l2_val,
        'hidden_size_rnn': hidden_1,
        'learning_rate': lr_val,
        'seed': seed,
        'normalization': norm,
        'model': combo
        },
        reinit=True)
            
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
        
        wandb.log(to_log)

        if valid_loss <= curr_batch_best_loss:
            curr_batch_best_loss = valid_loss
            
            best_moment_row = {
                'run_id': run.name,
                'dataset': dataset,
                'seed': seed,
                'window_size': window_size,
                'normalization': norm,
                'hidden_1': hidden_1,
                'hidden_2': hidden_2,
                'learning_rate': lr_val,
                'dropout': dropout,
                'l2_regularization': l2_val,
                'batch_size': batch_size,
                'num_layers': num_layer,
                'combo': combo,
                'trained_epochs': epoch + 1,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss,
                'train_avg_norm': train_avg_norm, 
                'val_avg_norm': val_avg_norm, 
                'test_avg_norm': test_avg_norm, 
                'train_avg_cosine_similarity': train_avg_cosine_similarity,
                'val_avg_cosine_similarity': val_avg_cosine_similarity, 
                'test_avg_cosine_similarity': test_avg_cosine_similarity,
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
    return best_moment_row['train_loss'], best_moment_row['valid_loss'], best_moment_row['test_loss'], best_pred_embeddings, best_real_embeddings


if __name__ == "__main__":
    # Load .env file
    load_dotenv()

    # Access API key
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY not found. Please set it in your .env file.")

    os.environ["WANDB_API_KEY"] = wandb_api_key

    my_utils = Utils()
    trial_num = -1
    trial_params_map = {}

    dropout_list = [0, 0.10]
    hidden_1_list = [64, 128, 256]
    hidden_2_list = [32, 64, 128]
    num_layers_list = [2, 3, 4]
    lr_val_list = [1e-4, 1e-3, 1e-2]
    l2_val_list = [0, 1e-5]
    batch_size_list = [16, 32]
    model_list = [
        "['LSTM', 'FC']",
        "['GRU', 'FC']",
        "['LSTM', 'GRU', 'FC']",
        "['LSTM', 'MLP']", 
        "['GRU', 'MLP']", 
        "['LSTM', 'GRU', 'MLP']"
    ]

    for batch_size in batch_size_list:
        for num_layer in num_layers_list:
            for dropout in dropout_list:
                for hidden_1 in hidden_1_list:
                    for hidden_2 in hidden_2_list:
                        for lr_val in lr_val_list:
                            for l2_val in l2_val_list:
                                for combo in model_list:
                                    for norm in [True, False]:
                                        trial_num += 1
                                        trial_params_map[trial_num] = {
                                            "batch_size": batch_size,
                                            "num_layers": num_layer,
                                            "dropout": dropout,
                                            "hidden_1": hidden_1,
                                            "hidden_2": hidden_2,
                                            "lr_val": lr_val,
                                            "l2_val": l2_val,
                                            "combo": combo,
                                            "norm": norm,
                                        }

    api = wandb.Api()
    entity = "rbuck8339-future-data-lab"
    project = "probs_deltas_pred"

    runs = api.runs(f"{entity}/{project}")

    dataset_best_trials = []

    for run in runs:
        run_name = run.name  # e.g., "mathoverflow_451"
        dataset_match = re.match(r"(.+?)_(\d+)$", run_name)
        if dataset_match:
            dataset_name = dataset_match.group(1)
            trial_num = int(dataset_match.group(2))
            train_loss = run.summary.get("train_loss", None)
            if train_loss is not None:
                dataset_best_trials.append({
                    "dataset": dataset_name,
                    "trial_num": trial_num,
                    "train_loss": train_loss
                })

    df = pd.DataFrame(dataset_best_trials)

    best_trials = df.loc[df.groupby("dataset")["train_loss"].idxmin()].reset_index(drop=True)


    for _, row in best_trials.iterrows():
        dataset = row["dataset"]
        trial_num = row["trial_num"]

        if trial_num not in trial_params_map:
            print(f"⚠️ Warning: trial {trial_num} not in param map, skipping {dataset}")
            continue

        params = trial_params_map[trial_num]

        train_loss, val_loss, test_loss, pred_embeddings, real_embeddings = train_and_eval_delta(
            dataset=dataset,
            window_size=7,
            num_back=5,
            norm=params["norm"],
            num_layer=params["num_layers"],
            dropout=params["dropout"],
            hidden_1=params["hidden_1"],
            hidden_2=params["hidden_2"],
            lr_val=params["lr_val"],
            l2_val=params["l2_val"],
            batch_size=params["batch_size"],
            combo=combo_map[params["combo"]],
            counter=999,
            seed=42,
        )

        print(f"Dataset: {dataset}")
        print(f"Best Trial: {trial_num}")
        print(f'Train loss was: {train_loss}')
        print(f'Valid loss was: {val_loss}')
        print(f' Test loss was: {test_loss}')
        
        save_dir = f"data/output/figures/deltaProbsNorm/{dataset}"
        os.makedirs(save_dir, exist_ok=True)
        
        prob_types = {
            "prob_old_nodes": 0,
            "prob_new_nodes": 1,
            "prob_oo": 2,
            "prob_nn": 3,
            "prob_on": 4,
            "prob_oon": 5, 
        }

        # Create list of column names sorted by their index
        col_names = [name for name, idx in sorted(prob_types.items(), key=lambda x: x[1])]

        # Convert embeddings to numpy arrays
        best_pred_embeddings = np.array(pred_embeddings)  # shape (num_samples, 6)
        best_real_embeddings = np.array(real_embeddings)  # shape (num_samples, 6)

        norm_pred_embeddings = np.array([my_utils.normalize_vector_by_groups(row) for row in best_pred_embeddings])
        norm_real_embeddings = np.array([my_utils.normalize_vector_by_groups(row) for row in best_real_embeddings])


        norm_pred_csv_path = os.path.join(save_dir, "norm_pred_embeddings.csv")
        norm_real_csv_path = os.path.join(save_dir, "norm_real_embeddings.csv")
        pd.DataFrame(norm_pred_embeddings[:, :len(col_names)], columns=col_names).to_csv(norm_pred_csv_path, index=False)
        pd.DataFrame(norm_real_embeddings[:, :len(col_names)], columns=col_names).to_csv(norm_real_csv_path, index=False)

        print(f"Saved normalized predictions to {norm_pred_csv_path}")
        print(f"Saved normalized real embeddings to {norm_real_csv_path}")

        os.makedirs(os.path.join(save_dir, "scatter"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "line"), exist_ok=True)

        # Plot normalized columns against each other
        for i, name in enumerate(col_names):
            # Scatter plot
            scatter_path = os.path.join(save_dir, f"scatter/Norm_Pred_vs_Real_{name}.png")
            plot_scatter(best_real_embeddings[:, i], best_pred_embeddings[:, i], scatter_path, mode="nodes" if i < 2 else "edges")

            # Line Plot
            line_path = os.path.join(save_dir, f"line/Norm_Pred_vs_Real_{name}_line.png")
            my_visualizer.plot_line_graph(norm_real_embeddings[:, i], norm_pred_embeddings[:, i], name, save_path=line_path)


        # Save CSVs with proper column names
        pred_csv_path = os.path.join(save_dir, "best_pred_embeddings.csv")
        real_csv_path = os.path.join(save_dir, "best_real_embeddings.csv")
        pd.DataFrame(best_pred_embeddings[:, :len(col_names)], columns=col_names).to_csv(pred_csv_path, index=False)
        pd.DataFrame(best_real_embeddings[:, :len(col_names)], columns=col_names).to_csv(real_csv_path, index=False)

        print(f"Saved predictions to {pred_csv_path}")
        print(f"Saved real embeddings to {real_csv_path}")

        # Plot each column against each other using names
        for i, name in enumerate(col_names):
            # Scatter plot
            scatter_path = os.path.join(save_dir, f"scatter/Pred_vs_Real_{name}.png")
            plot_scatter(best_real_embeddings[:, i], best_pred_embeddings[:, i], scatter_path, mode="nodes" if i < 2 else "edges")
            
            # Line Plot
            line_path = os.path.join(save_dir, f"line/Pred_vs_Real_{name}_line.png")
            my_visualizer.plot_line_graph(best_real_embeddings[:, i], best_pred_embeddings[:, i], name, save_path=line_path)