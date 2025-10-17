import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from dotenv import load_dotenv
print(torch.cuda.is_available())

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.dataset import DeltaEmbeddingDataset
from nn.custom_model import Decoder
from torch.utils.data import DataLoader, Dataset
from utils.utils import Utils
from utils.visualizer import Visualizer  # Assuming this is where your class lives


import wandb


RESULTS_PATH = 'data/input/cached/'
TRIALS_HISTORY_PATH = os.path.abspath('data/output/results/ProbabilityTesting/data/testing_toper_plus_probs.csv')

prob_types = {
    "prob_old_nodes": 0,
    "prob_new_nodes": 1,
    "prob_oo": 2,
    "prob_nn": 3,
    "prob_on": 4,
    "prob_oon": 5,
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
    
using_20_dim = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_and_eval(dataset, window_size, num_back, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 26
    input_dim = 26
    
    # i.e. we are including weight in the TopER vector
    if not using_20_dim:
        input_dim += 10
        output_dim += 10
    
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
        
    # Load the TopER and concatenate
    toper_embeddings, labels = my_loader.load_data(dataset, activation='Degree', include_weights=(not using_20_dim))
    embeddings = my_utils.concat_embeddings(toper_embeddings, embeddings)
        
    run_name = run_name + '_' + str(counter) 
        
    # Split data 80/10/10
    n = len(embeddings)
    train_end = int(0.8 * n)
    val_start = train_end - window_size
    val_end = int(0.9 * n)
    test_start = val_end - window_size

    X_train = np.array(embeddings[:train_end], dtype=np.float32)
    X_val = np.array(embeddings[val_start:val_end], dtype=np.float32)
    X_test = np.array(embeddings[test_start:], dtype=np.float32)

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
        project="toper_plus_probs", 
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
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout]).to(device)
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
            x = x.to(device)
            y_delta = y_delta.to(device)
            x_last = x_last.to(device)
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
            
            pred_embeddings.extend(predicted_embedding)
            real_embeddings.extend(real_embedding)

            for j in range(len(x)):
                try:
                    train_cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding[j], real_embedding[j]))
                except:
                    train_cosine_similarities.append(float('nan'))
                try:
                    train_norms.append(my_utils.compute_distances(predicted_embedding[j], real_embedding[j]))
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
                x = x.to(device)
                y_delta = y_delta.to(device)
                x_last = x_last.to(device)
                predicted_delta = model(x)
                predicted_delta = predicted_delta[:, -1, :]
                predicted_delta = predicted_delta.squeeze(1)
                y_delta = y_delta.float()
                loss = criterion(predicted_delta, y_delta)
                valid_loss += loss.item()
                
                # Reconstruct full vectors for metrics and logging
                predicted_embedding = x_last + predicted_delta
                real_embedding = x_last + y_delta
                
                pred_embeddings.extend(predicted_embedding)
                real_embeddings.extend(real_embedding)

                for j in range(len(x)):
                    try:
                        val_cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding[j], real_embedding[j]))
                    except:
                        val_cosine_similarities.append(float('nan'))
                    try:
                        val_norms.append(my_utils.compute_distances(predicted_embedding[j], real_embedding[j]))
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
                x = x.to(device)
                y_delta = y_delta.to(device)
                x_last = x_last.to(device)
            
                predicted_delta = model(x)
                predicted_delta = predicted_delta[:, -1, :]
                predicted_delta = predicted_delta.squeeze(1)
                y_delta = y_delta.float()
                loss = criterion(predicted_delta, y_delta)
                test_loss += loss.item()
                
                # Reconstruct full vectors for metrics and logging
                predicted_embedding = x_last + predicted_delta
                real_embedding = x_last + y_delta
                
                pred_embeddings.extend(predicted_embedding)
                real_embeddings.extend(real_embedding)
                
                for j in range(len(x)):
                    try:
                        test_cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding[j], real_embedding[j]))
                    except:
                        test_cosine_similarities.append(float('nan'))
                    try:
                        test_norms.append(my_utils.compute_distances(predicted_embedding[j], real_embedding[j]))
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





def main():
    my_visualizer = Visualizer()
    my_utils = Utils()
    
    df = pd.read_csv(TRIALS_HISTORY_PATH)
    
    df["train_loss"] = df["train_loss"].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    df["valid_loss"] = df["valid_loss"].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    
    # Filter out norm = True
    df = df[df['norm'] == False]
    
    # Strip the _# suffix to create a trial group
    df["trial_group"] = df["run_id"].str.rsplit("_", n=1).str[1]

    best_trials = df.loc[
        df.groupby("dataset")["train_loss"].idxmin()
    ].reset_index(drop=True)
                
    # Loop and print results as variable assignments
    for _, row in best_trials.iterrows():
        print(f"\n🔧 Best Trial for Dataset: {row['run_id']}\n{'-'*40}")

        # Extract parameters
        window_size = int(row['seed']) if int(row['window_size']) == 42 else int(row['window_size'])
        num_back = '5'
        dropout = float(row['dropout'])
        hidden_1 = int(row['hidden_1'])  
        hidden_2 = int(row['hidden_2'])  
        num_layers = int(row['num_layers'])
        lr_val = float(row['learning_rate'])
        l2_val = float(row['l2_regularization'])
        batch_size = int(row['batch_size'])
        model_str = row['combo']
        model = combo_map[model_str]
        dataset=row['dataset']
        norm=row['norm']
        
        train_loss, val_loss, test_loss, pred_embeddings, real_embeddings = train_and_eval(
            dataset=dataset,
            window_size=window_size,
            num_back=num_back,
            norm=norm,
            num_layer=num_layers,
            dropout=dropout,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            lr_val=lr_val,
            l2_val=l2_val,
            batch_size=batch_size,
            combo=model,
            counter=999,
            seed=42,
        )
    
        # Display results
        print(f'Train Loss: {train_loss}')
        print(f'Valid Loss: {val_loss}')
        print(f'Test Loss:  {test_loss}')

        res_path = RESULTS_PATH + dataset + '/PredProbsWithToper/'

        os.makedirs(res_path, exist_ok=True)
            
        pred_df_discrete = pd.DataFrame(pred_embeddings).iloc[:, 20:26]
        real_df_discrete = pd.DataFrame(real_embeddings).iloc[:, 20:26]
        
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
            print(f"Average row-wise RMSE: {avg_rmse}")
            print(f"Average row-wise MAE: {avg_mae}")
            print(f"Total Loss across all vectors: {total_loss}")

        # Run for each split
        for split_name, (pred_split, real_split) in splits.items():
            compute_metrics(pred_split, real_split, split_name)

        os.makedirs(res_path, exist_ok=True)
        plot_dir = os.path.join(res_path, 'normFalse/probs')
        os.makedirs(plot_dir, exist_ok=True)
        
        pred_df_discrete.to_csv(res_path + '/normFalse/pred_discrete_probs.csv', index=False)
        
        real_df_norm = real_df_discrete.apply(
            lambda row: my_utils.normalize_vector_by_groups(row.values),
            axis=1
        )
        # Normalize them
        pred_df_norm = pred_df_discrete.apply(
            lambda row: my_utils.normalize_vector_by_groups(row.values),
            axis=1
        )
        pred_df_norm.to_csv(res_path + '/normFalse/pred_norm_probs.csv', index=False)
        
        
        for prob_name, col_idx in prob_types.items():
            true_vals_discrete = real_df_discrete.iloc[:, col_idx]
            pred_vals_discrete = pred_df_discrete.iloc[:, col_idx]

            true_vals_norm = real_df_norm.iloc[:, col_idx]
            pred_vals_norm = pred_df_norm.iloc[:, col_idx]

            # Plot discrete comparison
            my_visualizer.plot_line_graph(true_vals_discrete, pred_vals_discrete, prob_type=prob_name, xlabel="Time Index", ylabel="Probability", title=f"True vs Predicted (Discrete) — {prob_name}", save_path=os.path.join(plot_dir, f"{prob_name}_discrete.png"))

            # Plot normalized comparison
            my_visualizer.plot_line_graph(true_vals_norm, pred_vals_norm, prob_type=prob_name, xlabel="Time Index", ylabel="Normalized Probability", title=f"True vs Predicted (Normalized) — {prob_name}", save_path=os.path.join(plot_dir, f"{prob_name}_norm.png"))        