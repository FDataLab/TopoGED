import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv

# visualization & metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path for your utils / nn modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.utils import Utils
from utils.dataset import DeltaEmbeddingDataset
from torch.utils.data import DataLoader
from utils.visualizer import Visualizer
from nn.custom_model import Decoder

import wandb


# Maps for parameters
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


class TopERGeneration:
    def __init__(self, with_grid_search=True, seed=42):
        self.with_grid_search = with_grid_search 
        self.seed = seed
        
        self.figures_output_path = 'data/output/figures/TopERScatter/'
        self.pred_embeddings_prefix = 'data/input/cached/'
        self.csv_output_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/output/results/RegressionTesting/data')), 'toper_generation_20dim_deltas.csv')
    
        if not os.path.isfile(self.csv_output_path):
            pd.DataFrame(columns=[
                'run_id', 'dataset', 'activation', 'window_size', 'seed', 'normalization',
                'hidden_1', 'hidden_2', 'learning_rate', 'dropout', 'l2_regularization',
                'batch_size', 'num_layers', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss',
                'train_avg_norm', 'val_avg_norm', 'test_avg_norm',
                'train_avg_cosine_similarity', 'val_avg_cosine_similarity', 'test_avg_cosine_similarity'
            ]).to_csv(self.csv_output_path, index=False)
    
        # helpers
        self.loader = Loader()
        self.utils = Utils()
        self.visualizer = Visualizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    
    def train_and_eval_delta(self, dataset, activations, window_size, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter):
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
            data, labels = self.loader.load_data(dataset, activation, include_weights=(not using_20dim))
            embeddings = self.utils.concat_embeddings(embeddings, data)
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
            X_train_scaled, X_val_scaled, X_test_scaled = self.utils.normalize_embeddings(X_train, X_val, X_test)
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
        run = wandb.init(
            project="toper_generation", 
            name = run_name, 
            config={
            'dataset': dataset,
            'activation': activation_name,
            'window_size': window_size,
            'num_layers': num_layer,
            'dropout': dropout,
            'l2_regularization': l2_val,
            'hidden_1': hidden_1,
            'hidden_2': hidden_2,
            'learning_rate': lr_val,
            'seed': self.seed,
            'normalization': norm,
            'model': combo
            },
            reinit=True)
                
        no_improvement_counter = 0
        model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
        criterion = nn.MSELoss().to(self.device)
        
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
                x, y_delta, x_last = x.to(self.device), y_delta.to(self.device), x_last.to(self.device)
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
                        train_cosine_similarities.append(self.utils.compute_cosine_similarity(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                    except:
                        train_cosine_similarities.append(float('nan'))
                    try:
                        train_norms.append(self.utils.compute_distances(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
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
                    x, y_delta, x_last = x.to(self.device), y_delta.to(self.device), x_last.to(self.device)
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
                            val_cosine_similarities.append(self.utils.compute_cosine_similarity(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                        except:
                            val_cosine_similarities.append(float('nan'))
                        try:
                            val_norms.append(self.utils.compute_distances(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
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
                    x, y_delta, x_last = x.to(self.device), y_delta.to(self.device), x_last.to(self.device)
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
                            test_cosine_similarities.append(self.utils.compute_cosine_similarity(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                        except:
                            test_cosine_similarities.append(float('nan'))
                        try:
                            test_norms.append(self.utils.compute_distances(predicted_embedding[j].detach().cpu().numpy(), real_embedding[j].detach().cpu().numpy()))
                        except:
                            test_norms.append(float('nan'))
                
            test_loss /= len(test_loader)
            test_avg_norm = np.nanmean(test_norms)
            test_avg_cosine_similarity = np.nanmean(test_cosine_similarities)
                    
            # We report RMSE:
            train_loss = np.sqrt(train_loss)
            valid_loss = np.sqrt(valid_loss)
            test_loss  = np.sqrt(test_loss)        
            
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
                    'run_id': counter,
                    'dataset': dataset,
                    'activation': activation_name,
                    'window_size': window_size,
                    'seed': self.seed,
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
                    'test_avg_cosine_similarity': test_avg_cosine_similarity
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
                
        pd.DataFrame([best_moment_row]).to_csv(self.csv_output_path, mode='a', header=False, index=False)
        
        # We return the best predicted and real embeddings (full vectors)
        return best_moment_row['train_loss'], best_moment_row['valid_loss'], best_pred_embeddings, best_real_embeddings

    
    def find_best_trials(self):
        df = pd.read_csv(self.csv_output_path)
        
        df["train_loss"] = df["train_loss"].replace([np.inf, -np.inf], np.finfo(np.float32).max)
        df["valid_loss"] = df["valid_loss"].replace([np.inf, -np.inf], np.finfo(np.float32).max)
        
        
        self.best_trials = df.loc[
            df.groupby("dataset")["train_loss"].idxmin()
        ].reset_index(drop=True)
    
    
    def generate_best(self):
        self.find_best_trials()
        for _, row in self.best_trials.iterrows():
            print(f"\n🔧 Best Trial for Dataset: {row['run_id']}\n{'-'*40}")

            # Extract parameters
            window_size = int(row['window_size'])
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
            activations = activation_map[row['activation'][:-1]]
            
            train_loss, val_loss, pred_embeddings, real_embeddings = self.train_and_eval_delta(
                dataset=dataset,
                window_size=window_size,
                activations=activations,
                norm=False,
                num_layer=num_layers,
                dropout=dropout,
                hidden_1=hidden_1,
                hidden_2=hidden_2,
                lr_val=lr_val,
                l2_val=l2_val,
                batch_size=batch_size,
                combo=model,
                counter=999,
            )
        
            res_path = self.pred_embeddings_prefix + dataset + '/PredTopER/'
            os.makedirs(res_path, exist_ok=True)
            os.makedirs(self.figures_output_path, exist_ok=True)   
            
            # In case we concatenated it with another embedding in the best trial
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
                # Convert to arrays
                pred = pred_df.to_numpy()
                real = real_df.to_numpy()

                # 1️⃣ Overall RMSE and MAE across all values
                total_rmse = np.sqrt(mean_squared_error(real, pred))
                total_mae = mean_absolute_error(real, pred)

                # 2️⃣ RMSE and MAE per sample (vector)
                rmse_per_row = np.sqrt(np.mean((pred - real) ** 2, axis=1))
                mae_per_row = np.mean(np.abs(pred - real), axis=1)

                avg_rmse = np.mean(rmse_per_row)
                avg_mae = np.mean(mae_per_row)

                # 3️⃣ Optional: last-column metrics if that’s meaningful for your dataset
                pred_col = pred_df.iloc[:, -2]
                real_col = real_df.iloc[:, -2]
                col_rmse = np.sqrt(mean_squared_error(real_col, pred_col))
                col_mae = mean_absolute_error(real_col, pred_col)

                print(f"\n📊 {split_name} Results")
                print(f"Total RMSE (global): {total_rmse:.6f}")
                print(f"Total MAE (global): {total_mae:.6f}")
                print(f"Mean per-sample RMSE: {avg_rmse:.6f}")
                print(f"Mean per-sample MAE: {avg_mae:.6f}")
                print(f"Last-column RMSE: {col_rmse:.6f}")
                print(f"Last-column MAE: {col_mae:.6f}")
                print(f"Last-column Mean: {pred_col.mean():.6f}")
                print(f"Last-column Std: {pred_col.std():.6f}")

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

            self.visualizer.plot_scatter(pred_col_nodes, real_col_nodes, self.figures_output_path + f'{dataset}_Nodes.png', mode="nodes")
            self.visualizer.plot_scatter(pred_col_edges, real_col_edges, self.figures_output_path + f'{dataset}_Edges.png', mode="edges")
        
        
    def grid_search(self):
        norm = False  # we won't use normalization
        window_size = 7  # Fixed window size
        dropout_list = [0, 0.1]
        hidden_1_list = [128, 256]
        hidden_2_list = [64, 128]
        num_layers_list = [2, 3]
        lr_val_list = [1e-4, 1e-3]
        l2_val_list = [1e-5, 1e-4]
        batch_size_list = [32, 64]
        activation_list =[
            'Degree',
            'Degree_Forman',
            'Degree_Weight',
            'Degree_Closeness',
        ]
        model_list = [
            "['LSTM', 'FC']",
            "['GRU', 'FC']",
            "['RNN', 'FC']",
            "['LSTM', 'GRU', 'MLP']"
        ]
        datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

        best_train_loss = float('inf')
        
        try:
            trial_num = 0  # Update as we go
            for batch_size in batch_size_list:
                for dropout in dropout_list:
                    for hidden_1 in hidden_1_list:
                        for hidden_2 in hidden_2_list:
                            for num_layers in num_layers_list:
                                for lr_val in lr_val_list:
                                    for l2_val in l2_val_list:
                                        for activation in activation_list:
                                            for model in model_list: 
                                                activations = activation_map[activation]
                                                model = combo_map[model]
                                                trial_num += 1
                                                
                                                for dataset in datasets: 
                                                    if dataset == 'Reddit_B' and 'Forman' in activations:
                                                        continue
                                                    
                                                    # Call your train_and_eval function for each dataset
                                                    res = self.train_and_eval_delta(
                                                        dataset=dataset,
                                                        window_size=window_size,
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
                                                        counter=trial_num,
                                                    )
                                                    
                                                    
        except Exception as e:
            print(f'Grid search failed with error: {e}')
            exit()
        
    
    def wandb_setup(self):
        # Load .env file
        load_dotenv()

        # Access API key
        wandb_api_key = os.getenv("WANDB_API_KEY")

        if wandb_api_key is None:
            raise ValueError("WANDB_API_KEY not found. Please set it in your .env file.")

        os.environ["WANDB_API_KEY"] = wandb_api_key
        
        wandb.init(project="Regression", name="toper_generation")
        
    
    def run(self):
        self.wandb_setup()
        
        if self.with_grid_search:
            self.grid_search()
            
        self.generate_best()
        
        
if __name__ == "__main__":
    my_generator = TopERGeneration(with_grid_search=True)
    my_generator.run()