import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import optuna
import sqlite3
import re
import matplotlib.pyplot as plt 
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path for your utils / nn modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.dataset import EmbeddingDataset
from utils.dataset import DeltaEmbeddingDataset
from nn.custom_model import Decoder
from torch.utils.data import DataLoader, Dataset
from utils.utils import Utils
from utils.visualizer import Visualizer

import wandb


# Maps for parameters
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

class ProbsGeneration:
    def __init__(self, with_grid_search=True, seed=42):
        self.with_grid_search = with_grid_search 
        self.seed = seed
        
        self.figures_output_path = 'data/output/figures/deltaProbsNorm/'
        self.pred_embeddings_prefix = 'data/input/cached/'
        self.csv_output_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/output/results/ProbabilityTesting/data')), 'probs_generation_normTrue.csv')
        
        if not os.path.isfile(self.csv_output_path):
            pd.DataFrame(columns=[
                'run_id', 'dataset', 'window_size', 'seed', 'normalization',
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
    
    
    def train_and_eval_delta(self, dataset, window_size, num_back, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter):
        # Setup
        output_dim = 6
        input_dim = 6
        patience = 25
        num_epochs = 500
        
        run_name = dataset
        
        # Load the probabilities for prediction
        probabilities_df = self.loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')  
        if norm:  # Chance to use proper loader at some point, one of them does normalized loading
            embeddings = probabilities_df.apply(
                lambda row: self.utils.normalize_vector_by_groups(row.values),
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
            'seed': self.seed,
            'normalization': norm,
            'model': combo
            },
            reinit=True)
                
        no_improvement_counter = 0
        model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
        model = model.to(self.device)
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
                    'run_id': run.name,
                    'dataset': dataset,
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
        
        pd.DataFrame([best_moment_row]).to_csv(self.csv_output_path, mode='a', header=False, index=False)
        
        # We return the best predicted and real embeddings (full vectors)
        return best_moment_row['train_loss'], best_moment_row['valid_loss'], best_moment_row['test_loss'], best_pred_embeddings, best_real_embeddings

    
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
            norm = row['normalization']
            seed = int(row['seed'])
            dropout = float(row['dropout'])
            hidden_1 = int(row['hidden_1'])  
            hidden_2 = int(row['hidden_2'])  
            num_layers = int(row['num_layers'])
            lr_val = float(row['learning_rate'])
            l2_val = float(row['l2_regularization'])
            batch_size = int(row['batch_size'])
            combo = row['combo']
            dataset=row['dataset']
            
            train_loss, val_loss, test_loss, pred_embeddings, real_embeddings = self.train_and_eval_delta(
                dataset=dataset,
                window_size=window_size, 
                num_back=5, 
                norm=norm,
                num_layer=num_layers, 
                dropout=dropout, 
                hidden_1=hidden_1, 
                hidden_2=hidden_2, 
                lr_val=lr_val, 
                l2_val=l2_val, 
                batch_size=batch_size,
                combo=combo_map[combo],
                counter=999, 
            )
        
            save_dir = self.pred_embeddings_prefix + dataset + '/PredProbabilitiesNorm/'
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

            norm_pred_embeddings = np.array([self.utils.normalize_vector_by_groups(row) for row in best_pred_embeddings])
            norm_real_embeddings = np.array([self.utils.normalize_vector_by_groups(row) for row in best_real_embeddings])

            norm_pred_csv_path = os.path.join(save_dir, "norm_pred_embeddings.csv")
            norm_real_csv_path = os.path.join(save_dir, "norm_real_embeddings.csv")
            pd.DataFrame(norm_pred_embeddings[:, :len(col_names)], columns=col_names).to_csv(norm_pred_csv_path, index=False)
            pd.DataFrame(norm_real_embeddings[:, :len(col_names)], columns=col_names).to_csv(norm_real_csv_path, index=False)

            os.makedirs(os.path.join(save_dir, "scatter"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "line"), exist_ok=True)

            # Plot normalized columns against each other
            for i, name in enumerate(col_names):
                # Scatter plot
                scatter_path = os.path.join(save_dir, f"scatter/Norm_Pred_vs_Real_{name}.png")
                self.visualizer.plot_scatter(best_real_embeddings[:, i], best_pred_embeddings[:, i], scatter_path, mode="nodes" if i < 2 else "edges")

                # Line Plot
                line_path = os.path.join(save_dir, f"line/Norm_Pred_vs_Real_{name}_line.png")
                self.visualizer.plot_line_graph(norm_real_embeddings[:, i], norm_pred_embeddings[:, i], name, save_path=line_path)


            # # Save CSVs with proper column names
            # pred_csv_path = os.path.join(save_dir, "best_pred_embeddings.csv")
            # real_csv_path = os.path.join(save_dir, "best_real_embeddings.csv")
            # pd.DataFrame(best_pred_embeddings[:, :len(col_names)], columns=col_names).to_csv(pred_csv_path, index=False)
            # pd.DataFrame(best_real_embeddings[:, :len(col_names)], columns=col_names).to_csv(real_csv_path, index=False)

            # print(f"Saved predictions to {pred_csv_path}")
            # print(f"Saved real embeddings to {real_csv_path}")

            # # Plot each column against each other using names
            # for i, name in enumerate(col_names):
            #     # Scatter plot
            #     scatter_path = os.path.join(save_dir, f"scatter/Pred_vs_Real_{name}.png")
            #     self.visualizer.plot_scatter(best_real_embeddings[:, i], best_pred_embeddings[:, i], scatter_path, mode="nodes" if i < 2 else "edges")
                
            #     # Line Plot
            #     line_path = os.path.join(save_dir, f"line/Pred_vs_Real_{name}_line.png")
            #     self.visualizer.plot_line_graph(best_real_embeddings[:, i], best_pred_embeddings[:, i], name, save_path=line_path)
            
            
    def grid_search(self):
        dropout_list = [0, 0.10]
        hidden_1_list = [64, 128]
        hidden_2_list = [32, 64]
        num_layers_list = [2, 3]
        lr_val_list = [1e-4, 1e-3]
        l2_val_list = [0, 1e-5]
        batch_size_list = [32, 64]
        model_list = [
            "['LSTM', 'FC']",
            "['GRU', 'FC']",
            "['LSTM', 'GRU', 'FC']",
            "['LSTM', 'GRU', 'MLP']"
        ]
        norms = [True]
            
        datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

        try:
            trial_num = -1
            for batch_size in batch_size_list:
                for num_layer in num_layers_list:
                    for dropout in dropout_list:
                        for hidden_1 in hidden_1_list:
                            for hidden_2 in hidden_2_list:
                                for lr_val in lr_val_list:
                                    for l2_val in l2_val_list:
                                        for combo in model_list:
                                            for norm in norms:
                                                trial_num += 1
                                                
                                                for dataset in datasets:
                                                    res = self.train_and_eval_delta(
                                                        dataset=dataset,
                                                        window_size=7, 
                                                        num_back=5, 
                                                        norm=norm,
                                                        num_layer=num_layer, 
                                                        dropout=dropout, 
                                                        hidden_1=hidden_1, 
                                                        hidden_2=hidden_2, 
                                                        lr_val=lr_val, 
                                                        l2_val=l2_val, 
                                                        batch_size=batch_size,
                                                        combo=combo_map[combo],
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
        
        wandb.init(project="Regression", name="probs_generation")
        
    
    def run(self):
        self.wandb_setup()
        
        if self.with_grid_search:
            self.grid_search()
            
        self.generate_best()
        
        
if __name__ == "__main__":
    my_generator = ProbsGeneration(with_grid_search=True)
    my_generator.run()