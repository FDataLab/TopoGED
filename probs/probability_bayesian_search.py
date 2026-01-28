import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from dotenv import load_dotenv
import sys
import wandb
import optuna

# Update path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.dataset import DeltaEmbeddingDataset
from nn.custom_model import Decoder
from torch.utils.data import DataLoader, Dataset
from utils.utils import Utils

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FLOAT_MAX = np.finfo(np.float32).max
RESULTS_CSV = 'data/output/ProbabilityTesting/data/bayesian_training_results.csv'

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

class ProbabilityWrapper(nn.Module):
    def __init__(self, base_model):
        super(ProbabilityWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        logits = self.base_model(x)
        if logits.dim() == 3:
            logits = logits[:, -1, :] 

        # Group 1: Indices 0, 1 | Group 2: Indices 2, 3, 4, 5
        group1 = torch.softmax(logits[:, :2], dim=1)
        group2 = torch.softmax(logits[:, 2:], dim=1)

        return torch.cat([group1, group2], dim=1)

def train_and_eval(dataset, window_size, num_back, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed):
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 6
    input_dim = 6
    patience = 25
    num_epochs = 500
    
    probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='all')  
    if norm:
        embeddings = probabilities_df.apply(
            lambda row: my_utils.normalize_vector_by_groups(row.values),
            axis=1
        )
        embeddings = np.vstack(embeddings.values).astype(np.float32) 
    else:
        embeddings = probabilities_df.values.astype(np.float32)
        
    run_name = f"{dataset}_{counter}"
    n = len(embeddings)
    train_end = int(0.70 * n)
    val_start = train_end - window_size
    val_end = int(0.85 * n)
    test_start = val_end - window_size

    X_train = embeddings[:train_end]
    X_val = embeddings[val_start:val_end]
    X_test = embeddings[test_start:]

    train_dataset = DeltaEmbeddingDataset(X_train, k=window_size)
    valid_dataset = DeltaEmbeddingDataset(X_val, k=window_size)
    test_dataset = DeltaEmbeddingDataset(X_test, k=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                                    
    # run = wandb.init(
    #     project="probs_direct_pred", 
    #     name=run_name, 
    #     config={
    #         'dataset': dataset, 'hidden_1': hidden_1, 'hidden_2': hidden_2,
    #         'window_size': window_size, 'num_layers': num_layer, 'dropout': dropout,
    #         'l2_regularization': l2_val, 'learning_rate': lr_val, 'seed': seed,
    #         'normalization': norm, 'model': combo
    #     },
    #     reinit=True)
            
    base_decoder = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], 
                           hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, 
                           bias=[True], dropout=[dropout]).to(device)
    model = ProbabilityWrapper(base_decoder).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    criterion = nn.MSELoss()
    
    curr_batch_best_loss = float('inf')    
    no_improvement_counter = 0
      
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        train_cosine_similarities, train_norms = [], []
        
        for i, (x, y_delta, x_last) in enumerate(train_loader):
            # Move data to device
            x, y_delta, x_last = x.to(device), y_delta.to(device), x_last.to(device)
            
            optimizer.zero_grad()
            y_target = x_last + y_delta 
            predicted_probs = model(x) 
            
            loss = criterion(predicted_probs, y_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            for j in range(len(x)):
                p_out = predicted_probs[j].detach().cpu().numpy()
                r_out = y_target[j].detach().cpu().numpy()
                try:
                    train_cosine_similarities.append(my_utils.compute_cosine_similarity(p_out, r_out))
                    train_norms.append(my_utils.compute_distances(p_out, r_out))
                except:
                    train_cosine_similarities.append(float('nan'))
                    train_norms.append(float('nan'))
        
        train_loss = (epoch_loss / len(train_loader))
                            
        model.eval()
        valid_loss = 0
        val_cosine_similarities, val_norms = [], []

        with torch.no_grad():
            for i, (x, y_delta, x_last) in enumerate(valid_loader):
                x, y_delta, x_last = x.to(device), y_delta.to(device), x_last.to(device)
                y_target = (x_last + y_delta).float()
                predicted_probs = model(x)
                
                loss = criterion(predicted_probs, y_target)
                valid_loss += loss.item()
                
                for j in range(len(x)):
                    p_out = predicted_probs[j].detach().cpu().numpy()
                    r_out = y_target[j].detach().cpu().numpy()
                    val_cosine_similarities.append(my_utils.compute_cosine_similarity(p_out, r_out))
                    val_norms.append(my_utils.compute_distances(p_out, r_out))
            
        valid_loss /= len(valid_loader)
        
        test_loss = 0
        test_cosine_similarities, test_norms = [], []

        with torch.no_grad():
            for i, (x, y_delta, x_last) in enumerate(test_loader):
                x, y_delta, x_last = x.to(device), y_delta.to(device), x_last.to(device)
                y_target = (x_last + y_delta).float()
                predicted_probs = model(x)
                
                loss = criterion(predicted_probs, y_target)
                test_loss += loss.item()
                
                for j in range(len(x)):
                    p_out = predicted_probs[j].detach().cpu().numpy()
                    r_out = y_target[j].detach().cpu().numpy()
                    test_cosine_similarities.append(my_utils.compute_cosine_similarity(p_out, r_out))
                    test_norms.append(my_utils.compute_distances(p_out, r_out))
            
        test_loss /= len(test_loader)
                
        # wandb.log({
        #     'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss, 'test_loss': test_loss,
        #     'val_avg_cosine_similarity': np.nanmean(val_cosine_similarities)
        # })

        if valid_loss <= curr_batch_best_loss:
            curr_batch_best_loss = valid_loss
            no_improvement_counter = 0
            best_moment_row = {
                'run_id': run_name, 'dataset': dataset, 'seed': seed, 'window_size': window_size,
                'normalization': norm, 'hidden_1': hidden_1, 'hidden_2': hidden_2,
                'learning_rate': lr_val, 'dropout': dropout, 'l2_regularization': l2_val,
                'batch_size': batch_size, 'num_layers': num_layer, 'combo': combo,
                'trained_epochs': epoch + 1, 'train_loss': train_loss, 'valid_loss': valid_loss, 'test_loss': test_loss,
                'val_avg_cosine_similarity': np.nanmean(val_cosine_similarities), 
                'test_avg_cosine_similarity': np.nanmean(test_cosine_similarities),
            }
        else:
            if epoch >= 100:
                no_improvement_counter += 1
        
        if no_improvement_counter >= patience:
            break
    
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df_result = pd.DataFrame([best_moment_row])
    df_result.to_csv(RESULTS_CSV, mode='a', index=False, header=not os.path.isfile(RESULTS_CSV))
    
    # wandb.finish()
    return best_moment_row.get('train_loss', FLOAT_MAX), best_moment_row.get('valid_loss', FLOAT_MAX)

def objective(trial, dataset):
    batch_size = trial.suggest_categorical("batch_size", [16])
    num_layer = trial.suggest_categorical("num_layer", [2, 3])
    dropout = trial.suggest_categorical("dropout", [0, 0.5, 0.10])
    hidden_1 = trial.suggest_categorical("hidden_1", [16, 32, 64])
    hidden_2 = trial.suggest_categorical("hidden_2", [8, 16, 32])
    lr_val = trial.suggest_categorical("lr_val", [1e-4, 1e-3, 1e-2])
    l2_val = trial.suggest_categorical("l2_val", [0, 1e-5, 1e-4])
    combo_key = trial.suggest_categorical("combo", list(combo_map.keys()))
    
    train_loss, val_loss = train_and_eval(
        dataset=dataset, window_size=5, num_back='all', norm=True, 
        num_layer=num_layer, dropout=dropout, hidden_1=hidden_1, 
        hidden_2=hidden_2, lr_val=lr_val, l2_val=l2_val, 
        batch_size=batch_size, combo=combo_map[combo_key],
        counter=trial.number, seed=42
    )
    return val_loss

if __name__ == "__main__":
    load_dotenv()
    # wandb_api_key = os.getenv("WANDB_API_KEY")
    # if wandb_api_key is None:
    #     raise ValueError("WANDB_API_KEY not found.")
    # os.environ["WANDB_API_KEY"] = wandb_api_key

    # datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 
    #             'networkcoindash', 'mathoverflow', 'Reddit_B', 'networkaragon', 
    #             'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']
    datasets = ['tgbl-wiki']
    for dataset in datasets:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, dataset), n_trials=250)