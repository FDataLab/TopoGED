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

# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

import wandb

# Constants
csv_file_path = os.path.abspath('data/output/results/RegressionTesting/data/embedding_testing_bayesian_regression_20dim_deltas_trainlossonly.csv')
seed = 42  # Can change
FLOAT_MAX = np.finfo(np.float32).max


# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['run_id', 'dataset', 'activation', 'seed', 'window_size', 'normalization', 'hidden_size_rnn', 'learning_rate', 'dropout', 'l2_regularization', 'batch_size', 'num_layers', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss', 'train_avg_norm', 'val_avg_norm', 'test_avg_norm', 'train_avg_cosine_similarity','val_avg_cosine_similarity', 'test_avg_cosine_similarity',]).to_csv(csv_file_path, index=False)


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


def train_and_eval_delta(dataset, activations, window_size, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed, csv_file_path):
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
    run = wandb.init(
        project="bayesian_testing_regression_updated_datasets", 
        name = run_name, 
        config={
        'dataset': dataset,
        'activation': activation_name,
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
    pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
    return best_moment_row['train_loss'], best_moment_row['valid_loss']


def main():
    # Load .env file
    load_dotenv()

    # Access API key
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY not found. Please set it in your .env file.")

    os.environ["WANDB_API_KEY"] = wandb_api_key
    
    wandb.init(project="Regression", name="bayesian_testing_regression_20dim_deltas")
    
    norm = False
    window_size = 7  # Fixed window size
    dropout_list = [0, 0.1, 0.2]
    hidden_1_list = [128, 256, 512, 1024]
    hidden_2_list = [64, 128, 256, 512]
    num_layers_list = [2, 3, 4]
    lr_val_list = [1e-5, 1e-4, 1e-3]
    l2_val_list = [1e-5, 1e-4, 1e-3]
    batch_size_list = [16, 32]
    activation_list =[
        'Degree',
        'Degree_Forman',
        'Degree_Weight',
        'Degree_Closeness',
    ]
    model_list = [
        "['RNN', 'FC']",
        "['LSTM', 'FC']",
        "['GRU', 'FC']",
        "['LSTM', 'GRU', 'FC']",
        "['LSTM', 'FC', 'FC']",
        "['GRU', 'FC', 'FC']",
        "['RNN', 'MLP']",
        "['LSTM', 'MLP']", 
        "['GRU', 'MLP']", 
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
                                            results = []                
                                            
                                            for dataset in datasets: 
                                                if dataset == 'Reddit_B' and 'Forman' in activations:
                                                    continue
                                                
                                                # Call your train_and_eval function for each dataset
                                                train_loss, val_loss = train_and_eval_delta(
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
                                                    seed=42,
                                                    csv_file_path=csv_file_path,
                                                )
                                                # Prevents errors later
                                                if np.isinf(train_loss) or np.isnan(train_loss):
                                                    train_loss = FLOAT_MAX
                                                if np.isinf(val_loss) or np.isnan(val_loss):
                                                    val_loss = FLOAT_MAX
                                                loss_score = train_loss  # Just doing train_loss for now
                                                results.append(loss_score)

                                            res = sum(results) / len(results)
                                            if res < best_train_loss:
                                                best_trial = {
                                                    'batch_size': batch_size, 
                                                    'dropout': dropout, 
                                                    'hidden_1': hidden_1, 
                                                    'hidden_2': hidden_2, 
                                                    'num_layers': num_layers, 
                                                    'lr_val': lr_val, 
                                                    'l2_val': l2_val, 
                                                    'activation': activation, 
                                                    'model': model, 
                                                }

    except:
        print(f'The best trial is: {best_trial}')
        print(f'On trial number: {trial_num}')
    
    
if __name__ == "__main__":
    main()