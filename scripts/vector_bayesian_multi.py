import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import optuna
import sqlite3

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from nn.custom_model import Decoder

from utils.utils import Utils


# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

import wandb

# Constants
csv_file_path = os.path.abspath('data/output/results/RegressionTesting/data/embedding_testing_bayesian_regression_new.csv')
model_dir = os.path.abspath('data/output/cached_model/RegressionTesting/EmbeddingTesting')
STORAGE = "sqlite:///./output/cached_model/RegressionTesting/bayesianSave/model_selection_regression_new.db"  # Where we save the study
os.makedirs(os.path.dirname('output/cached_model/RegressionTesting/bayesianSave/model_selection_regression_new.db'), exist_ok=True)
seed = 42  # Can change

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
}

def train_and_eval(dataset, activations, window_size, norm, num_layer, dropout, hidden_1, lr_val, l2_val, batch_size, combo, counter, seed, csv_file_path):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 0  # Regression (1 Activation)
    input_dim = 0  # Dynamic based on concatenations
    patience = 25  # Early stopping patience
    num_epochs = 500  # Max epochs to train
    
    run_name = dataset
    activation_name = ""
    
    # Set up embeddings
    embeddings = None  # Init
    for activation in activations:
        data, labels = my_loader.load_data(dataset, activation)  # Load embeddings and labels
        embeddings = my_utils.concat_embeddings(embeddings, data)  # Add the new data
        activation_name += activation + '_'
        
    input_dim = 30 * len(activations)
    output_dim = input_dim
        
    run_name = run_name + '_'+ activation_name + str(counter)    
        
    # Split data 70/15/15
    n = len(embeddings)

    # Calculate split indices
    train_end = int(0.8 * n)  # 80% train
    val_start = train_end - window_size  # val starts after gap
    val_end = int(0.9 * n)  # 10% val
    test_start = val_end - window_size  # test starts after gap

    embeddings = np.array([np.array(e, dtype=np.float32) for e in embeddings])

    X_train = embeddings[:train_end]
    X_val = embeddings[val_start:val_end]
    X_test = embeddings[test_start:]
                    
    if norm:
        X_train_scaled, X_val_scaled, X_test_scaled = my_utils.normalize_embeddings(X_train, X_val, X_test)
        
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test

    train_dataset = EmbeddingDataset(X_train_scaled, k=window_size)
    valid_dataset = EmbeddingDataset(X_val_scaled, k=window_size)
    test_dataset = EmbeddingDataset(X_test_scaled, k=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                                        
    # Initialize wandb
    run = wandb.init(
        project="bayesian_testing_regression_new", 
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
            
    no_improvement_counter = 0  # Number of epochs that we haven't seen an improvement in the validation AUCROC
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[output_dim], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    criterion = nn.MSELoss() 
    
    curr_batch_best_loss = float('inf')    
     
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        cosine_similarities = []
        norms = []
        time_index = 0  # Start time index at the beginning of the train set
        predicted_embeddings = []
        #predicted_embeddings_linfit = []
        real_embeddings = []

        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            output = output[:, -1, :]
            output = output.squeeze(1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Print time index, predicted embedding, and real embedding
            for i in range(len(x)):
                predicted_embedding = output[i].detach().cpu().numpy()
                real_embedding = y[i].detach().cpu().numpy()
                #predicted_embedding_linfit = my_utils.linear_fit(predicted_embedding)

                #predicted_embeddings_linfit.append(predicted_embedding_linfit)  # Fit a LinearRegression model for monotonically increasing behavior
                real_embeddings.append(real_embedding)
                predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                
                # Visualize 20-dim embeddings
                # predicted_linfit_str = '\t'.join(map(str, predicted_embedding_linfit))
                # predicted_str = '\t'.join(map(str, predicted_embedding))
                # real_str = '\t'.join(map(str, real_embedding))
                # print(f"Time Index:\t{time_index}\nPredicted Embedding:\t{predicted_str}\nLinear Fit Embedding:\t{predicted_linfit_str}\nReal Embedding:\t{real_str}")
                # print("-" * 50)
                
                try:
                    cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                except:
                    cosine_similarities.append(float('nan'))
                try:
                    norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))
                except:
                    norms.append(float('nan'))

                time_index += 1
        
        # my_visualizer = Visualizer(dataset="Cosine", task="Regression")
        # for i in range(3):
        #     my_visualizer.display_embeddings_once(predicted_embeddings[i], real_embeddings[i], predicted_embeddings_linfit[i])
        
        train_avg_norm = np.nanmean(norms)
        train_avg_cosine_similarity = np.nanmean(cosine_similarities)
        train_loss = (epoch_loss / len(train_loader))
                
        # Validation
        model.eval()
        valid_loss = 0
        cosine_similarities = []
        norms = []
        time_index = train_end  # Start time index at the beginning of the test set
        predicted_embeddings = []
        #predicted_embeddings_linfit = []
        real_embeddings = []

        with torch.no_grad():
            for x, y in valid_loader:
                output = model(x)  # Maintain hidden state across time steps
                output = output[:, -1, :]
                output = output.squeeze(1)
                y = y.float()
                loss = criterion(output, y)
                valid_loss += loss.item()
                
                # Print time index, predicted embedding, and real embedding
                for i in range(len(x)):
                    predicted_embedding = output[i].detach().cpu().numpy()
                    real_embedding = y[i].detach().cpu().numpy()
                    #predicted_embedding_linfit = my_utils.linear_fit(predicted_embedding)

                    #predicted_embeddings_linfit.append(predicted_embedding_linfit)  # Fit a LinearRegression model for monotonically increasing behavior
                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                    
                    # Visualize 20-dim embeddings
                    # predicted_linfit_str = '\t'.join(map(str, predicted_embedding_linfit))
                    # predicted_str = '\t'.join(map(str, predicted_embedding))
                    # real_str = '\t'.join(map(str, real_embedding))
                    # print(f"Time Index:\t{time_index}\nPredicted Embedding:\t{predicted_str}\nLinear Fit Embedding:\t{predicted_linfit_str}\nReal Embedding:\t{real_str}")
                    # print("-" * 50)
                    
                    try:
                        cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                    except:
                        cosine_similarities.append(float('nan'))
                    try:
                        norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))
                    except:
                        norms.append(float('nan'))

                    time_index += 1
            
        # my_visualizer = Visualizer(dataset="Cosine", task="Regression")
        # for i in range(3):
        #     my_visualizer.display_embeddings_once(predicted_embeddings[i], real_embeddings[i], predicted_embeddings_linfit[i])
        
        valid_loss /= len(valid_loader)
        val_avg_norm = np.nanmean(norms)
        val_avg_cosine_similarity = np.nanmean(cosine_similarities)
        
                
        # Testing        
        model.eval()
        test_loss = 0
        cosine_similarities = []
        norms = []
        time_index = val_end  # Start time index at the beginning of the test set
        predicted_embeddings = []
        #predicted_embeddings_linfit = []
        real_embeddings = []

        with torch.no_grad():
            for x, y in test_loader:
                output = model(x)  # Maintain hidden state across time steps
                output = output[:, -1, :]
                output = output.squeeze(1)
                y = y.float()
                loss = criterion(output, y)
                test_loss += loss.item()
                                # Print time index, predicted embedding, and real embedding
                for i in range(len(x)):
                    predicted_embedding = output[i].detach().cpu().numpy()
                    real_embedding = y[i].detach().cpu().numpy()
                    #predicted_embedding_linfit = my_utils.linear_fit(predicted_embedding)

                    #predicted_embeddings_linfit.append(predicted_embedding_linfit)  # Fit a LinearRegression model for monotonically increasing behavior
                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                    
                    # Visualize 20-dim embeddings
                    # predicted_linfit_str = '\t'.join(map(str, predicted_embedding_linfit))
                    # predicted_str = '\t'.join(map(str, predicted_embedding))
                    # real_str = '\t'.join(map(str, real_embedding))
                    # print(f"Time Index:\t{time_index}\nPredicted Embedding:\t{predicted_str}\nLinear Fit Embedding:\t{predicted_linfit_str}\nReal Embedding:\t{real_str}")
                    # print("-" * 50)
                    
                    try:
                        cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                    except:
                        cosine_similarities.append(float('nan'))
                    try:
                        norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))
                    except:
                        norms.append(float('nan'))            

                    time_index += 1
            
        # my_visualizer = Visualizer(dataset="Cosine", task="Regression")
        # for i in range(3):
        #     my_visualizer.display_embeddings_once(predicted_embeddings[i], real_embeddings[i], predicted_embeddings_linfit[i])
        
        test_loss /= len(test_loader)
        test_avg_norm = np.nanmean(norms)
        test_avg_cosine_similarity = np.nanmean(cosine_similarities)
                
        # Stores our current epoch 'steps' results
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
                
        # Log each epoch results
        wandb.log(to_log)

        # Optimize for the best aucroc
        if valid_loss <= curr_batch_best_loss:
            curr_batch_best_loss = valid_loss
            
            # Save for dataframe
            best_moment_row = {
                'run_id': run.name,  # For checking Wandb Logs
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
                    
                            
        # Early stopping only after 50 epochs
        if epoch >= 100:
            if valid_loss <= curr_batch_best_loss:
                no_improvement_counter = 0
                curr_batch_best_loss = valid_loss
            else:
                no_improvement_counter += 1
                
            if no_improvement_counter == patience:
                print(f'Training ending at epoch number: {epoch + 1}')
                break
    
    # Save the best moment from this training
    try:
        pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)

    except UnboundLocalError:
        # Fallback row with infinities where needed
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
            'trained_epochs': np.inf,
            'train_loss': np.inf,
            'valid_loss': np.inf,
            'test_loss': np.inf,
        }
        pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
    
    return best_moment_row['train_loss'], best_moment_row['valid_loss']
    

def objective(trial):
    # Suggest hyperparameters
    window_size = trial.suggest_int('window_size', 5, 30)
    dropout = trial.suggest_float('dropout', 0.01, 0.5)
    hidden_1 = trial.suggest_categorical('hidden_1', [32, 64, 128, 256, 512, 1024])  # Since it doesnt matter
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr_val = trial.suggest_float('lr_val', 1e-6, 1e-1, log=True)
    l2_val = trial.suggest_float('l2_val', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    activation = trial.suggest_categorical('activation', [
        'Degree',
        'Degree_Forman_Weight',
        'Degree_Betweenness_Closeness',
        'Degree_Forman_Closeness',
        'Degree_Weight_Closeness',
        'Degree_Forman',
        'Degree_Weight',
        'Degree_Betweenness',
        'Degree_Closeness',
    ])
    activations = activation_map[activation]
    model = trial.suggest_categorical('combo', [
        "['LSTM', 'ReLU']",
        "['GRU', 'ReLU']",
        "['LSTM', 'GRU', 'ReLU']",
        "['RNN']",
        "['GRU']",
        "['LSTM']",
        "['RNN', 'FC']",
        "['LSTM', 'FC']",
        "['GRU', 'FC']",
        "['LSTM', 'GRU', 'FC']",
    ])   
    
    bad_combos = [
        "['LSTM', 'ReLU']",
        "['GRU', 'ReLU']",
        "['LSTM', 'GRU', 'ReLU']",
        "['RNN']",
        "['GRU']",
        "['LSTM']"
    ]

    # Using because it already started
    if model in bad_combos:
        raise optuna.TrialPruned(f"Skipping combo {model} because it clips outputs")
    
    model = combo_map[model]
    norm = False
    
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']
    
    results = []

    # Evaluate on all datasets
    for dataset in datasets:
        if dataset == 'Reddit_B' and 'Forman' in activations:
            continue
        
        # Call your train_and_eval function for each dataset
        train_loss, val_loss = train_and_eval(
            dataset=dataset,
            window_size=window_size,
            activations=activations,
            norm=norm,
            num_layer=num_layers,
            dropout=dropout,
            hidden_1=hidden_1,
            lr_val=lr_val,
            l2_val=l2_val,
            batch_size=batch_size,
            combo=model,
            counter=trial.number,
            seed=42,
            csv_file_path=csv_file_path,
        )
        loss_score = (train_loss * 0.4 + val_loss * 0.6)  # Play with these numbers a bit, (0.2, 0.8) and (0.4, 0.6)
        results.append(loss_score)

    # Return the mean loss across datasets
    return sum(results) / len(results)
    

def main():
    os.environ["WANDB_API_KEY"] = "6a5ccf040a6c90944032e58878e46c19d673cdb0"
    wandb.init(project="Regression", name="bayesian_testing_regression_new")
    #optuna.delete_study(study_name="model_selection", storage=STORAGE)
    study = optuna.create_study(study_name="model_selection_new", storage=STORAGE, direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=500)

    print(f"Best trial: {study.best_trial}")
    
    
if __name__ == "__main__":
    main()