import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import optuna
import sqlite3

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
csv_file_path = os.path.abspath('data/output/results/RegressionTesting/data/probabilities_gs/embedding_testing_bayesian_individual_regression.csv')
model_dir = os.path.abspath('data/output/cached_model/RegressionTesting/probabilities_gs/EmbeddingTesting')
seed = 42  # Can change

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['run_id', 'dataset', 'activation', 'seed', 'normalization', 'hidden_size_rnn', 'learning_rate', 'dropout', 'l2_regularization', 'batch_size', 'num_layers', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss', 'train_avg_norm', 'val_avg_norm', 'test_avg_norm', 'train_avg_cosine_similarity','val_avg_cosine_similarity', 'test_avg_cosine_similarity',]).to_csv(csv_file_path, index=False)
    

combo_map = {
    "['LSTM', 'ReLU']": ['LSTM', 'ReLU'], 
    "['GRU', 'ReLU']": ['GRU', 'ReLU'], 
    "['LSTM', 'GRU', 'ReLU']": ['LSTM', 'GRU', 'ReLU'], 
    "['RNN']": ['RNN'],
    "['RNN', 'FC']": ['RNN', 'FC'],
    "['LSTM', 'FC']": ['LSTM', 'FC'],
    "['GRU', 'FC']": ['GRU', 'FC'],
    "['LSTM', 'GRU', 'FC']": ['LSTM', 'GRU', 'FC'],
}

def train_and_eval(dataset, activations, norm, num_layer, dropout, hidden_1, lr_val, l2_val, batch_size, combo, counter, seed, csv_file_path):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 30  # Regression (1 Activation)
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
        
    run_name = run_name + '_'+ activation_name + str(counter)    
        
    # Split data 70/15/15
    n = len(embeddings)

    # Calculate split indices
    train_end = int(0.7 * n)  # 70% for training
    val_end = int(0.85 * n)   # Next 15% for validation (70% + 15% = 85%)
    X_train = embeddings[:train_end]
    X_val = embeddings[train_end:val_end]
    X_test = embeddings[val_end:]
                    
    if norm:
        print('normalizing embeddings')
        X_train_scaled, X_val_scaled, X_test_scaled = my_utils.normalize_embeddings(X_train, X_val, X_test)
        
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test


    train_dataset = EmbeddingDataset(X_train_scaled)
    valid_dataset = EmbeddingDataset(X_val_scaled)
    test_dataset = EmbeddingDataset(X_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                                        
    # Initialize wandb
    run = wandb.init(
        project="bayesian_testing_regression", 
        name = run_name, 
        config={
        'dataset': dataset,
        'activation': activation_name,
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

        valid_loss /= len(valid_loader)
        
                
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
            if(dataset == 'cosine'):
                print(f'SAVING FROM RUN {run.name}')
                columns = [f"{prefix}_{i}" for prefix in ["node", "edges", "weights"] for i in range(1, 11)]
                tmp_df = pd.DataFrame(real_embeddings, columns = columns)
                tmp_df.to_csv('data/output/results/RegressionTesting/exampleEmbeddings/cosine_real_ex.csv')
                tmp_df = pd.DataFrame(predicted_embeddings, columns = columns)
                tmp_df.to_csv('data/output/results/RegressionTesting/exampleEmbeddings/cosine_pred_ex.csv')
                
                            
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
    pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
    
    return best_moment_row['train_loss'], best_moment_row['valid_loss']
    

def objective(trial, dataset):
    # Suggest hyperparameters
    dropout = trial.suggest_float('dropout', 0.01, 0.5)
    hidden_1 = trial.suggest_categorical('hidden_1', [32, 64, 128, 256, 512, 1024])  # Since it doesnt matter
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr_val = trial.suggest_float('lr_val', 1e-4, 1e-2, log=True)
    l2_val = trial.suggest_float('l2_val', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    activation = trial.suggest_categorical('activation', [
        "[Degree]",
        "[Betweenness]",
        "[Forman]",
        "[Closeness]",
        "[Weight]",
        # "['Betweenness', 'Closeness']", 
        # "['Degree', 'Forman', 'Weight']", 
        # "['Degree', 'Betweenness', 'Closeness']", 
        # "['Betweenness', 'Forman']", 
        # "['Closeness', 'Forman']", 
        # "['Degree', 'Betweenness', 'Closeness', 'Forman']", 
        # "['Betweenness', 'Closeness', 'Degree', 'Forman', 'Weight']",
    ])
    activations = activation_map[activation]
    model = trial.suggest_categorical('combo', [
        "['RNN']",
        "['RNN', 'FC']",
        "['LSTM', 'FC']",
        "['GRU', 'FC']",
        "['LSTM', 'GRU', 'FC']"
    ])   
    model = combo_map[model]
    norm = False
        

    if dataset == 'Reddit_B' and 'Forman' in activations:
        raise optuna.TrialPruned("Forman-based activation not supported for Reddit_B.")
    
    # Call your train_and_eval function for each dataset
    train_loss, val_loss = train_and_eval(
        dataset=dataset,
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
    return loss_score
    

def main():
    os.environ["WANDB_API_KEY"] = "6a5ccf040a6c90944032e58878e46c19d673cdb0"
    wandb.init(project="Regression", name="regression_bayesian_testing_individual_dataset_4060")
    
    datasets = ['cosine', 'CollegeMsg', 'networkcindicator', 'networkdgd', 'networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', ]    
    
    # Dictionary to store best results for each dataset
    best_results = {}

    # _4060 SIGNIFIES THAT WE ARE USING 40% TRAIN AND 60% VAL IN OUR BAYESIAN SCORING
    for dataset in datasets:
        STORAGE = f"sqlite:///./output/cached_model/RegressionTesting/bayesianSave/model_selection_{dataset}_4060.db"  # Where we save the study
        os.makedirs(os.path.dirname(f'output/cached_model/RegressionTesting/bayesianSave/model_selection_{dataset}_4060.db'), exist_ok=True)
        print(f"Optimizing for dataset: {dataset}")
        
        study_name = f"regression_model_selection_{dataset}_4060"  # Unique study name for each dataset
        
        optuna.delete_study(study_name=study_name, storage=STORAGE)
        
        study = optuna.create_study(
            study_name=study_name, 
            storage=STORAGE, 
            direction="minimize", 
            load_if_exists=True
        )
        
        # Pass dataset to the objective function
        study.optimize(lambda trial: objective(trial, dataset), n_trials=400)
        
        # Save the best trial for the current dataset
        best_results[dataset] = study.best_trial

        print(f"Best trial for {dataset}: {study.best_trial}")

    # Print all results
    print("\nBest results for each dataset:")
    for dataset, trial in best_results.items():
        print(f"{dataset}: {trial}")
    
    
if __name__ == "__main__":
    main()