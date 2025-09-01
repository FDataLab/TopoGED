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
from utils.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from nn.custom_model import Decoder

from utils.utils import Utils


RESULTS_PATH = 'data/input/cached/'
TRIALS_HISTORY_PATH = 'data/output/results/RegressionTesting/data/embedding_testing_bayesian_regression_updated_datasets.csv'


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

def train_and_eval(dataset, activations, window_size, norm, num_layer, dropout, hidden_1, lr_val, l2_val, batch_size, combo, counter, seed):\
    
    # Setup
    print(combo)
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
        print('Normalizing embeddings')
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
        #predicted_embeddings_linfit = []

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
        #predicted_embeddings_linfit = []

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
            best_pred_embeddings = predicted_embeddings
            best_real_embeddings = real_embeddings
                    
                            
        # Early stopping only after 50 epochs
        if epoch >= 100:
            if valid_loss <= curr_batch_best_loss:
                no_improvement_counter = 0
                curr_batch_best_loss = valid_loss
                best_train_loss = train_loss 
                best_val_loss = valid_loss
                best_pred_embeddings = predicted_embeddings
                best_real_embeddings = real_embeddings
            else:
                no_improvement_counter += 1
                
            if no_improvement_counter == patience:
                print(f'Training ending at epoch number: {epoch + 1}')
                break
    
    
    return best_moment_row['train_loss'], best_moment_row['valid_loss'], best_pred_embeddings, best_real_embeddings 



def main():
    df = pd.read_csv(TRIALS_HISTORY_PATH)
    
    # Necessary because of tanh clipping
    good_combos = [
        "['RNN', 'FC']",
        "['LSTM', 'FC']",
        "['GRU', 'FC']",
        "['LSTM', 'GRU', 'FC']"
    ]

    # Keep only trials with good combos
    df = df[df['combo'].isin(good_combos)]
    
    df["train_loss"] = df["train_loss"].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    df["valid_loss"] = df["valid_loss"].replace([np.inf, -np.inf], np.finfo(np.float32).max)
    
    # Compute a score (weighted combination)
    df["bayesian_score"] = 0.4 * df["train_loss"] + 0.6 * df["valid_loss"]
    
    # Strip the _# suffix to create a trial group
    df["trial_group"] = df["run_id"].str.rsplit("_", n=1).str[1]
    
    # Count entries per group
    group_counts = df.groupby("trial_group").size().reset_index(name="count")
    
    # Keep only groups with at least 13 entries
    valid_groups = group_counts[group_counts["count"] >= 13]["trial_group"]
    
    # Compute the mean score for valid groups only
    group_means = df[df["trial_group"].isin(valid_groups)].groupby("trial_group")["bayesian_score"].mean().reset_index()
    
    # Find the trial group with the lowest average score
    best_group = group_means.loc[group_means["bayesian_score"].idxmin()]["trial_group"]
    
    print(f"Best trial group across all datasets (>=13 entries): {best_group}\n{'-'*40}")
    
    # Optionally, display the best trials for each dataset within that group
    best_trials = df[df["trial_group"] == best_group]
    print(best_trials.head())
                    
    # Loop and print results as variable assignments
    for _, row in best_trials.iterrows():
        print(f"\n🔧 Best Trial for Dataset: {row['run_id']}\n{'-'*40}")

        # Extract parameters
        window_size = int(row['seed']) if int(row['window_size']) == 42 else int(row['window_size'])
        dropout = float(row['dropout'])
        hidden_1 = int(row['hidden_size_rnn'])  
        num_layers = int(row['num_layers'])
        lr_val = float(row['learning_rate'])
        l2_val = float(row['l2_regularization'])
        batch_size = int(row['batch_size'])
        model_str = row['combo']
        model = combo_map[model_str]
        dataset=row['dataset']
        activations = activation_map[row['activation'][:-1]]
        
        
        train_loss, val_loss, pred_embeddings, real_embeddings = train_and_eval(
            dataset=dataset,
            window_size=window_size,
            activations=activations,
            norm=False,
            num_layer=num_layers,
            dropout=dropout,
            hidden_1=hidden_1,
            lr_val=lr_val,
            l2_val=l2_val,
            batch_size=batch_size,
            combo=model,
            counter=999,
            seed=42,
        )
    
        loss_score = (train_loss * 0.4 + val_loss * 0.6)  # Play with these numbers a bit, (0.2, 0.8) and (0.4, 0.6)

        print(f"The old score was {row['bayesian_score']} and the new one was {loss_score}")

        res_path = RESULTS_PATH + dataset + '/PredTopERUpdated/'

        os.makedirs(res_path, exist_ok=True)

        # Form dataframes of pred, real, and a train_test split for graph construction
        pred_df_discrete = pd.DataFrame(pred_embeddings)
        real_df_discrete = pd.DataFrame(real_embeddings)
        
        # Take only the degree embedding
        pred_df_discrete = pred_df_discrete.iloc[:, :30]
        real_df_discrete = real_df_discrete.iloc[:, :30]
        
        pred_col = pred_df_discrete.iloc[:, -2]
        real_col = real_df_discrete.iloc[:, -2]

        # Compute RMSE
        rmse = mean_squared_error(real_col, pred_col, squared=False)  # squared=False gives RMSE

        # Compute MAE
        mae = mean_absolute_error(real_col, pred_col)

        print(f"Final Edge RMSE: {rmse}")
        print(f"Final Edge MAE: {mae}")  
        
        print(f"Final Edge Mean: {pred_col.mean()}")
        print(f"Final Edge STDDEV: {pred_col.std()}") 
        
        pred = pred_df_discrete.to_numpy()
        real = real_df_discrete.to_numpy()

        # RMSE per row, then average
        rmse_per_row = np.sqrt(np.mean((pred - real) ** 2, axis=1))
        avg_rmse = np.mean(rmse_per_row)

        # MAE per row, then average
        mae_per_row = np.mean(np.abs(pred - real), axis=1)
        avg_mae = np.mean(mae_per_row)
        
        print(f"Average row-wise RMSE: {avg_rmse}")
        print(f"Average row-wise MAE: {avg_mae}")

        pred_tensor = torch.tensor(pred_df_discrete.values, dtype=torch.float32)
        real_tensor = torch.tensor(real_df_discrete.values, dtype=torch.float32)

        # Define loss function (sum to mimic training accumulation)
        criterion = nn.MSELoss(reduction='sum')

        # Compute total loss
        total_loss = criterion(pred_tensor, real_tensor).item()

        
        print(f'Total Loss across all vectors: {total_loss}')
        
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
           
    
main()