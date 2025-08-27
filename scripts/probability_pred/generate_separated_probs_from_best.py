import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt


# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from utils.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from nn.custom_model import Decoder

from utils.utils import Utils


RESULTS_PATH = 'data/input/cached/'
TRIALS_HISTORY_PATH = 'data/output/results/ProbabilityTesting/data/probability_testing_bayesian_individual_regression_no_norm.csv'

combo_map = {
    "['RNN']": ["RNN"],
    "['RNN', 'FC']": ["RNN", "FC"],
    "['LSTM', 'FC']": ["LSTM", "FC"],
    "['GRU', 'FC']": ["GRU", "FC"],
    "['LSTM', 'GRU', 'FC']": ["LSTM", "GRU", "FC"],
    "['LSTM', 'ReLU']": ['LSTM', 'ReLU'], 
    "['GRU', 'ReLU']": ['GRU', 'ReLU'], 
    "['LSTM', 'GRU', 'ReLU']": ['LSTM', 'GRU', 'ReLU'], 
    "['GRU']": ['GRU'],
    "['LSTM']": ['LSTM'],
}


prob_types = {
    "prob_old_nodes": 0,
    "prob_new_nodes": 1,
    "prob_oo": 2,
    "prob_n": 3,
    "prob_on": 4,
    "prob_oon": 5,
}

# Utility function specific to probabilities

# Utility function specific to probabilities
def normalize_vector_by_groups(vec, tol=1e-8):
    vec = np.array(vec, dtype=np.float32)
    vec = np.maximum(vec, 0)

    # Normalize indices 0 and 1 for node type
    group1 = vec[0:2]
    sum1 = np.sum(group1)
    vec[0:2] = group1 / sum1

    if not np.isclose(np.sum(vec[0:2]), 1.0, atol=tol):
        print(f"Warning: Group 1 sum = {np.sum(vec[0:2])}, not 1!")

    # Normalize second group
    group2 = vec[2:6]
    sum2 = np.sum(group2)
    vec[2:6] = group2 / sum2

    if not np.isclose(np.sum(vec[2:6]), 1.0, atol=tol):
        print(f"Warning: Group 2 sum = {np.sum(vec[2:6])}, not 1!")

    return vec


# Need to generate with this now
def softmax_grouped(vec, tol=1e-8):
    vec = np.array(vec, dtype=np.float32)

    def softmax(x):
        e_x = np.exp(x - np.max(x))  # stable softmax
        return e_x / e_x.sum()

    # Apply softmax to first group (indices 0 and 1)
    vec[0:2] = softmax(vec[0:2])
    if not np.isclose(np.sum(vec[0:2]), 1.0, atol=tol):
        print(f"Warning: Group 1 sum = {np.sum(vec[0:2])}, not 1!")

    # Apply softmax to second group (indices 2-5)
    vec[2:6] = softmax(vec[2:6])
    if not np.isclose(np.sum(vec[2:6]), 1.0, atol=tol):
        print(f"Warning: Group 2 sum = {np.sum(vec[2:6])}, not 1!")

    return vec



def train_and_eval(dataset, prob_type, prob_type_idx, window_size, num_layer, dropout, hidden_1, lr_val, l2_val, batch_size, combo, counter, seed):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 1  # Outputting next vector
    input_dim = 1  
    patience = 25  # Early stopping patience
    num_epochs = 500  # Max epochs to train
    
    run_name = dataset
    
    # Set up probabilities
    probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities')  # Activation doesn't matter here
    probabilities = probabilities_df.iloc[:, prob_type_idx].to_numpy(dtype=np.float32).reshape(-1, 1)
    #probabilities = probabilities_df.values.tolist()


    # Probabilities to return
    all_real_embeddings = []
    all_pred_embeddings = []            
                
    run_name = run_name + '_' + str(counter)    
        
    # Split data 80/10/10
    n = len(probabilities)

    # Calculate split indices
    train_end = int(0.8 * n)  # 80% train
    val_start = train_end - window_size  # val starts after gap
    val_end = int(0.9 * n)  # 10% val
    test_start = val_end - window_size  # test starts after gap

    X_train = probabilities[:train_end]
    X_val = probabilities[val_start:val_end]
    X_test = probabilities[test_start:]

    train_dataset = EmbeddingDataset(X_train, k=window_size)
    valid_dataset = EmbeddingDataset(X_val, k=window_size)
    test_dataset = EmbeddingDataset(X_test, k=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                                        
    # Initialize wandb
    run = wandb.init(
        project=f"bayesian_testing_probabilities_multi_{prob_type}", 
        name = run_name, 
        config={
        'dataset': dataset,
        'num_layers': num_layer,
        'dropout': dropout,
        'l2_regularization': l2_val,
        'hidden_size_rnn': hidden_1,
        'learning_rate': lr_val,
        'seed': seed,
        'window_size': window_size,
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
        real_embeddings = []

        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            output = output[:, -1, :]
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Print time index, predicted embedding, and real embedding
            for i in range(len(x)):
                predicted_embedding = output[i].detach().cpu().numpy()
                real_embedding = y[i].detach().cpu().numpy()

                real_embeddings.append(real_embedding)
                predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                
                try:
                    cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                except:
                    cosine_similarities.append(float('nan'))
                try:
                    norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))
                except:
                    norms.append(float('nan'))

                all_pred_embeddings.append(predicted_embedding.tolist())
                all_real_embeddings.append(real_embedding.tolist())

                time_index += 1
        
        
        train_avg_norm = np.nanmean(norms)
        train_avg_cosine_similarity = np.nanmean(cosine_similarities)
        train_loss = (epoch_loss / len(train_loader))
                
        # Validation
        model.eval()
        valid_loss = 0
        cosine_similarities = []
        norms = []
        time_index = train_end  # Start time index at the beginning of the test set

        with torch.no_grad():
            for x, y in valid_loader:
                output = model(x)  # Maintain hidden state across time steps
                output = output[:, -1, :]
                y = y.float()
                loss = criterion(output, y)
                valid_loss += loss.item()
                
                # Print time index, predicted embedding, and real embedding
                for i in range(len(x)):
                    predicted_embedding = output[i].detach().cpu().numpy()
                    real_embedding = y[i].detach().cpu().numpy()

                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                    
                    try:
                        cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                    except:
                        cosine_similarities.append(float('nan'))
                    try:
                        norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))
                    except:
                        norms.append(float('nan'))

                    all_pred_embeddings.append(predicted_embedding.tolist())
                    all_real_embeddings.append(real_embedding.tolist())

                    time_index += 1
            
        valid_loss /= len(valid_loader)
        val_avg_norm = np.nanmean(norms)
        val_avg_cosine_similarity = np.nanmean(cosine_similarities)
        
                
        # Testing        
        model.eval()
        test_loss = 0
        cosine_similarities = []
        norms = []
        time_index = val_end  # Start time index at the beginning of the test set

        with torch.no_grad():
            for x, y in test_loader:
                output = model(x)  # Maintain hidden state across time steps
                output = output[:, -1, :]
                y = y.float()
                loss = criterion(output, y)
                test_loss += loss.item()
                                # Print time index, predicted embedding, and real embedding
                for i in range(len(x)):
                    predicted_embedding = output[i].detach().cpu().numpy()
                    real_embedding = y[i].detach().cpu().numpy()

                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                    
                    
                    try:
                        cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                    except:
                        cosine_similarities.append(float('nan'))
                    try:
                        norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))
                    except:
                        norms.append(float('nan'))            

                    all_pred_embeddings.append(predicted_embedding.tolist())
                    all_real_embeddings.append(real_embedding.tolist())

                    time_index += 1
            
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
            
            best_pred_embeddings = predicted_embeddings
            best_real_embeddings = real_embeddings
            
            # Save for dataframe
            best_moment_row = {
                'run_id': run.name,  # For checking Wandb Logs
                'dataset': dataset,
                'window_size': window_size,
                'seed': seed,
                'hidden_size_rnn': hidden_1,
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
            }
            if(dataset == 'cosine'):
                print(f'SAVING FROM RUN {run.name}')
                columns = [f"{prefix}_{i}" for prefix in ["node", "edges", "weights"] for i in range(1, 11)]
                tmp_df = pd.DataFrame(real_embeddings, columns = columns)
                tmp_df.to_csv(f'data/output/results/RegressionTesting/exampleEmbeddings/cosine_real_ex_multi_{prob_type}.csv')
                tmp_df = pd.DataFrame(predicted_embeddings, columns = columns)
                tmp_df.to_csv(f'data/output/results/RegressionTesting/exampleEmbeddings/cosine_pred_ex_multi_{prob_type}.csv')
                
                            
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
    
    
    return best_moment_row['train_loss'], best_moment_row['valid_loss'], best_pred_embeddings, best_real_embeddings




def main():
    path = 'data/output/results/ProbabilityTesting/data/probability_testing_bayesian_individual_regression_multi_'
    
    datasets = ['networkbancor', 'CollegeMsg', 'mathoverflow', 'networkaragon', 'networkaion', 'networkadex', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkaeternity', 'networkiconomi', 'networkcindicator', 'networkdgd']  # replace with actual dataset names
    dataset_dfs = {name: pd.DataFrame() for name in datasets}
    
    for prob_type in prob_types.keys():
        print(f'Working on: {prob_type}')
        try:
            df = pd.read_csv(path + prob_type + '.csv')
        except FileNotFoundError:
            print(f'Error: Type {prob_type} does not exist yet for analysis')
            continue 
        
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

        best_trials = df.loc[df.groupby("dataset")["bayesian_score"].idxmin()].reset_index(drop=True)
                    
        # Loop and print results as variable assignments
        for _, row in best_trials.iterrows():
            print(f"\n🔧 Best Trial for Dataset: {row['run_id']}\n{'-'*40}")

            # Extract parameters
            window_size = int(row['window_size'])
            dropout = float(row['dropout'])
            hidden_1 = int(row['hidden_size_rnn'])  
            num_layers = int(row['num_layers'])
            lr_val = float(row['learning_rate'])
            l2_val = float(row['l2_regularization'])
            batch_size = int(row['batch_size'])
            model_str = row['combo']
            model = combo_map[model_str]
            dataset=row['dataset']
            
            
            train_loss, val_loss, pred_embeddings, real_embeddings = train_and_eval(
                dataset=dataset,
                prob_type=prob_type,
                prob_type_idx=prob_types[prob_type],
                window_size = window_size,
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
            
            pred_series = pd.Series([pe[-1] for pe in pred_embeddings], name=prob_type)
            real_series = pd.Series([re[-1] for re in real_embeddings], name=prob_type)

            print(f"Real series shape: {real_series.shape}")
            print(f"Pred series shape: {pred_series.shape}")

            pred = pred_series.to_numpy()
            real = real_series.to_numpy()

            # MAE and RMSE
            mae = np.mean(np.abs(pred - real))
            rmse = np.sqrt(np.mean((pred - real) ** 2))

            # Mean and stddev of prediction series
            pred_mean = pred_series.mean()
            pred_std = pred_series.std()

            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}")
            print(f"Pred Series Mean: {pred_mean}")
            print(f"Pred Series StdDev: {pred_std}")

            # Add to dataframe for normalization
            if dataset_dfs[dataset].empty:
                dataset_dfs[dataset] = pred_series.to_frame()
            else:
                dataset_dfs[dataset] = pd.concat([dataset_dfs[dataset], pred_series], axis=1)
            
            loss_score = (train_loss * 0.4 + val_loss * 0.6)  # Play with these numbers a bit, (0.2, 0.8) and (0.4, 0.6)

            print(f"The old score was {row['bayesian_score']} and the new one was {loss_score}")
            
            res_path = RESULTS_PATH + dataset + '/PredProbabilitiesSeparated/'

            os.makedirs(res_path, exist_ok=True)
            
            plt.figure(figsize=(8, 4))
            plt.plot(real_series.values, label="Real", linewidth=1)
            plt.plot(pred_series.values, label="Pred", linewidth=1)

            plt.xlabel("Index")
            plt.ylabel("Probability")
            plt.title(f"{prob_type} - Real vs Predicted")
            plt.legend()
            plt.grid(True)

            # Save plot
            file_path = os.path.join(res_path, f"{prob_type}_discrete.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()


        
        
        # # best_pred_embeddings_norm = [normalize_vector_by_groups(vec) for vec in pred_embeddings]
        # # best_real_embeddings_norm = [normalize_vector_by_groups(vec) for vec in real_embeddings]

        # # # Convert to numpy arrays if you want
        # # best_pred_embeddings_norm = np.array(best_pred_embeddings_norm)
        # # best_real_embeddings_norm = np.array(best_real_embeddings_norm)
        
        # # for col in pred_df_discrete.columns:
        # #     rmse = np.sqrt(np.mean((pred_df_discrete[col] - real_df_discrete[col])**2))
        # #     # Normalized RMSE (range-based)
        # #     nrmse_range = rmse / (real_df_discrete[col].max() - real_df_discrete[col].min())
            
        # #     # Normalized RMSE (mean-based)
        # #     nrmse_mean = rmse / real_df_discrete[col].mean()
        # #     print(f"Dataset: {dataset} | Column: {col} | NRMSE (mean): {nrmse_mean:.6f}")

        #     # print(f"Dataset: {dataset} | Column: {col} | RMSE: {rmse:.6f} | NRMSE (range): {nrmse_range:.6f} | NRMSE (mean): {nrmse_mean:.6f}")
                    
        # real_part_len = int(len(real_df_discrete) * 0.7)
        # pred_part_len = len(real_df_discrete) - real_part_len 
        # real_part = real_df_discrete.iloc[:real_part_len, :].to_numpy()
        # pred_part = pred_df_discrete.iloc[-pred_part_len:, :].to_numpy()
        # hybrid_array = np.vstack([real_part, pred_part])
        # real_pred_df_discrete = pd.DataFrame(hybrid_array)
        
        
        # for col in pred_df.columns:
        #     rmse = np.sqrt(np.mean((pred_df[col] - real_df[col])**2))
        #     # Normalized RMSE (range-based)
        #     nrmse_range = rmse / (real_df[col].max() - real_df[col].min())
            
        #     # Normalized RMSE (mean-based)
        #     nrmse_mean = rmse / real_df[col].mean()
        #     print(f"Dataset: {dataset} | Column: {col} | NRMSE (mean): {nrmse_mean:.6f}")

        #     # print(f"Dataset: {dataset} | Column: {col} | RMSE: {rmse:.6f} | NRMSE (range): {nrmse_range:.6f} | NRMSE (mean): {nrmse_mean:.6f}")
                    
        # real_part_len = int(len(real_df) * 0.7)
        # pred_part_len = len(real_df) - real_part_len 
        # real_part = real_df.iloc[:real_part_len, :].to_numpy()
        # pred_part = pred_df.iloc[-pred_part_len:, :].to_numpy()
        # hybrid_array = np.vstack([real_part, pred_part])
        # real_pred_df = pd.DataFrame(hybrid_array)
        
        # for idx, col in enumerate(pred_df.columns):
        #     plt.figure(figsize=(5, 5))
        #     plt.scatter(real_df[col], pred_df[col], alpha=0.6)
        #     plt.xlabel("Real")
        #     plt.ylabel("Pred")
        #     plt.title(f"plot_type_{col}")
        #     plt.grid(True)
            
        #     plt.xlim(0, 1)
        #     plt.ylim(0, 1)
            
        #     # Save to file
        #     file_path = os.path.join(res_path, f"plot_type_{col}.png")
        #     plt.savefig(file_path, dpi=300, bbox_inches="tight")
        #     plt.close()

        # # Save the embeddings
        # pred_df_discrete.to_csv(os.path.join(res_path, f"{dataset}_pred_probabilities_discrete.csv"), index=False)
        # real_df_discrete.to_csv(os.path.join(res_path, f"{dataset}_real_probabilities_discrete.csv"), index=False) 
        # real_pred_df_discrete.to_csv(os.path.join(res_path, f"{dataset}_train_test_probabilities_discrete.csv"), index=False) 
        # # Save the embeddings
        # pred_df.to_csv(os.path.join(res_path, f"{dataset}_pred_probabilities.csv"), index=False)
        # real_df.to_csv(os.path.join(res_path, f"{dataset}_real_probabilities.csv"), index=False) 
        # real_pred_df.to_csv(os.path.join(res_path, f"{dataset}_train_test_probabilities.csv"), index=False) 
        
        # def plot_line_comparison(real_df, pred_df, res_path, suffix=""):
        #     for col in real_df.columns:
        #         plt.figure(figsize=(8, 4))
        #         plt.plot(real_df[col].values, label="Real", linewidth=2)
        #         plt.plot(pred_df[col].values, label="Pred", linewidth=2)
                
        #         plt.xlabel("Index")
        #         plt.ylabel("Probability")
        #         plt.title(f"{col} - Real vs Predicted")
        #         plt.legend()
        #         plt.grid(True)
        #         if suffix != "_discrete":
        #             plt.ylim(0, 1)

        #         # Save plot
        #         file_path = os.path.join(res_path, f"{col}{suffix}.png")
        #         plt.savefig(file_path, dpi=300, bbox_inches="tight")
        #         plt.close()


        # # --- Add after saving CSVs ---
        # plot_line_comparison(real_df_discrete, pred_df_discrete, res_path, suffix="_discrete")
        # plot_line_comparison(real_df, pred_df, res_path)
        
        # def print_pred_stats(pred_df, label=""):
        #     '''
        #     Print mean and std deviation for each column in a prediction dataframe.
        #     '''
        #     print(f"\n--- Predicted {label} Statistics ---")
        #     for col in pred_df.columns:
        #         mean_val = pred_df[col].mean()
        #         std_val = pred_df[col].std()
        #         print(f"{col}: mean = {mean_val:.6f}, std = {std_val:.6f}")


        # # --- Add this after creating pred_df_discrete and pred_df ---
        # print_pred_stats(pred_df_discrete, label="Discrete")
        # print_pred_stats(pred_df, label="Normalized")
        
    
           
    
main()