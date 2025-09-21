import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from utils.utils import Utils
from utils.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from nn.custom_model import Decoder

from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.metrics.point import RMSE
import pytorch_lightning as pl
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error



RESULTS_PATH = 'data/output/figures/scaleTrue2/'
#RESULTS_PATH = 'data/output/figures/scaleFalse2/'


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

# For plotting the predictions vs real values
def plot_line_comparison(pred_series, true_series, res_path, column_name, suffix="", lookback=7):
    pred = np.asarray(pred_series).flatten()
    true = np.asarray(true_series).flatten()
    
    os.makedirs(res_path, exist_ok=True)
    
    plt.figure(figsize=(8, 4))
    plt.plot(true_series[lookback:], label="Real", linewidth=1)
    plt.plot(pred_series, label="Pred", linewidth=1)
    
    plt.xlabel("Index")
    plt.ylabel("Count")
    plt.title(f"{column_name} - Real vs Predicted")
    plt.legend()
    plt.grid(True)
    if suffix != "_discrete":
        plt.ylim(0, 1)

    # Save plot
    file_path = os.path.join(res_path, f"{column_name}{suffix}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()
    



def plot_scatter(predicted, true, save_path, mode="nodes", xlabel="", ylabel=""):
    predicted = np.array(predicted, dtype=np.float32).flatten()
    true = np.array(true, dtype=np.float32).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(predicted, true, alpha=0.6)

    # Axis labels depending on mode
    plt.xlabel(r'$|\hat{p}|$')
    plt.ylabel(r'$|p|$')

    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Always start at (0,0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # No grid, no title, no legend
    plt.grid(False)

    # Make sure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    

probs_to_save = {
    # Dataset: df storing probs, 
    # ...
}

# For indexing the dataframe storing probabilities
prob_types = {
    "prob_old_nodes": 0,
    "prob_new_nodes": 1,
    "prob_oo": 2,
    "prob_nn": 3,
    # "prob_on": 4,
    # "prob_oon": 5,
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


# For a cleaner layout on the plots
renaming_probs_dict = {
    "prob_old_nodes": "Count Old Nodes",
    "prob_new_nodes": "Count New Nodes",
    "prob_oo": "Count Old-Old-Bank Edges",
    "prob_nn": "Count New-New Edges",
    "prob_on": "Count Old-New Edges",
    "prob_oon": "Count Old-Old-Nobank Edges",
}


def test_pytorch_tft(prob_type, learning_rate, hidden_size, attention_head_size, dropout, hidden_continuous_size, dataset):    
    batch_size = 32
    lookback = 7
    horizon = 1

    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(42)

    # Set up probabilities
    probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')  # Activation doesn't matter here
    
    prob_type_idx = prob_types[prob_type]
    series = probabilities_df.iloc[:, prob_type_idx].to_numpy(dtype=np.float32).flatten()
    
    N = len(series)
    train_end = int(N * 0.8)
    val_end = int(N * 0.9)

    # Create a dataframe with time index and series ID (for multiple series)
    df = pd.DataFrame({
        "series_id": 0,                 # unique id for this series
        "time_idx": np.arange(N),       # time steps
        "probability": series,          # target
        "static_feat": "0",             # dummy static categorical
        "future_feat": 0                # dummy known future feature
    })
    training_cutoff = int(N * 0.8)

    # Train dataset
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx < train_end],
        time_idx="time_idx",
        target="probability",
        group_ids=["series_id"],
        max_encoder_length=lookback,
        max_prediction_length=horizon,
        static_categoricals=["static_feat"],
        time_varying_known_reals=["future_feat"],
        time_varying_unknown_reals=["probability"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(center=False)
    )

    # Validation dataset (includes lookback from train end)
    validation = TimeSeriesDataSet(
        df[lambda x: (x.time_idx >= train_end - lookback) & (x.time_idx < val_end)],
        time_idx="time_idx",
        target="probability",
        group_ids=["series_id"],
        max_encoder_length=lookback,
        max_prediction_length=horizon,
        static_categoricals=["static_feat"],
        time_varying_known_reals=["future_feat"],
        time_varying_unknown_reals=["probability"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(center=False)
    )

    # Test dataset (includes lookback from val end)
    test_dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx >= val_end - lookback],
        time_idx="time_idx",
        target="probability",
        group_ids=["series_id"],
        max_encoder_length=lookback,
        max_prediction_length=horizon,
        static_categoricals=["static_feat"],
        time_varying_known_reals=["future_feat"],
        time_varying_unknown_reals=["probability"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(center=False)
    )

    # Dataloaders
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # Initialize TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=1,
        loss=RMSE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
            
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=20,
        verbose=True,
        mode="min"
    )

    # Trainer
    trainer = Trainer(
        max_epochs=150,
        callbacks=[early_stop_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=0.1,
        enable_progress_bar=False
    )

    # Train
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
        
    full_dataset_predict = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="probability",
        group_ids=["series_id"],
        max_encoder_length=lookback,
        max_prediction_length=horizon,
        static_categoricals=["static_feat"],
        time_varying_known_reals=["future_feat"],
        time_varying_unknown_reals=["probability"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    full_dataloader_predict = full_dataset_predict.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )

    print("Predicting the full dataset with the best model...")
    predictions_tensor = tft.predict(full_dataloader_predict)
    
    # Flatten the predictions to a single list
    single_list_predictions = predictions_tensor.detach().cpu().numpy().flatten().tolist()
    pred_series = single_list_predictions
    
    # To get a better column name
    column_mapper = {
        "prob_old_nodes": "Count Old Nodes",
        "prob_new_nodes": "Count New Nodes",
        "prob_oo": "Count OO",
        "prob_nn": "Count NN",
        "prob_on": "Count ON",
        "prob_oon": "Count OON",
    }
    column_name = column_mapper[prob_type]
    
    if dataset in probs_to_save.keys():
        probs_to_save[dataset][column_name] = pred_series 
    else:
        probs_to_save[dataset] = pd.DataFrame({column_name: pred_series})
        
    res_path = RESULTS_PATH + dataset + '/PredCounts/'
    os.makedirs(res_path, exist_ok=True)
    
    header = renaming_probs_dict[prob_type]  # Used for the plot
    
    plt.figure(figsize=(8, 4))
    plt.plot(series[lookback:], label="Real", linewidth=1)
    plt.plot(pred_series, label="Pred", linewidth=1)
    plt.xlabel("Index")
    plt.ylabel("Count")
    plt.title(f"{prob_type} - Real vs Predicted")
    plt.legend()
    plt.grid(True)

    # Save plot
    file_path = os.path.join(res_path, f"{header} Discrete.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    train_pred = tft.predict(train_loader).detach().cpu().numpy().flatten()
    val_pred   = tft.predict(val_loader).detach().cpu().numpy().flatten()
    test_pred  = tft.predict(test_loader).detach().cpu().numpy().flatten()

    # Ground truth
    train_true = df[lambda x: x.time_idx < train_end]["probability"].to_numpy()
    val_true   = df[lambda x: (x.time_idx >= train_end) & (x.time_idx < val_end)]["probability"].to_numpy()
    test_true  = df[lambda x: x.time_idx >= val_end]["probability"].to_numpy()

    # Metrics
    train_rmse = trainer.callback_metrics.get("train_loss")
    val_rmse   = trainer.callback_metrics.get("val_loss")
    test_rmse  = np.sqrt(mean_squared_error(test_true, test_pred))
        

    return train_rmse, val_rmse, test_rmse



def test_rnns(dataset, prob_type, prob_type_idx, num_back, window_size, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, scale, batch_size, combo, counter, seed):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 1  # Outputting next vector
    input_dim = 1  
    patience = 10  # Early stopping patience
    num_epochs = 500  # Max epochs to train
    

    run_name = dataset
    
    # Set up probabilities
    probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back=num_back)  # Activation doesn't matter here
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

    if scale:
        X_train_log = np.log1p(X_train)
        X_val_log   = np.log1p(X_val)
        X_test_log  = np.log1p(X_test)

        # ----------------------------
        # 2️⃣ Fit MinMaxScaler on training data only
        x_scaler = MinMaxScaler()
        X_train = x_scaler.fit_transform(X_train_log)
        X_val   = x_scaler.transform(X_val_log)
        X_test  = x_scaler.transform(X_test_log)


    train_dataset = EmbeddingDataset(X_train, k=window_size)
    valid_dataset = EmbeddingDataset(X_val, k=window_size)
    test_dataset = EmbeddingDataset(X_test, k=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
      
    no_improvement_counter = 0  # Number of epochs that we haven't seen an improvement in the validation AUCROC
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    criterion = nn.MSELoss() 
    
    curr_batch_best_loss = float('inf')    
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0

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

                if scale:
                    predicted_embedding = x_scaler.inverse_transform(predicted_embedding.reshape(-1, 1))
                    predicted_embedding = np.expm1(predicted_embedding).flatten()

                    real_embedding = x_scaler.inverse_transform(real_embedding.reshape(-1, 1))
                    real_embedding = np.expm1(real_embedding).flatten()

                real_embeddings.append(real_embedding)
                predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                
                all_pred_embeddings.append(predicted_embedding.tolist())
                all_real_embeddings.append(real_embedding.tolist())

                time_index += 1
        
        
        train_loss = (epoch_loss / len(train_loader))
                
        # Validation
        model.eval()
        valid_loss = 0
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

                    if scale:
                        predicted_embedding = x_scaler.inverse_transform(predicted_embedding.reshape(-1, 1))
                        predicted_embedding = np.expm1(predicted_embedding).flatten()

                        real_embedding = x_scaler.inverse_transform(real_embedding.reshape(-1, 1))
                        real_embedding = np.expm1(real_embedding).flatten()

                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction

                    all_pred_embeddings.append(predicted_embedding.tolist())
                    all_real_embeddings.append(real_embedding.tolist())

                    time_index += 1
            
        valid_loss /= len(valid_loader)         
                
        # Testing        
        model.eval()
        test_loss = 0
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

                    if scale:
                        predicted_embedding = x_scaler.inverse_transform(predicted_embedding.reshape(-1, 1))
                        predicted_embedding = np.expm1(predicted_embedding).flatten()

                        real_embedding = x_scaler.inverse_transform(real_embedding.reshape(-1, 1))
                        real_embedding = np.expm1(real_embedding).flatten()

                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction          

                    all_pred_embeddings.append(predicted_embedding.tolist())
                    all_real_embeddings.append(real_embedding.tolist())

                    time_index += 1
            
        test_loss /= len(test_loader)

        # Optimize for the best aucroc
        if valid_loss <= curr_batch_best_loss:
            curr_batch_best_loss = valid_loss
            
            best_pred_embeddings = predicted_embeddings
            best_real_embeddings = real_embeddings
            
            # Save for dataframe
            best_moment_row = {
                'trial': counter,  # For checking Wandb Logs
                'dataset': dataset,
                'seed': seed,
                'num_layer': num_layer,
                'dropout': dropout,
                'hidden_1': hidden_1,
                'hidden_2': hidden_2,
                'lr_val': lr_val,
                'l2_val': l2_val,
                'batch_size': batch_size,
                'scale': scale,
                'combo': combo,
                'trained_epochs': epoch,
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
            
        
    # To get a better column name
    column_mapper = {
        "prob_old_nodes": "Count Old Nodes",
        "prob_new_nodes": "Count New Nodes",
        "prob_oo": "Count OO",
        "prob_nn": "Count NN",
        "prob_on": "Count ON",
        "prob_oon": "Count OON",
    }
    column_name = column_mapper[prob_type]
    
    if dataset in probs_to_save.keys():
        probs_to_save[dataset][column_name] = best_pred_embeddings
    else:
        probs_to_save[dataset] = pd.DataFrame({column_name: best_pred_embeddings})

    res_path = RESULTS_PATH + dataset + '/PredCounts/'
    os.makedirs(res_path, exist_ok=True)

    header = renaming_probs_dict[prob_type]  # Used for the plot

    plt.figure(figsize=(8, 4))
    plt.plot(best_real_embeddings, label="Real", linewidth=1)
    plt.plot(best_pred_embeddings, label="Pred", linewidth=1)
    plt.xlabel("Index")
    plt.ylabel("Count")
    plt.title(f"{prob_type} - Real vs Predicted")
    plt.legend()
    plt.grid(True)

    # Save plot
    file_path = os.path.join(res_path, f"{header} Discrete.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
        
    return best_moment_row['train_loss'], best_moment_row['valid_loss'], best_moment_row['test_loss']


def main():
    rnn = True
    for prob_type in prob_types.keys(): 
        if rnn == True:
            TRIALS_HISTORY_PATH = f'data/output/results/ProbabilityTesting/data/grid_search_rnns_individual_regression_multi_{prob_type}_5back.csv'

        else:
            TRIALS_HISTORY_PATH = f'data/output/results/ProbabilityTesting/data/grid_search_tft_individual_regression_multi_{prob_type}_5back.csv'
        df = pd.read_csv(TRIALS_HISTORY_PATH)
        
        #df["bayesian_score"] = 0.4 * df["train_loss"] + 0.6 * df["valid_loss"]    
        df["bayesian_score"] = df["train_loss"]  # Just doing train loss now
        #df = df[df['scale'] == False]        
        # # Get the valid groups and find our best trial
        # group_counts = df.groupby("trial").size().reset_index(name="count")
        # valid_groups = group_counts[group_counts["count"] >= 13]["trial"]  # Keep only groups with at least 13 entries
        # group_means = df[df["trial"].isin(valid_groups)].groupby("trial")["bayesian_score"].mean().reset_index()  # Compute the mean score for valid groups only
        # best_group = group_means.loc[group_means["bayesian_score"].idxmin()]["trial"]  # Find the trial group with the lowest average score            
        # best_trials_group = df[df["trial"] == best_group]  # Optionally, display the best trials for each dataset within that group

        # # Our best trials
        # best_trials = best_trials_group.loc[
        #     best_trials_group.groupby("dataset")["bayesian_score"].idxmin()
        # ].reset_index(drop=True)
    
        best_trials = df.loc[
            df.groupby("dataset")["bayesian_score"].idxmin()
        ].reset_index(drop=True)
            
        # Loop and print results as variable assignments
        for _, row in best_trials.iterrows():
            print(f"\n🔧 Best Trial for Dataset: {row['trial']}\n{'-'*40}")

            if rnn == True:

                # Extract parameters
                num_layer = int(row['num_layer'])
                hidden_1 = int(row['hidden_1'])
                hidden_2 = int(row['hidden_2'])  
                dropout = float(row['dropout'])
                lr_val = float(row['lr_val'])
                l2_val = float(row['l2_val'])
                scale = row['scale']
                batch_size = int(row['batch_size'])
                combo = row['combo']
                dataset=row['dataset']
                train_loss, val_loss, test_loss = test_rnns(
                    dataset=dataset,
                    prob_type=prob_type, 
                    prob_type_idx=prob_types[prob_type], 
                    num_back=5, 
                    window_size=7, 
                    num_layer=num_layer, 
                    dropout=dropout, 
                    hidden_1=hidden_1, 
                    hidden_2=hidden_2, 
                    lr_val=lr_val, 
                    l2_val=l2_val, 
                    scale=scale,
                    batch_size=batch_size, 
                    combo=combo_map[combo],
                    counter=999, 
                    seed=42
                )


            else:
                # Extract parameters
                learning_rate = float(row['learning_rate'])
                hidden_size = int(row['hidden_size'])
                attention_head_size = int(row['attention_head_size'])  
                dropout = float(row['dropout'])
                hidden_continuous_size = int(row['hidden_continuous_size'])
                dataset=row['dataset']
                
                train_loss, val_loss, test_loss = test_pytorch_tft(
                    prob_type=prob_type, 
                    learning_rate=learning_rate, 
                    hidden_size=hidden_size, 
                    attention_head_size=attention_head_size, 
                    dropout=dropout, 
                    hidden_continuous_size=hidden_continuous_size,
                    dataset=dataset
                )
            
            print(f'RESULTS | {dataset} | {prob_type}')
            print(f'Train RMSE: {train_loss}')
            print(f'Valid RMSE: {val_loss}')
            print(f'Test RMSE: {test_loss}')

    # Process the predictions and display them
    for dataset, df in probs_to_save.items():
        res_path = RESULTS_PATH + dataset + '/PredCounts/'
        os.makedirs(res_path, exist_ok=True)
        
        csv_storage_path = os.path.join(res_path, f'pred_counts.csv')
        df.to_csv(csv_storage_path, index=False)
        
        window_size=7
        
        my_loader = Loader()
        
        # Set up probabilities
        probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')  # Activation doesn't matter here
    
        normalized_true_probs = probabilities_df.apply(
            lambda row: pd.Series(normalize_vector_by_groups(row)), axis=1
        )
        normalized_true_probs.columns = probabilities_df.columns  # keep same column names

        normalized_pred_df = df.apply(
            lambda row: pd.Series(normalize_vector_by_groups(row)), axis=1
        )
        normalized_pred_df.columns = df.columns
        csv_storage_path = os.path.join(res_path, f"normalized_pred_counts.csv")
        normalized_pred_df.to_csv(csv_storage_path, index=False)
        
        # Plot normalized columns
        for prob_type, prob_type_idx in prob_types.items():
            discrete_series_true = probabilities_df.iloc[:, prob_type_idx].to_numpy(dtype=np.float32).flatten()
            discrete_series_pred = df.iloc[:, prob_type_idx].to_numpy(dtype=np.float32).flatten()
            norm_series_true = normalized_true_probs.iloc[:, prob_type_idx].to_numpy(dtype=np.float32).flatten()
            norm_series_pred = normalized_pred_df.iloc[:, prob_type_idx].to_numpy(dtype=np.float32).flatten()

            plot_line_comparison(discrete_series_pred, discrete_series_true, res_path + 'DiscretePlots', column_name=prob_type, suffix="_discrete", lookback=window_size)
            plot_line_comparison(norm_series_pred, norm_series_true, res_path + 'NormalizedPlots', column_name=prob_type, lookback=window_size)

            scatter_path = 'data/output/figures/CountsScatter/'
            os.makedirs(res_path, exist_ok=True)

            plot_scatter(discrete_series_pred, discrete_series_true[7:], scatter_path + f'Discrete/{dataset}_{prob_type}_discrete.png')
            plot_scatter(norm_series_pred, norm_series_true[7:], scatter_path + f'Norm/{dataset}_{prob_type}_norm.png')
    
main()