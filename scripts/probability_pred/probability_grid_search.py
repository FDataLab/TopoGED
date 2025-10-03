import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import optuna
import sqlite3
from sklearn.preprocessing import MinMaxScaler

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import concurrent.futures

from utils.loader import Loader
from utils.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from nn.custom_model import Decoder
from torch.utils.data import DataLoader, Dataset
from utils.utils import Utils

import wandb

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pytorch_forecasting.metrics import MAE, QuantileLoss, SMAPE
from pytorch_forecasting.metrics.point import RMSE, PoissonLoss
from pytorch_forecasting import TimeSeriesDataSet, EncoderNormalizer, GroupNormalizer
from pytorch_forecasting.metrics.distributions import NegativeBinomialDistributionLoss
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from torch.nn import functional as F

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


def train_and_eval(prob_type, prob_type_idx, num_back, window_size, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, scale, batch_size, combo, counter, seed):
    csv_file_path = os.path.abspath(f'data/output/results/ProbabilityTesting/data/grid_search_rnns_individual_regression_multi_{prob_type}_5back.csv')
    
    if not os.path.isfile(csv_file_path):
        pd.DataFrame(columns=['trial', 'dataset', 'seed', 'num_layer', 'dropout', 'hidden_1', 'hidden_2', 'lr_val', 'l2_val', 'scale', 'batch_size', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss']).to_csv(csv_file_path, index=False)
        
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 1  # Outputting next vector
    input_dim = 1  
    patience = 10  # Early stopping patience
    num_epochs = 500  # Max epochs to train
        
    all_dataset_losses = []
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    for dataset in datasets:
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
                                            
        # Initialize wandb
        run = wandb.init(
            project=f"bayesian_testing_probabilities_rnn_multi_{prob_type}", 
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
            'combo': combo
            },
            reinit=True)
                
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

                        real_embeddings.append(real_embedding)
                        predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction          

                        all_pred_embeddings.append(predicted_embedding.tolist())
                        all_real_embeddings.append(real_embedding.tolist())

                        time_index += 1
                
            test_loss /= len(test_loader)
  
            # Stores our current epoch 'steps' results
            to_log = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss,
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
                
        pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
        all_dataset_losses.append(best_moment_row['train_loss'])
        
    return sum(all_dataset_losses) / len(all_dataset_losses)


def test_pytorch_tft(prob_type, trial_num, window_size, dropout, hidden_size, hidden_continuous_size, lr_val, batch_size, attention_head_size, seed=42):
    csv_file_path = os.path.abspath(f'data/output/results/ProbabilityTesting/data/grid_search_tft_individual_regression_multi_{prob_type}_5back.csv')
    
    if not os.path.isfile(csv_file_path):
        pd.DataFrame(columns=['trial', 'dataset', 'seed', 'learning_rate', 'hidden_size', 'attention_head_size', 'dropout', 'hidden_continuous_size', 'batch_size', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss']).to_csv(csv_file_path, index=False)
        
    # Static parameters
    lookback = 7
    horizon = 1
    
    all_dataset_losses = []
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    for dataset in datasets:
        run = wandb.init(
        project=f"rnn_multi_{prob_type}_{dataset}_5_back_grid_search", 
        name = dataset + '_' + str(trial_num) , 
        reinit=True)
        
        my_utils = Utils()
        my_utils.set_seeds(42)

        probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')
        prob_type_idx = prob_types[prob_type]
        series = probabilities_df.iloc[:, prob_type_idx].to_numpy(dtype=np.float32).flatten()
        N = len(series)

        df = pd.DataFrame({
            "series_id": 0,
            "time_idx": np.arange(N),
            "probability": series,
            "static_feat": "0",
            "future_feat": 0
        })
        training_cutoff = int(N * 0.8)

        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
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

        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
        train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

        # Initialize TFT model with optimized parameters
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=lr_val,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=RMSE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=15,
            verbose=True,
            mode="min"
        )

        # Trainer
        trainer = Trainer(
            max_epochs=250,
            callbacks=[early_stop_callback],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            gradient_clip_val=0.1,
        )

        # Train
        trainer.fit(
            tft,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        try:
            val_loss = trainer.callback_metrics["val_loss"].item()
            train_loss = trainer.callback_metrics["train_loss"].item()
            #all_dataset_losses.append(val_loss * 0.6 + train_loss * 0.4)
            all_dataset_losses.append(train_loss)  # Just train loss for now
            # history = trainer.logger.history
            # all_history = {
            #     'train_loss': history['train_loss'],
            #     'valid_loss': history['valid_loss'],
            # }
            
            # num_epochs = len(all_history['train_loss'])

            # # Iterate through each epoch's loss and log it to WandB
            # for i in range(num_epochs):
            #     wandb.log({
            #         "train_loss": all_history['train_loss'][i],
            #         "valid_loss": all_history['valid_loss'][i],
            #         "epoch": i
            #     })
            
            best_moment_row = {
                'trial': trial_num,  # For checking Wandb Logs
                'dataset': dataset,
                'seed': seed,
                'learning_rate': lr_val,
                'hidden_size': hidden_size,
                'attention_head_size': attention_head_size,
                'dropout': dropout,
                'hidden_continuous_size': hidden_continuous_size,
                'batch_size': batch_size,
                'trained_epochs': trainer.current_epoch + 1,
                'train_loss': train_loss,
                'valid_loss': val_loss,
            }
            
            pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
            
        except optuna.exceptions.TrialPruned:
            raise # Re-raise for Optuna to handle
        except Exception as e:
            print(f"Error for dataset {dataset}: {e}")
            all_dataset_losses.append(float('inf'))

    # Calculate and return the average loss across all datasets
    return sum(all_dataset_losses) / len(all_dataset_losses)


def runnerTft(prob_type, starting_point):  
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # import torch
    # torch.set_num_threads(1)    
  
    window_size = 7  # Trying to fix window size for now
    dropout_list = [0, 0.1, 0.2]
    hidden_size_list = [16, 32]
    hidden_continuous_size_list = [8, 16]
    lr_val_list = [1e-5, 1e-4, 1e-3]
    batch_size_list = [16]
    attention_head_size_list = [1, 2]
    
    dropout_list = [0.0, 0.05]  # keep almost none to avoid underfitting
    hidden_size_list = [64, 128, 256]  # much bigger than before
    hidden_continuous_size_list = [32, 64]  # more representational power
    attention_head_size_list = [2, 4, 6]  # richer temporal focus
    lr_val_list = [1e-3, 5e-3]  # avoid too small lr
    batch_size_list = [16]
    l2_val_list = [0]  # no weight decay for now
    # Target is 72 runs each now. since only using batch size of 16
    
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    best_train_loss = float('inf')
    
    try:
        trial_num = 1000  # Update as we go
        for batch_size in batch_size_list:
            for dropout in dropout_list:
                for hidden_size in hidden_size_list:
                    for hidden_continuous_size in hidden_continuous_size_list:
                        for attention_head_size in attention_head_size_list:
                            for lr_val in lr_val_list:
                                trial_num += 1
                                
                                if trial_num <= starting_point:
                                    continue
                                
                                res = test_pytorch_tft(
                                    prob_type=prob_type,
                                    window_size=window_size,
                                    dropout=dropout,
                                    hidden_size=hidden_size,
                                    hidden_continuous_size=hidden_continuous_size,
                                    lr_val=lr_val,
                                    batch_size=batch_size,
                                    attention_head_size=attention_head_size,
                                    trial_num=trial_num,
                                    seed=42,
                                )

                                if res < best_train_loss:
                                    best_trial = {
                                        'dropout': dropout, 
                                        'hidden_size': hidden_size, 
                                        'hidden_continuous_size': hidden_continuous_size, 
                                        'lr_val': lr_val, 
                                        'batch_size': batch_size, 
                                        'attention_head_size': attention_head_size,
                                        'trial_num': trial_num, 
                                    }

        print(f'Completed {prob_type}!')

    except Exception as e:
        print(e)
        print(f'{prob_type} | The best trial is: {best_trial}')
        print(f'{prob_type} | On trial number: {trial_num}')
    
    
def runnerRNNs(prob_type, starting_point):
    dropout_list = [0, 0.05]
    hidden_1_list = [128, 256]
    hidden_2_list = [64, 128]
    num_layers_list = [2, 3, 4]
    lr_val_list = [1e-4, 1e-3, 1e-2]
    l2_val_list = [0, 1e-5]
    batch_size_list = [8, 16]
    model_list = [
        "['LSTM', 'FC']",
        "['GRU', 'FC']",
        "['LSTM', 'GRU', 'FC']",
        "['LSTM', 'MLP']", 
        "['GRU', 'MLP']", 
        "['LSTM', 'GRU', 'MLP']"
    ]
    # Target is 72 runs each now. since only using batch size of 16
        
    try:
        trial_num = 1000  # Update as we go
        for batch_size in batch_size_list:
            for num_layer in num_layers_list:
                for dropout in dropout_list:
                    for hidden_1 in hidden_1_list:
                        for hidden_2 in hidden_2_list:
                            for lr_val in lr_val_list:
                                for l2_val in l2_val_list:
                                    for combo in model_list:
                                        for scale in [True, False]:
                                            trial_num += 1
                                            
                                            if trial_num <= starting_point:
                                                continue
                                            
                                            if num_layer <= 2:
                                                continue
                                            
                                            
                                            
                                            res = train_and_eval(
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
                                                counter=trial_num, 
                                                seed=42
                                            )

        print(f'Completed {prob_type}!')

    except Exception as e:
        print(e)
        print(f'{prob_type} | On trial number: {trial_num}')    
        
        
    
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class ltTopER(nn.Module):
    def __init__(self):
        super(ltTopER, self).__init__()

        self.alpha0 = nn.Parameter(torch.tensor(0.5))
        self.alpha1 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_last = x[:, -1, :]  # [batch_size, 1]
        x_hat = self.alpha0 + self.alpha1 * x_last
        return x_hat
    
        
def test_astrit(dataset, prob_type, prob_type_idx, num_back, window_size, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, scale, batch_size, counter, seed):
    csv_file_path = os.path.abspath(f'data/output/results/ProbabilityTesting/data/astrit_individual_regression_multi_{prob_type}_5back.csv')
    
    if not os.path.isfile(csv_file_path):
        pd.DataFrame(columns=['trial', 'dataset', 'seed', 'num_layer', 'dropout', 'hidden_1', 'hidden_2', 'lr_val', 'l2_val', 'scale', 'batch_size', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss']).to_csv(csv_file_path, index=False)
        
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
                                        
    # Initialize wandb
    run = wandb.init(
        project=f"astrit_multi_{prob_type}", 
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
        },
        reinit=True)
            
    no_improvement_counter = 0  # Number of epochs that we haven't seen an improvement in the validation AUCROC
    model = ltTopER()
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
                
                y = y.float()
                loss = criterion(output, y)
                valid_loss += loss.item()
                
                # Print time index, predicted embedding, and real embedding
                for i in range(len(x)):
                    predicted_embedding = output[i].detach().cpu().numpy()
                    real_embedding = y[i].detach().cpu().numpy()

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
                
                y = y.float()
                loss = criterion(output, y)
                test_loss += loss.item()
                                # Print time index, predicted embedding, and real embedding
                for i in range(len(x)):
                    predicted_embedding = output[i].detach().cpu().numpy()
                    real_embedding = y[i].detach().cpu().numpy()

                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction          

                    all_pred_embeddings.append(predicted_embedding.tolist())
                    all_real_embeddings.append(real_embedding.tolist())

                    time_index += 1
            
        test_loss /= len(test_loader)

        # Stores our current epoch 'steps' results
        to_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'test_loss': test_loss,
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
            
    pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
        
    return best_moment_row['train_loss'], best_moment_row['valid_loss'], best_pred_embeddings, best_real_embeddings


def run_astrit():
    prob_types = {
        "prob_old_nodes": 0,
        "prob_new_nodes": 1,
        # "prob_oo": 2,
        # "prob_nn": 3,
        # "prob_on": 4,
        # "prob_oon": 5,
    }
    dropout_list = [0, 0.05]
    hidden_1_list = [64, 128]
    hidden_2_list = [32, 64]
    num_layers_list = [2, 3]
    lr_val_list = [1e-3, 1e-2]
    l2_val_list = [0, 1e-5]
    batch_size_list = [16]
    scales = [True, False]
    # Target is 72 runs each now. since only using batch size of 16
        
    datasets = ['networkaeternity','networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon',  'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    for dataset in datasets:
        trial_num = -1  # Update as we go
        best_train_loss_scale_true = float('inf')
        best_train_loss_scale_false = float('inf')

        for batch_size in batch_size_list:
            for num_layer in num_layers_list:
                for dropout in dropout_list:
                    for hidden_1 in hidden_1_list:
                        for hidden_2 in hidden_2_list:
                            for lr_val in lr_val_list:
                                for l2_val in l2_val_list:
                                    for scale in scales:
                                        trial_num += 1
                                        for prob_type in prob_types.keys():
                                            train_loss, valid_loss, pred_embeddings, real_embeddings = test_astrit(
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
                                                counter=trial_num, 
                                                seed=42
                                            )
                                        
                                            if scale == True:
                                                if train_loss < best_train_loss_scale_true:
                                                    # Plot embeddings vs each other
                                                    best_pred_embeddings = np.array(pred_embeddings).flatten()
                                                    best_real_embeddings = np.array(real_embeddings).flatten()

                                                    plt.figure(figsize=(10, 6))
                                                    plt.plot(best_real_embeddings, color='blue', label='Real values')
                                                    plt.plot(best_pred_embeddings, color='orange', label='Predicted values')
                                                    plt.xlabel("Time index")
                                                    plt.ylabel("Value")
                                                    plt.title(f"{dataset}_{prob_type}")
                                                    plt.legend()
                                                    plot_dir = os.path.abspath("data/output/figures/astrit/dataset")
                                                    os.makedirs(plot_dir, exist_ok=True)
                                                    plot_path = os.path.join(plot_dir, f"{dataset}_{prob_type}_scale_true.png")
                                                    plt.savefig(plot_path)
                                                    plt.close()
                                            else:
                                                if train_loss < best_train_loss_scale_false:
                                                    # Plot embeddings vs each other
                                                    best_pred_embeddings = np.array(pred_embeddings).flatten()
                                                    best_real_embeddings = np.array(real_embeddings).flatten()

                                                    plt.figure(figsize=(10, 6))
                                                    plt.plot(best_real_embeddings, color='blue', label='Real values')
                                                    plt.plot(best_pred_embeddings, color='orange', label='Predicted values')
                                                    plt.xlabel("Time index")
                                                    plt.ylabel("Value")
                                                    plt.title(f"{dataset}_{prob_type}")
                                                    plt.legend()
                                                    plot_dir = os.path.abspath("data/output/figures/astrit/dataset")
                                                    os.makedirs(plot_dir, exist_ok=True)
                                                    plot_path = os.path.join(plot_dir, f"{dataset}_{prob_type}_scale_false.png")
                                                    plt.savefig(plot_path)
                                                    plt.close()
        
    
def runnerRNNs(prob_type, starting_point):
    dropout_list = [0, 0.05]
    hidden_1_list = [128, 256]
    hidden_2_list = [64, 128]
    num_layers_list = [2, 3, 4]
    lr_val_list = [1e-4, 1e-3, 1e-2]
    l2_val_list = [0, 1e-5]
    batch_size_list = [8, 16]
    # Target is 72 runs each now. since only using batch size of 16
        
    try:
        trial_num = 1000  # Update as we go
        for batch_size in batch_size_list:
            for num_layer in num_layers_list:
                for dropout in dropout_list:
                    for hidden_1 in hidden_1_list:
                        for hidden_2 in hidden_2_list:
                            for lr_val in lr_val_list:
                                for l2_val in l2_val_list:
                                    for combo in model_list:
                                        for scale in [True, False]:
                                            trial_num += 1
                                            
                                            if trial_num <= starting_point:
                                                continue
                                            
                                            if num_layer <= 2:
                                                continue
                                            
                                            
                                            
                                            res = train_and_eval(
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
                                                counter=trial_num, 
                                                seed=42
                                            )

        print(f'Completed {prob_type}!')

    except Exception as e:
        print(e)
        print(f'{prob_type} | On trial number: {trial_num}')    
    
if __name__ == "__main__":
    run_astrit()
# if __name__ == "__main__":
#     os.environ["WANDB_API_KEY"] = "6a5ccf040a6c90944032e58878e46c19d673cdb0"
#     wandb.init(project="Probabilities", name="gridsearch_testing_probabilities_tft")
    
#     prob_type_keys = list(prob_types.keys()) 
    
#     # TFT
#     starting_points = {
#         "prob_old_nodes": 1007,
#         "prob_new_nodes": 1008,
#         "prob_oo": 1011,
#         "prob_nn": 1022,
#         "prob_on": 1019,
#         "prob_oon": 1007,
#     }
    
#     # RNNs
#     starting_points = {
#         "prob_old_nodes": 1018,
#         "prob_new_nodes": 1018,
#         "prob_oo": 1023,
#         "prob_nn": 1022,
#         "prob_on": 1019,
#         "prob_oon": 1019,
#     }

#     with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
#         # Map each prob_type to its runner in parallel
#         futures = {executor.submit(runnerRNNs, pt, starting_points[pt]): pt for pt in prob_type_keys}

#         for future in concurrent.futures.as_completed(futures):
#             pt = futures[future]
#             try:
#                 future.result()  # Run and raise any exceptions
#                 print(f"{pt} finished successfully")
#             except Exception as e:
#                 print(f"{pt} raised an exception: {e}")