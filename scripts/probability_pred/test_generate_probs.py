import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import optuna

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
model_dir = os.path.abspath('data/output/cached_model/RegressionTesting/EmbeddingTesting')
STORAGE = "sqlite:///./output/cached_model/RegressionTesting/bayesianSave/model_selection_regression_20dim_deltas.db"  # Where we save the study
os.makedirs(os.path.dirname('output/cached_model/ProbabilityTesting/bayesianSave/model_selection_regression_20dim_deltas.db'), exist_ok=True)
os.makedirs('data/output/results/ProbabilityTesting/data/SeparatedModels', exist_ok=True)
os.makedirs('data/output/cached_model/ProbabilityTesting/SeparatedModels', exist_ok=True)

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
    "['GRU', 'LSTM', 'FC']": ['GRU', 'LSTM', 'FC'],
    "['RNN', 'LSTM', 'GRU', 'FC']": ['RNN', 'LSTM', 'GRU', 'FC'],
    "['LSTM', 'FC', 'FC']": ['LSTM', 'FC', 'FC'],
    "['GRU', 'FC', 'FC']": ['GRU', 'FC', 'FC'],
    "['LSTM', 'GRU', 'FC']": ['LSTM', 'GRU', 'FC'],
    "['RNN', 'MLP']": ['RNN', 'MLP'],
    "['LSTM', 'MLP', 'Sigmoid']": ['LSTM', 'MLP', 'Sigmoid'], 
    "['GRU', 'MLP', 'Sigmoid']": ['GRU', 'MLP', 'Sigmoid'], 
    "['LSTM', 'GRU', 'MLP', 'Sigmoid']": ['LSTM', 'GRU', 'MLP', 'Sigmoid']
}


prob_types = {
    "prob_old_nodes": 0,
    "prob_new_nodes": 1,
    "prob_oo": 2,
    "prob_n": 3,
    "prob_on": 4,
    #"prob_oon": 5,
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



def train_and_eval(dataset, prob_type, prob_type_idx, num_back, window_size, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed):
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
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
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
                'seed': seed,
                'window_size': window_size,
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


from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

def test_arima():
    datasets = ['networkbancor', 'CollegeMsg', 'mathoverflow', 'networkaragon', 'networkaion', 'networkadex', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkaeternity', 'networkiconomi', 'networkcindicator', 'networkdgd']  # replace with actual dataset names
    dataset_dfs = {name: pd.DataFrame() for name in datasets}
    
    for prob_type in prob_types.keys():
        datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

        my_loader = Loader()
        my_utils = Utils()
        my_utils.set_seeds(42)

        for dataset in datasets:
            # Set up probabilities
            probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')  # Activation doesn't matter here
            prob_type_idx = prob_types[prob_type]
            series = probabilities_df.iloc[:, prob_type_idx].to_numpy(dtype=np.float32).flatten()

            print(f"Processing dataset: {dataset} for problem type: {prob_type}")

            try:
                # Automatically select ARIMA order
                model_auto = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
                p, d, q = model_auto.order
                print(f"Selected ARIMA order for {dataset}: (p={p}, d={d}, q={q})")

                # Fit ARIMA
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                print('fit')
                pred_series = model_fit.predict(start=0, end=len(series)-1)  # fitted values
                
                res_path = RESULTS_PATH + dataset + '/PredProbabilitiesSeparatedTesting/'
                
                os.makedirs(res_path, exist_ok=True)
            
                plt.figure(figsize=(8, 4))
                plt.plot(series, label="Real", linewidth=1)
                plt.plot(pred_series, label="Pred", linewidth=1)

                plt.xlabel("Index")
                plt.ylabel("Probability")
                plt.title(f"{prob_type} - Real vs Predicted")
                plt.legend()
                plt.grid(True)

                # Save plot
                file_path = os.path.join(res_path, f"{prob_type}_discrete.png")
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()



            except Exception as e:
                print(f"Error processing {dataset}: {e}")


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pytorch_forecasting.metrics import MAE, QuantileLoss, SMAPE
from pytorch_forecasting.metrics.point import RMSE, PoissonLoss

# from fusionlab.nn.transformers import TemporalFusionTransformer


def test_temporalfusiontransformer():
    datasets = ['networkbancor', 'CollegeMsg', 'mathoverflow', 'networkaragon', 'networkaion', 'networkadex', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkaeternity', 'networkiconomi', 'networkcindicator', 'networkdgd']  # replace with actual dataset names
    dataset_dfs = {name: pd.DataFrame() for name in datasets}
    
    for prob_type in prob_types.keys():
        datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

        my_loader = Loader()
        my_utils = Utils()
        my_utils.set_seeds(42)

        for dataset in datasets:
            # Set up probabilities
            probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')  # Activation doesn't matter here
            prob_type_idx = prob_types[prob_type]
            series = probabilities_df.iloc[:, prob_type_idx].to_numpy(dtype=np.float32).flatten()
            
            lookback = 7   # history length
            horizon = 1   # predict one step ahead
            output_dim = 1
            
            

            # Reshape into (B, T_past, D_dyn)
            # Here: 1 feature (probability value)
            def create_windows(series, lookback, horizon):
                X, y = [], []
                for i in range(len(series) - lookback - horizon + 1):
                    past = series[i : i + lookback]
                    future = series[i + lookback : i + lookback + horizon]
                    X.append(past)
                    y.append(future)
                X = np.array(X)[..., np.newaxis]  # (N, lookback, 1)
                y = np.array(y)[..., np.newaxis]  # (N, horizon, 1)
                return X, y

            X, y = create_windows(series, lookback, horizon)
            
            N = X.shape[0]  # number of windows / samples
            D_stat = 2  # static dummy features
            D_fut = 2   # future dummy features

            # Create zero inputs
            static_in = np.zeros((N, D_stat), dtype=np.float32)
            future_in = np.zeros((N, horizon, D_fut), dtype=np.float32)         
            # === Model ===
            model = TemporalFusionTransformer(
                dynamic_input_dim=1,
                static_input_dim=D_stat,
                future_input_dim=D_fut,
                forecast_horizon=horizon,
                output_dim=output_dim,
                hidden_units=16,
                num_heads=1
            )

            # Compile + Train
            model.compile(optimizer="adam", loss="mse")
            model.fit([static_in, X, future_in], y, epochs=20, batch_size=32, verbose=1)

            # === Rolling Predictions ===
            predictions = []
            for i in range(len(series) - lookback):
                input_window = series[i : i + lookback][..., np.newaxis]       # (lookback, 1)
                input_window = np.expand_dims(input_window, axis=0)            # (1, lookback, 1)
                static_window = np.zeros((1, D_stat), dtype=np.float32)
                future_window = np.zeros((1, horizon, D_fut), dtype=np.float32)
                
                pred = model([static_window, input_window, future_window], training=False)
                predictions.append(pred.numpy().squeeze())

            predictions = np.array(predictions)  # shape (N, horizon)
            pred_series = np.concatenate(predictions)
                
            res_path = RESULTS_PATH + dataset + '/PredProbabilitiesSeparatedTesting/'

            os.makedirs(res_path, exist_ok=True)
            
            
            plt.figure(figsize=(8, 4))
            plt.plot(series, label="Real", linewidth=1)
            plt.plot(range(lookback, lookback + len(pred_series)), pred_series, label="Pred", linewidth=1)

            plt.xlabel("Index")
            plt.ylabel("Probability")
            plt.title(f"{prob_type} - Real vs Predicted")
            plt.legend()
            plt.grid(True)

            # Save plot
            file_path = os.path.join(res_path, f"{prob_type}_discrete.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()
            

from pytorch_forecasting import TimeSeriesDataSet, EncoderNormalizer, GroupNormalizer
from pytorch_forecasting.metrics.distributions import NegativeBinomialDistributionLoss
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from torch.nn import functional as F


probs_to_save = {
    # Dataset: df storing probs, 
    # ...
}
def test_pytorch_tft(prob_type, learning_rate, hidden_size, attention_head_size, dropout, hidden_continuous_size, dataset):    
    batch_size = 32
    lookback = 7
    horizon = 1

    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(42)

    # Load all 6 probability columns
    probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')  
    
    # Which column is the target
    prob_type_idx = prob_types[prob_type]
    target_col = probabilities_df.columns[prob_type_idx]

    # Convert to dataframe with time index + static
    df = probabilities_df.copy().reset_index(drop=True)
    df["series_id"] = 0
    df["time_idx"] = np.arange(len(df))
    df["static_feat"] = "0"   # dummy categorical
    df["future_feat"] = 0     # dummy known feature

    N = len(df)
    train_end = int(N * 0.8)
    val_end = int(N * 0.9)

    # Define feature sets
    known_reals = [c for c in probabilities_df.columns if c != target_col] + ["future_feat"]
    unknown_reals = [target_col]

    # Train dataset
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx < train_end],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=lookback,
        max_prediction_length=horizon,
        static_categoricals=["static_feat"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(center=False),
    )

    # Validation dataset
    validation = TimeSeriesDataSet(
        df[lambda x: (x.time_idx >= train_end - lookback) & (x.time_idx < val_end)],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=lookback,
        max_prediction_length=horizon,
        static_categoricals=["static_feat"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(center=False),
    )

    # Test dataset
    test_dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx >= val_end - lookback],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=lookback,
        max_prediction_length=horizon,
        static_categoricals=["static_feat"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(center=False),
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

    # Predict on full dataset
    full_dataset_predict = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        max_encoder_length=lookback,
        max_prediction_length=horizon,
        static_categoricals=["static_feat"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    full_dataloader_predict = full_dataset_predict.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )

    predictions_tensor = tft.predict(full_dataloader_predict)
    pred_series = predictions_tensor.detach().cpu().numpy().flatten().tolist()

    # For plotting
    series = df[target_col].to_numpy(dtype=np.float32).flatten()

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

    header = renaming_probs_dict[prob_type]

    plt.figure(figsize=(8, 4))
    plt.plot(series[lookback:], label="Real", linewidth=1)
    plt.plot(pred_series, label="Pred", linewidth=1)
    plt.xlabel("Index")
    plt.ylabel("Count")
    plt.title(f"{prob_type} - Real vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

    # Predictions per split
    train_pred = tft.predict(train_loader).detach().cpu().numpy().flatten()
    val_pred   = tft.predict(val_loader).detach().cpu().numpy().flatten()
    test_pred  = tft.predict(test_loader).detach().cpu().numpy().flatten()

    train_true = df[lambda x: x.time_idx < train_end][target_col].to_numpy()
    val_true   = df[lambda x: (x.time_idx >= train_end) & (x.time_idx < val_end)][target_col].to_numpy()
    test_true  = df[lambda x: x.time_idx >= val_end][target_col].to_numpy()

    # Metrics
    train_rmse = trainer.callback_metrics.get("train_loss")
    val_rmse   = trainer.callback_metrics.get("val_loss")
    test_rmse  = np.sqrt(mean_squared_error(test_true, test_pred))

    return train_rmse, val_rmse, test_rmse


def objective(trial, prob_type):
    csv_file_path = os.path.abspath(f'data/output/results/ProbabilityTesting/data/tft_individual_regression_multi_{prob_type}_5back.csv')
    
    if not os.path.isfile(csv_file_path):
        pd.DataFrame(columns=['trial', 'dataset', 'seed', 'learning_rate', 'hidden_size', 'attention_head_size', 'dropout', 'hidden_continuous_size', 'batch_size', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss']).to_csv(csv_file_path, index=False)

    
    # Set up hyperparameters to be optimized
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128])
    attention_head_size = trial.suggest_categorical('attention_head_size', [1, 2, 3, 4])
    dropout = trial.suggest_float('dropout', 0.0, 0.4)
    hidden_continuous_size = trial.suggest_categorical('hidden_continuous_size', [8, 16, 32, 64])
    
    # Static parameters
    lookback = 7
    horizon = 1
    batch_size = 32
    
    all_dataset_losses = []
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(42)
    for dataset in datasets:
        run = wandb.init(
        project=f"tft_multi_{prob_type}_{dataset}_5_back_new", 
        name = str(trial.number) , 
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
            learning_rate=learning_rate,
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
            all_dataset_losses.append(val_loss * 0.6 + train_loss * 0.4)
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
                'trial': trial.number,  # For checking Wandb Logs
                'dataset': dataset,
                'seed': 42,
                'learning_rate': learning_rate,
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
        


def main():
    path = 'data/output/results/ProbabilityTesting/data/probability_testing_bayesian_individual_regression_multi_'

    datasets = ['networkbancor', 'CollegeMsg', 'mathoverflow', 'networkaragon', 'networkaion', 'networkadex', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkaeternity', 'networkiconomi', 'networkcindicator', 'networkdgd']  # replace with actual dataset names
    dataset_dfs = {name: pd.DataFrame() for name in datasets}
    
    for prob_type in prob_types.keys():
        datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

        for dataset in datasets:
            # Extract parameters
            window_size = 7
            dropout = 0.3
            hidden_1 = 128 
            hidden_2 = 64
            num_layers = 3
            lr_val = 0.001
            l2_val = 1e-4
            batch_size = 32
            model = combo_map["['LSTM', 'GRU', 'MLP']"]
            dataset=dataset
            
            
            train_loss, val_loss, pred_embeddings, real_embeddings = train_and_eval(
                dataset=dataset,
                prob_type=prob_type,
                prob_type_idx=prob_types[prob_type],
                window_size=window_size,
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
                seed=42,
            )

            loss_score = (train_loss * 0.4 + val_loss * 0.6)  # Play with these numbers a bit, (0.2, 0.8) and (0.4, 0.6)

                
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

            #print(f"The old score was {row['bayesian_score']} and the new one was {loss_score}")
            
            res_path = RESULTS_PATH + dataset + '/PredProbabilitiesSeparatedTesting/'

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
    


if __name__ == "__main__":
    import multiprocessing as mp

    # loop over all prob_types
    # with mp.Pool(processes=len(prob_types)) as pool:
    #     pool.map(test_pytorch_tft, prob_types.keys())
        
    # for prob_type in prob_types.keys():
    #     test_pytorch_tft(prob_type)    
        
    # for dataset in probs_to_save.keys(): 
    #     res_path = RESULTS_PATH + dataset + '/PredProbabilitiesSeparatedTestingRMSE/'
    #     probs_to_save[dataset].to_csv(res_path + 'pred_probabilities.csv', index=False)
    #     print(f'Probabilities for {dataset} saved')
        
    for prob_type in prob_types.keys():
        os.environ["WANDB_API_KEY"] = "6a5ccf040a6c90944032e58878e46c19d673cdb0"
        wandb.init(project="Probabilities", name=f"tft_optimization_{prob_type}")
        #optuna.delete_study(study_name=f"tft_optimization_{prob_type}", storage=STORAGE)
        study = optuna.create_study(study_name=f"tft_optimization_{prob_type}", storage=STORAGE, direction="minimize", load_if_exists=True)
        #study.optimize(lambda trial: objective(trial, prob_type), n_trials=6)

        best_trial = study.best_trial
        print(f"\nBest trial for prob_type '{prob_type}':")
        print(f"  Trial number: {best_trial.number}")
        print(f"Best trial: {study.best_trial}")
        print(f"  Value: {best_trial.value}")
        print(f"  Params:")
        for param_name, param_value in best_trial.params.items():
            print(f"    {param_name}: {param_value}")
            
        params = best_trial.params
        test_pytorch_tft(
            prob_type=prob_type,
            learning_rate=params.get("learning_rate"),
            hidden_size=params.get("hidden_size"),
            attention_head_size=params.get("attention_head_size"),
            dropout=params.get("dropout"),
            hidden_continuous_size=params.get("hidden_continuous_size")
        )
        
    prob_type = "prob_oon" 
    test_pytorch_tft(
        prob_type=prob_type,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.05,
        hidden_continuous_size=8,
        
    )
        
    
    for dataset in probs_to_save.keys(): 
        res_path = RESULTS_PATH + dataset + '/PredProbabilities/'
        probs_to_save[dataset].to_csv(res_path + 'pred_probabilities.csv', index=False)
        print(f'Probabilities for {dataset} saved')

#test_arima()
#main()