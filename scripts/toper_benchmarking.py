import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# For benchmarking
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from darts import TimeSeries
from darts.models import TCNModel
from statsmodels.tsa.api import VAR
from pytorch_lightning import Trainer


# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from utils.utils import Utils

RESULTS_PATH = 'data/input/cached/'


def benchmarkVAR(dataset, X_train, X_val, X_test, window_size):
    X_train_2d = np.array(X_train).reshape(-1, X_train.shape[-1])
    X_val_2d = np.array(X_val).reshape(-1, X_val.shape[-1])
    X_test_2d = np.array(X_test).reshape(-1, X_test.shape[-1])

    model = VAR(X_train_2d)
    results = model.fit(maxlags=window_size)

    # Forecasts
    train_forecast = results.fittedvalues
    val_forecast = results.forecast(X_train_2d[-window_size:], steps=len(X_val_2d))
    test_forecast = results.forecast(X_val_2d[-window_size:], steps=len(X_test_2d))

    # RMSEs (skip first window_size points)
    train_rmse = np.sqrt(mean_squared_error(X_train_2d[window_size:], train_forecast))
    val_rmse = np.sqrt(mean_squared_error(X_val_2d, val_forecast))
    test_rmse = np.sqrt(mean_squared_error(X_test_2d, test_forecast))

    print(f'RESULTS | VAR | {dataset}')
    print(f'Train RMSE: {train_rmse}')
    print(f'Valid RMSE: {val_rmse}')
    print(f'Test RMSE: {test_rmse}')
    

def embeddings_to_timeseries(embeddings):
    """
    utility function for TCN
    """
    series_list = []
    for dim in range(embeddings.shape[1]):
        series_list.append(TimeSeries.from_values(embeddings[:, dim]))
    return series_list

def benchmarkTCN(dataset, X_train, X_val, X_test, window_size=7, n_epochs=200, batch_size=32, random_state=42):
    """
    window_size: input sequence length
    returns trained TCN model and train/val RMSE
    """    
    # Create Darts TimeSeries (multivariate)
    train_series = TimeSeries.from_values(X_train)
    val_series = TimeSeries.from_values(X_val)
    test_series = TimeSeries.from_values(X_test)

    # Disable Lightning output
    trainer = Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        max_epochs=250 
    )

    model = TCNModel(
        input_chunk_length=window_size,
        output_chunk_length=1,
        batch_size=batch_size,
        kernel_size=3,
        num_filters=64,
        dropout=0.1,
        random_state=random_state,
        likelihood=None
    )

    # Train for n_epochs quietly
    model.fit(train_series, val_series=val_series, verbose=False, trainer=trainer)

    # Predictions
    train_pred = model.predict(n=len(X_train)-window_size)
    val_pred = model.predict(n=len(X_val), series=train_series[-window_size:])
    test_pred = model.predict(n=len(X_test), series=TimeSeries.from_values(np.concatenate([X_train[-window_size:], X_val])))

    # RMSEs
    train_rmse = np.sqrt(mean_squared_error(X_train[window_size:], train_pred.values()))
    val_rmse = np.sqrt(mean_squared_error(X_val, val_pred.values()))
    test_rmse = np.sqrt(mean_squared_error(X_test, test_pred.values()))


    print(f'RESULTS | TCN | {dataset}')
    print(f'Train RMSE: {train_rmse}')
    print(f'Valid RMSE: {val_rmse}')
    print(f'Test RMSE: {test_rmse}')


def benchmarkBaseline(dataset, X_train, X_val, X_test, window_size=7):
    '''
    Choose the last probability and use it in place of the current
    '''
    # Get predictions and real values
    train_pred = X_train[:-1]
    train_true = X_train[1:]
    val_pred = np.concatenate([X_train[-1][None, :], X_val[:-1]], axis=0)
    val_true = X_val
    test_pred = np.concatenate([X_val[-1][None, :], X_test[:-1]], axis=0)
    test_true = X_test
    
    # Compute metrics, skip first window_size for train
    train_rmse = np.sqrt(mean_squared_error(train_true[window_size:], train_pred[window_size:]))
    val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
    test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    
    print(f'RESULTS | Baseline | {dataset}')
    print(f'Train RMSE: {train_rmse}')
    print(f'Valid RMSE: {val_rmse}')
    print(f'Test RMSE: {test_rmse}')


def compute_deltas(embeddings):
    return embeddings[1:] - embeddings[:-1]

if __name__ == "__main__":
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(42)
    window_size=7
    
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    for dataset in datasets:
        embeddings, labels = my_loader.load_data(dataset, 'Degree', include_weights=False)
                    
        # Split data 70/15/15
        n = len(embeddings)
        train_end = int(0.8 * n)
        val_start = train_end - window_size
        val_end = int(0.9 * n)
        test_start = val_end - window_size

        embeddings = np.array([np.array(e, dtype=np.float32) for e in embeddings])

        X_train = embeddings[:train_end]
        X_val = embeddings[val_start:val_end]
        X_test = embeddings[test_start:]
        
        # Convert to deltas for predictions
        X_train_delta = compute_deltas(X_train)
        X_val_delta = compute_deltas(X_val)
        X_test_delta = compute_deltas(X_test)
        
        # For benchmarking we don't need to predict the true values, just get losses for now
        
        benchmarkVAR(dataset, X_train_delta, X_val_delta, X_test_delta, window_size=7)
        benchmarkTCN(dataset, X_train_delta, X_val_delta, X_test_delta, window_size=7, n_epochs=250, batch_size=32)
        benchmarkBaseline(dataset, X_train_delta, X_val_delta, X_test_delta, window_size=7,)