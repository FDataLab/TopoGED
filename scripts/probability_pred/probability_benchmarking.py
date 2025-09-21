import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# For benchmarking
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*")
from utils.loader import Loader
from utils.utils import Utils

RESULTS_PATH = 'data/input/cached/'

count_types = {
    "count_old_nodes": 0,
    "count_new_nodes": 1,
    "count_oo": 2,
    "count_nn": 3,
    "count_on": 4,
    "count_oon": 5,
}


def benchmarkArima():
    datasets = ['networkbancor', 'CollegeMsg', 'mathoverflow', 'networkaragon', 'networkaion', 'networkadex', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkaeternity', 'networkiconomi', 'networkcindicator', 'networkdgd']  # replace with actual dataset names
    dataset_dfs = {name: pd.DataFrame() for name in datasets}
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(42)
    
    for count_type in count_types.keys():
        datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

        for dataset in datasets:
            # Set up probabilities
            counts_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')  # Activation doesn't matter here
            count_type_idx = count_types[count_type]
            series = counts_df.iloc[:, count_type_idx].to_numpy(dtype=np.float32).flatten()

            N = len(series)
            train_end = int(0.8 * N)
            val_end = int(0.9 * N)

            train, val, test = series[:train_end], series[train_end:val_end], series[val_end:]

            print(f"Processing dataset: {dataset} for count type: {count_type}")

            try:
                # Automatically select ARIMA order on training data
                model_auto = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
                p, d, q = model_auto.order
                print(f"Selected ARIMA order for {dataset}: (p={p}, d={d}, q={q})")

                # Fit ARIMA on training data
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()
                
                # Predictions
                # For training, we skip the first 7 days as before
                train_pred = model_fit.predict(start=7, end=len(train)-1)
                
                # Forecast validation and test sets
                val_pred = model_fit.forecast(steps=len(val))
                test_pred = model_fit.forecast(steps=len(val) + len(test))[-len(test):]  # only take test steps

                # Full prediction (optional)
                full_pred = np.concatenate([train_pred, val_pred, test_pred])

                # Metrics
                train_rmse = np.sqrt(mean_squared_error(train[7:], train_pred))
                train_mae  = mean_absolute_error(train[7:], train_pred)
                val_rmse = np.sqrt(mean_squared_error(val, val_pred))
                val_mae  = mean_absolute_error(val, val_pred)
                test_rmse = np.sqrt(mean_squared_error(test, test_pred))
                test_mae  = mean_absolute_error(test, test_pred)
                
                print(f'RESULTS | {dataset} | {count_type}')
                print(f'Train RMSE: {train_rmse}')
                print(f'Valid RMSE: {val_rmse}')
                print(f'Test RMSE: {test_rmse}')
                print(f'Train MAE: {train_mae}')
                print(f'Valid MAE: {val_mae}')
                print(f'Test MAE: {test_mae}')
                
                res_path = os.path.join(RESULTS_PATH, dataset, "Benchmarking", "Arima")
                os.makedirs(res_path, exist_ok=True)
            
                plt.figure(figsize=(8, 4))
                plt.plot(range(7, len(series)), series[7:], label="Real", linewidth=1)
                plt.plot(range(7, len(series)), full_pred, label="Pred", linewidth=1)

                plt.xlabel("Index")
                plt.ylabel("Count")
                plt.title(f"{count_type} - Real vs Predicted")
                plt.legend()
                plt.grid(True)
                
                # Store for viewing in a csv
                full_pred_df = pd.DataFrame(full_pred, columns=[count_type])
                if dataset_dfs[dataset].empty:
                    dataset_dfs[dataset] = full_pred_df
                else:
                    dataset_dfs[dataset] = pd.concat([dataset_dfs[dataset], full_pred_df], axis=1)

                # Save plot
                file_path = os.path.join(res_path, f"{count_type}_discrete.png")
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()


            except Exception as e:
                print(f"Error processing {dataset}: {e}") 

    # Save for later viewing
    for dataset in dataset_dfs.keys():
        storage_path = os.path.join(RESULTS_PATH, dataset, "Benchmarking", "Arima")
        os.makedirs(storage_path, exist_ok=True)  # Unnecessary but added for safety
        file_path = os.path.join(storage_path, "count_predictions.csv")
        dataset_dfs[dataset].to_csv(file_path, index=False)
    

def benchmarkExponentialSmoothing():
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    dataset_dfs = {name: pd.DataFrame() for name in datasets}
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(42)

    for count_type in count_types.keys():
        for dataset in datasets:
            # Load data
            counts_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')
            count_type_idx = count_types[count_type]
            series = counts_df.iloc[:, count_type_idx].to_numpy(dtype=np.float32).flatten()

            N = len(series)
            train_end = int(0.8 * N)
            val_end = int(0.9 * N)

            train, val, test = series[:train_end], series[train_end:val_end], series[val_end:]

            print(f"Processing dataset: {dataset} for count type: {count_type}")

            try:
                # Fit Exponential Smoothing (trend only for now)
                model = ExponentialSmoothing(train, trend="add", seasonal=None)
                model_fit = model.fit(optimized=True)

                # Predictions
                # Training predictions (skip first 7 days)
                train_pred = model_fit.predict(start=7, end=len(train)-1)

                # Validation forecast
                val_pred = model_fit.forecast(steps=len(val))

                # Test forecast: forecast total of val+test, then take only test portion
                test_pred = model_fit.forecast(steps=len(val) + len(test))[-len(test):]

                # Full prediction (optional, for plotting)
                full_pred = np.concatenate([model_fit.fittedvalues[7:], val_pred, test_pred])

                # Metrics
                train_rmse = np.sqrt(mean_squared_error(train[7:], train_pred))
                train_mae  = mean_absolute_error(train[7:], train_pred)
                val_rmse = np.sqrt(mean_squared_error(val, val_pred))
                val_mae  = mean_absolute_error(val, val_pred)
                test_rmse = np.sqrt(mean_squared_error(test, test_pred))
                test_mae  = mean_absolute_error(test, test_pred)
                
                print(f'RESULTS | {dataset} | {count_type}')
                print(f'Train RMSE: {train_rmse}')
                print(f'Valid RMSE: {val_rmse}')
                print(f'Test RMSE: {test_rmse}')
                print(f'Train MAE: {train_mae}')
                print(f'Valid MAE: {val_mae}')
                print(f'Test MAE: {test_mae}')

                res_path = os.path.join(RESULTS_PATH, dataset, "Benchmarking", "ExpSmoothing")
                os.makedirs(res_path, exist_ok=True)

                # Plot
                plt.figure(figsize=(8, 4))
                plt.plot(range(7, len(series)), series[7:], label="Real", linewidth=1)
                plt.plot(range(7, len(series)), full_pred, label="Pred", linewidth=1)
                plt.xlabel("Index")
                plt.ylabel("Count")
                plt.title(f"{count_type} - Real vs Predicted (ExpSmoothing)")
                plt.legend()
                plt.grid(True)

                # Store for viewing in a csv
                full_pred_df = pd.DataFrame(full_pred, columns=[count_type])
                if dataset_dfs[dataset].empty:
                    dataset_dfs[dataset] = full_pred_df
                else:
                    dataset_dfs[dataset] = pd.concat([dataset_dfs[dataset], full_pred_df], axis=1)

                # Save plot
                file_path = os.path.join(res_path, f"{count_type}_exp.png")
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()

            except Exception as e:
                print(f"Error processing {dataset}: {e}")

    # Save for later viewing
    for dataset, df in dataset_dfs.items():
        storage_path = os.path.join(RESULTS_PATH, dataset, "Benchmarking", "ExpSmoothing")
        os.makedirs(storage_path, exist_ok=True)
        file_path = os.path.join(storage_path, "all_predictions.csv")
        df.to_csv(file_path, index=False)


def benchmarkBaseline():
    '''
    Choose the last probability and use it in place of the current
    '''
    datasets = ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']

    dataset_dfs = {name: pd.DataFrame() for name in datasets}
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(42)

    for count_type in count_types.keys():
        for dataset in datasets:
            # Load data
            counts_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='5')
            count_type_idx = count_types[count_type]
            series = counts_df.iloc[:, count_type_idx].to_numpy(dtype=np.float32).flatten()

            N = len(series)
            train_end = int(0.8 * N)
            val_end = int(0.9 * N)

            train, val, test = series[:train_end], series[train_end:val_end], series[val_end:]

            print(f"Processing dataset: {dataset} for count type: {count_type}")

            try:
                # Predictions
                # We can just shift the values up one day, since that is how our baseline works
                # Effectively, just remove the last day of training
                train_pred = train[:-1]
                train_true = train[1:]
                val_pred = np.concatenate([[train[-1]], val[:-1]])
                val_true = val
                test_pred = np.concatenate([[val[-1]], test[:-1]])
                test_true = test
                full_pred = np.concatenate([train_pred, val_pred])
                full_true = np.concatenate([train_true, val_true])

                train_rmse = np.sqrt(mean_squared_error(train_true[7:], train_pred[7:]))
                train_mae  = mean_absolute_error(train_true[7:], train_pred[7:])
                val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
                val_mae  = mean_absolute_error(val_true, val_pred)
                test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
                test_mae  = mean_absolute_error(test_true, test_pred)
                

                print(f'RESULTS | {dataset} | {count_type}')
                print(f'Train RMSE: {train_rmse}')
                print(f'Valid RMSE: {val_rmse}')
                print(f'Test RMSE: {test_rmse}')
                print(f'Train MAE: {train_mae}')
                print(f'Valid MAE: {val_mae}')
                print(f'Test MAE: {test_mae}')

                res_path = os.path.join(RESULTS_PATH, dataset, "Benchmarking", "Baseline")
                os.makedirs(res_path, exist_ok=True)

                # Plot
                plt.figure(figsize=(8, 4))
                plt.plot(range(7, len(full_true)), full_true[7:], label="Real", linewidth=1)
                plt.plot(range(7, len(full_true)), full_pred[7:], label="Pred", linewidth=1)
                plt.xlabel("Index")
                plt.ylabel("Count")
                plt.title(f"{count_type} - Real vs Predicted (Baseline)")
                plt.legend()
                plt.grid(True)

                # Store for viewing in a csv
                full_pred_df = pd.DataFrame(full_pred, columns=[count_type])
                if dataset_dfs[dataset].empty:
                    dataset_dfs[dataset] = full_pred_df
                else:
                    dataset_dfs[dataset] = pd.concat([dataset_dfs[dataset], full_pred_df], axis=1)

                # Save plot
                file_path = os.path.join(res_path, f"{count_type}_baseline.png")
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()

            except Exception as e:
                print(f"Error processing {dataset}: {e}")

    # Save for later viewing
    for dataset, df in dataset_dfs.items():
        storage_path = os.path.join(RESULTS_PATH, dataset, "Benchmarking", "Baseline")
        os.makedirs(storage_path, exist_ok=True)
        file_path = os.path.join(storage_path, "all_predictions.csv")
        df.to_csv(file_path, index=False)



if __name__ == "__main__":
    print('Starting Arima benchmarking')
    benchmarkArima()
    print('Starting ExponentialSmoothing benchmarking')
    benchmarkExponentialSmoothing()
    print('Starting Baseline benchmarking')
    benchmarkBaseline()