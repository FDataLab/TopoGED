import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from darts import TimeSeries
from darts.models import TCNModel
from statsmodels.tsa.api import VAR
from pytorch_lightning import Trainer


class Benchmarker():
    """
    Utility class designed to benchmark for delta predictions across the following models: 
        - VAR (Vector Autoregression)
        - TCN (Temporal Convolutional Network)
        - Baseline Method: Assume yesterdays truth is todays prediction
    """
    def compute_deltas(self, embeddings):
        """
        Commpute the difference between each embedding
        
        params:
            embeddings: The embeddings to compute deltas for
            
        returns:
            embeddings[1:] - embeddings[:-1]: The difference between each embedding
        """
        return embeddings[1:] - embeddings[:-1]
    
    
    def embeddings_to_timeseries(self, embeddings):
        """
        utility function for TCN, required to feed inputs to the model
        
        params:
            embeddings: The embeddings to predict on
        
        returns:
            series_list: The embeddings in the proper format for TCN
        """
        series_list = []
        for dim in range(embeddings.shape[1]):
            series_list.append(TimeSeries.from_values(embeddings[:, dim]))
        return series_list


    def benchmarkVAR(self, dataset, X_train, X_val, X_test, window_size):
        """
        For benchmarking the VAR model, our main runner for it
        
        params:
            dataset (string): The dataset name, for visualization of losses
            X_train (list): The training split, typically first 80% of data
            X_val (list): The validation split, typically next 10% of data
            X_test (list): The testing split, typically the last 10% of data
            window_size (int): The window size to feed the model, a sliding window is used for training
            
        returns:
            None: Just display the losses and move on
        """
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
        

    def benchmarkTCN(self, dataset, X_train, X_val, X_test, window_size, n_epochs=200, batch_size=32, random_state=42):
        """
        For benchmarking the TCN model, our main runner for it
        
        params:
            dataset (string): The dataset name, for visualization of losses
            X_train (list): The training split, typically first 80% of data
            X_val (list): The validation split, typically next 10% of data
            X_test (list): The testing split, typically the last 10% of data
            window_size (int): The window size to feed the model, a sliding window is used for training
            n_epochs (int): Number of epochs to run for
            batch_size (int): The batch size to use for training
            random_state (int): The seed to use, we will use 42
            
        returns:
            None: Just display the losses and move on
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
            max_epochs=n_epochs 
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


    def benchmarkBaseline(self, dataset, X_train, X_val, X_test, window_size):
        """
        For benchmarking the baseline method, our main runner for it
        
        params:
            dataset (string): The dataset name, for visualization of losses
            X_train (list): The training split, typically first 80% of data
            X_val (list): The validation split, typically next 10% of data
            X_test (list): The testing split, typically the last 10% of data
            window_size (int): The window size to feed the model, a sliding window is used for training (we just ignore the first window_size days for uniformity)
            
        returns:
            None: Just display the losses and move on
        """
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
    
    
    def begin_benchmarking(self, dataset, X_train, X_val, X_test, window_size=7):
        """
        Begin the benchmarking, this will convert to deltas and run all necessary models
        
        params:
            dataset (string): The dataset name, for visualization of losses
            X_train (list): The training split, typically first 80% of data
            X_val (list): The validation split, typically next 10% of data
            X_test (list): The testing split, typically the last 10% of data
            window_size (int): The window size to feed the model, a sliding window is used for training
            
        returns:
            None: Just display the losses and move on
        """
        # Convert to deltas for predictions
        X_train_delta = self.compute_deltas(X_train)
        X_val_delta = self.compute_deltas(X_val)
        X_test_delta = self.compute_deltas(X_test)
        
        # Perform benchmarking
        self.benchmarkBaseline(dataset, X_train_delta, X_val_delta, X_test_delta, window_size=7)
        self.benchmarkTCN(dataset, X_train_delta, X_val_delta, X_test_delta, window_size=7, n_epochs=250, batch_size=32)
        self.benchmarkVAR(dataset, X_train_delta, X_val_delta, X_test_delta, window_size=7)
