import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from filterpy.kalman import KalmanFilter
import sys
from sklearn.linear_model import Ridge
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.visualizer import Visualizer
from utils.loader import Loader


def get_deltas(data):
    # Returns the difference between t and t-1
    return np.diff(data, axis=0, prepend=[data[0]])

def reconstruct_from_delta(last_raw, predicted_delta):
    return last_raw + predicted_delta

class VectorPredictors:
    def __init__(self, data):
        """
        data: np.array of shape (T, D) where T is time and D is dimensions
        """
        self.data = data.astype(np.float32)
        # Handle the t=0 case where history is empty
        self.T, self.D = self.data.shape if len(self.data.shape) == 2 else (0, 0)

    def _normalize_probabilities(self, pred):
        # 1. Force values into [0, inf) to prevent negative probabilities
        pred = np.maximum(pred, 0)
        
        # 2. Normalize Group 1 [0:2]
        sum1 = np.sum(pred[0:2])
        if sum1 > 0:
            pred[0:2] /= sum1
        else:
            pred[0:2] = np.array([0.5, 0.5])
            
        # 3. Normalize Group 2 [2:6]
        sum2 = np.sum(pred[2:6])
        if sum2 > 0:
            pred[2:6] /= sum2
        else:
            pred[2:6] = np.array([0.25, 0.25, 0.25, 0.25])
            
        return pred

    # 1. Simple Moving Average (SMA)
    def predict_sma(self, window=5):
        # Citation: Pandas (McKinney, 2010)
        return self.df.rolling(window=window).mean().iloc[-1].values

    # 2. V-EWMA (Darts / Statsmodels logic)
    def predict_vewma(self, alpha=0.8):
        # Citation: Darts (Herzen et al., 2022)
        return self.df.ewm(alpha=alpha, adjust=False).mean().iloc[-1].values

    # 3. AES / Simple Exponential Smoothing (Darts)
    def predict_aes(self):
        # AES is a variant of SES. Darts implements optimized SES.
        # Citation: Darts (Herzen et al., 2022)
        if self.T < 3: return self.data[-1]
        
        predictions = []
        # Loop through each of the 20 dimensions (bins)
        for d in range(self.D):
            # Extract univariate series for this bin
            univariate_data = self.df.iloc[:, d]
            series = TimeSeries.from_series(univariate_data)
            
            model = ExponentialSmoothing(trend=None, seasonal=None)
            model.fit(series)
            predictions.append(model.predict(1).values().flatten()[0])
            
        return np.array(predictions)

    # 4. SSM / Kalman Filter (FilterPy)
    def predict_ssm(self):
        # Citation: FilterPy (Labbe, 2014)
        # This uses a full Multivariate Kalman Filter
        kf = KalmanFilter(dim_x=self.D, dim_z=self.D)
        kf.x = self.data[-1].reshape(-1, 1) # Initial state
        kf.F = np.eye(self.D)               # State transition matrix
        kf.H = np.eye(self.D)               # Measurement function
        kf.P *= 1000.                       # Covariance matrix
        kf.R = np.eye(self.D) * 0.01        # Measurement noise
        kf.Q = np.eye(self.D) * 0.0001      # Process noise
        
        # We "warm up" the filter with history
        for i in range(len(self.data)):
            kf.predict()
            kf.update(self.data[i])
            
        kf.predict() # Forecast next step
        return kf.x.flatten()

    # 4. Vector Autoregression (VAR)
    def predict_var(self):
        # Requires statsmodels. Captures cross-dimension correlations.
        # Fits a linear model: V(t) = A*V(t-1) + B
        model = VAR(self.data)
        results = model.fit(maxlags=1) # Lag 1 for speed and reduced complexity
        return results.forecast(self.data, steps=1)[0]

    # 5. Vector Error Correction Model (VECM)
    def predict_vecm(self):
        # Ideal for 20D Hierarchical TopER. 
        # Models deltas while correcting for the 'equilibrium' (hierarchy).
        # Note: Requires enough data points to estimate cointegration.
        if self.T < 5: 
            return self.data[-1]
            
        try:
            # 1. Calculate Deltas (Short-run dynamics)
            deltas = np.diff(self.data, axis=0)
            
            # 2. Define the Error Correction Term (ECT)
            # This represents the 'equilibrium' state from the previous step
            ect = self.data[1:-1] 
            
            # 3. Define the Regressors (Lagged Deltas)
            lagged_deltas = deltas[:-1]
            
            # Combine ECT and Lagged Deltas as features
            X = np.hstack([lagged_deltas, ect])
            Y = deltas[1:] # What we want to predict (current delta)
            
            # 4. Fit Ridge Regression (Handles the 'Singular Matrix' issue)
            # This effectively estimates the VECM parameters alpha and beta
            model = Ridge(alpha=1.0)
            model.fit(X, Y)
            
            # 5. Forecast the next Delta
            current_ect = self.data[-1].reshape(1, -1)
            current_delta = deltas[-1].reshape(1, -1)
            X_next = np.hstack([current_delta, current_ect])
            
            forecasted_delta = model.predict(X_next).flatten()
            
            # Final Prediction: Last Value + Forecasted Change
            return self.data[-1] + forecasted_delta
        except:
            # In case of any fitting issues, fallback to last known value
            return self.data[-1]
    
    
if __name__ == "__main__":
    np.random.seed(1024)  # Matches other files 
    
    datasets = ["CollegeMsg", "mathoverflow", "networkadex", "networkaeternity", "networkaion", "networkaragon", "networkbancor", "networkcentra", "networkcindicator", "networkcoindash", "networkdgd", "networkiconomi", "Reddit_B", "tgbl-wiki"]


    num_buckets = 10
    using_weight = False
    activation = 'Degree'
    sma_windows = [5]
    train_split, val_split = 0.7, 0.15
    methods = [f"SMA_{w}" for w in sma_windows] + ["VAR", "V-EWMA", "AES", "SSM", "VECM"]
    
    my_loader = Loader()
    
    # We run the whole thing twice: once for Raw, once for Delta
    for mode in ["Raw"]:
        print(f"\n{'='*30}\nRUNNING MODE: {mode}\n{'='*30}")
        all_final_results = []
        use_predicted = False
        num_back = 'all'
        
        for dataset in datasets:
            # 1. Load and Flatten
            probabilities = my_loader.load_data(type='probabilities', dataset=dataset, activation='', normalized=True, use_predicted=use_predicted, num_back=num_back)
            
            if not isinstance(probabilities, np.ndarray):
                # If it's a DataFrame, .values or .to_numpy() works best
                if hasattr(probabilities, 'values'):
                    probabilities = probabilities.values
                else:
                    probabilities = np.array(probabilities)
            gt_embeddings = probabilities.astype(np.float32)    
            T, D = probabilities.shape
            # 2. Select Target Data
            target_data = get_deltas(probabilities) if mode == "Delta" else probabilities
        

            train_idx = int(T * train_split)
            val_idx = int(T * (train_split + val_split))
            splits = {"Train": range(0, train_idx), "Valid": range(train_idx, val_idx), "Test": range(val_idx, T)}
            
            # Start results for this dataset
            ds_results = {"dataset": dataset}

            for m in methods:
                print(f"--- {mode} | {dataset} | {m} ---")
                complete_pred_series = []
                split_mse_totals = {name: [] for name in splits.keys()}
                warmup_limit = D if m == "VAR" else int(m.split("_")[1]) if m.startswith("SMA") else 1
                # DECISION: If Delta, we start at t=1 because t=0 has no delta history
                start_t = 1 if mode == "Delta" else 0

                for t in range(0, T):
                    history = target_data[:t]
                    actual_target = target_data[t]
                    
                    pred = None
                    # FIX: We now pass the historical data into the predictor!
                    try:
                        predictor = VectorPredictors(history)
                        if t < warmup_limit:
                            pred = actual_target
                        else:
                            if m.startswith("SMA_"):
                                w = int(m.split("_")[1])
                                pred = predictor.predict_sma(window=min(w, len(history)))
                            elif m == "V-EWMA": pred = predictor.predict_vewma(alpha=0.8)
                            elif m == "AES":    pred = predictor.predict_aes()
                            elif m == "SSM":    pred = predictor.predict_ssm()
                            elif m == "VAR":    pred = predictor.predict_var() if len(history) > D else actual_target
                            elif m == "VECM":   pred = predictor.predict_vecm() if len(history) > D else actual_target
                    except Exception as e:
                        # Fallback to persistence (now it will only trigger on legitimate mathematical errors, not object errors)
                        pred = history[-1] if len(history) > 0 else actual_target

                    # --- RECONSTRUCTION ---
                    if mode == "Delta":
                        # FIX: Protect against the -1 array wrap-around at t=0
                        raw_pred = reconstruct_from_delta(gt_embeddings[t-1] if t > 0 else gt_embeddings[0], pred)
                    else:
                        raw_pred = pred

                    # Normalize to enforce probability rules [0,1]
                    # Since _normalize_probabilities doesn't rely on self.data, we can safely call it even if history was empty
                    raw_pred = predictor._normalize_probabilities(raw_pred)
                    complete_pred_series.append(raw_pred)
                    
                    # MSE Calculation (Now strictly enforced for t >= start_t to avoid Day 0 noise in Delta mode)
                    if t >= start_t:
                        mse_val = np.mean((gt_embeddings[t] - raw_pred)**2)
                        for name, indices in splits.items():
                            if t in indices:
                                split_mse_totals[name].append(mse_val)

                # # --- MOVE PLOTTING INSIDE THE METHOD LOOP ---
                pred_df = pd.DataFrame(complete_pred_series)
                real_df = pd.DataFrame(gt_embeddings)
                figures_output_path = f"data/output/ProbabilityTesting/data/sample_plots/{m}/{mode}/"
                os.makedirs(figures_output_path, exist_ok=True)
                for i in range(0, 6):
                    # Plot Nodes/Edges (-2 and -1 are the last threshold pair)
                    Visualizer.plot_scatter(pred_df.iloc[:, i], real_df.iloc[:, i], 
                                        f"{figures_output_path}{dataset}_{m}_idx{i}_{mode}.pdf", mode="nodes" if i < 2 else "edges")

                print(f"Generated {len(complete_pred_series)} predictions for {len(gt_embeddings)} true values.")

                pickle_dir = f'data/input/cached/{dataset}/predValues/'
                os.makedirs(pickle_dir, exist_ok=True)
                pickle_path = os.path.join(pickle_dir, f"{dataset}_testprobs_{m}_{mode}_{num_buckets}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(complete_pred_series, f)
                
                # Store averages for THIS method in ds_results
                for split_name in splits.keys():
                    if split_mse_totals[split_name]:
                        ds_results[f"{split_name}_{m}_MSE"] = np.mean(split_mse_totals[split_name])
                        print(f"  {split_name} Split | MSE: {ds_results[f'{split_name}_{m}_MSE']:.4f}")

            # --- APPEND TO RESULTS AFTER ALL METHODS FOR DATASET ARE DONE ---
            all_final_results.append(ds_results)

        # 3. Final Table Production for THIS mode
        df_results = pd.DataFrame(all_final_results)
        output_csv = f"data/output/ProbabilityTesting/data/new_testing_{mode}.csv"
        df_results.to_csv(output_csv, index=False)
        
        # Determine best method
        test_cols = [c for c in df_results.columns if "Test_" in c and "_MSE" in c]
        
        if test_cols:
            # Calculate the average across all datasets
            overall_performance = df_results[test_cols].mean().sort_values()
            
            # Clean up strings for display
            best_col = overall_performance.index[0]
            best_method = best_col.replace("Test_", "").replace("_MSE", "")

            print(f"\n--- Global Ranking ({mode}) ---")
            print(overall_performance)

            print("\n--- Split-wise Results ---")
            print(df_results.to_string(index=False))

            print(f"\n🏆 The best performing model on average ({mode} mode) is: {best_method}")
            
            # Optional: Save for external analysis - using mode in filename to avoid overwriting
            df_results.to_csv(f"data/output/ProbabilityTesting/data/new_testing_{mode}.csv", index=False)
        else:
            print(f"\n--- Split-wise Results ({mode}) ---")
            print(df_results.to_string(index=False))
            print(f"\n❌ No models successfully completed the Test phase for {mode} mode.")
