import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
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
        self.T, self.D = data.shape

    # 1. Simple Moving Average (SMA)
    def predict_sma(self, window=5):
        # Best guess for t+1 is the average of the last 'window' steps
        return np.mean(self.data[-window:], axis=0)

    # 2. Vectorized Exponentially Weighted Moving Average (V-EWMA)
    def predict_vewma(self, alpha=0.8):
        # High alpha (0.8) reduces lag; low alpha (0.2) increases smoothing
        # Result is weighted average: alpha * current + (1-alpha) * last_average
        result = self.data[0]
        for i in range(1, self.T):
            result = alpha * self.data[i] + (1 - alpha) * result
        return result

    # 3. Adaptive Exponential Smoothing (AES)
    def predict_aes(self):
        # Self-adjusting alpha based on error to eliminate lag during spikes
        # Formula: alpha_t = |error_t| / |absolute_error_t|
        res = self.data[0]
        alpha = 0.5
        error_sum = 0.01
        abs_error_sum = 0.01
        
        for i in range(1, self.T):
            error = self.data[i-1] - res
            error_sum = 0.9 * error_sum + 0.1 * error
            abs_error_sum = 0.9 * abs_error_sum + 0.1 * np.abs(error)
            
            # Dimensional-wise alpha
            alpha = np.abs(error_sum / (abs_error_sum + 1e-6))
            res = alpha * self.data[i] + (1 - alpha) * res
        return res

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
        model = VECM(self.data, k_ar_diff=1, coint_rank=min(self.D-1, 5))
        results = model.fit()
        return results.forecast(steps=1)[0]

    # 6. State Space Model (SSM) - Basic Kalman Filter Logic
    def predict_ssm(self):
        # Predict-Correct cycle. Great for 'weird patterns' without lag.
        # Simplied steady-state Kalman Filter
        state_est = self.data[0]
        error_cov = np.ones(self.D)
        process_var = 1e-4 # How much we trust the 'model'
        measurement_var = 1e-2 # How much we trust the 'weird' data
        
        for i in range(1, self.T):
            # Predict step
            pred_state = state_est
            pred_error_cov = error_cov + process_var
            
            # Update step (Kalman Gain)
            kalman_gain = pred_error_cov / (pred_error_cov + measurement_var)
            state_est = pred_state + kalman_gain * (self.data[i] - pred_state)
            error_cov = (1 - kalman_gain) * pred_error_cov
            
        return state_est
    
    
if __name__ == "__main__":
    np.random.seed(1024)  # Matches other files 
    
    datasets = ['networkiconomi']

    num_buckets = 10
    using_weight = False
    activation = 'Degree'
    sma_windows = [3, 5, 6]
    train_split, val_split = 0.7, 0.15
    methods = ["VAR"]
    
    my_loader = Loader()
    
    # We run the whole thing twice: once for Raw, once for Delta
    for mode in ["Raw", "Delta"]:
        print(f"\n{'='*30}\nRUNNING MODE: {mode}\n{'='*30}")
        all_final_results = []
        
        for dataset in datasets:
            for num_buckets in [5, 10, 15, 20]:
                # 1. Load and Flatten
                embeddings_raw, _ = my_loader.load_data(dataset, activation, 
                                                    include_weights=using_weight, 
                                                    num_buckets=num_buckets)
                gt_embeddings = np.array([np.array(snap).flatten() for snap in embeddings_raw])
                print(len(gt_embeddings))
                T, D = gt_embeddings.shape
                
                # 2. Select Target Data
                target_data = get_deltas(gt_embeddings) if mode == "Delta" else gt_embeddings

                train_idx = int(T * train_split)
                val_idx = int(T * (train_split + val_split))
                splits = {"Train": range(0, train_idx), "Valid": range(train_idx, val_idx), "Test": range(val_idx, T)}
                
                # Start results for this dataset
                ds_results = {"dataset": dataset}

                for m in methods:
                    print(f"--- {mode} | {dataset} | {m} ---")
                    complete_pred_series = []
                    split_mse_totals = {name: [] for name in splits.keys()}

                    for t in range(T):
                        history = target_data[:t]
                        actual_target = target_data[t]
                        
                        pred = None
                        try:
                            predictor = VectorPredictors(history)
                            if t == 0:
                                pred = actual_target
                            elif m.startswith("SMA_"):
                                w = int(m.split("_")[1])
                                pred = predictor.predict_sma(window=min(w, len(history)))
                            elif m == "V-EWMA": pred = predictor.predict_vewma(alpha=0.8)
                            elif m == "AES":    pred = predictor.predict_aes()
                            elif m == "SSM":    pred = predictor.predict_ssm()
                            elif m == "VAR":    pred = predictor.predict_var() if len(history) > D else actual_target
                            elif m == "VECM":   pred = predictor.predict_vecm() if len(history) > D else actual_target
                        except Exception:
                            # Fallback to Persistence
                            pred = history[-1] if len(history) > 0 else actual_target

                        # Reconstruct if in Delta mode
                        raw_pred = reconstruct_from_delta(gt_embeddings[t-1] if t > 0 else gt_embeddings[0], pred) if mode == "Delta" else pred
                        complete_pred_series.append(raw_pred)
                        
                        # MSE Calculation
                        mse_val = np.mean((gt_embeddings[t] - raw_pred)**2)
                        for name, indices in splits.items():
                            if t in indices:
                                split_mse_totals[name].append(mse_val)

                    # --- MOVE PLOTTING INSIDE THE METHOD LOOP ---
                    if num_buckets == 10:
                        pred_df = pd.DataFrame(complete_pred_series)
                        real_df = pd.DataFrame(gt_embeddings)
                        figures_output_path = f"data/output/TopERTesting/data/sample_plots/"
                        os.makedirs(figures_output_path, exist_ok=True)
                        
                        # Plot Nodes/Edges (-2 and -1 are the last threshold pair)
                        Visualizer.plot_scatter(pred_df.iloc[:, -2], real_df.iloc[:, -2], 
                                            f"{figures_output_path}{dataset}_{m}_Nodes_{mode}.pdf", mode="nodes")
                        Visualizer.plot_scatter(pred_df.iloc[:, -1], real_df.iloc[:, -1], 
                                            f"{figures_output_path}{dataset}_{m}_Edges_{mode}.pdf", mode="edges")
                    print(len(complete_pred_series), len(gt_embeddings), len(embeddings_raw))
                    pickle_dir = f'data/input/cached/{dataset}/predValues/'
                    os.makedirs(pickle_dir, exist_ok=True)
                    pickle_path = os.path.join(pickle_dir, f"{dataset}_testtoper_{m}_{mode}_{num_buckets}.pkl")
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(complete_pred_series, f)
                        
                    # Store averages for THIS method in ds_results
                    for split_name in splits.keys():
                        if split_mse_totals[split_name]:
                            ds_results[f"{split_name}_{m}_MSE"] = np.mean(split_mse_totals[split_name])

                # --- APPEND TO RESULTS AFTER ALL METHODS FOR DATASET ARE DONE ---
                all_final_results.append(ds_results)

        # 3. Final Table Production for THIS mode
        df_results = pd.DataFrame(all_final_results)
        output_csv = f"data/output/TopERTesting/data/new_testing_{mode}.csv"
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
            df_results.to_csv(f"data/output/TopERTesting/data/new_testing_{mode}.csv", index=False)
        else:
            print(f"\n--- Split-wise Results ({mode}) ---")
            print(df_results.to_string(index=False))
            print(f"\n❌ No models successfully completed the Test phase for {mode} mode.")
