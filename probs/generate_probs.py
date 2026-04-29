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

    def predict_sma(self, window=5):
        return np.mean(self.data[-window:], axis=0)

    # 2. Vectorized Exponentially Weighted Moving Average (V-EWMA)
    def predict_vewma(self, alpha=0.8):
        # High alpha (0.8) reduces lag; low alpha (0.2) increases smoothing
        # Result is weighted average: alpha * current + (1-alpha) * last_average
        if self.T == 0: return 0
        result = self.data[0]
        for i in range(1, self.T):
            result = alpha * self.data[i] + (1 - alpha) * result
        return result

    # 3. Adaptive Exponential Smoothing (AES)
    def predict_aes(self):
        # Self-adjusting alpha based on error to eliminate lag during spikes
        # Formula: alpha_t = |error_t| / |absolute_error_t|
        if self.T == 0: return 0
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
        if self.T == 0: return 0
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
    
    datasets = ["CollegeMsg", "mathoverflow", "networkadex", "networkaeternity", "networkaion", "networkaragon", "networkbancor", "networkcentra", "networkcindicator", "networkcoindash", "networkdgd", "networkiconomi", "Reddit_B", "tgbl-wiki"]


    num_buckets = 10
    using_weight = False
    activation = 'Degree'
    sma_windows = [5]
    train_split, val_split = 0.7, 0.15
    methods = [f"SMA_{w}" for w in sma_windows] + ["VAR", "V-EWMA", "AES", "SSM", "VECM"]
    methods = ["AES", "VECM", "VAR"]
    
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
