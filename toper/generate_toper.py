import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from filterpy.kalman import KalmanFilter
import sys
from sklearn.linear_model import Ridge
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
        self.df = pd.DataFrame(self.data)

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

def generate_latex(all_results, metric_key, caption, label):
    # Mapping internal method names to display names
    methods = ["SMA_5", "VAR", "V-EWMA", "SSM", "VECM"]
    display_names = ["SMA", "VAR", "V-EWMA", "SSM", "VECM"]
    
    latex = [
        r"\begin{table}[ht]",
        fr"\caption{{{caption}}}",
        fr"\label{{{label}}}",
        r"\centering",
        r"\begin{tabular}{l cccccc}",
        r"\toprule",
        r"\textbf{Dataset} & " + " & ".join([fr"\textbf{{{m}}}" for m in display_names]) + r" \\",
        r"\midrule"
    ]

    # To calculate column-wise averages
    column_data = {m: [] for m in methods}

    for ds_res in all_results:
        dataset_name = ds_res['dataset'].replace('network', '').replace('_', r'\_').capitalize()
        row = [dataset_name]
        
        # Get values for this row to determine bold/underline
        vals = [ds_res.get(f"Test_{m}_{metric_key}", 0.0) for m in methods]
        abs_vals = [abs(v) for v in vals]
        sorted_indices = np.argsort(abs_vals)
        best_idx, second_idx = sorted_indices[0], sorted_indices[1]

        for i, v in enumerate(vals):
            formatted = f"{v:.2f}"
            column_data[methods[i]].append(v)
            if i == best_idx:
                row.append(fr"\textbf{{{formatted}}}")
            elif i == second_idx:
                row.append(fr"\underline{{{formatted}}}")
            else:
                row.append(formatted)
        
        latex.append(" & ".join(row) + r" \\")

    # Add Average Row
    latex.append(r"\midrule[\heavyrulewidth]")
    avg_row = [r"\textbf{Average}"]
    avg_vals = [np.mean(column_data[m]) for m in methods]
    abs_avg_vals = [abs(v) for v in avg_vals]
    avg_sorted = np.argsort(abs_avg_vals)
    b_avg, s_avg = avg_sorted[0], avg_sorted[1]

    for i, v in enumerate(avg_vals):
        formatted = f"{v:.2f}"
        if i == b_avg:
            avg_row.append(fr"\textbf{{{formatted}}}")
        elif i == s_avg:
            avg_row.append(fr"\underline{{{formatted}}}")
        else:
            avg_row.append(formatted)
    
    latex.append(" & ".join(avg_row) + r" \\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)
    
def generate_single_latex_table(all_results, metric_key, caption, label):
    # Standard method order for NeurIPS baselines
    methods = ["SMA_5", "VAR", "V-EWMA", "SSM", "VECM"]
    display_names = ["SMA", "VAR", "V-EWMA", "SSM", "VECM"]
    
    latex = [
        r"\begin{table}[ht]",
        fr"\caption{{{caption}}}",
        fr"\label{{{label}}}",
        r"\centering",
        r"\begin{tabular}{l cccccc}",
        r"\toprule",
        r"\textbf{Dataset} & " + " & ".join([fr"\textbf{{{m}}}" for m in display_names]) + r" \\",
        r"\midrule"
    ]

    col_acc = {m: [] for m in methods}

    for ds_res in all_results:
        # Format dataset name: 'networkadex' -> 'Adex'
        name = ds_res['dataset'].replace('network', '').replace('_', r'\_').capitalize()
        row = [name]
        
        vals = [ds_res.get(f"Test_{m}_{metric_key}", 0.0) for m in methods]
        abs_vals = [abs(v) for v in vals]
        best_idx, second_idx = np.argsort(abs_vals)[:2]

        for i, v in enumerate(vals):
            col_acc[methods[i]].append(v)
            fmt = f"{v:.2f}"
            if i == best_idx: row.append(fr"\textbf{{{fmt}}}")
            elif i == second_idx: row.append(fr"\underline{{{fmt}}}")
            else: row.append(fmt)
        
        latex.append(" & ".join(row) + r" \\")

    # Average Calculation
    latex.append(r"\midrule[\heavyrulewidth]")
    avg_row = [r"\textbf{Average}"]
    avg_vals = [np.mean(col_acc[m]) for m in methods]
    b_avg, s_avg = np.argsort([abs(v) for v in avg_vals])[:2]

    for i, v in enumerate(avg_vals):
        fmt = f"{v:.2f}"
        if i == b_avg: avg_row.append(fr"\textbf{{{fmt}}}")
        elif i == s_avg: avg_row.append(fr"\underline{{{fmt}}}")
        else: avg_row.append(fmt)
    
    latex.append(" & ".join(avg_row) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(latex)

if __name__ == "__main__":
    np.random.seed(1024)
    datasets = ["CollegeMsg", "mathoverflow", "networkadex", "networkaeternity", "networkaion", "networkaragon", "networkbancor", "networkcentra", "networkcindicator", "networkcoindash", "networkdgd", "networkiconomi", "Reddit_B", "tgbl-wiki"]
    methods = ["VECM", "VAR", "SSM", "V-EWMA", "SMA_5"]
    my_loader = Loader()

    bucket_sizes = [10] 
    methods = ["VECM", "VAR", "SSM", "V-EWMA", "SMA_5"]
    my_loader = Loader()

    for num_buckets in bucket_sizes:
        print(f"\n{'#'*40}\nRUNNING FOR VECTOR LENGTH: {num_buckets}\n{'#'*40}")
        
        for mode in ["Raw"]:
            all_final_results = []
            
            for dataset in datasets:
                print(f"--- Processing Dataset: {dataset} | Buckets: {num_buckets} ---")
                
                embeddings_raw, _ = my_loader.load_data(dataset, 'Degree', 
                                                        include_weights=False, 
                                                        num_buckets=num_buckets)
                
                gt_embeddings = np.array([np.array(snap).flatten() for snap in embeddings_raw])
                T, D = gt_embeddings.shape
                val_idx = int(T * 0.85)
                ds_results = {"dataset": dataset}

                for m in methods:
                    split_err_nodes, split_err_edges = [], []
                    complete_pred_series = []

                    for t in range(T):
                        history = gt_embeddings[:t]
                        predictor = VectorPredictors(history)
                        
                        try:
                            if t == 0: 
                                pred = gt_embeddings[0]
                            elif m == "SMA_5": 
                                pred = predictor.predict_sma(window=min(5, t))
                            elif m == "V-EWMA": 
                                pred = predictor.predict_vewma(alpha=0.5)
                            elif m == "AES": 
                                pred = predictor.predict_aes(alpha=0.4)
                            elif m == "SSM": 
                                pred = predictor.predict_ssm()
                            elif m == "VAR": 
                                pred = predictor.predict_var()
                            elif m == "VECM": 
                                pred = predictor.predict_vecm()
                        except:
                            pred = gt_embeddings[t-1] if t > 0 else gt_embeddings[0]

                        complete_pred_series.append(pred)
                        
                        # Metrics logic
                        t_nodes, p_nodes = gt_embeddings[t][-2], pred[-2]
                        t_edges, p_edges = gt_embeddings[t][-1], pred[-1]
                        d_n = t_nodes if t_nodes != 0 else 1.0
                        d_e = t_edges if t_edges != 0 else 1.0

                        if t >= val_idx: 
                            # Multiply by 100 to convert decimal to actual percentage points
                            split_err_nodes.append(((p_nodes - t_nodes) / d_n) * 100)
                            split_err_edges.append(((p_edges - t_edges) / d_e) * 100)

                    # --- NEW: SAVE VECTORS TO PKL ---
                    # Logic: data/input/cached/{dataset}/predValues/{dataset}_{method}_{buckets}.pkl
                    pickle_dir = f'data/input/cached/{dataset}/predValues/'
                    os.makedirs(pickle_dir, exist_ok=True)
                    pickle_name = f"{dataset}_testtoper_{m}_{mode}_{num_buckets}.pkl"
                    pickle_path = os.path.join(pickle_dir, pickle_name)
                    
                    with open(pickle_path, 'wb') as f_pkl:
                        pickle.dump(complete_pred_series, f_pkl)
                    
                    # Store for LaTeX
                    ds_results[f"Test_{m}_Nodes"] = np.mean(split_err_nodes)
                    ds_results[f"Test_{m}_Edges"] = np.mean(split_err_edges)

                all_final_results.append(ds_results)

            # Save LaTeX tables after each bucket pass
            out_path = f"latex_tables/results_buckets_{num_buckets}_new.tex"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(generate_single_latex_table(all_final_results, "Nodes", 
                        f"Percent Error Nodes (D={num_buckets})", f"tab:err_nodes_{num_buckets}"))
                f.write("\n\n\\clearpage\n\n")
                f.write(generate_single_latex_table(all_final_results, "Edges", 
                        f"Percent Error Edges (D={num_buckets})", f"tab:err_edges_{num_buckets}"))

   