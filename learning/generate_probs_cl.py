import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from filterpy.kalman import KalmanFilter
import sys
from sklearn.linear_model import Ridge
import os
import pickle
import torch
import torch.nn as nn

# Always run relative to the project root so Loader paths resolve correctly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(_project_root)
sys.path.append(_project_root)

from utils.loader import Loader

# Redirect Loader to use learning/ paths so nothing is written outside learning/
_learning = os.path.join(_project_root, 'learning')
Loader.output_dir   = os.path.join(_learning, 'datasets', 'cached')
Loader.edgelist_dir = os.path.join(_learning, 'datasets', 'edgelist')
Loader.label_dir    = os.path.join(_learning, 'datasets', 'labels')
os.makedirs(Loader.output_dir, exist_ok=True)
os.makedirs(Loader.label_dir, exist_ok=True)


class _RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type):
        super().__init__()
        cell = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = cell(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, input_size)

    def forward(self, x, h=None):
        out, h_new = self.rnn(x, h)
        return self.fc(out[:, -1, :]), h_new


_RNN_BEST = {
    'RNN':  {'hidden_size':  32, 'lr': 5e-3, 'grad_steps': 5},
    'LSTM': {'hidden_size': 128, 'lr': 5e-3, 'grad_steps': 5},
    'GRU':  {'hidden_size':  64, 'lr': 5e-3, 'grad_steps': 5},
}


def run_online_recurrent(gt_embeddings, rnn_type):
    """
    Online learning with per-dimension z-score normalization.
    At each step t: feed normalized v_t -> predict normalized v_{t+1} ->
    take grad_steps gradient steps -> unnormalize prediction before storing.
    Evaluation is always on original scale.

    Normalization stats are estimated from the first 50% of the series only,
    so no future data leaks into the training signal.
    After each training loop, a clean eval-mode forward pass produces the
    prediction and the hidden state carried to the next step, preventing
    hidden state corruption from the gradient update iterations.
    """
    cfg        = _RNN_BEST[rnn_type]
    hidden_size = cfg['hidden_size']
    lr          = cfg['lr']
    grad_steps  = cfg['grad_steps']

    T, D   = gt_embeddings.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Estimate normalization stats from the first half of the series only
    warmup_n = max(10, T // 2)
    mean = gt_embeddings[:warmup_n].mean(axis=0)
    std  = gt_embeddings[:warmup_n].std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)
    norm = (gt_embeddings - mean) / std

    model     = _RecurrentNet(D, hidden_size, rnn_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    preds = [gt_embeddings[0].copy()]
    h     = None

    for t in range(T - 1):
        x      = torch.tensor(norm[t],     dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        target = torch.tensor(norm[t + 1], dtype=torch.float32).to(device)

        # Gradient update steps — h is intentionally not updated here to
        # avoid carrying a hidden state that was corrupted by repeated backprop
        model.train()
        for _ in range(grad_steps):
            optimizer.zero_grad()
            out, _    = model.rnn(x, h)
            pred_norm = model.fc(out.squeeze(0).squeeze(0))
            loss_fn(pred_norm, target).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Clean inference pass: this is what produces both the stored prediction
        # and the hidden state passed to t+1
        model.eval()
        with torch.no_grad():
            out, h_new = model.rnn(x, h)
            pred_norm  = model.fc(out.squeeze(0).squeeze(0))

        pred_orig = pred_norm.cpu().numpy() * std + mean
        preds.append(pred_orig)

        if isinstance(h_new, tuple):
            h = tuple(hh.detach() for hh in h_new)
        else:
            h = h_new.detach()

    return preds


class VectorPredictors:
    def __init__(self, data):
        """
        data: np.array of shape (T, D) where T is time and D is dimensions
        """
        self.data = data.astype(np.float32)
        self.T, self.D = data.shape
        self.df = pd.DataFrame(self.data)

    @staticmethod
    def normalize_probabilities(pred):
        """Enforce valid probability constraints in-place on a copy."""
        pred = np.maximum(pred, 0)

        # Group 1: node bins [0:2] must sum to 1
        sum1 = np.sum(pred[0:2])
        pred[0:2] = pred[0:2] / sum1 if sum1 > 0 else np.array([0.5, 0.5])

        # Group 2: edge bins [2:6] must sum to 1
        sum2 = np.sum(pred[2:6])
        pred[2:6] = pred[2:6] / sum2 if sum2 > 0 else np.array([0.25, 0.25, 0.25, 0.25])

        return pred

    def predict_sma(self, window=5):
        return self.df.rolling(window=window).mean().iloc[-1].values

    def predict_vewma(self, alpha=0.8):
        return self.df.ewm(alpha=alpha, adjust=False).mean().iloc[-1].values

    def predict_ssm(self):
        kf = KalmanFilter(dim_x=self.D, dim_z=self.D)
        kf.x = self.data[-1].reshape(-1, 1)
        kf.F = np.eye(self.D)
        kf.H = np.eye(self.D)
        kf.P *= 1000.
        kf.R = np.eye(self.D) * 0.01
        kf.Q = np.eye(self.D) * 0.0001
        for i in range(len(self.data)):
            kf.predict()
            kf.update(self.data[i])
        kf.predict()
        return kf.x.flatten()

    def predict_var(self):
        model = VAR(self.data)
        results = model.fit(maxlags=1)
        return results.forecast(self.data, steps=1)[0]

    def predict_vecm(self):
        if self.T < 5:
            return self.data[-1]
        try:
            deltas        = np.diff(self.data, axis=0)
            ect           = self.data[1:-1]
            lagged_deltas = deltas[:-1]
            X = np.hstack([lagged_deltas, ect])
            Y = deltas[1:]
            model = Ridge(alpha=1.0)
            model.fit(X, Y)
            current_ect   = self.data[-1].reshape(1, -1)
            current_delta = deltas[-1].reshape(1, -1)
            X_next = np.hstack([current_delta, current_ect])
            return self.data[-1] + model.predict(X_next).flatten()
        except:
            return self.data[-1]


def generate_single_latex_table(all_results, metric_key, caption, label):
    methods       = ["SMA_5", "VAR", "V-EWMA", "SSM", "VECM", "RNN", "LSTM", "GRU"]
    display_names = ["SMA",   "VAR", "V-EWMA", "SSM", "VECM", "RNN", "LSTM", "GRU"]

    latex = [
        r"\begin{table}[ht]",
        fr"\caption{{{caption}}}",
        fr"\label{{{label}}}",
        r"\centering",
        r"\begin{tabular}{l cccccccc}",
        r"\toprule",
        r"\textbf{Dataset} & " + " & ".join([fr"\textbf{{{m}}}" for m in display_names]) + r" \\",
        r"\midrule",
    ]

    col_acc = {m: [] for m in methods}

    for ds_res in all_results:
        name = ds_res['dataset'].replace('_', r'\_').capitalize()
        row  = [name]

        vals     = [ds_res.get(f"Test_{m}_{metric_key}", float('nan')) for m in methods]
        abs_vals = [abs(v) if not np.isnan(v) else float('inf') for v in vals]
        best_idx, second_idx = np.argsort(abs_vals)[:2]

        for i, v in enumerate(vals):
            col_acc[methods[i]].append(v)
            fmt = f"{v:.4f}" if not np.isnan(v) else "—"
            if i == best_idx:     row.append(fr"\textbf{{{fmt}}}")
            elif i == second_idx: row.append(fr"\underline{{{fmt}}}")
            else:                 row.append(fmt)

        latex.append(" & ".join(row) + r" \\")

    latex.append(r"\midrule[\heavyrulewidth]")
    avg_row  = [r"\textbf{Average}"]
    avg_vals = [np.nanmean(col_acc[m]) for m in methods]
    b_avg, s_avg = np.argsort([abs(v) for v in avg_vals])[:2]

    for i, v in enumerate(avg_vals):
        fmt = f"{v:.4f}"
        if i == b_avg:   avg_row.append(fr"\textbf{{{fmt}}}")
        elif i == s_avg: avg_row.append(fr"\underline{{{fmt}}}")
        else:            avg_row.append(fmt)

    latex.append(" & ".join(avg_row) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(latex)


if __name__ == "__main__":
    np.random.seed(1024)

    datasets     = ["sx-mathoverflow-700"]
    methods      = ["V-EWMA", "RNN"]
    bucket_sizes = [10]
    my_loader    = Loader()

    for num_buckets in bucket_sizes:
        print(f"\n{'#'*40}\nRUNNING FOR BUCKET SIZE: {num_buckets}\n{'#'*40}")

        for mode in ["Raw"]:
            all_final_results = []

            for dataset in datasets:
                print(f"--- Processing Dataset: {dataset} | Buckets: {num_buckets} ---")

                probabilities = my_loader.load_data(
                    type='probabilities', dataset=dataset,
                    activation='', normalized=True,
                    use_predicted=False, num_back='all'
                )
                if not isinstance(probabilities, np.ndarray):
                    probabilities = probabilities.values if hasattr(probabilities, 'values') else np.array(probabilities)

                gt_embeddings = probabilities.astype(np.float32)
                T, D          = gt_embeddings.shape
                val_idx       = int(T * 0.85)
                ds_results    = {"dataset": dataset}

                # Pre-compute online RNN predictions (full series, before the method loop)
                print(f"  Running online RNN/LSTM/GRU...")
                online_preds = {
                    arch: run_online_recurrent(gt_embeddings, arch)
                    for arch in ("RNN", "LSTM", "GRU")
                }
                print(f"  Done.")

                for m in methods:
                    split_mse            = []
                    complete_pred_series = []

                    for t in range(T):
                        history   = gt_embeddings[:t]
                        predictor = VectorPredictors(history) if t > 0 else None

                        try:
                            if t == 0:
                                pred = gt_embeddings[0].copy()
                            elif m == "SMA_5":
                                pred = predictor.predict_sma(window=min(5, t))
                            elif m == "V-EWMA":
                                # alpha=0.8 matches the paper pipeline (probs/generate_probs.py);
                                # toper uses alpha=0.5
                                pred = predictor.predict_vewma(alpha=0.8)
                            elif m == "SSM":
                                pred = predictor.predict_ssm()
                            elif m == "VAR":
                                # VAR needs at least D observations to fit
                                pred = predictor.predict_var() if t > D else gt_embeddings[t - 1]
                            elif m == "VECM":
                                pred = predictor.predict_vecm()
                            elif m in ("RNN", "LSTM", "GRU"):
                                pred = online_preds[m][t]
                        except Exception:
                            pred = gt_embeddings[t - 1] if t > 0 else gt_embeddings[0]

                        # Enforce probability constraints on every prediction
                        pred = VectorPredictors.normalize_probabilities(pred.copy())
                        complete_pred_series.append(pred)

                        if t >= val_idx:
                            split_mse.append(np.mean((gt_embeddings[t] - pred) ** 2))

                    # Save full prediction series
                    pickle_dir  = os.path.join('learning', 'pred_vectors', dataset)
                    os.makedirs(pickle_dir, exist_ok=True)
                    pickle_path = os.path.join(pickle_dir, f"{dataset}_testprobs_{m}_{mode}_{num_buckets}.pkl")
                    with open(pickle_path, 'wb') as f_pkl:
                        pickle.dump(complete_pred_series, f_pkl)

                    ds_results[f"Test_{m}_MSE"] = float(np.mean(split_mse)) if split_mse else float('nan')
                    print(f"  {m} | Test MSE: {ds_results[f'Test_{m}_MSE']:.6f}")

                all_final_results.append(ds_results)

            # Save LaTeX table
            out_path = os.path.join('learning', 'latex_tables', f"probs_results_buckets_{num_buckets}_{mode}.tex")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(generate_single_latex_table(
                    all_final_results, "MSE",
                    f"Probability Vector Prediction MSE (D={num_buckets})",
                    f"tab:probs_mse_{num_buckets}"
                ))

            # Print ranking
            df_results = pd.DataFrame(all_final_results)
            test_cols  = [c for c in df_results.columns if "Test_" in c and "_MSE" in c]
            if test_cols:
                ranking = df_results[test_cols].mean().sort_values()
                print(f"\n--- Global MSE Ranking ({mode}, D={num_buckets}) ---")
                print(ranking.to_string())
                best = ranking.index[0].replace("Test_", "").replace("_MSE", "")
                print(f"\nBest method: {best}")

            print(f"\nLaTeX table saved to: {out_path}")
