"""
Normalized TopER forecasting (supervisor's question: does normalizing the TopER
embedding by node/edge count change MAR and the downstream graph metrics?).

Idea (shape / scale decomposition):
  Raw TopER vector per snapshot is interleaved cumulative counts
      [n1,e1, n2,e2, ..., n10,e10]
  where bucket 10 is the whole graph, so n10 = total_nodes, e10 = total_edges.

  Normalize each snapshot by its own totals:
      n_i_norm = n_i / total_nodes      e_i_norm = e_i / total_edges
  This turns each snapshot into a monotone shape curve in [0,1] (ending at 1.0),
  separating "shape" (distribution across the filtration) from "scale" (size).

  We then:
    1. forecast the normalized vector with method M  -> shape forecast
    2. forecast the 2-D totals series [total_nodes, total_edges] SEPARATELY with M
       (Option A: honest forecast from the past only, no leakage. For V-EWMA this
        is identical to the raw arm's n10/e10 forecast.)
    3. reconstruct raw counts:  n_i = n_i_norm * predicted_total_nodes  (same for edges)

  The reconstructed 20-D raw-count series is written in the EXACT same pickle
  format the construction script consumes, tagged "-norm", so the downstream code
  path is unchanged.

This file does NOT modify generate_toper_cl.py; it imports its forecasters so the
methods are byte-identical to the raw arm.
"""
import os
import sys
import pickle
import numpy as np

# Importing generate_toper_cl sets up project-root chdir + Loader path redirection
# and gives us the exact same forecasters used by the raw pipeline.
import generate_toper_cl as base
from generate_toper_cl import run_online_recurrent, VectorPredictors
from utils.loader import Loader

_learning = os.path.dirname(os.path.abspath(__file__))

# Match the raw pipeline's smoothing constant for TopER (alpha=0.5).
_VEWMA_ALPHA = 0.5


def forecast_series(X, method):
    """Next-step forecast for every t, replicating generate_toper_cl exactly.
    Returns array (T, D); row 0 is the first ground-truth vector (persistence start)."""
    X = np.asarray(X, dtype=np.float32)
    T, _ = X.shape
    if method in ("RNN", "LSTM", "GRU"):
        return np.array(run_online_recurrent(X, method))
    if method == "V-EWMA":
        out = [X[0].copy()]
        for t in range(1, T):
            out.append(VectorPredictors(X[:t]).predict_vewma(alpha=_VEWMA_ALPHA))
        return np.array(out)
    raise ValueError(f"unknown method {method}")


def normalize(gt):
    """gt: (T,20) interleaved counts -> (norm (T,20), totals (T,2))."""
    nodes = gt[:, 0::2]          # (T,10)  cumulative node counts
    edges = gt[:, 1::2]          # (T,10)  cumulative edge counts
    tot_n = gt[:, -2].copy()     # total nodes  (= n10)
    tot_e = gt[:, -1].copy()     # total edges  (= e10)
    dn = np.where(tot_n > 0, tot_n, 1.0)[:, None]
    de = np.where(tot_e > 0, tot_e, 1.0)[:, None]
    norm = np.empty_like(gt)
    norm[:, 0::2] = nodes / dn
    norm[:, 1::2] = edges / de
    totals = np.stack([tot_n, tot_e], axis=1)
    return norm, totals


def reconstruct(pred_norm, pred_tot):
    """Rebuild 20-D raw counts from normalized shape forecast + totals forecast.
    Guards: clip fractions to [0,1] then cumulative-max to keep the curve monotone."""
    T = pred_norm.shape[0]
    fn = np.clip(pred_norm[:, 0::2], 0.0, 1.0)
    fe = np.clip(pred_norm[:, 1::2], 0.0, 1.0)
    fn = np.maximum.accumulate(fn, axis=1)
    fe = np.maximum.accumulate(fe, axis=1)
    pn = np.clip(pred_tot[:, 0], 0.0, None)[:, None]
    pe = np.clip(pred_tot[:, 1], 0.0, None)[:, None]
    recon = np.empty((T, 20), dtype=np.float64)
    recon[:, 0::2] = fn * pn
    recon[:, 1::2] = fe * pe
    recon[:, -2] = pn[:, 0]          # totals exact
    recon[:, -1] = pe[:, 0]
    return recon


def mar(pred, true, lo, hi):
    """Mean Absolute Relative error over test rows [lo:hi], dims with true>0."""
    p, t = pred[lo:hi], true[lo:hi]
    mask = t > 0
    rel = np.abs(p - t) / np.where(mask, t, 1.0)
    return float(rel[mask].mean())


def pct_err(pred, true, lo, hi, col):
    """Signed mean percent error on one column (matches existing node/edge metric)."""
    p, t = pred[lo:hi, col], true[lo:hi, col]
    d = np.where(t != 0, t, 1.0)
    return float((((p - t) / d) * 100).mean())


if __name__ == "__main__":
    import torch
    np.random.seed(1024)
    torch.manual_seed(1024)

    datasets = ["mathoverflow", "networkaion", "networkdgd", "sx-mathoverflow-700"]
    methods = ["V-EWMA", "RNN", "LSTM", "GRU"]
    num_buckets = 10
    loader = Loader()

    rows = []  # (dataset, method, MAR_raw, MAR_norm, MARtot_raw, MARtot_norm)

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        emb, _ = loader.load_data(dataset, "Degree", include_weights=False, num_buckets=num_buckets)
        gt = np.array([np.array(s).flatten() for s in emb], dtype=np.float64)
        T = gt.shape[0]
        lo, hi = int(T * 0.85), T

        norm, totals = normalize(gt)

        for m in methods:
            raw_pred = forecast_series(gt, m)                 # raw arm (for MAR baseline)
            pred_norm = forecast_series(norm, m)             # normalized shape forecast
            pred_tot = forecast_series(totals, m)            # separate totals forecast
            recon = reconstruct(pred_norm, pred_tot)         # normalized arm prediction

            mar_raw = mar(raw_pred, gt, lo, hi)
            mar_norm = mar(recon, gt, lo, hi)
            # totals-only MAR (the part that drives node/edge budgets)
            tot_idx = np.array([18, 19])
            mar_tot_raw = mar(raw_pred[:, tot_idx], gt[:, tot_idx], lo, hi)
            mar_tot_norm = mar(recon[:, tot_idx], gt[:, tot_idx], lo, hi)
            rows.append((dataset, m, mar_raw, mar_norm, mar_tot_raw, mar_tot_norm))
            print(f"  {m:7s} MAR raw={mar_raw:.4f} norm={mar_norm:.4f} | "
                  f"MAR_totals raw={mar_tot_raw:.4f} norm={mar_tot_norm:.4f}")

            # Save normalized prediction in the construction's pickle format.
            name = f"{m}-norm"
            out_dir = os.path.join(_learning, "pred_vectors", dataset)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{dataset}_testtoper_{name}_Raw_{num_buckets}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump([r for r in recon], f)

    # ---- Markdown MAR table ----
    print("\n\n#### MAR: raw vs normalized TopER forecast (test split)\n")
    print("| Dataset | Method | MAR raw | MAR norm | MAR_totals raw | MAR_totals norm |")
    print("|---|---|---|---|---|---|")
    for ds, m, a, b, c, d in rows:
        print(f"| {ds} | {m} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.4f} |")

    # ---- Save CSV ----
    out_csv = os.path.join(_learning, "latex_tables", "mar_normalized.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("dataset,method,mar_raw,mar_norm,mar_totals_raw,mar_totals_norm\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nSaved {out_csv}")
