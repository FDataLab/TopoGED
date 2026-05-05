import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
import networkx as nx
import numpy as np
import pickle
import argparse
import copy
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.utils import to_dense_adj

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from benchmarkers.benchmarker_utils.dataset_setup import load_data
from benchmarkers.tgcn.model import TGCNModel

import torch
import numpy as np
import random
seed = random.randint(1, 500)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Seed set to: {seed}")

import time
import psutil
import gc

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved(0) / 1024**2 # MB
    return 0

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2 # MB


@torch.no_grad()
def optimize_threshold(model, val_snaps, train_snaps, node_count, hidden_dim, device, is_directed=True):
    model.eval()
    all_probs, all_targets = [], []
    
    # Warm up hidden state with training history
    h = torch.zeros(node_count, hidden_dim, device=device)
    for snap in train_snaps:
        x = snap.x.to_dense() if snap.x.is_sparse else snap.x
        _, h = model(x.to(device), snap.edge_index.to(device), h)

    # Validate using the next-step forecasting logic
    for i in range(len(val_snaps) - 1):
        snap, target_snap = val_snaps[i], val_snaps[i+1]
        x = snap.x.to_dense() if snap.x.is_sparse else snap.x
        z, h = model(x.to(device), snap.edge_index.to(device), h)
        
        # Sample-based validation (Positive edges from target, 1:1 Negatives)
        pos_idx = target_snap.edge_index.to(device)
        neg_idx = torch.randint(0, node_count, (2, pos_idx.size(1)), device=device)
        
        # Inner product decoding for samples
        pos_scores = torch.sigmoid(torch.sum(z[pos_idx[0]] * z[pos_idx[1]], dim=1))
        neg_scores = torch.sigmoid(torch.sum(z[neg_idx[0]] * z[neg_idx[1]], dim=1))
        
        all_probs.append(torch.cat([pos_scores, neg_scores]))
        all_targets.append(torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]))

    y_scores = torch.cat(all_probs).cpu().numpy()
    y_true = torch.cat(all_targets).cpu().numpy()
    
    thresholds = np.linspace(0.05, 0.99, 95)
    best_f1, best_threshold = 0, 0.5
    
    for t in thresholds:
        preds = (y_scores > t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, t
    print(f"--- Probability Distribution Check ---")
    print(f"Positives | Mean: {pos_scores.mean():.6f} | Std: {pos_scores.std():.6f} | Max: {pos_scores.max():.6f} | Min: {pos_scores.min():.6f}")
    print(f"Negatives | Mean: {neg_scores.mean():.6f} | Std: {neg_scores.std():.6f} | Max: {neg_scores.max():.6f} | Min: {neg_scores.min():.6f}")
                
    print(f"Optimal Threshold: {best_threshold:.2f} | Sampled Val F1: {best_f1:.4f}")
    return best_threshold

@torch.no_grad()
def construct_predicted_graphs(model, current_snaps, previous_snaps, node_count, hidden_dim, device, dataset, file_path, is_directed=True, threshold=0.5):
    import scipy.sparse as sp
    import gc
    import pickle
    import psutil
    import os
    import numpy as np
    import torch
    import networkx as nx

    def get_ram_usage():
        return psutil.Process().memory_info().rss / (1024 * 1024)

    model.eval()
    
    # 1. Load Ground Truth for Metrics and Dynamic Capping
    from GraphGeneration.scripts.load_data import load_data
    _, _, _, target_graphs = load_data(
        dataset, '', '', '', 'all', 
        use_predicted=False, num_buckets=10, use_test_style=None
    )
    
    # Flatten buckets and calculate edge counts
    target_graphs_flat = [bucket[-1] for bucket in target_graphs]
    num_edges_in_targets = [g.number_of_edges() for g in target_graphs_flat]

    # Calculate the global starting index for the test set
    total_snaps = len(target_graphs_flat)
    test_start_idx = total_snaps - len(current_snaps)

    # 2. Warm up hidden state
    h = torch.zeros(node_count, hidden_dim, device=device)
    for snap in previous_snaps:
        x = snap.x.to_dense() if snap.x.is_sparse else snap.x
        _, h = model(x.to(device), snap.edge_index.to(device), h)
            
    predicted_networks = []

    # 3. Sequential Inference and Standardized Construction
    print(f"--- Starting TGCN Sparse Construction (5x Dynamic Cap): {len(current_snaps)} Snapshots ---")
    for t_local, snap in enumerate(current_snaps):
        # Global index for target alignment
        t_global = test_start_idx + t_local
        
        x = snap.x.to_dense() if snap.x.is_sparse else snap.x
        z, h = model(x.to(device), snap.edge_index.to(device), h)
        
        # CHUNKED CANDIDATE COLLECTION
        all_rows, all_cols, all_scores = [], [], []
        chunk_size = 512
        
        for i in range(0, node_count, chunk_size):
            end_i = min(i + chunk_size, node_count)
            logits_chunk = torch.mm(z[i:end_i], z.t())
            probs_chunk = torch.sigmoid(logits_chunk)
            
            # Zero out self-loops
            diag_idx = torch.arange(i, end_i, device=device)
            probs_chunk[torch.arange(end_i - i), diag_idx] = 0
            
            mask = probs_chunk >= threshold
            rows, cols = torch.where(mask)
            scores = probs_chunk[mask]
            
            all_rows.append((rows + i).cpu())
            all_cols.append(cols.cpu())
            all_scores.append(scores.cpu())
            del logits_chunk, probs_chunk, mask

        # Consolidate candidates
        full_rows_tensor = torch.cat(all_rows)
        full_cols_tensor = torch.cat(all_cols)
        full_scores_tensor = torch.cat(all_scores)
        num_threshold_passed = full_scores_tensor.numel()

        # 4. STANDARDIZED DYNAMIC CAPPING (5x edges of T-1)
        max_num_edges = max(num_edges_in_targets[t_global - 1] * 5, 1000)
        
        # Capture raw/uncapped arrays BEFORE capping
        raw_rows = full_rows_tensor.numpy()
        raw_cols = full_cols_tensor.numpy()
        
        if num_threshold_passed > max_num_edges:
            _, top_k_idx = torch.topk(full_scores_tensor, max_num_edges)
            final_rows = full_rows_tensor[top_k_idx].numpy()
            final_cols = full_cols_tensor[top_k_idx].numpy()
            status = "CAPPED"
        else:
            final_rows = raw_rows
            final_cols = raw_cols
            status = "ACCEPTED"

        # 5. PREPARE GROUND TRUTH (Keep it sparse!)
        true_graph = target_graphs_flat[t_global].copy()
        true_graph.add_nodes_from(range(node_count))
        
        true_adj_sp = nx.to_scipy_sparse_array(true_graph, nodelist=range(node_count), format='csr')
        num_true_edges = true_adj_sp.nnz

        # --- HELPER FUNCTION FOR METRICS (FIXED FOR NUMPY 1.23+) ---
        def get_metrics(pred_rows, pred_cols, N):
            if len(pred_rows) > 0:
                # FIX: Explicitly copy indices to ensure they own their memory and are writeable
                r_idx = np.array(pred_rows).copy()
                c_idx = np.array(pred_cols).copy()
                
                matched = np.array(true_adj_sp[r_idx, c_idx]).flatten()
                tp = np.sum(matched > 0)
                fp = len(pred_rows) - tp
                fn = num_true_edges - tp
                tn = (int(N) * (int(N) - 1)) - (tp + fp + fn)
                return tp, fp, tn, fn
            else:
                return 0, 0, (int(N) * (int(N) - 1)) - num_true_edges, num_true_edges

        # Calculate both sets of metrics
        tp_raw, fp_raw, tn_raw, fn_raw = get_metrics(raw_rows, raw_cols, node_count)
        tp_cap, fp_cap, tn_cap, fn_cap = get_metrics(final_rows, final_cols, node_count)

        # 6. BUILD FINAL SPARSE MATRIX
        adj_final = sp.csr_matrix(
            (np.ones(len(final_rows), dtype=np.int8), (final_rows, final_cols)),
            shape=(node_count, node_count)
        )

        # 7. PRINT DUAL METRICS
        print(f"\nSnap {t_global} | {status} | True Edges: {num_true_edges} | Cap Limit: {max_num_edges}")
        print(f"  [UNCAPPED] Pred: {len(raw_rows):<7} | TP: {tp_raw:<5} | FP: {fp_raw:<7} | TN: {tn_raw:<7} | FN: {fn_raw:<5}")
        print(f"  [CAPPED]   Pred: {adj_final.nnz:<7} | TP: {tp_cap:<5} | FP: {fp_cap:<7} | TN: {tn_cap:<7} | FN: {fn_cap:<5}")

        if not is_directed:
            adj_final = adj_final + adj_final.T
            adj_final.data[:] = 1

        predicted_networks.append(adj_final)
        
        # Cleanup
        del z, all_rows, all_cols, all_scores, full_rows_tensor, full_cols_tensor, full_scores_tensor, raw_rows, raw_cols, final_rows, final_cols
        gc.collect()

    # 8. Save Logic with _5xCap suffix
    save_path = f"data/output/predicted/TGCN/{file_path}_threshold_5xCap.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump({'graphs': predicted_networks, 'node_count': node_count}, f)
    
    print(f"Saved memory-safe TGCN sparse graphs to {save_path}")
    return predicted_networks

@torch.no_grad()
def validate_link_prediction(model, target_snaps, history_snaps, node_count, hidden_dim, device):
    """Standardized helper for T-GCN Link Prediction AUC."""
    model.eval()
    all_scores, all_targets = [], []
    
    # 1. Warm up hidden state through the provided history
    h = torch.zeros(node_count, hidden_dim, device=device)
    for snap in history_snaps:
        x = snap.x.to_dense() if snap.x.is_sparse else snap.x
        _, h = model(x.to(device), snap.edge_index.to(device), h)

    # 2. Evaluate target snapshots
    for i in range(len(target_snaps) - 1):
        snap, target_snap = target_snaps[i], target_snaps[i+1]
        x = snap.x.to_dense() if snap.x.is_sparse else snap.x
        z, h = model(x.to(device), snap.edge_index.to(device), h)
        
        pos_idx = target_snap.edge_index.to(device)
        if pos_idx.size(1) == 0: continue
        
        # Standard 1:1 sampling
        num_pos = min(pos_idx.size(1), 10000)
        neg_idx = torch.randint(0, node_count, (2, num_pos), device=device)
        
        p_p = torch.sigmoid(torch.sum(z[pos_idx[0, :num_pos]] * z[pos_idx[1, :num_pos]], dim=1))
        n_p = torch.sigmoid(torch.sum(z[neg_idx[0]] * z[neg_idx[1]], dim=1))
        
        all_scores.append(torch.cat([p_p, n_p]).cpu().numpy().flatten())
        all_targets.append(np.concatenate([np.ones(p_p.size(0)), np.zeros(n_p.size(0))]))

    return roc_auc_score(np.concatenate(all_targets), np.concatenate(all_scores))

def train_model(dataset_snaps, node_count, node_features, dataset, hidden_dim, lr, lambda_loss, is_directed=True, device='cuda'):
    model = TGCNModel(node_count, node_features, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n = len(dataset_snaps)
    train_snaps = dataset_snaps[:int(n*0.7)]
    val_snaps = dataset_snaps[int(n*0.7):int(n*0.85)] 
    test_snaps = dataset_snaps[int(n*0.85):]

    best_val_auc = 0.0
    best_train_auc_at_val = 0.0
    patience, no_improve, best_model_wts = 15, 0, None
    
    start_train = time.time()

    for epoch in range(200): 
        model.train()
        h = torch.zeros(node_count, hidden_dim).to(device) 
        epoch_loss = 0
        
        # 1. Training Loop
        for i in range(len(train_snaps) - 1):
            snap = train_snaps[i].to(device)
            target_snap = train_snaps[i+1].to(device)
            
            z, h = model(snap.x, snap.edge_index, h)
            
            pos_idx = target_snap.edge_index
            pos_scores = torch.sum(z[pos_idx[0]] * z[pos_idx[1]], dim=1)
            neg_idx = torch.randint(0, node_count, (2, pos_idx.size(1)), device=device)
            neg_scores = torch.sum(z[neg_idx[0]] * z[neg_idx[1]], dim=1)
            
            loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                    F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            
            reg_loss = lambda_loss * sum(p.pow(2).sum() for p in model.parameters())
            total_batch_loss = loss + reg_loss
            
            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_batch_loss.item()
            h = h.detach()

        # 2. Dual AUC Evaluation
        # Train AUC (Warm up with first, evaluate rest)
        train_auc = validate_link_prediction(model, train_snaps[1:], [train_snaps[0]], node_count, hidden_dim, device)
        # Val AUC (Warm up with full train history)
        val_auc = validate_link_prediction(model, val_snaps, train_snaps, node_count, hidden_dim, device)
        
        avg_loss = epoch_loss / (len(train_snaps) - 1)

        # 3. Early Stopping Logic on Val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_train_auc_at_val = train_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
            marker = "*"
        else:
            no_improve += 1
            marker = ""

        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} {marker}")
        
        if no_improve >= patience:
            print(f"Early stopping triggered. Best Val AUC: {best_val_auc:.4f}")
            break
            
    # --- FINAL SUMMARY PRINT ---
    print(f"\n[FINAL TRAINING SUMMARY - T-GCN]")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Corresponding Train AUC: {best_train_auc_at_val:.4f}")
    print(f"---------------------------\n")
            
    t1 = time.time() - start_train
    g1 = torch.cuda.max_memory_reserved(device) / 1024**2 if torch.cuda.is_available() else 0
    r1 = get_ram_usage()

    # === PHASE 2: THRESHOLD OPTIMIZATION ===
    start_opt = time.time()
    model.load_state_dict(best_model_wts)
    opt_threshold = optimize_threshold(model, val_snaps, train_snaps, node_count, hidden_dim, device, is_directed)
    t2 = time.time() - start_opt
    g2 = torch.cuda.max_memory_reserved(device) / 1024**2 if torch.cuda.is_available() else 0
    r2 = get_ram_usage()

    # === PHASE 3: CONSTRUCTION ===
    start_cons = time.time()
    file_path = f"{dataset}_{hidden_dim}_{lr}_{lambda_loss}_{'directed' if is_directed else 'undirected'}"
    construct_predicted_graphs(model, test_snaps, train_snaps + val_snaps, node_count, hidden_dim, device, dataset, file_path, is_directed, threshold=opt_threshold)
    t3 = time.time() - start_cons
    g3 = torch.cuda.max_memory_reserved(device) / 1024**2 if torch.cuda.is_available() else 0
    r3 = get_ram_usage()

    print(f"\n--- DATASET: {dataset} (T-GCN) METRICS ---")
    print(f"TRAIN:  Time={t1:.2f}s, GPU={g1:.2f}MB, RAM={r1:.2f}MB")
    print(f"THRESH: Time={t2:.2f}s, GPU={g2:.2f}MB, RAM={r2:.2f}MB")
    print(f"CONST:  Time={t3:.2f}s, GPU={g3:.2f}MB, RAM={r3:.2f}MB")
    print(f"-------------------------------------------\n")

    gc.collect(); torch.cuda.empty_cache()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_loss", type=float, default=0.0015)
    parser.add_argument("--window_size", type=int, default=10)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dict = load_data('tgcn', args.dataset)

    train_model(data_dict['snapshots'], data_dict['node_count'], data_dict['feature_dim'], 
                args.dataset, args.hidden_dim, args.lr, args.lambda_loss, not args.undirected, device)
