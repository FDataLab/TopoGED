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

# Ensure model path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from benchmarkers.benchmarker_utils.dataset_setup import load_data
from benchmarkers.gclstm.model import GCLSTMModel

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

import time, psutil, gc

def get_gpu_memory(device_id=0):
    if torch.cuda.is_available():
        # Handle cases where 'device' (the torch.device object) is passed
        if isinstance(device_id, torch.device):
            index = device_id.index if device_id.index is not None else 0
        else:
            index = device_id
        return torch.cuda.memory_reserved(index) / 1024**2
    return 0

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

@torch.no_grad()
def optimize_threshold(model, val_snaps, train_snaps, node_count, device, is_directed=True, window_size=10):
    model.eval()
    all_probs, all_targets = [], []
    history = train_snaps[-window_size:] if train_snaps else []
    full_sequence = history + val_snaps
    
    for i in range(window_size, len(full_sequence)):
        h, c = None, None
        for j in range(i - window_size, i - 1):
            snap = full_sequence[j].to(device)
            x = snap.x.to_dense() if snap.x.is_sparse else snap.x
            _, h, c = model(x, snap.edge_index, h, c)
        
        prev_snap = full_sequence[i - 1].to(device)
        x_prev = prev_snap.x.to_dense() if prev_snap.x.is_sparse else prev_snap.x
        z, h, c = model(x_prev, prev_snap.edge_index, h, c)
        
        target_snap = full_sequence[i].to(device)
        pos_idx = target_snap.edge_index
        neg_idx = torch.randint(0, node_count, (2, pos_idx.size(1)), device=device)
        
        pos_scores = torch.sigmoid(torch.sum(z[pos_idx[0]] * z[pos_idx[1]], dim=1))
        neg_scores = torch.sigmoid(torch.sum(z[neg_idx[0]] * z[neg_idx[1]], dim=1))
        
        all_probs.append(torch.cat([pos_scores, neg_scores]))
        all_targets.append(torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]))

    y_scores = torch.cat(all_probs).cpu().numpy()
    y_true = torch.cat(all_targets).cpu().numpy()
    thresholds = np.linspace(0.05, 0.99, 95)
    best_f1, best_threshold = 0, 0.01
    
    for t in thresholds:
        preds = (y_scores > t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1: 
            best_f1, best_threshold = score, t
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]

    print(f"--- Probability Distribution Check ---")
    print(f"Positives | Mean: {pos_scores.mean():.6f} | Std: {pos_scores.std():.6f} | Max: {pos_scores.max():.6f} | Min: {pos_scores.min():.6f}")
    print(f"Negatives | Mean: {neg_scores.mean():.6f} | Std: {neg_scores.std():.6f} | Max: {neg_scores.max():.6f} | Min: {neg_scores.min():.6f}")
    
    print(f"Optimal Threshold: {best_threshold:.2f} | Sampled Val F1: {best_f1:.4f}")
    return best_threshold


@torch.no_grad()
def construct_predicted_graphs(model, current_snaps, previous_snaps, node_count, device, dataset, file_path, is_directed=True, window_size=10, threshold=0.5):
    import scipy.sparse as sp
    import gc
    import pickle
    import os
    import numpy as np
    import torch
    import networkx as nx

    model.eval()
    
    # 1. Load Ground Truth for Metrics and Dynamic Capping
    from GraphGeneration.scripts.load_data import load_data
    _, _, _, target_graphs = load_data(
        dataset, '', '', '', 'all', 
        use_predicted=False, num_buckets=10, use_test_style=None
    )
    
    # Flatten buckets: target_graphs_flat[i] is the ground truth for snapshot i
    target_graphs_flat = [bucket[-1] for bucket in target_graphs]
    num_edges_in_targets = [g.number_of_edges() for g in target_graphs_flat]
    
    test_start_idx = len(target_graphs_flat) - len(current_snaps)

    history = previous_snaps[-window_size:] if previous_snaps else []
    full_sequence = history + current_snaps
    
    predicted_networks = []
    
    print(f"--- Starting GC-LSTM Sparse Construction (5x Dynamic Cap) ---")
    
    for i in range(window_size, len(full_sequence)):
        global_idx = test_start_idx + (i - window_size)
        h, c = None, None 
        
        # RNN Warm up
        for j in range(i - window_size, i - 1):
            snap = full_sequence[j].to(device)
            x = snap.x.to_dense() if snap.x.is_sparse else snap.x
            _, h, c = model(x, snap.edge_index, h, c)
        
        # Get embeddings for target snapshot
        prev_snap = full_sequence[i - 1].to(device)
        x_prev = prev_snap.x.to_dense() if prev_snap.x.is_sparse else prev_snap.x
        z, _, _ = model(x_prev, prev_snap.edge_index, h, c)

        # CHUNKED CANDIDATE COLLECTION
        all_rows, all_cols, all_scores = [], [], []
        chunk_size = 512

        for row_start in range(0, node_count, chunk_size):
            row_end = min(row_start + chunk_size, node_count)
            logits_chunk = torch.mm(z[row_start:row_end], z.t())
            probs_chunk = torch.sigmoid(logits_chunk)
            
            # Zero out self-loops
            diag_idx = torch.arange(row_start, row_end, device=device)
            probs_chunk[torch.arange(row_end - row_start), diag_idx] = 0
            
            mask = probs_chunk >= threshold
            rows, cols = torch.where(mask)
            scores = probs_chunk[mask]
            
            all_rows.append((rows + row_start).cpu())
            all_cols.append(cols.cpu())
            all_scores.append(scores.cpu())
            del logits_chunk, probs_chunk, mask

        # CONSOLIDATE CANDIDATES
        final_rows_tensor = torch.cat(all_rows)
        final_cols_tensor = torch.cat(all_cols)
        final_scores_tensor = torch.cat(all_scores)
        num_threshold_passed = final_scores_tensor.numel()

        # 2. DYNAMIC CAPPING LOGIC (5x edges of T-1)
        max_num_edges = num_edges_in_targets[global_idx - 1] * 5
        
        # Capture raw/uncapped arrays BEFORE capping
        raw_rows = final_rows_tensor.numpy()
        raw_cols = final_cols_tensor.numpy()
        
        if num_threshold_passed > max_num_edges:
            _, top_k_idx = torch.topk(final_scores_tensor, max_num_edges)
            final_rows = final_rows_tensor[top_k_idx].numpy()
            final_cols = final_cols_tensor[top_k_idx].numpy()
            status = "CAPPED"
        else:
            final_rows = raw_rows
            final_cols = raw_cols
            status = "ACCEPTED"

        # 3. PREPARE GROUND TRUTH
        true_graph = target_graphs_flat[global_idx].copy()
        true_graph.add_nodes_from(range(node_count))
        
        true_adj_sp = nx.to_scipy_sparse_array(true_graph, nodelist=range(node_count), format='csr')
        num_true_edges = true_adj_sp.nnz

        # --- HELPER FUNCTION FOR METRICS (FIXED) ---
        def get_metrics(pred_rows, pred_cols, N):
            if len(pred_rows) > 0:
                # FIX: Force explicit copy for writeable flag compatibility
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

        # 4. BUILD FINAL SPARSE MATRIX
        adj_final = sp.csr_matrix(
            (np.ones(len(final_rows), dtype=np.int8), (final_rows, final_cols)),
            shape=(node_count, node_count)
        )

        # 5. PRINT DUAL METRICS
        print(f"\nSnap {global_idx} | {status} | True Edges: {num_true_edges} | Cap Limit: {max_num_edges}")
        print(f"  [UNCAPPED] Pred: {len(raw_rows):<7} | TP: {tp_raw:<5} | FP: {fp_raw:<7} | TN: {tn_raw:<7} | FN: {fn_raw:<5}")
        print(f"  [CAPPED]   Pred: {adj_final.nnz:<7} | TP: {tp_cap:<5} | FP: {fp_cap:<7} | TN: {tn_cap:<7} | FN: {fn_cap:<5}")

        if not is_directed:
            adj_final = adj_final + adj_final.T
            adj_final.data[:] = 1

        predicted_networks.append(adj_final)
        
        # Cleanup
        del z, all_rows, all_cols, all_scores, final_rows_tensor, final_cols_tensor, final_scores_tensor
        gc.collect()
    # 6. Save Logic
    base_name = file_path.replace("_threshold", "")
    save_path = f"data/output/predicted/GCLSTM/{base_name}_threshold_5xCap.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump({'graphs': predicted_networks, 'node_count': node_count}, f)
    
    print(f"Saved sparse graphs to {save_path}")
    return predicted_networks


@torch.no_grad()
def validate_link_prediction(model, val_snaps, train_snaps, node_count, device, window_size=5):
    """Computes AUC on validation set to guide early stopping."""
    model.eval()
    all_probs, all_targets = [], []
    history = train_snaps[-window_size:] if train_snaps else []
    full_sequence = history + val_snaps
    
    for i in range(window_size, len(full_sequence)):
        h, c = None, None
        # Warm up
        for j in range(i - window_size, i - 1):
            snap = full_sequence[j].to(device)
            x = snap.x.to_dense() if snap.x.is_sparse else snap.x
            _, h, c = model(x, snap.edge_index, h, c)
        
        # Predict
        prev_snap = full_sequence[i - 1].to(device)
        x_prev = prev_snap.x.to_dense() if prev_snap.x.is_sparse else prev_snap.x
        z, h, c = model(x_prev, prev_snap.edge_index, h, c)
        
        target_snap = full_sequence[i].to(device)
        pos_idx = target_snap.edge_index
        # Sample 1:1 negatives
        neg_idx = torch.randint(0, node_count, (2, pos_idx.size(1)), device=device)
        
        pos_scores = torch.sigmoid(torch.sum(z[pos_idx[0]] * z[pos_idx[1]], dim=1))
        neg_scores = torch.sigmoid(torch.sum(z[neg_idx[0]] * z[neg_idx[1]], dim=1))
        
        all_probs.append(torch.cat([pos_scores, neg_scores]).cpu())
        all_targets.append(torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu())

    y_true = torch.cat(all_targets).numpy()
    y_scores = torch.cat(all_probs).numpy()
    return roc_auc_score(y_true, y_scores)

def train_model(dataset_snaps, node_count, node_features, dataset, beta, hidden_dim, K, lr, is_directed=True, device='cuda'):
    model = GCLSTMModel(node_count, node_features, hidden_dim=hidden_dim, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    window_size = 5
        
    n = len(dataset_snaps)
    train_snaps = dataset_snaps[:int(n*0.7)]
    val_snaps = dataset_snaps[int(n*0.7):int(n*0.85)]
    test_snaps = dataset_snaps[int(n*0.85):]

    start_train = time.time()
    
    # 1. Initialize tracking variables for the "Best" state
    best_val_auc = 0.0
    best_train_auc_at_val = 0.0  # New tracker
    best_model_wts = None
    no_improve = 0
    patience = 15

    for epoch in range(200):
        model.train()
        epoch_loss = 0
        
        # --- Standard Training Pass ---
        for i in range(window_size, len(train_snaps)):
            h, c = None, None
            for j in range(i - window_size, i - 1):
                snap = train_snaps[j].to(device)
                x = snap.x.to_dense() if snap.x.is_sparse else snap.x
                _, h, c = model(x, snap.edge_index, h, c)
            
            prev_snap = train_snaps[i - 1].to(device)
            x_prev = prev_snap.x.to_dense() if prev_snap.x.is_sparse else prev_snap.x
            z, h, c = model(x_prev, prev_snap.edge_index, h, c)
            
            target_snap = train_snaps[i].to(device)
            pos_idx = target_snap.edge_index
            neg_idx = torch.randint(0, node_count, (2, pos_idx.size(1)), device=device)
            
            pos_logits = torch.sum(z[pos_idx[0]] * z[pos_idx[1]], dim=1)
            neg_logits = torch.sum(z[neg_idx[0]] * z[neg_idx[1]], dim=1)
            
            loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) + \
                   F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            
            loss += beta * sum(p.pow(2.0).sum() for p in model.parameters())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # 2. Evaluation Phase
        train_auc = validate_link_prediction(model, train_snaps[window_size:], train_snaps[:window_size], node_count, device, window_size)
        val_auc = validate_link_prediction(model, val_snaps, train_snaps, node_count, device, window_size)
        
        avg_loss = epoch_loss / (len(train_snaps) - window_size)
        
        # 3. Update "Best" state based ONLY on Val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_train_auc_at_val = train_auc  # Capture corresponding train metric
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
    print(f"\n[FINAL TRAINING SUMMARY]")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Corresponding Train AUC: {best_train_auc_at_val:.4f}")
    print(f"---------------------------\n")

    t1, g1, r1 = time.time() - start_train, get_gpu_memory(device), get_ram_usage()

    # === PHASE 2: THRESHOLD OPTIMIZATION ===
    start_opt = time.time()
    model.load_state_dict(best_model_wts)
    opt_threshold = optimize_threshold(model, val_snaps, train_snaps, node_count, device, is_directed, window_size)
    t2, g2, r2 = time.time() - start_opt, get_gpu_memory(device), get_ram_usage()

    # === PHASE 3: CONSTRUCTION ===
    start_cons = time.time()
    file_path = f"{dataset}_{beta}_{hidden_dim}_{K}_{lr}_{'directed' if is_directed else 'undirected'}"
    construct_predicted_graphs(model, test_snaps, val_snaps, node_count, device, dataset, file_path, is_directed, window_size, threshold=opt_threshold)
    t3, g3, r3 = time.time() - start_cons, get_gpu_memory(device), get_ram_usage()

    print(f"\n--- DATASET: {dataset} (GC-LSTM) METRICS ---")
    print(f"TRAIN:  Time={t1:.2f}s, GPU={g1:.2f}MB, RAM={r1:.2f}MB")
    print(f"THRESH: Time={t2:.2f}s, GPU={g2:.2f}MB, RAM={r2:.2f}MB")
    print(f"CONST:  Time={t3:.2f}s, GPU={g3:.2f}MB, RAM={r3:.2f}MB")
    print(f"---------------------------------------------\n")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    is_directed = not args.undirected 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dict = load_data('gclstm', args.dataset)

    train_model(data_dict['snapshots'], data_dict['node_count'], data_dict['feature_dim'], args.dataset, args.beta, args.hidden_dim, args.K, args.lr, is_directed, device)
