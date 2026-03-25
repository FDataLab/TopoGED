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

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
random.seed(seed)
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
    thresholds = np.linspace(0.01, 0.99, 50)
    best_f1, best_threshold = 0, 0.01
    
    for t in thresholds:
        preds = (y_scores > t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1: 
            best_f1, best_threshold = score, t
            
    print(f"Optimal Threshold: {best_threshold:.2f} | Sampled Val F1: {best_f1:.4f}")
    return best_threshold

@torch.no_grad()
def construct_predicted_graphs(model, current_snaps, previous_snaps, node_count, device, dataset, file_path, is_directed=True, window_size=10, threshold=0.5):
    from benchmarkers.benchmarker_utils.k_values_extractor import get_topk
    model.eval()
    history = previous_snaps[-window_size:] if previous_snaps else []
    full_sequence = history + current_snaps
    all_embeddings = []
    
    for i in range(window_size, len(full_sequence)):
        h, c = None, None 
        for j in range(i - window_size, i - 1):
            snap = full_sequence[j].to(device)
            x = snap.x.to_dense() if snap.x.is_sparse else snap.x
            _, h, c = model(x, snap.edge_index, h, c)
        
        prev_snap = full_sequence[i - 1].to(device)
        x_prev = prev_snap.x.to_dense() if prev_snap.x.is_sparse else prev_snap.x
        z, _, _ = model(x_prev, prev_snap.edge_index, h, c)
        all_embeddings.append(z)

    try:
        top_k_values = get_topk(dataset, use_true=False)
        test_top_k = top_k_values[-len(current_snaps):]
        strategies = [True, False]
    except:
        strategies = [False]

    for using_topk in strategies:
        predicted_networks = []
        for t, z in enumerate(all_embeddings):
            # CHUNKED RECONSTRUCTION to prevent N x N allocation crash
            adj_matrix = np.zeros((node_count, node_count), dtype=np.int8)
            chunk_size = 512
            
            if using_topk:
                k = int(test_top_k[t])
                all_vals, all_inds = [], []

            for i in range(0, node_count, chunk_size):
                end_i = min(i + chunk_size, node_count)
                logits_chunk = torch.mm(z[i:end_i], z.t())
                probs_chunk = torch.sigmoid(logits_chunk)
                
                # Zero out self-loops
                diag_idx = torch.arange(i, end_i, device=device)
                probs_chunk[torch.arange(end_i - i), diag_idx] = 0
                
                if using_topk:
                    ck = min(k, probs_chunk.numel())
                    v, l = torch.topk(probs_chunk.view(-1), ck)
                    all_vals.append(v)
                    all_inds.append(l + (i * node_count))
                else:
                    mask = (probs_chunk > threshold).cpu().numpy()
                    adj_matrix[i:end_i] = mask.astype(np.int8)

            if using_topk:
                top_v = torch.cat(all_vals)
                top_i = torch.cat(all_inds)
                _, global_locs = torch.topk(top_v, min(k, top_v.numel()))
                final_inds = top_i[global_locs]
                rows = (final_inds // node_count).cpu().numpy()
                cols = (final_inds % node_count).cpu().numpy()
                adj_matrix[rows, cols] = 1

            if not is_directed: adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
            predicted_networks.append(nx.from_numpy_array(adj_matrix, create_using=(nx.DiGraph() if is_directed else nx.Graph())))

        strategy = 'topk' if using_topk else 'threshold'
        save_path = f"data/output/predicted/GCLSTM/{file_path}_{strategy}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'graphs': predicted_networks, 'node_count': node_count}, f)
        print(f"Saved GC-LSTM graphs ({strategy}) to {save_path}")


def train_model(dataset_snaps, node_count, node_features, dataset, beta, hidden_dim, K, lr, is_directed=True, device='cuda'):
    model = GCLSTMModel(node_count, node_features, hidden_dim=hidden_dim, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    window_size, patience = 5, 15
    best_loss, no_improve, best_model_wts = float('inf'), 0, None
    
    n = len(dataset_snaps)
    train_snaps = dataset_snaps[:int(n*0.7)]
    val_snaps = dataset_snaps[int(n*0.7):int(n*0.85)]
    test_snaps = dataset_snaps[int(n*0.85):]

    start_train = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        all_preds, all_targets = [], []
        
        for i in range(window_size, len(train_snaps)):
            h, c = None, None
            # History window unrolling
            for j in range(i - window_size, i - 1):
                snap = train_snaps[j].to(device)
                x = snap.x.to_dense() if snap.x.is_sparse else snap.x
                _, h, c = model(x, snap.edge_index, h, c)
            
            # Prediction step
            prev_snap = train_snaps[i - 1].to(device)
            x_prev = prev_snap.x.to_dense() if prev_snap.x.is_sparse else prev_snap.x
            z, h, c = model(x_prev, prev_snap.edge_index, h, c)
            
            # --- SAMPLED LOSS (Sparse) ---
            target_snap = train_snaps[i].to(device)
            pos_idx = target_snap.edge_index
            neg_idx = torch.randint(0, node_count, (2, pos_idx.size(1)), device=device)
            
            pos_logits = torch.sum(z[pos_idx[0]] * z[pos_idx[1]], dim=1)
            neg_logits = torch.sum(z[neg_idx[0]] * z[neg_idx[1]], dim=1)
            
            loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) + \
                   F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            
            # Regularization
            loss += beta * sum(p.pow(2.0).sum() for p in model.parameters())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if epoch % 10 == 0:
                all_preds.append(torch.cat([pos_logits.sigmoid(), neg_logits.sigmoid()]).detach())
                all_targets.append(torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]).detach())

        avg_loss = epoch_loss / (len(train_snaps) - window_size)
        if epoch % 10 == 0: 
            y_true = torch.cat(all_targets).cpu().numpy()
            y_scores = torch.cat(all_preds).cpu().numpy()
            print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.6f} | Sampled Train AUC: {roc_auc_score(y_true, y_scores):.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience: break

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