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
    
    thresholds = np.linspace(0.01, 0.99, 50)
    best_f1, best_threshold = 0, 0.5
    
    for t in thresholds:
        preds = (y_scores > t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, t
            
    print(f"Optimal Threshold: {best_threshold:.2f} | Sampled Val F1: {best_f1:.4f}")
    return best_threshold

@torch.no_grad()
def construct_predicted_graphs(model, current_snaps, previous_snaps, node_count, hidden_dim, device, dataset, file_path, is_directed=True, threshold=0.5):
    from benchmarkers.benchmarker_utils.k_values_extractor import get_topk
    model.eval()
    
    # Warm up
    h = torch.zeros(node_count, hidden_dim, device=device)
    for snap in previous_snaps:
        x = snap.x.to_dense() if snap.x.is_sparse else snap.x
        _, h = model(x.to(device), snap.edge_index.to(device), h)
            
    # Generate embeddings for forecasting
    all_embeddings = []
    for snap in current_snaps:
        x = snap.x.to_dense() if snap.x.is_sparse else snap.x
        z, h = model(x.to(device), snap.edge_index.to(device), h)
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
            # CHUNKED RECONSTRUCTION to avoid N x N crash
            adj_matrix = np.zeros((node_count, node_count), dtype=np.int8)
            chunk_size = 512
            
            if using_topk:
                k = int(test_top_k[t])
                all_vals, all_inds = [], []

            for i in range(0, node_count, chunk_size):
                end_i = min(i + chunk_size, node_count)
                logits_chunk = torch.mm(z[i:end_i], z.t())
                probs_chunk = torch.sigmoid(logits_chunk)
                
                # Zero out self-loops in chunk
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

            if not is_directed:
                adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
            
            predicted_networks.append(nx.from_numpy_array(adj_matrix, create_using=(nx.DiGraph() if is_directed else nx.Graph())))

        strategy = 'topk' if using_topk else 'threshold'
        save_path = f"data/output/predicted/TGCN/{file_path}_{strategy}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'graphs': predicted_networks, 'node_count': node_count}, f)
        print(f"Saved T-GCN predicted graphs ({strategy}) to {save_path}")
            
    return predicted_networks

# --- 3. MAIN TRAINER ---

def train_model(dataset_snaps, node_count, node_features, dataset, hidden_dim, lr, lambda_loss, is_directed=True, device='cuda'):
    model = TGCNModel(node_count, node_features, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    patience, best_loss, no_improve, best_model_wts = 15, float('inf'), 0, None
    n = len(dataset_snaps)
    train_snaps = dataset_snaps[:int(n*0.7)]
    val_snaps = dataset_snaps[int(n*0.7):int(n*0.85)] 
    test_snaps = dataset_snaps[int(n*0.85):]

    start_train = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(200): 
        model.train()
        h = torch.zeros(node_count, hidden_dim).to(device) 
        optimizer.zero_grad()
        epoch_loss = 0
        all_preds, all_targets = [], []
        
        for i in range(len(train_snaps) - 1):
            snap = train_snaps[i].to(device)
            target_snap = train_snaps[i+1].to(device)
            
            # Forward: Get embeddings
            z, h = model(snap.x, snap.edge_index, h)
            
            # Sparse Link Prediction Loss
            pos_idx = target_snap.edge_index
            pos_scores = torch.sum(z[pos_idx[0]] * z[pos_idx[1]], dim=1)
            
            neg_idx = torch.randint(0, node_count, (2, pos_idx.size(1)), device=device)
            neg_scores = torch.sum(z[neg_idx[0]] * z[neg_idx[1]], dim=1)
            
            loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                   F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            
            # Regularization
            reg_loss = lambda_loss * sum(p.pow(2).sum() for p in model.parameters())
            total_batch_loss = loss + reg_loss
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += total_batch_loss.item()
            h = h.detach()

            if epoch % 10 == 0:
                all_preds.append(torch.cat([torch.sigmoid(pos_scores), torch.sigmoid(neg_scores)]).detach())
                all_targets.append(torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).detach())

        avg_loss = epoch_loss / (len(train_snaps) - 1)
        if epoch % 10 == 0:
            y_t = torch.cat(all_targets).cpu().numpy()
            y_p = torch.cat(all_preds).cpu().numpy()
            train_auc = roc_auc_score(y_t, y_p)
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | Sampled AUC: {train_auc:.4f}")

        if avg_loss < best_loss:
            best_loss, best_model_wts, no_improve = avg_loss, copy.deepcopy(model.state_dict()), 0
        else: no_improve += 1
        if no_improve >= patience: break
            
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