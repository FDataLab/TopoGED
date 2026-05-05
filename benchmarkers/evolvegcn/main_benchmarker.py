import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import pickle
import argparse
import os
import sys
import math
from sklearn.metrics import roc_auc_score, f1_score
import networkx as nx

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from benchmarkers.benchmarker_utils.dataset_setup import load_data

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

def get_gpu_memory():
    return torch.cuda.memory_reserved(0) / 1024**2 if torch.cuda.is_available() else 0

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

# --- 1. MOCK UTILS ---
class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def pad_with_last_val(vect, k):
    device = vect.device
    pad_size = k - vect.size(0)
    if pad_size > 0:
        last_val = vect[-1]
        padding = torch.tensor([last_val] * pad_size, device=device)
        return torch.cat([vect, padding])
    return vect

def reset_param(t):
    stdv = 1. / math.sqrt(t.size(1))
    t.data.uniform_(-stdv, stdv)

import types
u = types.ModuleType("utils")
u.Namespace = Namespace
u.pad_with_last_val = pad_with_last_val
u.reset_param = reset_param
sys.modules['utils'] = u

import egcn_h
import egcn_o
import models

# --- 3. MAIN RUNNER ---
class EvolveGCNRunner:
    def __init__(self, args, data_dict):
        self.args = args
        self.is_directed = not args.undirected
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.A_list = [a.to(self.device) for a in data_dict['A_list']]
        self.Nodes_list = [n.to(self.device) for n in data_dict['Nodes_list']]
        self.node_count = data_dict['node_count']
        self.feature_dim = data_dict['feature_dim']
        
        n = len(self.A_list)
        # Shifted indices for link prediction (T predicts T+1)
        self.train_idx = range(1, int(n * 0.7))
        self.val_idx = range(int(n * 0.7), int(n * 0.85))
        self.test_idx = range(int(n * 0.85), n)
        
        self.build_model()
        
    def build_model(self):
        gcn_args = u.Namespace({
            'feats_per_node': self.feature_dim,
            'layer_1_feats': self.args.layer_1_feats,
            'layer_2_feats': self.args.layer_2_feats, 
            'in_feats': self.feature_dim, 
            'k_top_grcu': min(self.node_count, 200),
            'k_top_gcn': min(self.node_count, 200) 
        })
        
        if self.args.model == 'egcn_h':
            self.encoder = egcn_h.EGCN(gcn_args, activation=nn.RReLU(), device=self.device)
        else:
            self.encoder = egcn_o.EGCN(gcn_args, activation=nn.RReLU(), device=self.device)

        cls_args = u.Namespace({
            'gcn_parameters': {'layer_2_feats': self.args.layer_2_feats, 'cls_feats': self.args.cls_feats},
            'experiment_type': 'standard'
        })
        self.classifier = models.Classifier(cls_args, out_features=1, in_features=(self.args.layer_2_feats * 2)).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()), 
            lr=self.args.lr,
            weight_decay=self.args.l2_reg
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

    def get_window_and_mask(self, t, window_size=5):
        # Predict snapshot T using data from snapshots [T-window_size : T-1]
        start = max(0, t - window_size)
        end = t # slice is exclusive, so this takes up to t-1
        A_window = self.A_list[start:end]
        Nodes_window = self.Nodes_list[start:end]
        
        if len(A_window) == 0:
            A_window, Nodes_window = [self.A_list[0]], [self.Nodes_list[0]]
            
        mask_list = [torch.zeros(self.node_count, 1).to(self.device) for _ in range(len(A_window))]
        return A_window, Nodes_window, mask_list

    @torch.no_grad()
    # Provided by Gemini, helps with CUDA OOM
    def get_adj_scores_chunked(self, emb, chunk_size=512):
        """Computes probabilities in chunks to avoid N x N expansion OOM."""
        n = emb.size(0)
        all_probs = []
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            # (chunk_size, 1, dim)
            u_chunk = emb[i:end_i].unsqueeze(1).expand(-1, n, -1)
            # (1, n, dim) -> (chunk_size, n, dim)
            v_all = emb.unsqueeze(0).expand(end_i - i, -1, -1)
            
            # Predict
            logits = self.classifier(torch.cat([u_chunk, v_all], dim=2)).squeeze(-1)
            all_probs.append(torch.sigmoid(logits).cpu()) # Move to RAM immediately
            
        probs = torch.cat(all_probs, dim=0)
        if not self.is_directed:
            probs = (probs + probs.T) / 2.0
        return probs

    @torch.no_grad()
    def evaluate_link_prediction(self, indices):
        """Helper to calculate AUC on a specific set of snapshot indices."""
        self.encoder.eval()
        self.classifier.eval()
        all_scores, all_targets = [], []

        for t in indices:
            A_win, N_win, Masks = self.get_window_and_mask(t)
            emb = self.encoder(A_win, N_win, Masks)
            
            adj_t = self.A_list[t].coalesce() if self.A_list[t].is_sparse else self.A_list[t]
            pos_edges = adj_t.indices().t() if adj_t.is_sparse else adj_t.nonzero()
            
            if pos_edges.size(0) == 0: continue
            
            # Use same sampling logic as training for consistency
            num_pos = min(pos_edges.size(0), 10000) 
            neg_edges = torch.randint(0, self.node_count, (num_pos, 2)).to(self.device)
            
            pos_p = torch.sigmoid(self.classifier(torch.cat([emb[pos_edges[:num_pos,0]], emb[pos_edges[:num_pos,1]]], dim=1)))
            neg_p = torch.sigmoid(self.classifier(torch.cat([emb[neg_edges[:,0]], emb[neg_edges[:,1]]], dim=1)))
            
            all_scores.append(torch.cat([pos_p, neg_p]).cpu().numpy().flatten())
            all_targets.append(np.concatenate([np.ones(pos_p.size(0)), np.zeros(neg_p.size(0))]))

        return roc_auc_score(np.concatenate(all_targets), np.concatenate(all_scores))

    def train(self):
        print(f"--- Training {self.args.model} | Task: Future Link Prediction ---")
        start_train = time.time()
        
        # Track metrics for the "Best" state
        best_val_auc = 0.0
        best_train_auc_at_val = 0.0
        patience = 15
        no_improve = 0
        self.best_encoder, self.best_classifier = None, None

        for epoch in range(1, 201):
            # 1. Training Phase
            self.encoder.train()
            self.classifier.train()
            epoch_loss = 0
            
            for t in self.train_idx:
                self.optimizer.zero_grad()
                A_win, N_win, Masks = self.get_window_and_mask(t)
                emb = self.encoder(A_win, N_win, Masks)
                
                # Fetch positive edges
                adj_t = self.A_list[t].coalesce() if self.A_list[t].is_sparse else self.A_list[t]
                pos_edges = adj_t.indices().t() if adj_t.is_sparse else adj_t.nonzero()
                
                if pos_edges.size(0) == 0: continue
                
                # Subsample if necessary for memory, then sample 1:1 negatives
                num_pos = min(pos_edges.size(0), 20000)
                perm = torch.randperm(pos_edges.size(0))[:num_pos]
                pos_edges = pos_edges[perm]
                neg_edges = torch.randint(0, self.node_count, (num_pos, 2)).to(self.device)
                
                # Compute Loss
                pos_scores = self.classifier(torch.cat([emb[pos_edges[:,0]], emb[pos_edges[:,1]]], dim=1))
                neg_scores = self.classifier(torch.cat([emb[neg_edges[:,0]], emb[neg_edges[:,1]]], dim=1))
                
                loss = self.bce_loss(pos_scores, torch.ones_like(pos_scores)) + \
                       self.bce_loss(neg_scores, torch.zeros_like(neg_scores))
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # 2. AUC Evaluation Phase (Both Train and Val)
            train_auc = self.evaluate_link_prediction(self.train_idx)
            val_auc = self.evaluate_link_prediction(self.val_idx)
            avg_epoch_loss = epoch_loss / len(self.train_idx)

            # 3. Update Best State based on Validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_train_auc_at_val = train_auc # Capture the corresponding train metric
                self.best_encoder = copy.deepcopy(self.encoder)
                self.best_classifier = copy.deepcopy(self.classifier)
                no_improve = 0
                status = "*" 
            else:
                no_improve += 1
                status = ""

            print(f"Epoch {epoch:03d} | Loss: {avg_epoch_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} {status}")
            
            if no_improve >= patience:
                print(f"Early stopping triggered. Best Val AUC: {best_val_auc:.4f}")
                break

        # --- FINAL SUMMARY PRINT ---
        print(f"\n[FINAL TRAINING SUMMARY - {self.args.model}]")
        print(f"Best Validation AUC: {best_val_auc:.4f}")
        print(f"Corresponding Train AUC: {best_train_auc_at_val:.4f}")
        print(f"---------------------------\n")

        # Restore best weights for Phase 2 & 3
        if self.best_encoder is not None:
            self.encoder, self.classifier = self.best_encoder, self.best_classifier
        
        t1, g1, r1 = time.time() - start_train, get_gpu_memory(), get_ram_usage()

        # === PHASE 2 & 3 ===
        start_opt = time.time()
        opt_thresh = self.optimize_threshold()
        t2, g2, r2 = time.time() - start_opt, get_gpu_memory(), get_ram_usage()

        start_cons = time.time()
        self.construct_graphs(opt_thresh, self.args.dataset)
        t3, g3, r3 = time.time() - start_cons, get_gpu_memory(), get_ram_usage()

        print(f"\n--- DATASET: {self.args.dataset} (EvolveGCN) METRICS ---")
        print(f"TRAIN:  Time={t1:.2f}s, GPU={g1:.2f}MB, RAM={r1:.2f}MB")
        print(f"THRESH: Time={t2:.2f}s, GPU={g2:.2f}MB, RAM={r2:.2f}MB")
        print(f"CONST:  Time={t3:.2f}s, GPU={g3:.2f}MB, RAM={r3:.2f}MB")
        gc.collect(); torch.cuda.empty_cache()

    @torch.no_grad()
    def optimize_threshold(self):
        """Sample-based threshold optimization to replace N x N dense validation."""
        self.encoder.eval(); self.classifier.eval()
        all_probs, all_targets = [], []
        
        for t in self.val_idx:
            A_win, N_win, Masks = self.get_window_and_mask(t)
            emb = self.encoder(A_win, N_win, Masks)
            
            # Use positive edges from the actual snapshot
            adj_t = self.A_list[t]
            if adj_t.is_sparse:
                # Crucial: coalesce first to merge duplicate entries and enable .indices()
                coalesced_t = adj_t.coalesce()
                # Also filter for values > 0 in case your sparse tensor includes zeros
                indices = coalesced_t.indices()
                values = coalesced_t.values()
                mask = values > 0
                pos_edges = indices[:, mask].t()
            else:
                pos_edges = adj_t.nonzero()
            
            # Sample 1:1 negative edges
            neg_edges = torch.randint(0, self.node_count, (pos_edges.size(0), 2)).to(self.device)
            
            # Get scores for samples only
            pos_scores = torch.sigmoid(self.classifier(torch.cat([emb[pos_edges[:,0]], emb[pos_edges[:,1]]], dim=1)))
            neg_scores = torch.sigmoid(self.classifier(torch.cat([emb[neg_edges[:,0]], emb[neg_edges[:,1]]], dim=1)))
            
            all_probs.append(torch.cat([pos_scores, neg_scores]).cpu().numpy().flatten())
            all_targets.append(np.concatenate([np.ones(pos_scores.size(0)), np.zeros(neg_scores.size(0))]))
        
        y_scores = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)
        
        best_f1, best_tau = 0, 0.5
        thresholds = np.linspace(0.05, 0.99, 95)
        for tau in thresholds:
            preds = (y_scores > tau).astype(int)
            # Manual F1 to avoid sklearn overhead on large arrays
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            if f1 > best_f1: best_f1, best_tau = f1, tau

        pos_scores = y_scores[y_true == 1]
        neg_scores = y_scores[y_true == 0]

        print(f"--- Probability Distribution Check ---")
        print(f"Positives | Mean: {pos_scores.mean():.6f} | Std: {pos_scores.std():.6f} | Max: {pos_scores.max():.6f} | Min: {pos_scores.min():.6f}")
        print(f"Negatives | Mean: {neg_scores.mean():.6f} | Std: {neg_scores.std():.6f} | Max: {neg_scores.max():.6f} | Min: {neg_scores.min():.6f}")
        
        print(f"Optimal Threshold: {best_tau:.2f} | Sampled Val F1: {best_f1:.4f}")

        return best_tau

    def construct_graphs(self, threshold, dataset):
        """
        Memory-safe EvolveGCN graph construction with a 5x Dynamic Cap
        and Dual Metric Printing (Capped vs Uncapped).
        """
        import scipy.sparse as sp
        import gc
        import pickle
        import os
        import numpy as np
        import torch
        import networkx as nx

        self.encoder.eval()
        self.classifier.eval()
        
        # Used to get number of edges to cap the datasets with X edges (for safety)    
        from GraphGeneration.scripts.load_data import load_data
        _, _, _, target_graphs = load_data(
            dataset, '', '', '', 'all', 
            use_predicted=False, num_buckets=10, use_test_style=None
        )
        
        target_graphs_flat = [bucket[-1] for bucket in target_graphs]
        num_edges_in_targets = [g.number_of_edges() for g in target_graphs_flat]
        
        predicted_networks = []

        with torch.no_grad():
            for t_idx in self.test_idx:
                # 1. Generate embeddings and probabilities
                adjacency_window, node_window, masks = self.get_window_and_mask(t_idx)
                embeddings = self.encoder(adjacency_window, node_window, masks)
                probabilities = self.get_adj_scores_chunked(embeddings)
                
                if isinstance(probabilities, np.ndarray):
                    probabilities = torch.from_numpy(probabilities)
                
                n = probabilities.shape[0]
                probabilities.view(-1)[::n+1] = 0
                
                # --- PREPARE GROUND TRUTH (Keep it sparse!) ---
                true_graph = target_graphs_flat[t_idx].copy()
                true_graph.add_nodes_from(range(self.node_count))
                
                true_adj_sp = nx.to_scipy_sparse_array(true_graph, nodelist=range(self.node_count), format='csr')
                num_true_edges = true_adj_sp.nnz

                # 2. RAW THRESHOLD PASS
                mask_raw = probabilities >= threshold
                num_raw = mask_raw.sum().item()
                indices_raw = torch.where(mask_raw)
                
                # Capture the raw/uncapped arrays BEFORE capping
                raw_rows = indices_raw[0].cpu().numpy()
                raw_cols = indices_raw[1].cpu().numpy()
                
                # 3. APPLY CAP (T-1 logic)
                max_num_edges = max(num_edges_in_targets[t_idx - 1] * 5, 1000)
                
                if num_raw > max_num_edges:
                    scores = probabilities[mask_raw]
                    _, top_k_idx = torch.topk(scores, max_num_edges)
                    final_rows = indices_raw[0][top_k_idx].cpu().numpy()
                    final_cols = indices_raw[1][top_k_idx].cpu().numpy()
                    status = "CAPPED"
                else:
                    final_rows = raw_rows
                    final_cols = raw_cols
                    status = "ACCEPTED"

                # --- HELPER FUNCTION FOR METRICS ---
                def get_metrics(pred_rows, pred_cols, N):
                    if len(pred_rows) > 0:
                        matched = np.array(true_adj_sp[pred_rows, pred_cols]).flatten()
                        tp = np.sum(matched > 0)
                        fp = len(pred_rows) - tp
                        fn = num_true_edges - tp
                        tn = (N * (N - 1)) - (tp + fp + fn)
                        return tp, fp, tn, fn
                    else:
                        return 0, 0, (N * (N - 1)) - num_true_edges, num_true_edges

                # Calculate both sets of metrics
                tp_raw, fp_raw, tn_raw, fn_raw = get_metrics(raw_rows, raw_cols, self.node_count)
                tp_cap, fp_cap, tn_cap, fn_cap = get_metrics(final_rows, final_cols, self.node_count)

                # 4. BUILD FINAL SPARSE MATRIX (Only save the capped version)
                adj_final = sp.csr_matrix(
                    (np.ones(len(final_rows), dtype=np.int8), (final_rows, final_cols)), shape=(n, n)
                )

                # 5. PRINT DUAL METRICS
                print(f"\nSnapshot {t_idx} | {status} | True Edges: {num_true_edges} | Cap Limit: {max_num_edges}")
                print(f"  [UNCAPPED] Pred: {len(raw_rows):<7} | TP: {tp_raw:<5} | FP: {fp_raw:<7} | TN: {tn_raw:<7} | FN: {fn_raw:<5}")
                print(f"  [CAPPED]   Pred: {adj_final.nnz:<7} | TP: {tp_cap:<5} | FP: {fp_cap:<7} | TN: {tn_cap:<7} | FN: {fn_cap:<5}")
                
                # Cleanup huge tensors
                del probabilities, indices_raw, mask_raw, raw_rows, raw_cols, final_rows, final_cols
                
                if not self.is_directed:
                    adj_final = adj_final + adj_final.T
                    adj_final.data[:] = 1
                
                predicted_networks.append(adj_final)
                gc.collect()

        # 6. Prepare naming and save
        file_params = (
            f"{self.args.dataset}_{self.args.model}_"
            f"{self.args.layer_1_feats}_{self.args.layer_2_feats}_{self.args.cls_feats}_"
            f"{self.args.lr}_{self.args.l2_reg}_"
            f"{'directed' if self.is_directed else 'undirected'}"
        )
        
        save_path = f"data/output/predicted/EvolveGCN/{file_params}_threshold_5xCap.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({'graphs': predicted_networks, 'node_count': self.node_count}, f)
        
        print(f"Saved memory-safe EvolveGCN sparse graphs to {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="egcn_h")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--layer_1_feats", type=int, default=64)
    parser.add_argument("--layer_2_feats", type=int, default=32)
    parser.add_argument("--cls_feats", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    args = parser.parse_args()
    data_dict = load_data('evolvegcn', args.dataset)
    EvolveGCNRunner(args, data_dict).train()
