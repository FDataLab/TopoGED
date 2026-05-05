import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import copy
import networkx as nx
from torch_geometric.utils import negative_sampling, to_dense_adj
from sklearn.metrics import roc_auc_score, f1_score
import pickle
import sys
import time, psutil, gc # FIX: Ensure time is imported

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from benchmarkers.benchmarker_utils.dataset_setup import load_data
from benchmarkers.roland.run.roland_model import ROLAND

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

def get_gpu_memory(device):
    return torch.cuda.memory_reserved(device) / 1024**2 if torch.cuda.is_available() else 0

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

class RolandRunner:
    def __init__(self, args, data_dict, dataset_name, device, is_directed=True):
        self.args = args
        self.device = device
        self.dataset_name = dataset_name
        self.is_directed = is_directed
        self.snapshots = data_dict['snapshots']
        self.node_count = data_dict['node_count']
        self.feat_dim = data_dict['feature_dim']
        
        # Model architecture requires at least 1 edge dim
        self.edge_dim = data_dict.get('edge_dim', 0)
        model_edge_dim = self.edge_dim if self.edge_dim > 0 else 1
        
        n = len(self.snapshots)
        self.train_end = int(n * 0.7)
        self.val_end = int(n * 0.85)
        
        actual_feat_dim = self.node_count if self.feat_dim is None else self.feat_dim

        self.model = ROLAND(self.node_count, actual_feat_dim, args.hidden_dim, 
                    edge_dim=model_edge_dim, num_layers=args.num_layers).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_model = None

    def get_snapshot(self, t):
        data = self.snapshots[t]
        edge_index = data.edge_index.to(self.device)
        if data.x is None:
            x = torch.eye(self.node_count, device=self.device)
        elif data.x.is_sparse:
            x = data.x.to_dense().to(self.device)
        else:
            x = data.x.to(self.device)
        
        if data.edge_attr is not None:
            edge_attr = data.edge_attr.to(self.device)
            if edge_attr.dim() == 1: edge_attr = edge_attr.unsqueeze(-1)
        else:
            edge_attr = torch.ones((edge_index.size(1), 1), device=self.device)
        return x, edge_index, edge_attr

    @torch.no_grad()
    def get_probs_chunked(self, z, chunk_size=500):
        """Row-wise chunking to prevent OOM on large datasets like networkadex"""
        all_probs = []
        for i in range(0, self.node_count, chunk_size):
            end_idx = min(i + chunk_size, self.node_count)
            z_src_chunk = z[i:end_idx].unsqueeze(1).expand(-1, self.node_count, -1)
            z_dst_all = z.unsqueeze(0).expand(end_idx - i, -1, -1)
            
            logits = self.model.pred_head(torch.cat([z_src_chunk, z_dst_all], dim=2)).squeeze(-1)
            all_probs.append(torch.sigmoid(logits).cpu()) # Move to RAM
        return torch.cat(all_probs, dim=0)

    @torch.no_grad()
    def optimize_threshold(self):
        self.model.eval()
        # Warm up hidden states through train history
        states = self.model.init_states(self.node_count, self.device)
        for t in range(self.train_end - 1):
            x, edge_index, edge_attr = self.get_snapshot(t)
            states = self.model(x, edge_index, edge_attr, states)
        
        all_y_scores, all_y_true = [], []
        # Predict validation snapshots using sampled negatives
        for t in range(self.train_end - 1, self.val_end - 1):
            x, edge_index, edge_attr = self.get_snapshot(t)
            states = self.model(x, edge_index, edge_attr, states)
            z = states[-1]
            
            target_snap = self.snapshots[t+1]
            pos_edges = target_snap.edge_index.to(self.device)
            neg_edges = torch.randint(0, self.node_count, (2, pos_edges.size(1)), device=self.device)
            
            # Predict only for sampled pairs
            pos_logits = self.model.predict_links(z, pos_edges)
            neg_logits = self.model.predict_links(z, neg_edges)
            
            all_y_scores.append(torch.cat([pos_logits.sigmoid(), neg_logits.sigmoid()]).cpu())
            all_y_true.append(torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]).cpu())

        y_scores = torch.cat(all_y_scores)
        y_true = torch.cat(all_y_true)
        scores_np = y_scores.detach().numpy()
        true_np = y_true.detach().numpy()
        
        pos_scores = scores_np[true_np == 1]
        neg_scores = scores_np[true_np == 0]

        print(f"--- Probability Distribution Check ---")
        print(f"Positives | Mean: {pos_scores.mean():.6f} | Std: {pos_scores.std():.6f} | Max: {pos_scores.max():.6f} | Min: {pos_scores.min():.6f}")
        print(f"Negatives | Mean: {neg_scores.mean():.6f} | Std: {neg_scores.std():.6f} | Max: {neg_scores.max():.6f} | Min: {neg_scores.min():.6f}")
        # -------------------------------------------------

        thresholds = np.unique(np.percentile(y_scores, np.linspace(0, 100, 100)))
        best_f1, best_tau = 0, 0.5
        for tau in thresholds:
            preds = (y_scores > tau).float()
            tp, fp, fn = (preds * y_true).sum(), (preds * (1 - y_true)).sum(), ((1 - preds) * y_true).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            if f1 > best_f1: best_f1, best_tau = f1.item(), tau.item()
        
        print(f"--- Probability Distribution Check ---")
        print(f"Positives | Mean: {pos_scores.mean():.6f} | Std: {pos_scores.std():.6f} | Max: {pos_scores.max():.6f} | Min: {pos_scores.min():.6f}")
        print(f"Negatives | Mean: {neg_scores.mean():.6f} | Std: {neg_scores.std():.6f} | Max: {neg_scores.max():.6f} | Min: {neg_scores.min():.6f}")
                
        print(f"Optimal Threshold: {best_tau:.2f} | Sampled Val F1: {best_f1:.4f}")
        return best_tau

    @torch.no_grad()
    def validate(self, snapshot_indices, initial_states):
        """Helper to compute AUC on a specific split while maintaining state history."""
        self.model.eval()
        all_preds, all_targets = [], []
        states = [s.clone() for s in initial_states] # Start from the end of the previous split
        
        for t in range(snapshot_indices[0], snapshot_indices[-1] + 1):
            x, edge_index, edge_attr = self.get_snapshot(t)
            states = self.model(x, edge_index, edge_attr, states)
            z = states[-1]
            
            # Target is t+1 (Future Link Prediction)
            if t + 1 < len(self.snapshots):
                next_data = self.snapshots[t+1]
                pos_idx = next_data.edge_index.to(self.device)
                neg_idx = negative_sampling(pos_idx, num_nodes=self.node_count)
                
                pos_score = self.model.predict_links(z, pos_idx).sigmoid()
                neg_score = self.model.predict_links(z, neg_idx).sigmoid()
                
                all_preds.append(torch.cat([pos_score, neg_score]).cpu())
                all_targets.append(torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).cpu())
            
            states = [s.detach() for s in states]
            
        y_true = torch.cat(all_targets).numpy()
        y_scores = torch.cat(all_preds).numpy()
        return roc_auc_score(y_true, y_scores)

    def run(self):
        import torch
        from benchmarkers.benchmarker_utils.k_values_extractor import get_topk
        start_train = time.time()
        
        # Track metrics for the "Best" state
        best_val_auc = 0.0
        best_train_auc_at_val = 0.0
        patience = 15
        no_improve = 0
        self.best_model = None 

        print(f"--- Training ROLAND ({self.args.dataset}) | Persistence: {patience} ---")
        for epoch in range(1, 201):
            self.model.train()
            epoch_loss = 0
            states = self.model.init_states(self.node_count, self.device)
            
            # 1. Training Phase
            for t in range(self.train_end - 1):
                x, edge_index, edge_attr = self.get_snapshot(t)
                next_data = self.snapshots[t+1]
                next_edge_index = next_data.edge_index.to(self.device)
                
                detached_states = [s.detach() for s in states]
                for _ in range(self.args.num_updates_per_snapshot): 
                    self.optimizer.zero_grad()
                    new_states = self.model(x, edge_index, edge_attr, detached_states)
                    z = new_states[-1]
                    
                    neg_edge_index = negative_sampling(next_edge_index, num_nodes=self.node_count)
                    pos_score = self.model.predict_links(z, next_edge_index)
                    neg_score = self.model.predict_links(z, neg_edge_index)
                    
                    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                    preds = torch.cat([pos_score, neg_score])
                    loss = self.criterion(preds, labels)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                states = [s.detach() for s in new_states]

            # 2. Evaluation Phase (Train and Val AUC)
            # Train AUC uses the states generated during the training pass
            train_auc = self.validate(range(self.train_end - 1), self.model.init_states(self.node_count, self.device))
            # Val AUC starts from the final training states
            val_auc = self.validate(range(self.train_end - 1, self.val_end - 1), states)
            
            avg_loss = epoch_loss / ((self.train_end - 1) * self.args.num_updates_per_snapshot)
            
            # 3. Update Best State based on Validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_train_auc_at_val = train_auc
                self.best_model = copy.deepcopy(self.model.state_dict())
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
        print(f"\n[FINAL TRAINING SUMMARY - ROLAND]")
        print(f"Best Validation AUC: {best_val_auc:.4f}")
        print(f"Corresponding Train AUC: {best_train_auc_at_val:.4f}")
        print(f"---------------------------\n")

        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)

        t1, g1, r1 = time.time() - start_train, get_gpu_memory(self.device), get_ram_usage()
        torch.cuda.reset_peak_memory_stats(self.device)

        start_opt = time.time()
        best_tau = self.optimize_threshold()
        t2, g2, r2 = time.time() - start_opt, get_gpu_memory(self.device), get_ram_usage()
        torch.cuda.reset_peak_memory_stats(self.device)
        
        
        start_cons = time.time()
        import scipy.sparse as sp
        import gc
        import pickle
        import os
        import torch
        import numpy as np
        import networkx as nx

        self.model.eval()
        # 1. Load Ground Truth for Metrics and Dynamic Capping
        from GraphGeneration.scripts.load_data import load_data
        _, _, _, target_graphs = load_data(
            self.args.dataset, '', '', '', 'all', 
            use_predicted=False, num_buckets=10, use_test_style=None
        )
        
        # Flatten buckets: target_graphs_flat[t] is the ground truth for snapshot t
        target_graphs_flat = [bucket[-1] for bucket in target_graphs]
        num_edges_in_targets = [g.number_of_edges() for g in target_graphs_flat]

        predicted_networks = []

        with torch.no_grad():
            # Warm up hidden states
            states = self.model.init_states(self.node_count, self.device)
            for t in range(self.val_end - 1):
                x, edge_index, edge_attr = self.get_snapshot(t)
                states = self.model(x, edge_index, edge_attr, states)
            
            print(f"--- Starting ROLAND Sparse Construction (5x Dynamic Cap) ---")
            # 2. Process test snapshots sequentially
            # t here is the snapshot index in the sequence
            for t in range(self.val_end - 1, len(self.snapshots) - 1):
                x, edge_index, edge_attr = self.get_snapshot(t)
                states = self.model(x, edge_index, edge_attr, states)
                
                # Retrieve scores
                probs = self.get_probs_chunked(states[-1]) 
                if isinstance(probs, np.ndarray):
                    probs = torch.from_numpy(probs).to(self.device)

                # Standardized Block Start
                # ---------------------------------------------------------
                # A. Mask diagonal
                n = probs.shape[0]
                probs.view(-1)[::n+1] = 0

                if not self.is_directed:
                    probs = (probs + probs.t()) / 2.0
                
                # B. Prepare Ground Truth (Sparse comparison)
                # We compare current prediction (t) against target at t
                true_graph = target_graphs_flat[t].copy()
                true_graph.add_nodes_from(range(self.node_count))
                
                true_adj_sp = nx.to_scipy_sparse_array(true_graph, nodelist=range(self.node_count), format='csr')
                num_true_edges = true_adj_sp.nnz

                # C. Raw Threshold Pass
                mask_raw = probs >= best_tau
                indices_raw = torch.where(mask_raw)
                
                # Capture raw arrays BEFORE capping
                raw_rows = indices_raw[0].cpu().numpy()
                raw_cols = indices_raw[1].cpu().numpy()
                num_raw = len(raw_rows)
                
                # D. Dynamic Capping (5x edges of T-1)
                max_num_edges = max(num_edges_in_targets[t - 1] * 5, 1000)
                
                if num_raw > max_num_edges:
                    scores_raw = probs[mask_raw]
                    _, top_k_idx = torch.topk(scores_raw, max_num_edges)
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

                # E. Build Final Sparse Matrix (Only save capped version)
                adj_final = sp.csr_matrix(
                    (np.ones(len(final_rows), dtype=np.int8), (final_rows, final_cols)),
                    shape=(n, n)
                )

                # F. Print Snapshot Summary
                print(f"\nSnap {t} | {status} | True Edges: {num_true_edges} | Cap Limit: {max_num_edges}")
                print(f"  [UNCAPPED] Pred: {len(raw_rows):<7} | TP: {tp_raw:<5} | FP: {fp_raw:<7} | TN: {tn_raw:<7} | FN: {fn_raw:<5}")
                print(f"  [CAPPED]   Pred: {adj_final.nnz:<7} | TP: {tp_cap:<5} | FP: {fp_cap:<7} | TN: {tn_cap:<7} | FN: {fn_cap:<5}")
                
                # Cleanup huge tensors
                del probs, mask_raw, indices_raw, raw_rows, raw_cols, final_rows, final_cols
                if 'scores_raw' in locals(): del scores_raw
                # ---------------------------------------------------------

                if not self.is_directed:
                    adj_final = adj_final + adj_final.T
                    adj_final.data[:] = 1

                predicted_networks.append(adj_final)
                gc.collect()

        # 3. Save Logic
        strategy_str = 'threshold_5xCap'
        file_params = (
            f"{self.dataset_name}_{self.args.hidden_dim}_{self.args.lr}_"
            f"{self.args.num_layers}_{self.args.num_updates_per_snapshot}_"
            f"{'directed' if self.is_directed else 'undirected'}"
        )
        
        save_path = f"data/output/predicted/ROLAND/{file_params}_{strategy_str}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({'graphs': predicted_networks, 'node_count': self.node_count}, f)
            
        print(f"Saved 5x-Capped ROLAND sparse graphs to {save_path}")

        t3, g3, r3 = time.time() - start_cons, get_gpu_memory(self.device), get_ram_usage()
        print(f"\n--- DATASET: {self.dataset_name} (ROLAND) METRICS ---")
        print(f"TRAIN:  Time={t1:.2f}s, GPU={g1:.2f}MB, RAM={r1:.2f}MB")
        print(f"THRESH: Time={t2:.2f}s, GPU={g2:.2f}MB, RAM={r2:.2f}MB")
        print(f"CONST:  Time={t3:.2f}s, GPU={g3:.2f}MB, RAM={r3:.2f}MB\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True); parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=128); parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_layers", type=int, default=2); parser.add_argument("--num_updates_per_snapshot", type=int, default=5)
    
    args = parser.parse_args(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dict = load_data('roland', args.dataset)
    RolandRunner(args, data_dict, args.dataset, device, is_directed=not args.undirected).run()
