import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import time
import pickle
import os
import sys
import networkx as nx
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops

# Ensure path is set for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from benchmarkers.benchmarker_utils.dataset_setup import load_data
from benchmarkers.sfdyg.models.hawkes import HGNNLP 

import time, psutil, gc

def get_gpu_memory(device_id=0):
    if torch.cuda.is_available():
        if isinstance(device_id, torch.device):
            idx = device_id.index if device_id.index is not None else 0
        else:
            idx = device_id
        return torch.cuda.max_memory_reserved(idx) / 1024**2
    return 0

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SFDyGPipeline:
    def __init__(self, args, data_dict, device, is_directed=True):
        self.args = args
        self.device = device
        self.is_directed = is_directed
        self.snapshots = data_dict['snapshots']
        self.node_count = data_dict['node_count']
        self.feature_dim = data_dict['feature_dim']
        self.window = args.window
        self.n_neg_train = args.n_neg_train
        self.clip_grad_norm = 2.0
        
        n = len(self.snapshots)
        self.train_idx = list(range(0, int(n * 0.7)))
        self.val_idx = list(range(int(n * 0.7), int(n * 0.85)))
        self.test_idx = list(range(int(n * 0.85), n))
        
        self.model = HGNNLP(
            n_node=self.node_count,          
            n_feat=self.feature_dim,      
            n_edge=1, 
            n_hidden=args.n_hidden, 
            dropout=args.dropout, 
            bias=args.bias, 
            name=args.model, 
            layers=args.n_layers, 
            heads=args.heads, 
            batch_norm=args.bn, 
            norm=args.norm_type
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.best_state = None

    @torch.no_grad()
    def fast_negative_sampling(self, edge_index, num_nodes, num_neg_samples=1):
        """Original SFDyG Logic: Ensuring negatives aren't true edges."""
        avoid_edge_index, _ = add_remaining_self_loops(edge_index)
        # Numerical encoding of edges to find overlaps
        scale = pow(10, len(str(num_nodes)))
        avoid = (avoid_edge_index[0] * scale + avoid_edge_index[1]).to(self.device)

        # Generate candidates (20% extra to account for overlaps)
        u = edge_index[0].repeat(int(num_neg_samples * 1.2))
        v = torch.randint(0, num_nodes, (len(u),), device=self.device)
        
        e = u * scale + v
        mask = torch.isin(e, avoid)
        
        # Filter and slice to exact size needed
        neg_u, neg_v = u[~mask], v[~mask]
        return torch.stack([neg_u[:edge_index.size(1)*num_neg_samples], 
                            neg_v[:edge_index.size(1)*num_neg_samples]])

    def get_fused_data(self, t):
        """Implements SFDyG Input Fusion."""
        start = max(0, t - self.window)
        snaps = self.snapshots[start:t]
        
        all_edges, all_times = [], []
        for i, snap in enumerate(snaps):
            edges = snap.edge_index.to(self.device)
            if edges.size(1) > 0:
                times = torch.full((edges.size(1), 1), float(i + 1) / self.window, 
                                   dtype=torch.float32, device=self.device)
                all_edges.append(edges)
                all_times.append(times)
                
        if len(all_edges) > 0:
            fused_edge_index = torch.cat(all_edges, dim=1)
            fused_edge_attr = torch.cat(all_times, dim=0)
        else:
            fused_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            fused_edge_attr = torch.empty((0, 1), dtype=torch.float32, device=self.device)
            
        raw_x = self.snapshots[0].x
        if raw_x is not None:
            x = raw_x.to_dense().to(self.device) if raw_x.is_sparse else raw_x.to(self.device)
        else:
            x = torch.ones((self.node_count, 1), dtype=torch.float32, device=self.device)
                
        return Data(x=x, edge_index=fused_edge_index, edge_attr=fused_edge_attr).to(self.device)

    @torch.no_grad()
    def get_probs_chunked(self, h, chunk_size=512):
        """Memory-safe chunked NxN probability generation for 32k nodes."""
        num_nodes = h.size(0)
        all_probs = []
        for i in range(0, num_nodes, chunk_size):
            end_i = min(i + chunk_size, num_nodes)
            h_chunk = h[i:end_i]
            logits_chunk = -torch.mm(h_chunk, h.t()) / 50.0
            all_probs.append(torch.sigmoid(logits_chunk).cpu())
        return torch.cat(all_probs, dim=0)

    @torch.no_grad()
    def optimize_threshold(self):
        print("--- Optimizing Threshold (Sampled Sparse) ---")
        self.model.eval()
        all_y_scores, all_y_true = [], []
        
        for t in self.val_idx:
            fused_data = self.get_fused_data(t)
            h = self.model.encoder(fused_data) if hasattr(self.model, 'encoder') else self.model(fused_data)
            if isinstance(h, tuple): h = h[0]
            
            target_snap = self.snapshots[t]
            pos_edges = target_snap.edge_index.to(self.device)
            neg_edges = self.fast_negative_sampling(pos_edges, self.node_count, 1)

            # Sparse scoring instead of NxN
            scale = 50.0 # Standard transformer/GNN scaling
            p_scores = torch.sigmoid((h[pos_edges[0]] * h[pos_edges[1]]).sum(-1) / scale)
            n_scores = torch.sigmoid((h[neg_edges[0]] * h[neg_edges[1]]).sum(-1) / scale)
            
            all_y_scores.append(torch.cat([p_scores, n_scores]).cpu())
            all_y_true.append(torch.cat([torch.ones_like(p_scores), torch.zeros_like(n_scores)]).cpu())

        y_scores = torch.cat(all_y_scores).numpy()
        y_true = torch.cat(all_y_true).numpy()

        pos_scores = y_scores[y_true == 1]
        neg_scores = y_scores[y_true == 0]

        print(f"--- Probability Distribution Check ---")
        print(f"Positives | Mean: {pos_scores.mean():.6f} | Std: {pos_scores.std():.6f} | Max: {pos_scores.max():.6f} | Min: {pos_scores.min():.6f}")
        print(f"Negatives | Mean: {neg_scores.mean():.6f} | Std: {neg_scores.std():.6f} | Max: {neg_scores.max():.6f} | Min: {neg_scores.min():.6f}")
                
        best_f1, best_threshold = 0, 0.5
        thresholds = np.linspace(0.05, 0.99, 95)
        print(thresholds)
        for t in thresholds:
            preds = (y_scores > t).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1: best_f1, best_threshold = score, t
        print(f"Optimal Threshold: {best_threshold:.2f} | Sampled Val F1: {best_f1:.4f}") 
        return best_threshold
    
    @torch.no_grad()
    def evaluate_link_prediction(self, indices):
        """Helper to calculate AUC on a specific set of snapshot indices."""
        self.model.eval()
        all_scores, all_targets = [], []

        for t in indices:
            if t == 0: continue # Skip first snapshot as there is no history
            fused_data = self.get_fused_data(t)
            h = self.model.encoder(fused_data) if hasattr(self.model, 'encoder') else self.model(fused_data)
            if isinstance(h, tuple): h = h[0]
            
            target_snap = self.snapshots[t]
            pos_edges = target_snap.edge_index.to(self.device)
            # Use 1:1 negative sampling for standard AUC evaluation
            neg_edges = self.fast_negative_sampling(pos_edges, self.node_count, 1)
            
            scale = 50.0
            p_s = torch.sigmoid((h[pos_edges[0]] * h[pos_edges[1]]).sum(-1) / scale)
            n_s = torch.sigmoid((h[neg_edges[0]] * h[neg_edges[1]]).sum(-1) / scale)
            
            all_scores.append(torch.cat([p_s, n_s]).cpu())
            all_targets.append(torch.cat([torch.ones_like(p_s), torch.zeros_like(n_s)]).cpu())

        y_true = torch.cat(all_targets).numpy()
        y_scores = torch.cat(all_scores).numpy()
        return roc_auc_score(y_true, y_scores)

    def train(self):
        print(f"--- Training SFDyG ({self.args.model}) | Validation Early Stopping ---")
        best_val_auc = 0.0
        best_train_auc_at_val = 0.0
        patience = 15
        no_improve = 0
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)

        for epoch in range(1, 1 + self.args.epochs):
            self.model.train()
            epoch_loss = 0
            
            # 1. Training Pass
            for t in self.train_idx:
                if t == 0: continue 
                fused_data = self.get_fused_data(t)
                pos_edges = self.snapshots[t].edge_index.to(self.device)
                neg_edges = self.fast_negative_sampling(pos_edges, self.node_count, self.n_neg_train)
                
                self.optimizer.zero_grad()
                loss = self.model.train_step(fused_data, pos_edges, neg_edges)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)                
                self.optimizer.step()
                epoch_loss += loss.item()

            lr_scheduler.step()
            avg_loss = epoch_loss / len(self.train_idx)

            # 2. Evaluation Phase
            # We evaluate both splits to monitor generalization
            train_auc = self.evaluate_link_prediction(self.train_idx)
            val_auc = self.evaluate_link_prediction(self.val_idx)

            # 3. Early Stopping Logic on Validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_train_auc_at_val = train_auc # Capture corresponding train AUC
                self.best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
                marker = "*"
            else:
                no_improve += 1
                marker = ""

            if epoch % self.args.eval_steps == 0:
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} {marker}")
            
            if no_improve >= patience:
                print(f"Early stopping triggered. Best Val AUC: {best_val_auc:.4f}")
                break
                    
        # --- FINAL SUMMARY PRINT ---
        print(f"\n[FINAL TRAINING SUMMARY - SFDyG]")
        print(f"Best Validation AUC: {best_val_auc:.4f}")
        print(f"Corresponding Train AUC: {best_train_auc_at_val:.4f}")
        print(f"---------------------------\n")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            
    def construct_graphs(self, threshold):
        """
        Memory-safe SFDyG graph construction with standardized 5x Dynamic Capping 
        and Dual Sparse Metric printing (Capped vs Uncapped).
        """
        import scipy.sparse as sp
        import gc
        import pickle
        import os
        import numpy as np
        import torch
        import networkx as nx

        self.model.eval()
        
        # 1. Load Ground Truth for Metrics and Dynamic Capping
        from GraphGeneration.scripts.load_data import load_data
        # Note: ensuring 'dataset' is used from self.args
        _, _, _, target_graphs = load_data(
            self.args.dataset, '', '', '', 'all', 
            use_predicted=False, num_buckets=10, use_test_style=None
        )
        
        # Flatten buckets and calculate edge counts for the cap
        target_graphs_flat = [bucket[-1] for bucket in target_graphs]
        num_edges_in_targets = [g.number_of_edges() for g in target_graphs_flat]

        predicted_networks = []

        print(f"--- Starting SFDyG Sparse Construction (5x Dynamic Cap) ---")
        with torch.no_grad():
            for t in self.test_idx:
                # 1. Generate embeddings
                fused_data = self.get_fused_data(t)
                h = self.model.encoder(fused_data) if hasattr(self.model, 'encoder') else self.model(fused_data)
                if isinstance(h, tuple): 
                    h = h[0]
                
                # 2. CHUNKED CANDIDATE COLLECTION
                all_rows, all_cols, all_scores = [], [], []
                chunk_size = 512
                
                for i in range(0, self.node_count, chunk_size):
                    end_i = min(i + chunk_size, self.node_count)
                    
                    # Logits with SFDyG specific temperature scaling (/ 50.0)
                    logits = torch.mm(h[i:end_i], h.t()) / 50.0
                    probs = torch.sigmoid(logits)
                    
                    # Zero out self-loops
                    diag_idx = torch.arange(i, end_i, device=self.device)
                    probs[torch.arange(end_i - i), diag_idx] = 0
                    
                    # Identify candidates passing threshold
                    mask = probs >= threshold
                    rows, cols = torch.where(mask)
                    scores = probs[mask]
                    
                    all_rows.append((rows + i).cpu())
                    all_cols.append(cols.cpu())
                    all_scores.append(scores.cpu())
                    del logits, probs, mask

                # Consolidate candidates
                full_rows_tensor = torch.cat(all_rows)
                full_cols_tensor = torch.cat(all_cols)
                full_scores_tensor = torch.cat(all_scores)
                num_threshold_passed = full_scores_tensor.numel()

                # 3. STANDARDIZED DYNAMIC CAPPING (5x edges of T-1)
                # t is the index of the snapshot we are currently predicting
                max_num_edges = max(num_edges_in_targets[t - 1] * 5, 1000)
                
                # Capture the raw/uncapped arrays BEFORE capping
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

                # 4. PREPARE GROUND TRUTH (Keep it sparse!)
                true_graph = target_graphs_flat[t].copy()
                true_graph.add_nodes_from(range(self.node_count))
                
                true_adj_sp = nx.to_scipy_sparse_array(true_graph, nodelist=range(self.node_count), format='csr')
                num_true_edges = true_adj_sp.nnz

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

                # 5. BUILD FINAL SPARSE MATRIX (We still only save the capped version)
                adj_final = sp.csr_matrix(
                    (np.ones(len(final_rows), dtype=np.int8), (final_rows, final_cols)),
                    shape=(self.node_count, self.node_count)
                )

                # 6. PRINT DUAL METRICS
                print(f"\nSnap {t} | {status} | True Edges: {num_true_edges} | Cap Limit: {max_num_edges}")
                print(f"  [UNCAPPED] Pred: {len(raw_rows):<7} | TP: {tp_raw:<5} | FP: {fp_raw:<7} | TN: {tn_raw:<7} | FN: {fn_raw:<5}")
                print(f"  [CAPPED]   Pred: {adj_final.nnz:<7} | TP: {tp_cap:<5} | FP: {fp_cap:<7} | TN: {tn_cap:<7} | FN: {fn_cap:<5}")

                if not self.is_directed:
                    adj_final = adj_final + adj_final.T
                    adj_final.data[:] = 1

                predicted_networks.append(adj_final)
                
                # Cleanup
                del h, all_rows, all_cols, all_scores, full_rows_tensor, full_cols_tensor, full_scores_tensor, raw_rows, raw_cols, final_rows, final_cols
                gc.collect()

        # 7. Save Logic
        strategy = 'threshold_5xCap'
        file_name = f"{self.args.dataset}_{self.args.model}_{self.args.window}_{self.args.heads}_{self.args.lr}_{self.args.n_hidden}_{self.args.n_neg_train}_{self.args.dropout}"
        save_path = f"data/output/predicted/SFDyG/{file_name}_{strategy}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({'graphs': predicted_networks, 'node_count': self.node_count}, f)
        
        print(f"Saved 5x-Capped SFDyG sparse graphs to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="hgat")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--n_neg_train", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--norm_type", type=str, default='snorm')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=5)
    
    args = parser.parse_args()
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dict = load_data('sfdyg', args.dataset)
    
    runner = SFDyGPipeline(args, data_dict, device, is_directed=not args.undirected)
    
    start_train = time.time()
    runner.train()
    t1, g1, r1 = time.time() - start_train, get_gpu_memory(device), get_ram_usage()

    start_opt = time.time()
    opt_tau = runner.optimize_threshold()
    t2, g2, r2 = time.time() - start_opt, get_gpu_memory(device), get_ram_usage()

    start_cons = time.time()
    runner.construct_graphs(opt_tau)
    t3, g3, r3 = time.time() - start_cons, get_gpu_memory(device), get_ram_usage()

    print(f"\n--- DATASET: {args.dataset} (SFDyG) METRICS ---")
    print(f"TRAIN:  Time={t1:.2f}s, GPU={g1:.2f}MB, RAM={r1:.2f}MB")
    print(f"THRESH: Time={t2:.2f}s, GPU={g2:.2f}MB, RAM={r2:.2f}MB")
    print(f"CONST:  Time={t3:.2f}s, GPU={g3:.2f}MB, RAM={r3:.2f}MB\n")
