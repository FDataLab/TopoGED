import os
import sys
import time
import torch
import numpy as np
import pickle
import copy
import networkx as nx
from math import isnan
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
import scipy.sparse as sp
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from benchmarkers.benchmarker_utils.dataset_setup import load_data

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

def get_gpu_memory(device):
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved(device) / 1024**2
    return 0

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

class Runner(object):
    def __init__(self, data_dict, is_directed=True):
        self.is_directed = is_directed
        self.snapshots = data_dict['snapshots'] 
        self.len = len(self.snapshots)
        args.num_nodes = data_dict['node_count']
        
        n = self.len
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        self.train_shots = list(range(0, train_end))
        self.val_shots = list(range(train_end, val_end))
        self.test_shots = list(range(val_end, n))
        
        self.load_feature()
        self.model = load_model(args).to(args.device)
        
        # Ensure manifold and curvature parameters are on the correct device
        for m in self.model.modules():
            for attr_name in ['c', 'c_in', 'c_out', 'curvature']:
                if hasattr(m, attr_name):
                    val = getattr(m, attr_name)
                    if isinstance(val, torch.Tensor):
                        setattr(m, attr_name, val.to(args.device))
                    elif isinstance(val, (float, int)):
                        new_tensor = torch.tensor([float(val)], device=args.device)
                        setattr(m, attr_name, new_tensor)
        
        if hasattr(self.model, 'manifold') and hasattr(self.model.manifold, 'c'):
             self.model.manifold.c = self.model.manifold.c.to(args.device)
        
        # HTGN uses ReconLoss which traditionally expects dense targets. 
        # We will wrap it or call its internal manifold logic sparsely.
        self.loss = ReconLoss(args) 
        self.best_model_state = None
        logger.info(f'Split: Train({len(self.train_shots)}), Val({len(self.val_shots)}), Test({len(self.test_shots)})')

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
        else:
            raw_x = self.snapshots[0].x
            # Keep as sparse to save memory; model usually projects it to dense inside forward()
            self.x = raw_x.to(args.device) 
            args.nfeat = self.x.size(1)

    @torch.no_grad()
    def get_probs_chunked(self, z, threshold=None, chunk_size=300):
        """Hyperbolic probability estimation using chunked sqdist."""
        num_nodes = z.size(0)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int8)
        
        for i in range(0, num_nodes, chunk_size):
            end_idx = min(i + chunk_size, num_nodes)
            z_chunk = z[i:end_idx]
            
            # Compute hyperbolic distance: d_h(u, v)
            # Broadcasting: (chunk, 1, dim) vs (1, N, dim)
            dist = self.loss.manifold.sqdist(z_chunk.unsqueeze(1), z.unsqueeze(0), c=args.curvature)
            
            # Link probability in Hyperbolic space is inverse to distance
            probs_chunk = torch.sigmoid(-dist / 10.0)
            
            if threshold is not None:
                mask = (probs_chunk >= threshold).cpu().numpy()
                adj_matrix[i:end_idx] = mask.astype(np.int8)
            
        return adj_matrix

    @torch.no_grad()
    def optimize_threshold(self):
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()
        self.model.init_hiddens()
        
        # Warm up hidden state
        for t in range(self.val_shots[0]):
            snap = self.snapshots[t].to(args.device)
            z = self.model(snap.edge_index, self.x)
            self.model.update_hiddens_all_with(z)
        
        all_y_scores, all_y_true = [], []
        for t_idx in self.val_shots:
            if t_idx >= self.len: break
            
            # Predict T using state from T-1
            current_snap = self.snapshots[t_idx-1].to(args.device)
            z = self.model(current_snap.edge_index, self.x)
            
            # Sparse Target Evaluation
            target_snap = self.snapshots[t_idx].to(args.device)
            pos_edges = target_snap.edge_index
            neg_edges = torch.randint(0, args.num_nodes, (2, pos_edges.size(1)), device=args.device)
            
            # Hyperbolic dot-like similarity via sqdist
            p_dist = self.loss.manifold.sqdist(z[pos_edges[0]], z[pos_edges[1]], c=args.curvature)
            n_dist = self.loss.manifold.sqdist(z[neg_edges[0]], z[neg_edges[1]], c=args.curvature)
            
            all_y_scores.append(torch.cat([torch.sigmoid(-p_dist / 10.0), torch.sigmoid(-n_dist / 10.0)]).cpu())
            all_y_true.append(torch.cat([torch.ones(p_dist.size(0)), torch.zeros(n_dist.size(0))]))
            
            self.model.update_hiddens_all_with(z)

        y_scores = torch.cat(all_y_scores).numpy()
        y_true = torch.cat(all_y_true).numpy()
        
        thresholds = np.linspace(0.05, 0.99, 95)
        best_f1, best_tau = 0, 0.01
        for t in thresholds:
            preds = (y_scores > t).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1, best_tau = score, t
        pos_scores = y_scores[y_true == 1]
        neg_scores = y_scores[y_true == 0]
        print(f"--- HTGN Probability Distribution Check ---")
        print(f"Positives | Mean: {pos_scores.mean():.6f} | Std: {pos_scores.std():.6f} | Max: {pos_scores.max():.6f} | Min: {pos_scores.min():.6f}")
        print(f"Negatives | Mean: {neg_scores.mean():.6f} | Std: {neg_scores.std():.6f} | Max: {neg_scores.max():.6f} | Min: {neg_scores.min():.6f}")
        print(f"Optimal Threshold: {best_tau:.2f} | Sampled Val F1: {best_f1:.4f}")
        return best_tau

        
    def construct_graphs(self, threshold, file_path, dataset_name):
        """
        Memory-safe HTGN construction with standardized 5x Dynamic Capping 
        and Sparse Metric printing (TP, FP, TN, FN).
        """

        # 1. Load Ground Truth for Metrics and Dynamic Capping
        from GraphGeneration.scripts.load_data import load_data
        _, _, _, target_graphs = load_data(
            dataset_name, '', '', '', 'all', 
            use_predicted=False, num_buckets=10, use_test_style=None
        )
        
        # Flatten buckets and calculate edge counts for the cap
        target_graphs_flat = [bucket[-1] for bucket in target_graphs]
        num_edges_in_targets = [g.number_of_edges() for g in target_graphs_flat]

        TEMP = 10.0 
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()
        
        # Warm up hidden states
        with torch.no_grad():
            self.model.init_hiddens()
            warmup_shots = self.train_shots + self.val_shots
            for t in warmup_shots:
                snap = self.snapshots[t].to(args.device)
                z = self.model(snap.edge_index, self.x)
                self.model.update_hiddens_all_with(z)
        
        predicted_networks = []

        # 2. Process test snapshots with Dynamic Capping
        with torch.no_grad():
            for t in self.test_shots:
                # HTGN predicts snapshot T using state at T-1
                snap = self.snapshots[t-1].to(args.device)
                z = self.model(snap.edge_index, self.x)
                self.model.update_hiddens_all_with(z)

                all_rows, all_cols, all_scores = [], [], []
                chunk_size = 300 
                
                for i in range(0, args.num_nodes, chunk_size):
                    end_i = min(i + chunk_size, args.num_nodes)
                    
                    # Hyperbolic distance calculation
                    dist = self.loss.manifold.sqdist(
                        z[i:end_i].unsqueeze(1), 
                        z.unsqueeze(0), 
                        c=args.curvature
                    )
                    probs = torch.sigmoid(-dist / TEMP)
                    
                    # Zero diagonal (Self-loops)
                    diag_idx = torch.arange(i, end_i, device=args.device)
                    probs[torch.arange(end_i - i), diag_idx] = 0
                    
                    mask = probs >= threshold
                    rows, cols = torch.where(mask)
                    scores = probs[mask]
                    
                    all_rows.append((rows + i).cpu())
                    all_cols.append(cols.cpu())
                    all_scores.append(scores.cpu())
                    del dist, probs, mask

                # Consolidate candidates
                full_rows_tensor = torch.cat(all_rows)
                full_cols_tensor = torch.cat(all_cols)
                full_scores_tensor = torch.cat(all_scores)
                num_threshold_passed = full_scores_tensor.numel()

                # 3. STANDARDIZED DYNAMIC CAPPING (5x edges of T-1)
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

                # 4. PREPARE GROUND TRUTH
                true_graph = target_graphs_flat[t].copy()
                true_graph.add_nodes_from(range(args.num_nodes))
                
                true_adj_sp = nx.to_scipy_sparse_array(true_graph, nodelist=range(args.num_nodes), format='csr')
                num_true_edges = true_adj_sp.nnz

                # --- NEW: HELPER FUNCTION FOR METRICS (FIXED) ---
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
                tp_raw, fp_raw, tn_raw, fn_raw = get_metrics(raw_rows, raw_cols, args.num_nodes)
                tp_cap, fp_cap, tn_cap, fn_cap = get_metrics(final_rows, final_cols, args.num_nodes)

                # 5. BUILD FINAL SPARSE MATRIX
                adj_final = sp.csr_matrix(
                    (np.ones(len(final_rows), dtype=np.int8), (final_rows, final_cols)),
                    shape=(args.num_nodes, args.num_nodes)
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
                del z, all_rows, all_cols, all_scores, full_rows_tensor, full_cols_tensor, full_scores_tensor
                gc.collect()

        # 6. Save with _5xCap suffix for clarity
        save_path = f"data/output/predicted/HTGN/{file_path}_threshold_5xCap.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({'graphs': predicted_networks, 'node_count': args.num_nodes}, f)
            
        print(f"Saved 5x-Capped HTGN sparse graphs to {save_path}")



    @torch.no_grad()
    def evaluate_link_prediction(self, indices, warm_up_indices=None):
        """Helper to calculate Hyperbolic AUC on a specific set of snapshot indices."""
        self.model.eval()
        self.model.init_hiddens()
        
        # Warm up hidden state if history is provided
        if warm_up_indices:
            for t in warm_up_indices:
                snap = self.snapshots[t].to(args.device)
                z = self.model(snap.edge_index, self.x)
                self.model.update_hiddens_all_with(z)
        
        all_scores, all_targets = [], []
        for t_idx in indices:
            # Predict T using state from T-1
            # For train split, we unroll; for val, we continue from warm_up
            prev_snap = self.snapshots[t_idx-1].to(args.device)
            z = self.model(prev_snap.edge_index, self.x)
            
            target_snap = self.snapshots[t_idx].to(args.device)
            pos_edges = target_snap.edge_index
            neg_edges = torch.randint(0, args.num_nodes, (2, pos_edges.size(1)), device=args.device)
            
            # Hyperbolic similarity via sqdist
            p_dist = self.loss.manifold.sqdist(z[pos_edges[0]], z[pos_edges[1]], c=args.curvature)
            n_dist = self.loss.manifold.sqdist(z[neg_edges[0]], z[neg_edges[1]], c=args.curvature)
            
            # Sigmoid of negative distance is our "probability"
            scores = torch.cat([torch.sigmoid(-p_dist / 10.0), torch.sigmoid(-n_dist / 10.0)])
            targets = torch.cat([torch.ones(p_dist.size(0)), torch.zeros(n_dist.size(0))])
            
            all_scores.append(scores.cpu())
            all_targets.append(targets.cpu())
            self.model.update_hiddens_all_with(z)

        y_true = torch.cat(all_targets).numpy()
        y_scores = torch.cat(all_scores).numpy()
        return roc_auc_score(y_true, y_scores)

    def run(self):
        import geoopt
        optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_val_auc = 0.0
        best_train_auc_at_val = 0.0
        no_improve = 0
        
        start_train = time.time()
        for epoch in range(1, args.max_epoch + 1):
            epoch_losses = []
            self.model.init_hiddens()
            self.model.train()
            
            # 1. Training Pass
            for t in range(len(self.train_shots) - 1):
                snap = self.snapshots[self.train_shots[t]].to(args.device)
                next_snap = self.snapshots[self.train_shots[t+1]].to(args.device)
                
                optimizer.zero_grad()
                z = self.model(snap.edge_index, self.x)
                
                pos_edges = next_snap.edge_index
                neg_edges = torch.randint(0, args.num_nodes, (2, pos_edges.size(1)), device=args.device)
                
                p_dist = self.loss.manifold.sqdist(z[pos_edges[0]], z[pos_edges[1]], c=args.curvature)
                n_dist = self.loss.manifold.sqdist(z[neg_edges[0]], z[neg_edges[1]], c=args.curvature)
                
                loss = F.binary_cross_entropy_with_logits(-p_dist, torch.ones_like(p_dist)) + \
                       F.binary_cross_entropy_with_logits(-n_dist, torch.zeros_like(n_dist))
                
                if args.use_htc != 0:
                    loss += self.model.htc(z)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                self.model.update_hiddens_all_with(z.detach())

            # 2. Evaluation Phase
            # Training AUC (Warm up with first snapshot, evaluate the rest)
            train_auc = self.evaluate_link_prediction(self.train_shots[1:], warm_up_indices=[self.train_shots[0]])
            # Validation AUC (Warm up with all training history)
            val_auc = self.evaluate_link_prediction(self.val_shots, warm_up_indices=self.train_shots)
            
            avg_loss = np.mean(epoch_losses)

            # 3. Early Stopping on Validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_train_auc_at_val = train_auc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
                marker = "*"
            else:
                no_improve += 1
                marker = ""

            if epoch % args.log_interval == 0:
                logger.info(f"Epoch:{epoch:03d} | Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} {marker}")
            
            if no_improve >= args.patience:
                logger.info(f"Early stopping triggered. Best Val AUC: {best_val_auc:.4f}")
                break

        # FINAL SUMMARY
        print(f"\n[FINAL TRAINING SUMMARY - HTGN]")
        print(f"Best Validation AUC: {best_val_auc:.4f}")
        print(f"Corresponding Train AUC: {best_train_auc_at_val:.4f}")
        print(f"---------------------------\n")
        
        print('Training done, optimizing threshold now')
        # Tracking metrics...
        t1, g1, r1 = time.time() - start_train, get_gpu_memory(args.device), get_ram_usage()
        
        # Evaluation...
        start_opt = time.time()
        opt_tau = self.optimize_threshold()
        
        t2, g2, r2 = time.time() - start_opt, get_gpu_memory(args.device), get_ram_usage()
        
        start_cons = time.time()
        file_path = f"{args.dataset}_{args.lr}_{args.nhid}_{args.nout}_{args.curvature}_{'directed' if self.is_directed else 'undirected'}"
        self.construct_graphs(opt_tau, file_path, args.dataset)
        t3, g3, r3 = time.time() - start_cons, get_gpu_memory(args.device), get_ram_usage()

        print(f"\n--- DATASET: {args.dataset} (HTGN) METRICS ---")
        print(f"TRAIN:  Time={t1:.2f}s, GPU={g1:.2f}MB, RAM={r1:.2f}MB")
        print(f"THRESH: Time={t2:.2f}s, GPU={g2:.2f}MB, RAM={r2:.2f}MB")
        print(f"CONST:  Time={t3:.2f}s, GPU={g3:.2f}MB, RAM={r3:.2f}MB")
        gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    from benchmarkers.htgn.script.config import args
    from benchmarkers.htgn.script.utils.util import logger, init_logger
    from benchmarkers.htgn.script.models.load_model import load_model
    from benchmarkers.htgn.script.loss import ReconLoss
    
    data_dict = load_data('htgn', args.dataset)
    log_path = args.output_folder + args.dataset + '_htgn.txt'
    os.makedirs(os.path.dirname(log_path), exist_ok=True) 
    init_logger(log_path)
    
    runner = Runner(data_dict, is_directed=not args.undirected)
    runner.run()
