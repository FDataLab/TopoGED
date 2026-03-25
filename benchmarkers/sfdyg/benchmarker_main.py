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
        """Sparse negative sampling for large-scale training."""
        num_pos = edge_index.size(1)
        num_samples = num_pos * num_neg_samples
        return torch.randint(0, num_nodes, (2, num_samples), device=self.device)

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
            logits_chunk = torch.mm(h_chunk, h.t())
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
            p_scores = torch.sigmoid((h[pos_edges[0]] * h[pos_edges[1]]).sum(-1))
            n_scores = torch.sigmoid((h[neg_edges[0]] * h[neg_edges[1]]).sum(-1))
            
            all_y_scores.append(torch.cat([p_scores, n_scores]).cpu())
            all_y_true.append(torch.cat([torch.ones_like(p_scores), torch.zeros_like(neg_scores)]).cpu())

        y_scores = torch.cat(all_y_scores).numpy()
        y_true = torch.cat(all_y_true).numpy()
        
        best_f1, best_threshold = 0, 0.5
        for t in np.linspace(0.01, 0.99, 50):
            preds = (y_scores > t).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1: best_f1, best_threshold = score, t
                
        return best_threshold
    
    def train(self):
        print(f"--- Training SFDyG ({self.args.model}) | Sparse mode ---")
        best_loss = float('inf')
        patience = 20
        no_improve = 0
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)

        for epoch in range(1, 1 + self.args.epochs):
            self.model.train()
            epoch_loss = 0
            all_preds, all_targets = [], []
            
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

                if epoch % 10 == 0:
                    with torch.no_grad():
                        h = self.model.encoder(fused_data) if hasattr(self.model, 'encoder') else self.model(fused_data)
                        if isinstance(h, tuple): h = h[0]
                        p_s = torch.sigmoid((h[pos_edges[0]] * h[pos_edges[1]]).sum(-1))
                        n_s = torch.sigmoid((h[neg_edges[0]] * h[neg_edges[1]]).sum(-1))
                        all_preds.append(torch.cat([p_s, n_s]).cpu())
                        all_targets.append(torch.cat([torch.ones_like(p_s), torch.zeros_like(n_s)]).cpu())
                
            lr_scheduler.step()
            avg_loss = epoch_loss / len(self.train_idx)

            if epoch % self.args.eval_steps == 0:
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")
                if len(all_preds) > 0:
                    auc = roc_auc_score(torch.cat(all_targets), torch.cat(all_preds))
                    print(f"Sampled Train AUC: {auc:.4f}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    no_improve = 0
                else: no_improve += self.args.eval_steps
                
                if no_improve >= patience: break
                    
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def construct_graphs(self, threshold):
        from benchmarkers.benchmarker_utils.k_values_extractor import get_topk
        self.model.eval()
        
        try:
            top_k_values = get_topk(self.args.dataset, use_true=False)
            test_top_k = top_k_values[-len(self.test_idx):]
            strategies = [True, False]
        except: strategies = [False]

        all_embeddings = []
        with torch.no_grad():
            for t in self.test_idx:
                fused_data = self.get_fused_data(t)
                h = self.model.encoder(fused_data) if hasattr(self.model, 'encoder') else self.model(fused_data)
                if isinstance(h, tuple): h = h[0]
                all_embeddings.append(h)

        for using_topk in strategies:
            predicted_networks = []
            for t_idx, h in enumerate(all_embeddings):
                adj_matrix = np.zeros((self.node_count, self.node_count), dtype=np.int8)
                
                if using_topk:
                    k = int(test_top_k[t_idx])
                    all_vals, all_inds = [], []
                    chunk_size = 512
                    for i in range(0, self.node_count, chunk_size):
                        end_i = min(i + chunk_size, self.node_count)
                        p_chunk = torch.sigmoid(torch.mm(h[i:end_i], h.t()))
                        # Zero diagonal
                        p_chunk[torch.arange(end_i-i), torch.arange(i, end_i, device=self.device)] = 0
                        v, l = torch.topk(p_chunk.view(-1), min(k, p_chunk.numel()))
                        all_vals.append(v); all_inds.append(l + (i * self.node_count))
                    
                    top_v = torch.cat(all_vals)
                    top_i = torch.cat(all_inds)
                    _, gl = torch.topk(top_v, min(k, top_v.numel()))
                    final_i = top_i[gl].cpu().numpy()
                    adj_matrix[final_i // self.node_count, final_i % self.node_count] = 1
                else:
                    # Memory-safe chunked thresholding
                    for i in range(0, self.node_count, 512):
                        end_i = min(i + 512, self.node_count)
                        mask = (torch.sigmoid(torch.mm(h[i:end_i], h.t())) > threshold).cpu().numpy()
                        adj_matrix[i:end_i] = mask.astype(np.int8)

                if not self.is_directed: adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
                predicted_networks.append(nx.from_numpy_array(adj_matrix, create_using=(nx.DiGraph() if self.is_directed else nx.Graph())))

            strategy = 'topk' if using_topk else 'threshold'
            file_path = f"{self.args.dataset}_{self.args.model}_{self.args.window}"
            save_path = f"data/output/predicted/SFDyG/{file_path}_{strategy}.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({'graphs': predicted_networks, 'node_count': self.node_count}, f)
            print(f"Saved {strategy} graphs to {save_path}")

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