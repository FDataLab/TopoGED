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
        thresholds = torch.linspace(0.01, 0.99, 50)
        best_f1, best_tau = 0, 0.5
        for tau in thresholds:
            preds = (y_scores > tau).float()
            tp, fp, fn = (preds * y_true).sum(), (preds * (1 - y_true)).sum(), ((1 - preds) * y_true).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            if f1 > best_f1: best_f1, best_tau = f1.item(), tau.item()
        return best_tau

    def run(self):
        from benchmarkers.benchmarker_utils.k_values_extractor import get_topk
        start_train = time.time()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(self.device)
        best_train_loss = float('inf'); patience = 15; no_improve = 0
        
        for epoch in range(1, 201):
            self.model.train(); epoch_loss = 0
            all_preds, all_targets = [], []
            states = self.model.init_states(self.node_count, self.device)
            
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
                    loss = self.criterion(preds, labels); loss.backward(); self.optimizer.step()
                    epoch_loss += loss.item()

                if epoch % 10 == 0 or epoch == 1:
                    all_preds.append(preds.detach().sigmoid().cpu()); all_targets.append(labels.cpu())
                states = [s.detach() for s in new_states]
            
            avg_loss = epoch_loss / ((self.train_end - 1) * self.args.num_updates_per_snapshot)
            if epoch % 10 == 0 or epoch == 1:
                train_auc = roc_auc_score(torch.cat(all_targets), torch.cat(all_preds))
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | Train AUC: {train_auc:.4f}")

            if avg_loss < best_train_loss:
                best_train_loss = avg_loss; self.best_model = copy.deepcopy(self.model); no_improve = 0
            else: no_improve += 1
            if no_improve >= patience: break

        t1, g1, r1 = time.time() - start_train, get_gpu_memory(self.device), get_ram_usage()
        # torch.cuda.reset_peak_memory_stats(self.device)

        start_opt = time.time()
        best_tau = self.optimize_threshold()
        t2, g2, r2 = time.time() - start_opt, get_gpu_memory(self.device), get_ram_usage()
        # torch.cuda.reset_peak_memory_stats(self.device)
        
        start_cons = time.time()
        all_test_probs = []
        with torch.no_grad():
            states = self.model.init_states(self.node_count, self.device)
            for t in range(self.val_end - 1): # Re-bridge to start of test
                x, edge_index, edge_attr = self.get_snapshot(t)
                states = self.model(x, edge_index, edge_attr, states)
            
            for t in range(self.val_end - 1, len(self.snapshots) - 1):
                x, edge_index, edge_attr = self.get_snapshot(t)
                states = self.model(x, edge_index, edge_attr, states)
                probs = self.get_probs_chunked(states[-1]).numpy() # Chunked to avoid OOM
                if not self.is_directed: probs = (probs + probs.T) / 2.0
                np.fill_diagonal(probs, 0); all_test_probs.append(probs)

        try:
            top_k_values = get_topk(self.dataset_name, use_true=False)
            test_top_k = top_k_values[-(len(self.snapshots) - self.val_end):]
            strategies = [True, False]
        except Exception as e:
            print(f"Top-K fetch failed, using threshold only. Error: {e}")
            strategies = [False]

        for using_topk in strategies:
            predicted_networks = []
            for t, probs in enumerate(all_test_probs):
                if using_topk:
                    k = int(test_top_k[t])
                    adj = np.zeros_like(probs, dtype=int)
                    if k > 0:
                        flat = np.argsort(probs, axis=None)[-k:]
                        r, c = np.unravel_index(flat, probs.shape)
                        adj[r, c] = 1
                else:
                    adj = (probs > best_tau).astype(int)
                
                if not self.is_directed: adj = np.maximum(adj, adj.T)
                predicted_networks.append(nx.from_numpy_array(adj, create_using=(nx.DiGraph() if self.is_directed else nx.Graph())))
            
            strategy_str = 'topk' if using_topk else 'threshold'
            file_params = f"{self.dataset_name}_{self.args.hidden_dim}_{self.args.lr}_{self.args.num_layers}_{self.args.num_updates_per_snapshot}_{'directed' if self.is_directed else 'undirected'}"
            save_path = f"data/output/predicted/ROLAND/{file_params}_{strategy_str}.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({'graphs': predicted_networks, 'node_count': self.node_count}, f)
            print(f"Saved ROLAND graphs ({strategy_str}) to {save_path}")
            
        t3, g3, r3 = time.time() - start_cons, get_gpu_memory(self.device), get_ram_usage()
        print(f"\n--- DATASET: {self.dataset_name} (ROLAND) METRICS ---\nTRAIN: {t1:.2f}s, {g1:.2f}MB\nTHRESH: {t2:.2f}s, {g2:.2f}MB\nCONST: {t3:.2f}s, {g3:.2f}MB\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True); parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=128); parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_layers", type=int, default=2); parser.add_argument("--num_updates_per_snapshot", type=int, default=5)
    
    args = parser.parse_args(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dict = load_data('roland', args.dataset)
    RolandRunner(args, data_dict, args.dataset, device, is_directed=not args.undirected).run()