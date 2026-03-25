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

    def train(self):
        print(f"--- Training {self.args.model} | Task: Future Link Prediction ---")
        start_train = time.time()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(self.device)
        
        best_train_loss = float('inf')
        patience, no_improve = 15, 0
        self.best_encoder, self.best_classifier = None, None

        for epoch in range(1, 201):
            self.encoder.train(); self.classifier.train()
            epoch_loss = 0
            all_epoch_preds, all_epoch_targets = [], []
            
            for t in self.train_idx:
                self.optimizer.zero_grad()
                # Use history BEFORE t to get embeddings
                A_win, N_win, Masks = self.get_window_and_mask(t)
                emb = self.encoder(A_win, N_win, Masks)
                
                # Target is the current snapshot t
                if self.A_list[t].is_sparse:
                    sparse_t = self.A_list[t].coalesce() 
                    indices = sparse_t.indices()
                    values = sparse_t.values()
                    mask = values > 0
                    pos_edges = indices[:, mask].t()
                else:
                    pos_edges = self.A_list[t].nonzero()
                if pos_edges.size(0) == 0: continue
                
                num_pos = pos_edges.size(0)
                MAX_NEG_EDGES = 20000  # should stop/limit OOM risk
                if num_pos > MAX_NEG_EDGES:
                    # Randomly shuffle and take the top K
                    perm = torch.randperm(num_pos)[:MAX_NEG_EDGES]
                    pos_edges = pos_edges[perm]
                    num_pos = MAX_NEG_EDGES

                neg_edges = torch.randint(0, self.node_count, (num_pos, 2)).to(self.device)
                
                pos_scores = self.classifier(torch.cat([emb[pos_edges[:,0]], emb[pos_edges[:,1]]], dim=1))
                neg_scores = self.classifier(torch.cat([emb[neg_edges[:,0]], emb[neg_edges[:,1]]], dim=1))
                
                loss = self.bce_loss(pos_scores, torch.ones_like(pos_scores)) + \
                       self.bce_loss(neg_scores, torch.zeros_like(neg_scores))
                
                loss.backward(); self.optimizer.step(); epoch_loss += loss.item()

                if epoch % 10 == 0 or epoch == 1:
                    with torch.no_grad():
                        preds = torch.cat([torch.sigmoid(pos_scores), torch.sigmoid(neg_scores)]).cpu().numpy().flatten()
                        targets = np.concatenate([np.ones(pos_scores.size(0)), np.zeros(neg_scores.size(0))])
                        all_epoch_preds.append(preds)
                        all_epoch_targets.append(targets)

            avg_epoch_loss = epoch_loss / len(self.train_idx)
            
            if avg_epoch_loss < best_train_loss:
                best_train_loss = avg_epoch_loss
                self.best_encoder = copy.deepcopy(self.encoder)
                self.best_classifier = copy.deepcopy(self.classifier)
                no_improve = 0
            else: no_improve += 1
            
            if epoch % 10 == 0 or epoch == 1:
                y_t = np.concatenate(all_epoch_targets)
                y_p = np.concatenate(all_epoch_preds)
                print(f"Epoch {epoch:03d} | Loss: {avg_epoch_loss:.6f} | Train AUC: {roc_auc_score(y_t, y_p):.4f}")
            if no_improve >= patience: break

        t1, g1, r1 = time.time() - start_train, get_gpu_memory(), get_ram_usage()
        if self.best_encoder is not None:
            self.encoder, self.classifier = self.best_encoder, self.best_classifier

        # === PHASE 2 & 3 ===
        start_opt = time.time()
        opt_thresh = self.optimize_threshold()
        t2, g2, r2 = time.time() - start_opt, get_gpu_memory(), get_ram_usage()

        start_cons = time.time()
        self.construct_graphs(opt_thresh)
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
            pos_edges = adj_t.indices().t() if adj_t.is_sparse else adj_t.nonzero()
            
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
        for tau in np.arange(0.1, 0.9, 0.05):
            preds = (y_scores > tau).astype(int)
            # Manual F1 to avoid sklearn overhead on large arrays
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            if f1 > best_f1: best_f1, best_tau = f1, tau
        return best_tau

    def construct_graphs(self, threshold):
        """Use the chunked score function for final output."""
        from benchmarkers.benchmarker_utils.k_values_extractor import get_topk
        self.encoder.eval(); self.classifier.eval()
        all_probs = []
        with torch.no_grad():
            for t in self.test_idx:
                A_win, N_win, Masks = self.get_window_and_mask(t)
                emb = self.encoder(A_win, N_win, Masks)
                # Use the chunked helper
                probs = self.get_adj_scores_chunked(emb).numpy()
                np.fill_diagonal(probs, 0)
                all_probs.append(probs)
                
        try:
            top_k_values = get_topk(self.args.dataset, use_true=False)
            test_top_k = top_k_values[-len(self.test_idx):]
            strategies = [True, False]
        except: strategies = [False]

        for using_topk in strategies:
            predicted_networks = []
            for t, probs in enumerate(all_probs):
                if using_topk:
                    k = min(int(test_top_k[t]), probs.size)
                    adj_matrix = np.zeros_like(probs, dtype=int)
                    if k > 0:
                        flat_indices = np.argsort(probs, axis=None)[-k:]
                        r_idx, c_idx = np.unravel_index(flat_indices, probs.shape)
                        adj_matrix[r_idx, c_idx] = 1
                else:
                    adj_matrix = (probs > threshold).astype(int)
                if not self.is_directed: adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
                predicted_networks.append(nx.from_numpy_array(adj_matrix, create_using=(nx.DiGraph() if self.is_directed else nx.Graph())))

            strategy = 'topk' if using_topk else 'threshold'
            file_path = f"{self.args.dataset}_{self.args.model}_{self.args.layer_1_feats}_{self.args.layer_2_feats}_{self.args.cls_feats}_{self.args.lr}_{self.args.l2_reg}_{'directed' if self.is_directed else 'undirected'}"
            save_path = f"data/output/predicted/EvolveGCN/{file_path}_{strategy}.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({'graphs': predicted_networks, 'node_count': self.node_count}, f)

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