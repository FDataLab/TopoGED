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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from benchmarkers.utils.dataset_setup import load_data

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

# --- 1. MOCK UTILS (To satisfy egcn_h/o imports) ---
class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def pad_with_last_val(vect, k):
    """Needed for egcn_h.py TopK pooling"""
    device = vect.device
    pad_size = k - vect.size(0)
    if pad_size > 0:
        last_val = vect[-1]
        padding = torch.tensor([last_val] * pad_size, device=device)
        return torch.cat([vect, padding])
    return vect

def reset_param(t):
    """Needed for models.py"""
    stdv = 1. / math.sqrt(t.size(1))
    t.data.uniform_(-stdv, stdv)

# Inject mocks into sys.modules so imports works
import types
u = types.ModuleType("utils")
u.Namespace = Namespace
u.pad_with_last_val = pad_with_last_val
u.reset_param = reset_param
sys.modules['utils'] = u

# --- 2. IMPORT PROVIDED MODEL FILES ---
# Ensure egcn_h.py, egcn_o.py, and models.py are in the same folder
import egcn_h
import egcn_o
import models

# --- 3. MAIN RUNNER (Modified main.py + trainer.py) ---
class EvolveGCNRunner:
    def __init__(self, args, data_dict):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Data
        self.A_list = [a.to(self.device) for a in data_dict['A_list']]
        self.Nodes_list = [n.to(self.device) for n in data_dict['Nodes_list']]
        self.node_count = data_dict['node_count']
        self.feature_dim = data_dict['feature_dim']
        
        # 70/15/15 Split
        n = len(self.A_list)
        self.train_idx = range(0, int(n * 0.7))
        self.val_idx = range(int(n * 0.7), int(n * 0.85))
        self.test_idx = range(int(n * 0.85), n)
        
        # Build Models
        self.build_model()
        
    def build_model(self):
        # Parameters expected by EGCN classes
        # - Embedding size set same to reduce tuning
        gcn_args = u.Namespace({
            'feats_per_node': self.feature_dim,
            'layer_1_feats': 128,
            'layer_2_feats': 64, 
            'in_feats': self.feature_dim, # For GRCU
            'out_feats': 64 # For GRCU
        })
        
        if self.args.model == 'egcn_h':
            self.encoder = egcn_h.EGCN(gcn_args, activation=nn.RReLU(), device=self.device)
        elif self.args.model == 'egcn_o':
            self.encoder = egcn_o.EGCN(gcn_args, activation=nn.RReLU(), device=self.device)
        else:
            raise ValueError(f"Unknown model: {self.args.model}")

        # Classifier from models.py
        # EvolveGCN uses an MLP decoder on concatenated node embeddings
        cls_args = u.Namespace({
            'gcn_parameters': {'layer_2_feats': 64, 'cls_feats': 32},
            'experiment_type': 'standard'
        })
        # input dim = 64 (emb) * 2 (concat) = 128
        self.classifier = models.Classifier(cls_args, out_features=1, in_features=128).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()), 
            lr=0.001
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

    def get_window_and_mask(self, t, window_size=5):
        """Prepare input window and dummy mask for EGCN-H"""
        start = max(0, t - window_size + 1)
        A_window = self.A_list[start:t+1]
        Nodes_window = self.Nodes_list[start:t+1]
        
        # Create zero mask (assuming all nodes are valid in the fixed-size adjacency)
        # mask is used in egcn_h.py TopK to ignore nodes. 0 means include.
        mask_list = [torch.zeros(self.node_count, 1).to(self.device) for _ in range(len(A_window))]
        
        return A_window, Nodes_window, mask_list

    def train(self):
        best_val_auc = 0
        best_state = None
        
        print(f"--- Training {self.args.model} ---")
        for epoch in range(1, 201):
            self.encoder.train()
            self.classifier.train()
            epoch_loss = 0
            
            for t in self.train_idx:
                self.optimizer.zero_grad()
                
                # 1. Forward Pass
                A_win, N_win, Masks = self.get_window_and_mask(t)
                
                # Note: EGCN-H signature is (A, Nodes, mask), EGCN-O is (A, Nodes, mask) (modified in file)
                if self.args.model == 'egcn_h':
                    emb = self.encoder(A_win, N_win, Masks)
                else:
                    # egcn_o.py forward definition in snippet: forward(self, A_list, Nodes_list, nodes_mask_list)
                    # even though logic might ignore mask, signature expects it based on provided file
                    emb = self.encoder(A_win, N_win, Masks)
                
                # 2. Link Prediction Loss (Self-Supervised on Snapshot t)
                # Sample edges from current snapshot adjacency
                adj_t = self.A_list[t].to_dense()
                pos_edges = adj_t.nonzero()
                if pos_edges.size(0) == 0: continue
                
                # Sample Negatives
                neg_edges = torch.randint(0, self.node_count, (pos_edges.size(0), 2)).to(self.device)
                
                # Decode
                pos_src, pos_dst = emb[pos_edges[:,0]], emb[pos_edges[:,1]]
                neg_src, neg_dst = emb[neg_edges[:,0]], emb[neg_edges[:,1]]
                
                pos_scores = self.classifier(torch.cat([pos_src, pos_dst], dim=1))
                neg_scores = self.classifier(torch.cat([neg_src, neg_dst], dim=1))
                
                loss = self.bce_loss(pos_scores, torch.ones_like(pos_scores)) + \
                       self.bce_loss(neg_scores, torch.zeros_like(neg_scores))
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Validation
            val_auc = self.evaluate_auc(self.val_idx)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {
                    'enc': copy.deepcopy(self.encoder.state_dict()),
                    'cls': copy.deepcopy(self.classifier.state_dict())
                }
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Final Test
        print("--- Testing ---")
        self.encoder.load_state_dict(best_state['enc'])
        self.classifier.load_state_dict(best_state['cls'])
        
        opt_thresh = self.optimize_threshold()
        self.construct_graphs(opt_thresh)

    def get_adj_scores(self, emb):
        """Reconstruct full N x N probability matrix using Classifier"""
        # Efficient broadcasting for MLP: (N, 1, F) concat (1, N, F) -> (N, N, 2F)
        n = emb.size(0)
        u = emb.unsqueeze(1).repeat(1, n, 1)
        v = emb.unsqueeze(0).repeat(n, 1, 1)
        pair_emb = torch.cat([u, v], dim=2)
        
        # Pass through classifier
        logits = self.classifier(pair_emb).squeeze()
        return torch.sigmoid(logits)

    def evaluate_auc(self, indices):
        self.encoder.eval()
        self.classifier.eval()
        auc_scores = []
        with torch.no_grad():
            for t in indices:
                A_win, N_win, Masks = self.get_window_and_mask(t)
                emb = self.encoder(A_win, N_win, Masks)
                
                scores = self.get_adj_scores(emb).cpu().numpy().flatten()
                targets = self.A_list[t].to_dense().cpu().numpy().flatten()
                
                auc_scores.append(roc_auc_score(targets, scores))
        return np.mean(auc_scores)

    def optimize_threshold(self):
        self.encoder.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for t in self.val_idx:
                A_win, N_win, Masks = self.get_window_and_mask(t)
                emb = self.encoder(A_win, N_win, Masks)
                
                probs = self.get_adj_scores(emb).cpu().numpy().flatten()
                targets = self.A_list[t].to_dense().cpu().numpy().flatten()
                all_probs.append(probs)
                all_targets.append(targets)
        
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        
        best_f1, best_tau = 0, 0.5
        for tau in np.arange(0.01, 1.0, 0.01):
            preds = (all_probs > tau).astype(int)
            f1 = f1_score(all_targets, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
        print(f"Optimal Threshold: {best_tau:.2f} (F1: {best_f1:.4f})")
        return best_tau

    def construct_graphs(self, threshold):
        self.encoder.eval()
        graphs = []
        with torch.no_grad():
            for t in self.test_idx:
                A_win, N_win, Masks = self.get_window_and_mask(t)
                emb = self.encoder(A_win, N_win, Masks)
                
                probs = self.get_adj_scores(emb)
                adj = (probs > threshold).cpu().numpy().astype(int)
                # Remove self loops for final graph
                np.fill_diagonal(adj, 0)
                graphs.append(nx.from_numpy_array(adj))
        
        save_path = f"data/output/predicted/{self.args.dataset}_evolvegcn_predicted.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'graphs': graphs, 'node_count': self.node_count}, f)
        print(f"Saved {len(graphs)} graphs to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="egcn_h", choices=['egcn_h', 'egcn_o'])
    args = parser.parse_args()
    
    data_dict = load_data('evolvegcn', args.dataset)
        
    runner = EvolveGCNRunner(args, data_dict)
    runner.train()