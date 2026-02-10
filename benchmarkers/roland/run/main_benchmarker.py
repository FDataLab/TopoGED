import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import networkx as nx
from torch_geometric.utils import negative_sampling, to_dense_adj
from sklearn.metrics import roc_auc_score, f1_score
import pickle
import sys


# Import the model architecture we defined previously
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from benchmarkers.utils.dataset_setup import load_data
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

class RolandRunner:
    def __init__(self, data_dict, dataset_name, device):
        self.device = device
        self.dataset_name = dataset_name
        self.snapshots = data_dict['snapshots']
        self.node_count = data_dict['node_count']
        self.feat_dim = data_dict['feature_dim']
        self.edge_dim = data_dict.get('edge_dim', 0)
        
        # 1. 70/15/15 Split
        n = len(self.snapshots)
        self.train_end = int(n * 0.7)
        self.val_end = int(n * 0.85)
        
        # Hyperparameters (Paper default)
        self.hidden_dim = 128
        self.lr = 0.01 
        
        # Initialize Model
        self.model = ROLAND(self.node_count, self.feat_dim, self.hidden_dim, 
                            edge_dim=self.edge_dim, num_layers=2).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def get_snapshot(self, t):
        """
        Retrieves snapshot t. 
        CRITICAL: Generates Identity features on-the-fly if x is None to save RAM.
        """
        data = self.snapshots[t]
        
        # 1. Handle Structure
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device) if data.edge_attr is not None else None
        
        # 2. Handle Features (Memory Optimization)
        if data.x is None:
            # Generate local identity for this batch/snapshot
            x = torch.eye(self.node_count).to(self.device)
        else:
            x = data.x.to(self.device)
            
        return x, edge_index, edge_attr

    def run(self):
        # Initialize History H_0
        current_states = self.model.init_states(self.node_count, self.device)
        
        print(f"--- Phase 1: Incremental Training (0 -> {self.train_end}) ---")
        self.model.train()
        
        # Incremental Training Loop
        for t in range(self.train_end):
            x, edge_index, edge_attr = self.get_snapshot(t)
            
            # Detach History (Truncated BPTT for Scalability)
            detached_states = [s.detach() for s in current_states]
            
            # Fine-tune on current snapshot
            # ROLAND paper suggests a few epochs per snapshot during live update
            for _ in range(5): 
                self.optimizer.zero_grad()
                
                # Forward to get H_t
                new_states = self.model(x, edge_index, edge_attr, detached_states)
                final_emb = new_states[-1]
                
                # Link Prediction Loss
                neg_edge_index = negative_sampling(edge_index, num_nodes=self.node_count)
                pos_score = self.model.predict_links(final_emb, edge_index)
                neg_score = self.model.predict_links(final_emb, neg_edge_index)
                
                labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).to(self.device)
                preds = torch.cat([pos_score, neg_score]).squeeze()
                
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()
            
            # Update History for t+1
            with torch.no_grad():
                current_states = self.model(x, edge_index, edge_attr, detached_states)
                
            if t % 10 == 0: print(f"Snapshot {t}/{self.train_end} | Loss: {loss.item():.4f}")

        # Save states for validation
        train_states = [s.detach() for s in current_states]
        
        print("--- Phase 2: Optimizing Threshold ---")
        self.model.eval()
        current_states = train_states
        val_probs, val_labels = [], []
        
        for t in range(self.train_end, self.val_end):
            x, edge_index, edge_attr = self.get_snapshot(t)
            with torch.no_grad():
                current_states = self.model(x, edge_index, edge_attr, current_states)
                z = current_states[-1]
                
                # Generate probabilities for thresholding
                # Optimization: We use sparse reconstruction or sampling if N is huge
                # Here we simulate full N*N for standard benchmarks
                z_src = z.unsqueeze(1).repeat(1, self.node_count, 1)
                z_dst = z.unsqueeze(0).repeat(self.node_count, 1, 1)
                logits = self.model.pred_head(torch.cat([z_src, z_dst], dim=2)).squeeze()
                probs = torch.sigmoid(logits)
                
                # Ground Truth Dense Adjacency
                true_adj = to_dense_adj(edge_index, max_num_nodes=self.node_count).squeeze()
                
                val_probs.append(probs.cpu().numpy().flatten())
                val_labels.append(true_adj.cpu().numpy().flatten())

        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        
        best_tau, best_f1 = 0.5, 0
        for tau in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(val_labels, (val_probs > tau).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        print(f"Optimal Threshold: {best_tau:.2f} (F1: {best_f1:.4f})")

        print("--- Phase 3: Constructing Test Graphs ---")
        predicted_graphs = []
        
        for t in range(self.val_end, len(self.snapshots)):
            x, edge_index, edge_attr = self.get_snapshot(t)
            with torch.no_grad():
                current_states = self.model(x, edge_index, edge_attr, current_states)
                z = current_states[-1]
                
                # Construct Graph
                z_src = z.unsqueeze(1).repeat(1, self.node_count, 1)
                z_dst = z.unsqueeze(0).repeat(self.node_count, 1, 1)
                logits = self.model.pred_head(torch.cat([z_src, z_dst], dim=2)).squeeze()
                adj_binary = (torch.sigmoid(logits) > best_tau).cpu().numpy().astype(int)
                np.fill_diagonal(adj_binary, 0)
                
                predicted_graphs.append(nx.from_numpy_array(adj_binary))

        save_path = f"data/output/predicted/{self.dataset_name}_roland_predicted.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'graphs': predicted_graphs, 'node_count': self.node_count}, f)
        print(f"Saved {len(predicted_graphs)} graphs to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # NEW: Load data on-the-fly (x=None inside snapshots to save RAM)
    data_dict = load_data('roland', args.dataset)
    
    RolandRunner(data_dict, args.dataset, device).run()