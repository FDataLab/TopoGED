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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
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

class Runner(object):
    def __init__(self, data_dict):
        # 1. Load from Cached Dict
        self.snapshots = data_dict['snapshots'] 
        self.len = len(self.snapshots)
        args.num_nodes = data_dict['node_count']
        
        # 2. 70/15/15 Split
        n = self.len
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        self.train_shots = list(range(0, train_end))
        self.val_shots = list(range(train_end, val_end))
        self.test_shots = list(range(val_end, n))
        
        # 3. Model Setup
        self.load_feature()
        self.model = load_model(args).to(args.device)
        self.loss = ReconLoss(args) # HTGN uses ReconLoss for link prediction [cite: 779]
        
        self.best_model_state = None
        logger.info(f'Split: Train({len(self.train_shots)}), Val({len(self.val_shots)}), Test({len(self.test_shots)})')

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
        else:
            self.x = self.snapshots[0].x.to(args.device)
            args.nfeat = self.x.size(1)

    def evaluate_auc(self, shots):
        """Standard AUC evaluation for monitoring."""
        self.model.eval()
        auc_scores = []
        with torch.no_grad():
            for t in shots:
                snap = self.snapshots[t].to(args.device)
                z = self.model(snap.edge_index, self.x)
                auc, _ = self.loss.predict(z, snap.edge_index, None) 
                auc_scores.append(auc)
        return np.mean(auc_scores)

    def optimize_threshold(self):
        """Finds the best threshold using Fermi-Dirac probabilities on the Val split."""
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()
        all_probs, all_targets = [], []

        with torch.no_grad():
            for t in self.val_shots:
                snap = self.snapshots[t].to(args.device)
                z = self.model(snap.edge_index, self.x)
                
                # Fetch full NxN probability matrix using custom get_probs in loss.py
                probs = self.loss.get_probs(z) 
                target = to_dense_adj(snap.edge_index, max_num_nodes=args.num_nodes).squeeze().cpu().numpy()
                
                all_probs.append(probs.cpu().numpy().flatten())
                all_targets.append(target.flatten())

        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)

        best_tau, best_f1 = 0.01, 0
        # Search 0.01 intervals as with GCLSTM/TGCN
        thresholds = np.arange(0.01, 1.0, 0.01)
        for t in thresholds:
            preds = (all_probs > t).astype(int)
            score = f1_score(all_targets, preds, zero_division=0)
            if score > best_f1:
                best_f1, best_tau = score, t
        
        logger.info(f"Optimal HTGN Threshold: {best_tau:.2f} | Val F1: {best_f1:.4f}")
        return best_tau

    def construct_graphs(self, threshold):
        """Constructs final NetworkX graphs for the test sequence."""
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()
        predicted_graphs = []
        
        with torch.no_grad():
            for t in self.test_shots:
                snap = self.snapshots[t].to(args.device)
                z = self.model(snap.edge_index, self.x)
                probs = self.loss.get_probs(z)
                adj = (probs > threshold).cpu().numpy().astype(int)
                predicted_graphs.append(nx.from_numpy_array(adj))
        
        save_path = f"data/output/predicted/{args.dataset}_htgn_predicted.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'graphs': predicted_graphs, 'node_count': args.num_nodes}, f)
        logger.info(f"Saved {len(predicted_graphs)} graphs to {save_path}")

    def run(self):
        # RiemannianAdam is required for hyperbolic parameters [cite: 923]
        optimizer = self.optimizer(using_riemannianAdam=True) 
        best_train_auc = 0
        no_improve_epochs = 0
        
        for epoch in range(1, args.max_epoch + 1):
            epoch_losses = []
            self.model.init_hiddens()
            self.model.train()
            
            for t in self.train_shots:
                snap = self.snapshots[t].to(args.device)
                optimizer.zero_grad()
                z = self.model(snap.edge_index, self.x)
                
                # Equation 15: Unified loss [cite: 909]
                if args.use_htc == 0:
                    loss = self.loss(z, snap.edge_index)
                else:
                    loss = self.loss(z, snap.edge_index) + self.model.htc(z)
                
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                self.model.update_hiddens_all_with(z)

            # Early Stopping Check (Based on Train AUC plateau)
            train_auc = self.evaluate_auc(self.train_shots)
            if train_auc > best_train_auc:
                best_train_auc = train_auc
                no_improve_epochs = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= args.patience:
                logger.info(f"Stopping early at epoch {epoch} (Train AUC: {train_auc:.4f})")
                break
            
            if epoch % args.log_interval == 0:
                logger.info(f"Epoch:{epoch} | Loss: {np.mean(epoch_losses):.4f} | Train AUC: {train_auc:.4f}")

        # Post-training: Optimize threshold and build graphs
        opt_tau = self.optimize_threshold()
        self.construct_graphs(opt_tau)

    def optimizer(self, using_riemannianAdam=True):
        if using_riemannianAdam:
            import geoopt
            return geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if __name__ == '__main__':
    from benchmarkers.htgn.script.config import args
    from benchmarkers.htgn.script.utils.util import set_random, logger, init_logger
    from benchmarkers.htgn.script.models.load_model import load_model
    from benchmarkers.htgn.script.loss import ReconLoss, VGAEloss
    
    # CLI integration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    cli_args, unknown = parser.parse_known_args()
    
    # Load cached file
    data_dict = load_data('htgn', args.dataset)

    # set_random(args.seed)
    init_logger(args.output_folder + cli_args.dataset + '_htgn.txt')
    
    args.dataset = cli_args.dataset
    runner = Runner(data_dict)
    runner.run()