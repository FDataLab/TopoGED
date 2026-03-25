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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
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
            dist = self.loss.manifold.sqdist(z_chunk.unsqueeze(1), z.unsqueeze(0), c=1.0)
            
            # Link probability in Hyperbolic space is inverse to distance
            probs_chunk = torch.sigmoid(-dist)
            
            if threshold is not None:
                mask = (probs_chunk > threshold).cpu().numpy()
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
            p_dist = self.loss.manifold.sqdist(z[pos_edges[0]], z[pos_edges[1]], c=1.0)
            n_dist = self.loss.manifold.sqdist(z[neg_edges[0]], z[neg_edges[1]], c=1.0)
            
            all_y_scores.append(torch.cat([torch.sigmoid(-p_dist), torch.sigmoid(-n_dist)]).cpu())
            all_y_true.append(torch.cat([torch.ones(p_dist.size(0)), torch.zeros(n_dist.size(0))]))
            
            self.model.update_hiddens_all_with(z)

        y_scores = torch.cat(all_y_scores).numpy()
        y_true = torch.cat(all_y_true).numpy()
        
        thresholds = np.linspace(0.01, 0.99, 50)
        best_f1, best_tau = 0, 0.01
        for t in thresholds:
            preds = (y_scores > t).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1, best_tau = score, t
        
        print(f"Optimal Threshold: {best_tau:.2f} | Sampled Val F1: {best_f1:.4f}")
        return best_tau

    def construct_graphs(self, threshold, file_path):
        from benchmarkers.benchmarker_utils.k_values_extractor import get_topk
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()
        
        with torch.no_grad():
            self.model.init_hiddens()
            warmup_shots = self.train_shots + self.val_shots
            for t in warmup_shots:
                snap = self.snapshots[t].to(args.device)
                z = self.model(snap.edge_index, self.x)
                self.model.update_hiddens_all_with(z)
        
        all_embeddings = []
        with torch.no_grad():
            for t in self.test_shots:
                snap = self.snapshots[t-1].to(args.device)
                z = self.model(snap.edge_index, self.x)
                all_embeddings.append(z)
                self.model.update_hiddens_all_with(z)
                
        try:
            top_k_values = get_topk(args.dataset, use_true=False)
            test_top_k = top_k_values[-len(self.test_shots):]
            strategies = [True, False]
        except:
            strategies = [False]

        for using_topk in strategies:
            predicted_networks = []
            for t, z in enumerate(all_embeddings):
                if using_topk:
                    # For Top-K on 32k nodes, we must chunk to find top scores without massive allocation
                    k = int(test_top_k[t])
                    adj_matrix = np.zeros((args.num_nodes, args.num_nodes), dtype=np.int8)
                    all_vals, all_inds = [], []
                    chunk_size = 512
                    for i in range(0, args.num_nodes, chunk_size):
                        end_i = min(i + chunk_size, args.num_nodes)
                        dist = self.loss.manifold.sqdist(z[i:end_i].unsqueeze(1), z.unsqueeze(0), c=1.0)
                        probs = torch.sigmoid(-dist)
                        # Zero diagonal
                        diag_idx = torch.arange(i, end_i, device=args.device)
                        probs[torch.arange(end_i - i), diag_idx] = 0
                        
                        v, l = torch.topk(probs.view(-1), min(k, probs.numel()))
                        all_vals.append(v)
                        all_inds.append(l + (i * args.num_nodes))
                    
                    top_v = torch.cat(all_vals)
                    top_i = torch.cat(all_inds)
                    _, global_l = torch.topk(top_v, min(k, top_v.numel()))
                    final_i = top_i[global_l]
                    adj_matrix[final_i // args.num_nodes, final_i % args.num_nodes] = 1
                else:
                    adj_matrix = self.get_probs_chunked(z, threshold=threshold, chunk_size=300)
                
                if not self.is_directed: 
                    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
                
                predicted_networks.append(nx.from_numpy_array(adj_matrix, create_using=(nx.DiGraph() if self.is_directed else nx.Graph())))

            strategy_name = 'topk' if using_topk else 'threshold'
            save_path = f"data/output/predicted/HTGN/{file_path}_{strategy_name}.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({'graphs': predicted_networks, 'node_count': args.num_nodes}, f)
            print(f"Saved HTGN predicted graphs ({strategy_name}) to {save_path}")

    def run(self):
        # RiemannianAdam is critical for HTGN to stay on the manifold
        import geoopt
        optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = float('inf')
        no_improve = 0
        
        start_train = time.time()
        for epoch in range(1, args.max_epoch + 1):
            epoch_losses = []
            self.model.init_hiddens()
            self.model.train()
            
            for t in range(len(self.train_shots) - 1):
                snap = self.snapshots[self.train_shots[t]].to(args.device)
                next_snap = self.snapshots[self.train_shots[t+1]].to(args.device)
                
                optimizer.zero_grad()
                z = self.model(snap.edge_index, self.x)
                
                # --- SPARSE HYPERBOLIC LOSS ---
                # Sample negative edges for hyperbolic ranking loss
                pos_edges = next_snap.edge_index
                neg_edges = torch.randint(0, args.num_nodes, (2, pos_edges.size(1)), device=args.device)
                
                # We calculate distances instead of dot products
                p_dist = self.loss.manifold.sqdist(z[pos_edges[0]], z[pos_edges[1]], c=1.0)
                n_dist = self.loss.manifold.sqdist(z[neg_edges[0]], z[neg_edges[1]], c=1.0)
                
                # HTGN usually uses a ranking loss or Fermi-Dirac BCE
                loss = F.binary_cross_entropy_with_logits(-p_dist, torch.ones_like(p_dist)) + \
                       F.binary_cross_entropy_with_logits(-n_dist, torch.zeros_like(n_dist))
                
                if args.use_htc != 0:
                    loss += self.model.htc(z)
                
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                self.model.update_hiddens_all_with(z.detach())

            avg_loss = np.mean(epoch_losses)
            if avg_loss < best_loss:
                best_loss, no_improve = avg_loss, 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                no_improve += 1

            if epoch % args.log_interval == 0:
                logger.info(f"Epoch:{epoch:03d} | Avg Loss: {avg_loss:.4f}")
            if no_improve >= args.patience: break
        
        # Tracking metrics...
        t1, g1, r1 = time.time() - start_train, get_gpu_memory(args.device), get_ram_usage()
        
        # Evaluation...
        start_opt = time.time()
        opt_tau = self.optimize_threshold()
        t2, g2, r2 = time.time() - start_opt, get_gpu_memory(args.device), get_ram_usage()
        
        start_cons = time.time()
        file_path = f"{args.dataset}_{args.lr}_{args.nhid}_{args.nout}_{args.curvature}_{'directed' if self.is_directed else 'undirected'}"
        self.construct_graphs(opt_tau, file_path)
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