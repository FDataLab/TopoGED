import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
import networkx as nx
import numpy as np
import pickle
import argparse
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.utils import to_dense_adj

# Ensure model path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from benchmarkers.utils.dataset_setup import load_data
from benchmarkers.tgcn.model import TGCNModel

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

# --- THRESHOLD OPTIMIZATION ---
def optimize_threshold(model, val_snaps, train_snaps, node_count, device, window_size=12):
    """Finds best threshold on val set after training is complete."""
    model.eval()
    all_probs, all_targets = [], []
    history = train_snaps[-window_size:] if train_snaps else []
    full_sequence = history + val_snaps
    
    with torch.no_grad():
        for i in range(window_size, len(full_sequence)):
            h = torch.zeros(node_count, 100).to(device)
            for j in range(i - window_size, i):
                snap = full_sequence[j].to(device)
                _, h = model(snap.x, snap.edge_index, h)
            
            snap_to_predict = full_sequence[i].to(device)
            output, _ = model(snap_to_predict.x, snap_to_predict.edge_index, h)
            
            target = to_dense_adj(snap_to_predict.edge_index, max_num_nodes=node_count).flatten().cpu().numpy()
            probs = output.flatten().cpu().numpy()
            all_targets.append(target)
            all_probs.append(probs)

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    best_threshold, best_f1 = 0.01, 0
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    for t in thresholds:
        preds = (all_probs > t).astype(int)
        score = f1_score(all_targets, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, t
            
    print(f"Optimal Threshold: {best_threshold:.2f} | Val F1: {best_f1:.4f}")
    return best_threshold

# --- GRAPH CONSTRUCTION ---
def construct_predicted_graphs(model, current_snaps, previous_snaps, node_count, device, window_size=12, threshold=0.5):
    """Generates NetworkX graphs for the final test sequence."""
    model.eval()
    predicted_networks = []
    history = previous_snaps[-window_size:] if previous_snaps else []
    full_sequence = history + current_snaps
    
    with torch.no_grad():
        for i in range(window_size, len(full_sequence)):
            h = torch.zeros(node_count, 100).to(device) 
            for j in range(i - window_size, i):
                snap = full_sequence[j].to(device)
                _, h = model(snap.x, snap.edge_index, h)
            
            snap_to_predict = full_sequence[i].to(device)
            output, _ = model(snap_to_predict.x, snap_to_predict.edge_index, h)
            adj_matrix = (output.cpu().detach().numpy() > threshold).astype(int)
            predicted_networks.append(nx.from_numpy_array(adj_matrix))
            
    return predicted_networks

# --- TRAINING PIPELINE ---
def train_model(dataset_snaps, node_count, node_features, device='cuda'):
    # AUTHOR-STRICT SETTINGS
    hidden_dim = 100 #
    lr = 0.001 #
    lambda_loss = 0.0015 # From original author script
    window_size = 12 # Author default seq_len
    patience = 15
    best_train_auc, no_improve_epochs = 0, 0
    
    model = TGCNModel(node_count, node_features, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n = len(dataset_snaps)
    # 70/15/15 Split
    train_snaps = dataset_snaps[:int(n*0.7)]
    val_snaps = dataset_snaps[int(n*0.7):int(n*0.85)]
    test_snaps = dataset_snaps[int(n*0.85):]

    for epoch in range(100): 
        model.train()
        epoch_loss, y_true_train, y_pred_train = 0, [], []
        
        # Sliding Window Training
        for i in range(window_size, len(train_snaps)):
            h = torch.zeros(node_count, hidden_dim).to(device) 
            for j in range(i - window_size, i):
                snap = train_snaps[j].to(device)
                _, h = model(snap.x, snap.edge_index, h)
            
            target_snap = train_snaps[i].to(device)
            output, h = model(target_snap.x, target_snap.edge_index, h)
            target_adj = to_dense_adj(target_snap.edge_index, max_num_nodes=node_count).squeeze().to(device)
            
            # Loss = 0.5 * L2 distance + Lambda * L2 Regularization
            # 0.5 factor matches tf.nn.l2_loss behavior
            l2_dist = 0.5 * torch.sum((output - target_adj) ** 2)
            reg_loss = lambda_loss * sum(torch.sum(p ** 2) / 2 for p in model.parameters())
            loss = l2_dist + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            y_true_train.extend(target_adj.flatten().detach().cpu().numpy())
            y_pred_train.extend(output.flatten().detach().cpu().numpy())

        # MONITORING: Early stopping based on Training AUC plateau
        train_auc = roc_auc_score(y_true_train, y_pred_train)
        if train_auc > best_train_auc:
            best_train_auc, no_improve_epochs = train_auc, 0
        else:
            no_improve_epochs += 1
            
        print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Train AUC: {train_auc:.4f}")
        if no_improve_epochs >= patience:
            print("Stopping early: Training AUC has plateaued.")
            break

    # POST-TRAIN: Optimize threshold on Validation, then Construct Test Graphs
    opt_threshold = optimize_threshold(model, val_snaps, train_snaps, node_count, device, window_size)
    test_graphs = construct_predicted_graphs(model, test_snaps, val_snaps, node_count, device, window_size, threshold=opt_threshold)
    
    return model, test_graphs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T-GCN for Link Prediction.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dict = load_data('tgcn', args.dataset)

    print(f"--- Running T-GCN Benchmark: {args.dataset} ---")
    model, predicted_graphs = train_model(
        data_dict['snapshots'], 
        data_dict['node_count'], 
        data_dict['feature_dim'], 
        device=device
    )

    save_path = f"data/output/predicted/{args.dataset}_tgcn_predicted.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({'graphs': predicted_graphs, 'node_count': data_dict['node_count']}, f)
    print(f"Saved {len(predicted_graphs)} T-GCN predicted graphs to {save_path}")