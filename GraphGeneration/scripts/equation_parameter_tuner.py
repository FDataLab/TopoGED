import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
from json import encoder
from math import e
import random
import numpy as np
import argparse
import os
import sys
from sklearn.metrics import recall_score
from sqlalchemy import all_
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from sympy import use
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from GraphGeneration.utils.Evaluator import Evaluator
from nn.custom_model import Decoder
from GraphGeneration.utils.graph_construction_utils import compute_reappearance_probabilities, generate_tgcn_node_features, get_node_features, update_degrees, generate_gnn_node_embeddings
from GraphGeneration.models.model import setupMLP, load_encoder_model
from process_data import modifyGraphIds, build_edgebanks_from_start
from GraphGeneration.scripts.compute_embedding import compute_node2vec_embeddings

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
num_back = 1000

# =====================================================
#  MODEL: Learnable alpha, beta, decay_factor
# =====================================================
class ReappearanceModel(nn.Module):
    def __init__(self, alpha=3.0, beta=5.0, decay_factor=1.0):
        super().__init__()

        # Trainable scalars
        # self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor([beta], dtype=torch.float32))
        self.decay_factor = nn.Parameter(torch.tensor([decay_factor], dtype=torch.float32))

    def forward(self, graphs, options):
        """
        graphs: list of networkx graphs (snapshots)
        options: list of node IDs to compute probabilities for
        Returns: torch tensor shape [len(options)] with probabilities
        """

        # Collect node stats
        nodes = {}
        freq = defaultdict(int)

        for t, G in enumerate(graphs):
            for node in G.nodes():
                nodes[node] = (t, G.degree(node))
                freq[node] += 1

        #max_degree = max(deg for _, (_, deg) in nodes.items()) if nodes else 1

        # Positivity constraints (important!)
        #alpha = F.softplus(self.alpha)
        beta  = F.softplus(self.beta)
        decay = F.softplus(self.decay_factor) + 1e-6

        t_curr = len(graphs)

        # Compute scores
        scores = []

        for node in options:
            last_seen, degree = nodes[node]

            recency = torch.exp(-(t_curr - last_seen) / decay)
            #degree_score = (degree / max_degree) ** alpha
            degree_score = 1
            freq_score = (freq[node] ** beta)

            scores.append((recency * degree_score * freq_score).unsqueeze(0))

        scores = torch.cat(scores)  # shape [len(options)]

        # Normalize → probability distribution
        probs = scores / (scores.sum() + 1e-12)
        return probs




# =====================================================
#  LOSS FUNCTION
#  Encourage model to place high probability on
#  the correct "old" nodes
# =====================================================
def reappearance_loss(probs, options, ground_truth_old):
    """
    probs: tensor [len(options)]
    options: list of node IDs
    ground_truth_old: set of correct node IDs
    """
    mask = torch.tensor([1 if node in ground_truth_old else 0
                         for node in options], dtype=torch.float32)

    p_correct = probs[mask.bool()]  # probabilities of correct nodes

    if len(p_correct) == 0:
        # no positives → no loss
        return torch.tensor(0.0, requires_grad=True)

    # maximize ∏ p(correct)  → minimize negative mean log-prob
    return -torch.log(p_correct).mean()




# =====================================================
#  TRAINING LOOP
# =====================================================
def train_model(model, training_data, validation_data, epochs=50, lr=1e-3,
                patience=5):
    """
    training_data: list of tuples (graphs, options, ground_truth_old, k)
    validation_data: same format as training_data
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None

    for epoch in range(epochs):
        # -----------------------------
        # TRAINING
        # -----------------------------
        model.train()
        total_loss = 0.0

        for graphs, options, ground_truth_old, k in training_data:
            optimizer.zero_grad()

            probs = model(graphs, options)
            loss = reappearance_loss(probs, options, ground_truth_old)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graphs, options, ground_truth_old, k in validation_data:
                probs = model(graphs, options)
                loss = reappearance_loss(probs, options, ground_truth_old)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {total_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        # -----------------------------
        # EARLY STOPPING CHECK
        # -----------------------------
        if val_loss < best_val_loss - 1e-6:  # tiny tolerance
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

    # Restore best model state
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print("\nLearned parameters:")
    print("beta  =", F.softplus(model.beta))
    print("decay =", F.softplus(model.decay_factor))



# =====================================================
#  INFERENCE (sampling k nodes)
# =====================================================
def predict_k_nodes(model, graphs, options, k, top_k=False):
    """
    Weighted sampling using learned probabilities.
    """
    probs = model(graphs, options).detach().cpu().numpy()
    if top_k:
        top_indices = np.argsort(probs)[-k:][::-1]  # descending order
        selected = [options[i] for i in top_indices]
    else:
        selected = random.choices(options, weights=probs, k=k)
    return selected


def prepare_data(graphs, probabilities):
    train_percent = 0.70
    valid_percent = 0.15
    test_percent = 0.15
    
    data = []

    for i in range(2, len(graphs)):
        # Take previous num_back graphs as history
        history = [graphs[j][-1] for j in range(max(0, i - num_back), i)]

        # Candidate options = all nodes seen in history
        options = set()
        for g in history:
            options.update(g.nodes())
        options = list(options)

        # Ground truth = nodes in current graph
        targets = set(graphs[i][-1].nodes())

        # k value (can be used later for sampling)
        k = int(probabilities[i][0])

        data.append((history, options, targets, k))

    # Split into train/val/test
    n_total = len(data)
    n_train = int(train_percent * n_total)
    n_val   = int(valid_percent * n_total)

    training_data = data[:n_train]
    val_data      = data[n_train:n_train+n_val]
    test_data     = data[n_train+n_val:]

    return training_data, val_data, test_data



seed = 42

if __name__ == "__main__":
    my_loader = Loader()
    num_back = 5
    use_predicted = False 
    starting_graph = 2
    
    
    
    model_res_path = ""
    performance_path = ""
    
    # Load nodes and compute reappearance probabilities for different parameters
    # for dataset in ['networkadex', 'networkaion', 'networkbancor', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']:
    # for dataset in ['networkadex','CollegeMsg', ]:
    for dataset in ['CollegeMsg', 'networkaeternity', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'networkaragon', ]:
        probabilities_df = my_loader.load_data(type='probabilities', dataset=dataset, activation='', normalized=True, use_predicted=use_predicted, num_back=num_back)
        probabilities = probabilities_df.values.tolist()
        thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=False)
        graph_descriptions, _ = my_loader.load_data(dataset, activation='Degree', type='features', use_predicted=use_predicted, include_weights=False)
        graph_descriptions = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in graph_descriptions]
        target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)
        target_graphs, _ = modifyGraphIds(target_graphs, thresholds, num_back)
                
        training_data, val_data, test_data = prepare_data(target_graphs, probabilities)
        
        for top_k in [True]:
            print(f'Dataset: {dataset}')
            print(f'top_k is: {top_k}')
            model = ReappearanceModel(alpha=0, beta=3.0, decay_factor = 1.0)
            train_model(model, training_data, val_data, epochs=50, lr=1e-3, patience=5)