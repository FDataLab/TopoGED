from json import encoder
from math import e
import random
import numpy as np
import argparse
import os
import sys
import optuna
import torch
import matplotlib
matplotlib.use("Agg")
from collections import defaultdict
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader


import os
import math
import random
import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
    

def compute_reappearance_probabilities_new(graphs, current_target_count_old_nodes, decay_factor, alpha, beta, snapshot_seed, epsilon=1e-10):
    # FIXED: Added snapshot_seed to the signature and implemented local_rng
    local_rng = np.random.RandomState(snapshot_seed)
    
    nodes = dict()
    frequency = defaultdict(int)
    
    for t, G in enumerate(graphs):
        # Extract networkx object if it's inside a bucket list
        actual_G = G[-1] if isinstance(G, list) else G
        for node in actual_G.nodes():
            nodes[node] = (t, actual_G.degree(node))
            frequency[node] += 1
    
    if not nodes:
        return []

    max_degree = max(degree for _, (_, degree) in nodes.items())
    t_curr = len(graphs)
    
    node_ids = list(nodes.keys())
    scores = []
    
    for node_id in node_ids:
        last_seen, degree = nodes[node_id]
        recency_score = np.exp(-(t_curr - last_seen) / (decay_factor + epsilon))
        degree_score = (degree / max_degree) ** alpha if max_degree > 0 else 1.0
        frequency_score = (frequency[node_id] ** beta)
        
        scores.append(max(recency_score * degree_score * frequency_score, epsilon))

    # Normalize to make a valid probability distribution
    weights = np.array(scores)
    weights /= weights.sum()

    # Safety check for sample size vs population size
    sample_size = min(len(node_ids), current_target_count_old_nodes)
    
    # Use local_rng instead of np.random to ensure Optuna-friendly determinism
    sampled_old_nodes = local_rng.choice(node_ids, size=sample_size, replace=False, p=weights)
    
    return sampled_old_nodes.tolist()


def evaluate_best_params(params, target_graphs_flat, graph_descriptions, probabilities, starting_graph, num_back):
    """ Helper to run the simulation once and split results by Train/Val/Test """
    f1s = []
    for i in range(starting_graph, len(probabilities)):
        v_total = int(graph_descriptions[i][-1][0])
        target_count = int(round(probabilities[i][0] * v_total))
        history = target_graphs_flat[max(i - num_back, 0):i]
        
        sampled = compute_reappearance_probabilities_new(
            history, target_count, 
            params['decay_factor'], params['alpha'], params['beta'], 
            snapshot_seed=i
        )
        
        pred_set = set(sampled)
        true_set = set(target_graphs_flat[i].nodes())
        tp = len(pred_set & true_set)
        p = tp / len(pred_set) if len(pred_set) > 0 else 0
        r = tp / len(true_set) if len(true_set) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1s.append(f1)
    
    # Calculate Split Indices
    total_len = len(f1s)
    idx_train = int(0.7 * total_len)
    idx_val = int(0.85 * total_len)
    
    train_f1 = np.mean(f1s[:idx_train]) if f1s[:idx_train] else 0
    val_f1 = np.mean(f1s[idx_train:idx_val]) if f1s[idx_train:idx_val] else 0
    test_f1 = np.mean(f1s[idx_val:]) if f1s[idx_val:] else 0
    
    return train_f1, val_f1, test_f1

def objective(trial, dataset, target_graphs_flat, graph_descriptions, probabilities, starting_graph, num_back):
    decay_factor = trial.suggest_float("decay_factor", 0.0, 1.0)
    alpha = trial.suggest_float("alpha", 0.0, 10.0)
    beta = trial.suggest_float("beta", 0.0, 10.0)
    # 0.15, 0.21 smth like that
    f1s = []
    for i in range(starting_graph, len(probabilities)):
        v_total = int(graph_descriptions[i][-1][0])
        target_count = int(round(probabilities[i][0] * v_total))
        history = target_graphs_flat[max(i - num_back, 0):i]
        sampled = compute_reappearance_probabilities_new(history, target_count, decay_factor, alpha, beta, snapshot_seed=i)
        
        pred_set, true_set = set(sampled), set(target_graphs_flat[i].nodes())
        tp = len(pred_set & true_set)
        p, r = (tp / len(pred_set) if len(pred_set) > 0 else 0), (tp / len(true_set) if len(true_set) > 0 else 0)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        
    return np.mean(f1s)
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    my_loader = Loader()
    num_back, n_trials, starting_graph = 5, 500, 2
    
    datasets = ['networkadex', 'networkaion', 'networkbancor', 'networkcentra', 'networkcoindash', 
                'mathoverflow', 'Reddit_B', 'networkaragon', 'networkaeternity', 'networkiconomi', 
                'CollegeMsg', 'networkcindicator', 'networkdgd', 'tgbl-wiki']

    for dataset in datasets:
        print(f"\n--- Optuna Study: {dataset} ---")
        
        # 1. Load and Flatten Data
        probs_df = my_loader.load_data(type='probabilities', activation='Degree', dataset=dataset, num_back=num_back)
        probs = probs_df.values.tolist()
        thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds')
        g_desc, _ = my_loader.load_data(dataset, activation='Degree', type='features')
        g_desc = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in g_desc]
        raw_target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs')
        target_graphs_flat = [bucket[-1] for bucket in raw_target_graphs]

        # 2. Run Optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, dataset, target_graphs_flat, g_desc, probs, starting_graph, num_back), n_trials=n_trials)

        # 3. Final Split Evaluation
        train_f1, val_f1, test_f1 = evaluate_best_params(study.best_params, target_graphs_flat, g_desc, probs, starting_graph, num_back)

        print(f"DONE: {dataset}")
        print(f"  > Best Params: {study.best_params}")
        print(f"  > Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
        print("-" * 40)

        # Save results
        res_df = study.trials_dataframe()
        res_path = f"GraphGeneration/output/results/old_node_optimization/{dataset}_optuna_search.csv"
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        res_df.to_csv(res_path, index=False)