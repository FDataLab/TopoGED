import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import random
import torch
from collections import defaultdict

# --- NeurIPS Professional Styling ---
rcParams['font.family'] = 'serif'
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['figure.dpi'] = 300
sns.set_style("whitegrid", {'axes.grid': False})
line_color = '#0984e3'
cmap = "blues" # Professional, perceptually uniform palette

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.loader import Loader

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


if __name__ == "__main__":
    my_loader = Loader()
    num_back, n_trials, starting_graph = 5, 500, 2
    
    datasets = [
        'networkadex', 
        #'networkaion', 
        #'networkbancor', 
        #'networkcentra', 
        #'networkcoindash', 
        #'mathoverflow', 
        #'Reddit_B', 
        #'networkaragon', 
        #'networkaeternity', 
        #'networkiconomi', 
        #'CollegeMsg', 
        #'networkcindicator', 
        #'networkdgd', 
        #'tgbl-wiki'
        ]

    fixed_params = {
        'CollegeMsg': {'decay_factor': 0.31, 'alpha': 3.88, 'beta': 8.81},
        'mathoverflow': {'decay_factor': 0.15, 'alpha': 5.94, 'beta': 7.46},
        'networkadex': {'decay_factor': 0.85, 'alpha': 2.64, 'beta': 8.94},
        'networkaion': {'decay_factor': 0.60, 'alpha': 1.94, 'beta': 8.75},
        'networkaeternity': {'decay_factor': 0.53, 'alpha': 3.53, 'beta': 9.24},
        'networkaragon': {'decay_factor': 0.55, 'alpha': 3.53, 'beta': 9.99},
        'networkbancor': {'decay_factor': 0.64, 'alpha': 4.19, 'beta': 9.51},
        'networkcentra': {'decay_factor': 0.35, 'alpha': 3.25, 'beta': 9.78},
        'networkcoindash': {'decay_factor': 0.63, 'alpha': 2.97, 'beta': 9.47},
        'networkiconomi': {'decay_factor': 0.37, 'alpha': 2.94, 'beta': 9.70},
        'networkcindicator': {'decay_factor': 0.51, 'alpha': 3.74, 'beta': 9.24},
        'networkdgd': {'decay_factor': 0.31, 'alpha': 1.32, 'beta': 9.96},
        'Reddit_B': {'decay_factor': 0.96, 'alpha': 3.92, 'beta': 9.96},
        'tgbl-wiki': {'decay_factor': 0.21, 'alpha': 4.67, 'beta': 9.32},
    }
    
    # Setup
    output_dir = "paper_plots"
    os.makedirs(output_dir, exist_ok=True)


    custom_cmap = 'Blues' 
    line_color = '#0984e3' # A professional, high-contrast blue

    for dataset in datasets:
        print(f'Starting dataset: {dataset}')
        probs_df = my_loader.load_data(type='probabilities', activation='Degree', dataset=dataset, num_back=num_back)
        probs = probs_df.values.tolist()
        thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds')
        g_desc, _ = my_loader.load_data(dataset, activation='Degree', type='features')
        g_desc = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in g_desc]
        raw_target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs')
        target_graphs_flat = [bucket[-1] for bucket in raw_target_graphs]
        
        base = fixed_params[dataset]

        # ---------------------------------------------------------
        # 1. 1D LINE PLOTS (Vary one, Fix two)
        # ---------------------------------------------------------
        # param_configs = [
        #     ('decay_factor', np.arange(0.01, 1.01, 0.01), r'Decay Factor ($\lambda$)'),
        #     ('alpha', np.arange(0.0, 10.01, 0.1), r'Alpha ($\alpha$)'),
        #     ('beta', np.arange(0.0, 15.01, 0.1), r'Beta ($\beta$)')
        # ]

        # for var_name, space, label in param_configs:
        #     f1_scores = []
        #     for val in space:
        #         current_params = base.copy()
        #         current_params[var_name] = val
        #         _, _, test_f1 = evaluate_best_params(current_params, target_graphs_flat, g_desc, probs, starting_graph, num_back)
        #         f1_scores.append(test_f1)
            
        #     plt.figure(figsize=(5, 4))
        #     # CHANGED: Color set to blue
        #     plt.plot(space, f1_scores, color=line_color, lw=2.5, label='Test F1')
        #     plt.axvline(base[var_name], color='black', linestyle='--', alpha=0.5, label='Selected Value')
        #     plt.xlabel(label)
        #     plt.ylabel('F1 Score')
        #     plt.title(f'Sensitivity of {label}')
        #     plt.legend()
        #     plt.savefig(f"{output_dir}/{dataset}_1D_{var_name}.pdf", bbox_inches='tight')
        #     plt.close()

        # ---------------------------------------------------------
        # 2. 2D HEATMAPS (Vary two, Fix one)
        # ---------------------------------------------------------
        heatmap_configs = [
            ('alpha', 'beta', np.arange(0, 10.01, 0.2), np.arange(0, 15.01, 0.2), r'$\alpha$', r'$\beta$'),
            ('beta', 'decay_factor', np.arange(0, 15.01, 0.2), np.arange(0.01, 1.01, 0.02), r'$\beta$', r'$\lambda$'),
            ('alpha', 'decay_factor', np.arange(0, 10.01, 0.2), np.arange(0.01, 1.01, 0.02), r'$\alpha$', r'$\lambda$')
        ]

        for x_name, y_name, x_space, y_space, x_label, y_label in heatmap_configs:
            print(f"  > Plotting {x_name} vs {y_name}...")
            z = np.zeros((len(y_space), len(x_space)))
            
            for i, x_val in enumerate(x_space):
                for j, y_val in enumerate(y_space):
                    current_params = base.copy()
                    current_params[x_name] = x_val
                    current_params[y_name] = y_val
                    _, _, test_f1 = evaluate_best_params(current_params, target_graphs_flat, g_desc, probs, starting_graph, num_back)
                    z[j, i] = test_f1

            plt.figure(figsize=(6, 5))
            # CHANGED: cmap set to custom_cmap (shades of blue)
            cp = plt.contourf(x_space, y_space, z, levels=20, cmap=custom_cmap)
            plt.colorbar(cp, label='Test F1 Score')
            
            # CHANGED: Marker color changed to a bright cyan or yellow to stand out against dark blue
            plt.scatter(base[x_name], base[y_name], color='#00d2d3', marker='*', s=150, edgecolors='black', label='Optimal')
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f'Interaction: {x_label} and {y_label}')
            plt.legend()
            plt.savefig(f"{output_dir}/{dataset}_2D_{x_name}_{y_name}.pdf", bbox_inches='tight')
            plt.close()

    print(f"\nAll plots saved to: {output_dir}")