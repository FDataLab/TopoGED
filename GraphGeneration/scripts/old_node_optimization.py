import random
import numpy as np
import argparse
import os
import sys
from sklearn.metrics import recall_score
import torch
import pandas as pd
from collections import defaultdict

from sympy import use
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from GraphGeneration.utils.Evaluator import Evaluator
from nn.custom_model import Decoder

np.random.seed(42)
random.seed(42)
    

def compute_reappearance_probabilities_new(graphs, current_target_count_old_nodes, decay_factor, alpha, beta, epsilon=1e-10):
    nodes = dict()
    frequency = defaultdict(int)
    # Create the nodes dict degree history
    # nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing probabilities
    
    for t, G in enumerate(graphs):
        for node in G[-1].nodes():
            nodes[node] = (t, G[-1].degree(node))
            frequency[node] += 1
    
    max_degree = max(degree for _, (_, degree) in nodes.items())
    
    probs = {}
    t_curr = len(graphs)  # Makes the formula work best
    
    for node_id, (last_seen, degree) in nodes.items():
        recency_score = np.exp(-(t_curr - last_seen) / decay_factor) if decay_factor > 0 else 1.0
        degree_score = (degree / max_degree) ** alpha if max_degree > 0 else 1.0
        frequency_score = (frequency[node_id] ** beta) if beta > 0 else 1.0
        
        raw_score = recency_score * degree_score * frequency_score
        
        probs[node_id] = max(raw_score, epsilon)  # Apply epsilon floor to avoid exact 0

    # Normalize to make a valid probability distribution
    total = sum(probs.values())
    for node in probs:
        probs[node] /= total

    node_ids = list(probs.keys())
    weights = list(probs.values())

    sampled_old_nodes = list(np.random.choice(node_ids, size=current_target_count_old_nodes, replace=False, p=np.array(weights)/np.sum(weights)))  # Makes sure that we select only unique nodes each time
    
    return sampled_old_nodes
    
    
# def train_mlp_for_node_classification(encoder_model, train_data, val_data, test_data, input_dim=64, output_dim=1, hidden_2=32, num_layer=2, combo=['MLP'], dropout=0.1, epochs=250, batch_size=32):
#     # Make the MLP 
#     model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    
#     # Training



# def prepare_data_for_mlp(encoder_model, target_graphs, starting_graph, num_back):
#     train_part = 0.80
#     val_part = 0.10
#     test_part = 0.10
    
    
    
#     return train_data, val_data, test_data

def numpy_mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    index = np.argmax(counts)
    return values[index]


if __name__ == "__main__":
    my_loader = Loader()
    num_back = 5
    use_predicted = False 
    starting_graph = 2
    
    
    
    mlp_res_path = ""
    
    # Load nodes and compute reappearance probabilities for different parameters
    for dataset in ['networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']:
        probabilities_df = my_loader.load_data(type='probabilities', dataset=dataset, activation='', normalized=True, use_predicted=use_predicted, num_back=num_back)
        probabilities = probabilities_df.values.tolist()
        target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)
        graph_descriptions, _ = my_loader.load_data(dataset, activation='Degree', type='features', use_predicted=use_predicted, include_weights=False)
        graph_descriptions = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in graph_descriptions]
        
        equation_res_path = f"GraphGeneration/output/results/old_node_optimization/{dataset}_equation_results.csv"
        os.makedirs(os.path.dirname(equation_res_path), exist_ok=True)
        
        for decay_factor in [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0]:
            for alpha in [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0]:
                for beta in [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0]:
                    if decay_factor == 0 and alpha == 0 and beta == 0:
                        continue  # Skip trivial case
                                        
                    # Predicted nodes, and true nodes
                    preds = []
                    trues = []
                    
                    for i in range(starting_graph, len(probabilities)): 
                        current_target_snapshot = i
                                                                        
                        V_total = int(graph_descriptions[current_target_snapshot][-1][0])  # Used to convert probabilities
                        
                        # Get the true count of 4 edges type and number of new, old nodes of the target snapshot (probabilities are fed in as percents)
                        current_target_count_old_nodes = int(round(probabilities[current_target_snapshot][0] * V_total))
                    
                        sampled_old_nodes = compute_reappearance_probabilities_new(target_graphs[max(i - num_back, 0):i], current_target_count_old_nodes, decay_factor, alpha, beta)
                                            
                        # For later evaluation
                        preds.append(sampled_old_nodes)
                        trues.append(list(target_graphs[current_target_snapshot][-1].nodes()))
                    
                    # For results storage
                    precisions = []
                    recalls = []
                    f1s = [] 
                    
                    for pred_nodes, true_nodes in zip(preds, trues):
                        pred_set = set(pred_nodes)
                        true_set = set(true_nodes)

                        tp = len(pred_set & true_set)
                        fp = len(pred_set - true_set)
                        fn = len(true_set - pred_set)

                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

                        if precision + recall > 0:
                            f1 = 2 * precision * recall / (precision + recall)
                        else:
                            f1 = 0

                        precisions.append(precision)
                        recalls.append(recall)
                        f1s.append(f1)
                    
                    precisions = np.array(precisions)
                    recalls = np.array(recalls)
                    f1s = np.array(f1s)
                    
                    # Return mean, std, mode, min, max of precision, recall, accuracy, f1 and input variables
                    res = {
                        'dataset': dataset,
                        'decay_factor': decay_factor,
                        'alpha': alpha,
                        'beta': beta,
                        'mean_precision': np.mean(precisions),
                        'std_precision': np.std(precisions),
                        'mode_precision': numpy_mode(precisions),
                        'min_precision': np.min(precisions),
                        'max_precision': np.max(precisions),
                        'mean_recall': np.mean(recalls),
                        'std_recall': np.std(recalls),
                        'mode_recall': numpy_mode(recalls),
                        'min_recall': np.min(recalls),
                        'max_recall': np.max(recalls),
                        'mean_f1': np.mean(f1s),
                        'std_f1': np.std(f1s),
                        'mode_f1': numpy_mode(f1s),
                        'min_f1': np.min(f1s),
                        'max_f1': np.max(f1s)
                    }
                    
                    # Output res to the csv for analysis
                    write_header = not os.path.exists(equation_res_path)
                    pd.DataFrame([res]).to_csv(equation_res_path, mode='a', header=write_header, index=False) 
                
        # # Try the MLP training for node classification
        # for setup in ["GCN_binary_lr0.001"]:
            
            
            
        #     for lr in [0.01, 0.001, 0.0001]:
        #         encoder_path = f"data/input/cached/{dataset}/saved_data_gnn_{setup}/saved_models/embedder_1024"
        #         encoder_model = torch
                
        #         res = train_mlp_for_node_classification(encoder_model, train_data, val_data, test_data, epochs=250, batch_size=32)
                
        #         pd.DataFrame([res]).to_csv(mlp_res_path, mode='a', header=False, index=False) 