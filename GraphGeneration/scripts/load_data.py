import pandas as pd 
import torch
import os
import sys
import pickle
import numpy as np
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader


def load_data(dataset, embedding, mlpEncoding, embeddingType, num_back, use_predicted=False, num_buckets=10, use_test_style=None):
    my_loader = Loader()
    output_dir = os.path.abspath(f'data/input/cached/{dataset}')
    
    # Create cached file path for cached model, cached data training
    cached_model_dataset_folder = os.path.join(output_dir, 'saved_models/')
    cached_data_dataset_folder = os.path.join(output_dir, 'saved_data/')

    # Construct output evaluation csv
#     structure_pred_file_path = f'GraphGeneration/output/results/structure/{dataset}/topoGED_embedding{embedding}_mlpEncoding{mlpEncoding}_embeddingType{embeddingType}/structure_pred.csv'
#     structure_true_file_path = f'GraphGeneration/output/results/structure/{dataset}/topoGED_embedding{embedding}_mlpEncoding{mlpEncoding}_embeddingType{embeddingType}/structure_true.csv'
#     structure_diff_file_path = f'GraphGeneration/output/results/structure/{dataset}/topoGED_embedding{embedding}_mlpEncoding{mlpEncoding}_embeddingType{embeddingType}/structure_diff.csv'
#     kernel_pred_file_path = f'GraphGeneration/output/results/kernel/{dataset}/topoGED_embedding{embedding}_mlpEncoding{mlpEncoding}_embeddingType{embeddingType}/kernel_pred.csv'
#     kernel_true_file_path = f'GraphGeneration/output/results/kernel/{dataset}/topoGED_embedding{embedding}_mlpEncoding{mlpEncoding}_embeddingType{embeddingType}/kernel_true.csv'
#     edge_file_path = f'GraphGeneration/output/results/structure/{dataset}/topoGED_embedding{embedding}_mlpEncoding{mlpEncoding}_embeddingType{embeddingType}/edge_analysis.csv'
#     topER_file_path = f'GraphGeneration/output/results/topER/{dataset}/topoGED_embedding{embedding}_mlpEncoding{mlpEncoding}_embeddingType{embeddingType}/toper_diff.csv'
#     animation_path = f'GraphGeneration/output/results/animations/{dataset}/topoGED_embedding{embedding}_mlpEncoding{mlpEncoding}_embeddingType{embeddingType}/pred_vs_true.mp4'

#     # Create file paths if needed
#     for path in [structure_pred_file_path, structure_true_file_path, structure_diff_file_path, kernel_pred_file_path, 
#                 kernel_true_file_path, edge_file_path, topER_file_path, animation_path, cached_model_dataset_folder, cached_data_dataset_folder]:
#         os.makedirs(os.path.dirname(path), exist_ok=True)

#     columns_structure = ['Graph Number', 'Average Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortivity Coefficient',
#             'Clustering Coefficient', 'Density', 'Number of Weakly Connected Components',
#             'Number of Strongly Connected Components', 'Number of Nodes', 'Number of Edges',
#             'Eigenvalue_1', 'Eigenvalue_2', 'Eigenvalue_3', 'Eigenvalue_4', 'Eigenvalue_5', ]
#     removed = ['Betweenness Centrality', 'Closeness Centrality', 'Number of Cliques', 'Diameter', 'Number of 3-Motifs',  'Number of Cycles', ]

#     # Write the header and empty content
#     pd.DataFrame(columns=columns_structure).to_csv(structure_pred_file_path, index=False)
#     pd.DataFrame(columns=columns_structure).to_csv(structure_true_file_path, index=False)
#     pd.DataFrame(columns=columns_structure + ['Kernel Distance']).to_csv(structure_diff_file_path, index=False)

# #     columns_edges = ['Graph Number', 'precision overall', 'recall overall', 'tp_overall', 'fp_overall','tn_overall','fn_overall', 'precision oo', 'recall oo', 'tp_oo', 'fp_oo','tn_oo','fn_oo', 'precision oon', 'recall oon', 'tp_oon', 'fp_oon','tn_oon','fn_oon',  'precision on', 'recall on', 'tp_on', 'fp_on','tn_on','fn_on', 'precision nn', 'recall nn', 'tp_nn', 'fp_nn','tn_nn','fn_nn', 
# #                         'Correct Node IDs', 'Correct Old Node IDs', 'Precision Old IDs', 'Recall Old IDs',  'Correct New Node IDs', 'Precision New IDs', 
# #                         'Recall New IDs', 'Correct Overall IDs', 'Precision Overall IDs', 'Recall Overall IDs']

#     columns_edges = ['Graph Number','precision oo', 'recall oo', 'tp_oo', 'fp_oo','tn_oo','fn_oo', 'precision oon', 'recall oon', 'tp_oon', 'fp_oon','tn_oon','fn_oon']

#     # Write the header and empty content
#     pd.DataFrame(columns=columns_edges).to_csv(edge_file_path, index=False)

#     columns_kernel = [f"Graphlet{i}" for i in range(1, 22)]

#     # Write the header and empty content
#     pd.DataFrame(columns=columns_kernel).to_csv(kernel_pred_file_path, index=False)
#     pd.DataFrame(columns=columns_kernel).to_csv(kernel_true_file_path, index=False)

    print(f'Loading data for {dataset} with use_predicted={use_predicted}, num_back={num_back}, num_buckets={num_buckets}, use_test_style={use_test_style}')
    # Load probabilities
    if use_predicted:
        print(f'USING TEST WITH {use_test_style} FOR LOADING PROBS AND TOPER')
        with open(f'data/input/cached/{dataset}/predValues/{dataset}_testprobs_{use_test_style}_Raw_10.pkl', 'rb') as f:
            probabilities = pickle.load(f)
    else:
        print(f'USING TRUE VALS FOR LOADING PROBS AND TOPER')
        probabilities = my_loader.load_data(type='probabilities', dataset=dataset, activation='', normalized=True, use_predicted=use_predicted, num_back=num_back)

        

    # FIX: Only call .values.tolist() if it's actually a DataFrame/Series
    if not use_predicted:
        if hasattr(probabilities, 'values'):
            probabilities = probabilities.values.tolist()
        # If it's already a list, we don't need to do anything.
        elif isinstance(probabilities, list):
            pass 
        else:
            # Optional: handle numpy arrays if they appear
            probabilities = np.array(probabilities).tolist()

    # Load all features, thresholds, and target subgraphs
    if use_predicted:
        with open(f'data/input/cached/{dataset}/predValues/{dataset}_testtoper_{use_test_style}_Raw_{num_buckets}.pkl', 'rb') as f:
            features = pickle.load(f)
    else:
        features, _ = my_loader.load_data(dataset, activation='Degree', type='features', use_predicted=use_predicted, include_weights=False, num_buckets=num_buckets)

        
    thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=False, num_buckets=num_buckets)
    target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False, num_buckets=num_buckets)
    
    return probabilities, features, thresholds, target_graphs



def generate_negative_edges(G, num_samples, edge_type, old_nodes, is_directed, edgebank=None):
    all_nodes = list(G.nodes())
    new_nodes = list(set(all_nodes) - set(old_nodes))
    old_nodes_list = list(old_nodes)
    negatives = set()
    
    G_edges = set(G.edges())
    eb_lookup = edgebank if edgebank is not None else {}

    # Safety to prevent infinite loops if the graph is nearly complete
    max_attempts = num_samples * 150
    attempts = 0

    while len(negatives) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # 1. Targeted Sampling based on edge_type
        # We pick nodes from the correct pools immediately rather than filtering later
        if edge_type == 'o-o-bank' or edge_type == 'o-o-nobank':
            if len(old_nodes_list) < 2: continue
            u, v = random.sample(old_nodes_list, 2)
            
            if edge_type == 'o-o-bank' and v not in eb_lookup.get(u, set()):
                continue
            if edge_type == 'o-o-nobank' and v in eb_lookup.get(u, set()):
                continue
                
        elif edge_type == 'n-n':
            if len(new_nodes) < 2: continue
            u, v = random.sample(new_nodes, 2)
            
        elif edge_type == 'o-n':
            if not old_nodes_list or not new_nodes: continue
            # Randomly decide direction if directed, otherwise just pick one from each
            u = random.choice(old_nodes_list)
            v = random.choice(new_nodes)
            if is_directed and random.random() > 0.5:
                u, v = v, u

        # 2. Validity Check
        if (u, v) in G_edges:
            continue
        if not is_directed and (v, u) in G_edges:
            continue
            
        negatives.add((u, v))

    if len(negatives) < num_samples:
        print(f"Only {len(negatives)} unique negative edges found for type {edge_type}")

    return list(negatives)


def generate_training_data(training_graphs, all_edgebanks, days_back, is_directed):
    sorted_samples = {
        'o-o-bank': {'X': [], 'y': []},
        'o-o-nobank': {'X': [], 'y': []},
        'o-n': {'X': [], 'y': []},
        'n-n': {'X': [], 'y': []},
        }  # A dict to sort embeddings for multiheaded MLP training
    
    # Generate embedding inputs and labels
    for i, graph in enumerate(training_graphs): 
        # Old nodes of 5 days before 
        old_nodes_days = set().union(*[g.nodes() for g in training_graphs[max(i - days_back, 0): i]]) 
        new_edges_count = {
            'o-o-bank': 0,
            'o-o-nobank': 0,
            'o-n': 0,
            'n-n': 0,
        }
        
        for edge_type in ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']:
            sorted_samples[edge_type]['X'].append([])
            sorted_samples[edge_type]['y'].append([])
        
        # Generate positive labels
        for u, v in graph.edges(data=False):
            try:
                edge_type = 'any'  # Default/fallback type

                # Determine edge type based on node categories and edgebank
                if u in old_nodes_days and v in old_nodes_days:
                    
                    if v in all_edgebanks[i].get(u, set()):
                        edge_type = 'o-o-bank'
                    else:  # We haven't seen this edge in the edgebank
                        edge_type = 'o-o-nobank'
                elif (u not in old_nodes_days and v in old_nodes_days) or (u in old_nodes_days and v not in old_nodes_days):
                    edge_type = 'o-n'
                elif u not in old_nodes_days and v not in old_nodes_days:
                    edge_type = 'n-n'
                    

                # Store the sample
                if edge_type != "any":
                    new_edges_count[edge_type] += 1
                    sorted_samples[edge_type]['X'][-1].append(torch.tensor([u, v]))
                    sorted_samples[edge_type]['y'][-1].append(1)
                    if not is_directed:
                        sorted_samples[edge_type]['X'][-1].append(torch.tensor([v, u]))
                        sorted_samples[edge_type]['y'][-1].append(1)
                        
            except Exception as e:
                print(f"[FATAL] Unexpected failure at outer loop for edge ({u}, {v}): {type(e).__name__} - {e}")
            
        # Generate negative samples for each edge type
        for edge_type in ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']:
            negative_edges = generate_negative_edges(
                graph,
                num_samples=new_edges_count[edge_type],
                edge_type=edge_type,
                old_nodes=old_nodes_days,
                is_directed=is_directed,
                edgebank=all_edgebanks[i]
            )

            tmp_samples = [torch.tensor([u, v]) for u, v in negative_edges]

            # Add negative samples to the corresponding lists
            sorted_samples[edge_type]['X'][-1].extend(tmp_samples)
            sorted_samples[edge_type]['y'][-1].extend([0] * len(negative_edges))

    
    return sorted_samples


def generate_training_data_cached(training_graphs, all_edgebanks, days_back, MAX_SAMPLES, dataset, seed, is_directed, saved_data_file_path):
    dir_string = 'Directed' if is_directed else 'Undirected'
    cache_path = saved_data_file_path + "/" + dataset + "_" + str(days_back) + "days_" + str(seed) + "_" + dir_string
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            print(f"Loading training data from cache: {cache_path}")
            return pickle.load(f)

    # Generate the data
    data = generate_training_data(training_graphs, all_edgebanks, days_back, is_directed)

    # Save it
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
        print(f"Saved training data to cache: {cache_path}")

    return data


def generate_validation_data(training_graphs, days_back, is_directed, all_edgebanks, MAX_SAMPLES):
    sorted_samples = {
        'o-o-bank': {'X': [], 'y': []},
        'o-o-nobank': {'X': [], 'y': []},
        'o-n': {'X': [], 'y': []},
        'n-n': {'X': [], 'y': []},
        }  # A dict to sort embeddings for multiheaded MLP training
    
    # Generate embedding inputs and labels
    for i, graph in enumerate(training_graphs):
        # Skip the first `days_back` graphs for sample generation
        if i < days_back:
            continue

        # Old nodes of the last `days_back` graphs
        old_nodes_days = set().union(*[g.nodes() for g in training_graphs[max(i - days_back, 0): i]])
        
        new_edges_count = {
            'o-o-bank': 0,
            'o-o-nobank': 0,
            'o-n': 0,
            'n-n': 0,
        }
        
        for edge_type in ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']:
            sorted_samples[edge_type]['X'].append([])
            sorted_samples[edge_type]['y'].append([])

        # Generate positive labels
        for u, v in graph.edges(data=False):
            try:
                edge_type_label = 'any'

                if u in old_nodes_days and v in old_nodes_days:
                    if v in all_edgebanks[max(i - days_back, 0)].get(u, set()):
                        edge_type_label = 'o-o-bank'
                    else:
                        edge_type_label = 'o-o-nobank'
                elif (u in old_nodes_days) != (v in old_nodes_days):
                    edge_type_label = 'o-n'
                elif u not in old_nodes_days and v not in old_nodes_days:
                    edge_type_label = 'n-n'

                if edge_type_label != 'any':
                    new_edges_count[edge_type_label] += 1
                    sorted_samples[edge_type_label]['X'][-1].append(torch.tensor([u, v]))
                    sorted_samples[edge_type_label]['y'][-1].append(1)
                    if not is_directed:
                        sorted_samples[edge_type_label]['X'][-1].append(torch.tensor([v, u]))
                        sorted_samples[edge_type_label]['y'][-1].append(1)

            except Exception as e:
                print(f"[FATAL] Unexpected failure at outer loop for edge ({u}, {v}): {type(e).__name__} - {e}")
    
        # Generate negative samples for each edge type
        for edge_type in ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']:
            negative_edges = generate_negative_edges(
                graph,
                num_samples=new_edges_count[edge_type],
                edge_type=edge_type,
                old_nodes=old_nodes_days,
                is_directed=is_directed,
                edgebank=all_edgebanks[max(i - days_back, 0)]
            )

            tmp_samples = [torch.tensor([u, v]) for u, v in negative_edges]

            # Add negative samples to the corresponding lists
            sorted_samples[edge_type]['X'][-1].extend(tmp_samples)
            sorted_samples[edge_type]['y'][-1].extend([0] * len(negative_edges))
    
    return sorted_samples


def generate_validation_data_cached(training_graphs, days_back, all_edgebanks, MAX_SAMPLES, dataset, seed, is_directed, type_data, saved_data_file_path):
    dir_string = 'Directed' if is_directed else 'Undirected'
    cache_path = saved_data_file_path + "/" + dataset + "_" + type_data + "_" + str(days_back) + "_days" + str(seed) + "_" + dir_string
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            print(f"Loading training validation data from cache: {cache_path}")
            return pickle.load(f)

    # Generate the data
    data = generate_validation_data(training_graphs, days_back, is_directed, all_edgebanks, MAX_SAMPLES)

    # Save it
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
        print(f"Saved validation data to cache: {cache_path}")

    return data