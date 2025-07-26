import pandas as pd 
import torch
import math
import os
import sys
import numpy as np
import pickle
from GraphGeneration.utils.casting_type import to_tensor
from GraphGeneration.models.temporal_gnn.script.config import args
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader


def load_data(dataset, strategy, embedding, mlpEncoding, embedOld, trainingStyle, embeddingType):
    my_loader = Loader()
    
    # Construct csv
    structure_pred_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/structure_pred.csv'
    structure_true_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/structure_true.csv'
    structure_diff_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/structure_diff.csv'
    kernel_pred_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/kernel_pred.csv'
    kernel_true_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/kernel_true.csv'
    edge_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/edge_analysis.csv'
    topER_file_path = f'GraphGeneration/output/results/topER/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/toper_diff.csv'
    animation_path = f'GraphGeneration/output/results/animations/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/pred_vs_true.mp4'

    # Create file paths if needed
    for path in [structure_pred_file_path, structure_true_file_path, structure_diff_file_path, kernel_pred_file_path, 
                kernel_true_file_path, edge_file_path, topER_file_path, animation_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    columns_structure = ['Graph Number', 'Average Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortivity Coefficient',
            'Clustering Coefficient', 'Density', 'Number of Weakly Connected Components',
            'Number of Strongly Connected Components', 'Number of Nodes', 'Number of Edges',
            'Eigenvalue_1', 'Eigenvalue_2', 'Eigenvalue_3', 'Eigenvalue_4', 'Eigenvalue_5', ]
    removed = ['Betweenness Centrality', 'Closeness Centrality', 'Number of Cliques', 'Diameter', 'Number of 3-Motifs',  'Number of Cycles', ]

    # Write the header and empty content
    pd.DataFrame(columns=columns_structure).to_csv(structure_pred_file_path, index=False)
    pd.DataFrame(columns=columns_structure).to_csv(structure_true_file_path, index=False)
    pd.DataFrame(columns=columns_structure + ['Kernel Distance']).to_csv(structure_diff_file_path, index=False)

#     columns_edges = ['Graph Number', 'precision overall', 'recall overall', 'tp_overall', 'fp_overall','tn_overall','fn_overall', 'precision oo', 'recall oo', 'tp_oo', 'fp_oo','tn_oo','fn_oo', 'precision oon', 'recall oon', 'tp_oon', 'fp_oon','tn_oon','fn_oon',  'precision on', 'recall on', 'tp_on', 'fp_on','tn_on','fn_on', 'precision nn', 'recall nn', 'tp_nn', 'fp_nn','tn_nn','fn_nn', 
#                         'Correct Node IDs', 'Correct Old Node IDs', 'Precision Old IDs', 'Recall Old IDs',  'Correct New Node IDs', 'Precision New IDs', 
#                         'Recall New IDs', 'Correct Overall IDs', 'Precision Overall IDs', 'Recall Overall IDs']

    columns_edges = ['Graph Number','precision oo', 'recall oo', 'tp_oo', 'fp_oo','tn_oo','fn_oo', 'precision oon', 'recall oon', 'tp_oon', 'fp_oon','tn_oon','fn_oon']

    # Write the header and empty content
    pd.DataFrame(columns=columns_edges).to_csv(edge_file_path, index=False)

    columns_kernel = [f"Graphlet{i}" for i in range(1, 22)]

    # Write the header and empty content
    pd.DataFrame(columns=columns_kernel).to_csv(kernel_pred_file_path, index=False)
    pd.DataFrame(columns=columns_kernel).to_csv(kernel_true_file_path, index=False)

    # Load probabilities
    probabilities_df = my_loader.load_data(type='probabilities', dataset=dataset, activation='')
    probabilities = probabilities_df.values.tolist()

    # Load all features, thresholds, and target subgraphs
    features, _ = my_loader.load_data(dataset, activation='Degree', type='features', include_weights=True)
    thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=True)
    target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)
    
    return probabilities, features, thresholds, target_graphs

def generate_negative_edges(G, num_samples, edge_type, edgebank=None):
    """
    For training the MLP, we need some negative edges that did not occur in the graph to predict
    
    Args:
        G (nx.DiGraph): The graph we are trying to generate samples on, we use its structure to check what edges dont exist
        num_samples (int): How many negative samples we want to create (we aim for equal amounts of positive and negative)
        edge_type (string): The type of edge we are attempting to generate negative samples for
        edgebank (dict):  A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        
    Returns:
        list(negatives) (list): A list of negative edges for training the MLP
    """
    all_nodes = list(G.nodes())
    negatives = set()
    
    # Remove if unnecessary
    max_attempts = 250000
    attempts = 0
    print(f'For edge type {edge_type}')
    while len(negatives) < num_samples and attempts < max_attempts:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        
        # Skip if u == v (self-loops not allowed)
        if u == v:
            continue
        
        # Filter edges based on edge_type
        if edge_type == 'o-o-bank':
            # This may stall, so there is a precaution to stop this
            if G.nodes[u]['feat']['type'] == 0 and G.nodes[v]['feat']['type'] == 0 and v in edgebank.get(u, []):
                if not G.has_edge(u, v):
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'o-o-nobank':
            if G.nodes[u]['feat']['type'] == 0 and G.nodes[v]['feat']['type'] == 0 and v not in edgebank.get(u, []):
                if not G.has_edge(u, v):
                    negatives.add((u, v))
            else:
                attempts += 1
                    
    negatives = list(negatives)
    if len(negatives) < num_samples:
        print(f"Only {len(negatives)} unique negative edges found for type {edge_type}, requested {num_samples}")

    return negatives

def generate_training_data(training_graphs, old_nodes, all_edgebanks, MAX_SAMPLES):
    sorted_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
            }  # A dict to sort embeddings for multiheaded MLP training
    
    # Generate embedding inputs and labels
    for i, graph in enumerate(training_graphs):    
        new_edges_count = {
            'o-o-bank': 0,
            'o-o-nobank': 0,
        }
        
        for edge_type in ['o-o-bank', 'o-o-nobank']:
            sorted_samples[edge_type]['X'].append([])
            sorted_samples[edge_type]['y'].append([])
        
        # Generate positive labels
        for u, v in graph.edges(data=False):
            try:
                edge_type = 'any'  # Default/fallback type

                # Determine edge type based on node categories and edgebank
                if u in old_nodes and v in old_nodes:
                    if v in all_edgebanks[i].get(u, []):
                        edge_type = 'o-o-bank'
                        new_edges_count[edge_type] += 1
                    else:
                        edge_type = 'o-o-nobank'
                        new_edges_count[edge_type] += 1

                # Store the sample
                if edge_type != "any":
                    sorted_samples[edge_type]['X'][i].append(torch.tensor([u, v]))
                    sorted_samples[edge_type]['y'][i].append(1)

            except Exception as e:
                print(f"[FATAL] Unexpected failure at outer loop for edge ({u}, {v}): {type(e).__name__} - {e}")
            
        # Generate an equal amount of negative labels for each type of edge
        negative_edges_oo = generate_negative_edges(graph, new_edges_count['o-o-bank'], edge_type='o-o-bank', edgebank=all_edgebanks[i])
        negative_edges_oon = generate_negative_edges(graph, new_edges_count['o-o-nobank'], edge_type='o-o-nobank', edgebank=all_edgebanks[i])

        tmp_samples_oo = [torch.tensor([u, v]) for u, v in negative_edges_oo]
        tmp_samples_oon = [torch.tensor([u, v]) for u, v in negative_edges_oon]
        
        # Add to our samples
        sorted_samples['o-o-bank']['X'][i].extend(tmp_samples_oo)
        sorted_samples['o-o-bank']['y'][i].extend([0 for _ in range(len(negative_edges_oo))])
        sorted_samples['o-o-nobank']['X'][i].extend(tmp_samples_oon)
        sorted_samples['o-o-nobank']['y'][i].extend([0 for _ in range(len(negative_edges_oon))])
        
        old_nodes.update(graph.nodes())  # Add the old nodes
    
    print(len(sorted_samples['o-o-bank']['X']))
    print(len(sorted_samples['o-o-bank']['X'][0]))
    
    return sorted_samples, new_edges_count


def generate_training_data_cached(training_graphs, old_nodes, all_edgebanks, MAX_SAMPLES, dataset, seed):
    cache_path = dataset + "_" + str(seed)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            print(f"Loading training data from cache: {cache_path}")
            return pickle.load(f)

    # Generate the data
    data = generate_training_data(training_graphs, old_nodes, all_edgebanks, MAX_SAMPLES)

    # Save it
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
        print(f"Saved training data to cache: {cache_path}")

    return data

def generate_validation_data(training_graphs, old_training_nodes, all_edgebanks, MAX_SAMPLES):
    sorted_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
            }  # A dict to sort embeddings for multiheaded MLP training
    
    # Generate embedding inputs and labels
    for i, graph in enumerate(training_graphs[1:]):  # Since we go one graph back for predictions  
        new_edges_count = {
            'o-o-bank': 0,
            'o-o-nobank': 0,
        }

        # Generate positive labels
        for u, v in graph.edges(data=False):
            try:
                edge_type = 'any'  # Default/fallback type

                # Determine edge type based on node categories and edgebank
                if u in old_training_nodes and v in old_training_nodes:
                    if v in all_edgebanks.get(u, []):
                        edge_type = 'o-o-bank'
                        new_edges_count[edge_type] += 1
                    else:
                        edge_type = 'o-o-nobank'
                        new_edges_count[edge_type] += 1

                # Store the sample
                if edge_type != "any":
                    sorted_samples[edge_type]['X'].append(torch.tensor([u, v]))
                    sorted_samples[edge_type]['y'].append(1)

            except Exception as e:
                print(f"[FATAL] Unexpected failure at outer loop for edge ({u}, {v}): {type(e).__name__} - {e}")
            
        # Generate an equal amount of negative labels for each type of edge
        negative_edges_oo = generate_negative_edges(graph, new_edges_count['o-o-bank'], edge_type='o-o-bank', edgebank=all_edgebanks)
        negative_edges_oon = generate_negative_edges(graph, new_edges_count['o-o-nobank'], edge_type='o-o-nobank', edgebank=all_edgebanks)

        tmp_samples_oo = [torch.tensor([u, v]) for u, v in negative_edges_oo]
        tmp_samples_oon = [torch.tensor([u, v]) for u, v in negative_edges_oon]
        
        # Add to our samples
        sorted_samples['o-o-bank']['X'].extend(tmp_samples_oo)
        sorted_samples['o-o-bank']['y'].extend([0 for _ in range(len(negative_edges_oo))])
        sorted_samples['o-o-nobank']['X'].extend(tmp_samples_oon)
        sorted_samples['o-o-nobank']['y'].extend([0 for _ in range(len(negative_edges_oon))])
    
    # If we need to remove some samples to prevent OOM crashes
    total_samples = sum(len(sorted_samples[key]['X']) for key in sorted_samples)
    if total_samples > MAX_SAMPLES:
        print(f"Total samples exceed {MAX_SAMPLES}. Truncating samples.")
        # Randomly sample to reduce to MAX_SAMPLES
        for edge_type in sorted_samples:
            num_samples_to_remove = total_samples - MAX_SAMPLES
            if num_samples_to_remove > 0:
                indices_to_remove = random.sample(range(len(sorted_samples[edge_type]['X'])), num_samples_to_remove)
                sorted_samples[edge_type]['X'] = [x for i, x in enumerate(sorted_samples[edge_type]['X']) if i not in indices_to_remove]
                sorted_samples[edge_type]['y'] = [y for i, y in enumerate(sorted_samples[edge_type]['y']) if i not in indices_to_remove]
                total_samples = sum(len(sorted_samples[key]['X']) for key in sorted_samples)
                
    return sorted_samples, new_edges_count

def generate_validation_data_cached(training_graphs, old_training_nodes, all_edgebanks, MAX_SAMPLES, dataset, seed, type_data):
    cache_path = dataset + "_" + type_data + "_" + str(seed)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            print(f"Loading training validation data from cache: {cache_path}")
            return pickle.load(f)

    # Generate the data
    data = generate_validation_data(training_graphs, old_training_nodes, all_edgebanks, MAX_SAMPLES)

    # Save it
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
        print(f"Saved validation data to cache: {cache_path}")

    return data