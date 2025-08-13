import pandas as pd 
import torch
import os
import sys
import pickle
from GraphGeneration.models.temporal_gnn.script.config import args
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader


def load_data(dataset, strategy, embedding, mlpEncoding, embedOld, trainingStyle, embeddingType):
    my_loader = Loader()
    output_dir = os.path.abspath(f'data/input/cached/{dataset}')
    
    # Create cached file path for cached model, cached data training
    cached_model_dataset_folder = os.path.join(output_dir, 'saved_models/')
    cached_data_dataset_folder = os.path.join(output_dir, 'saved_data/')

    # Construct output evaluation csv
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
                kernel_true_file_path, edge_file_path, topER_file_path, animation_path, cached_model_dataset_folder, cached_data_dataset_folder]:
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

def generate_negative_edges(G, num_samples, edge_type, old_nodes, edgebank=None):
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
            if u in old_nodes and v in old_nodes and v in edgebank.get(u, []):
                if not G.has_edge(u, v):
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'o-o-nobank':
            if u in old_nodes and v in old_nodes and v not in edgebank.get(u, []):
                if not G.has_edge(u, v):
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'n-n':
            if u not in old_nodes and v not in old_nodes:
                if not G.has_edge(u, v) and (u, v) not in negatives:
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'o-n':
            if (u in old_nodes and v not in old_nodes) or (u not in old_nodes and v in old_nodes):
                if not G.has_edge(u, v) and (u, v) not in negatives:
                    negatives.add((u, v))
            else:
                attempts += 1
                    
    negatives = list(negatives)
    if len(negatives) < num_samples:
        print(f"Only {len(negatives)} unique negative edges found for type {edge_type}, requested {num_samples}")

    return negatives

def generate_training_data(training_graphs, all_edgebanks):
    sorted_samples = {
        'o-o-bank': {'X': [], 'y': []},
        'o-o-nobank': {'X': [], 'y': []},
        'o-n': {'X': [], 'y': []},
        'n-n': {'X': [], 'y': []},
        }  # A dict to sort embeddings for multiheaded MLP training
    
    old_nodes = set()
    
    # Generate embedding inputs and labels
    for i, graph in enumerate(training_graphs):    
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
                if u in old_nodes and v in old_nodes:
                    if v in all_edgebanks[i].get(u, []):
                        edge_type = 'o-o-bank'
                        new_edges_count[edge_type] += 1
                    else:
                        edge_type = 'o-o-nobank'
                        new_edges_count[edge_type] += 1
                elif (u in old_nodes and v not in old_nodes):
                    edge_type = 'o-n'
                    new_edges_count[edge_type] += 1
                elif (u not in old_nodes and v in old_nodes):
                    edge_type = 'o-n'
                    new_edges_count[edge_type] += 1
                    u, v = v, u
                elif u not in old_nodes and v not in old_nodes:
                    edge_type = 'n-n'
                    new_edges_count[edge_type] += 1

                # Store the sample
                if edge_type != "any":
                    sorted_samples[edge_type]['X'][i].append(torch.tensor([u, v]))
                    sorted_samples[edge_type]['y'][i].append(1)

            except Exception as e:
                print(f"[FATAL] Unexpected failure at outer loop for edge ({u}, {v}): {type(e).__name__} - {e}")
            
        # Generate an equal amount of negative labels for each type of edge
        negative_edges_oo = generate_negative_edges(graph, new_edges_count['o-o-bank'], edge_type='o-o-bank', edgebank=all_edgebanks[i], old_nodes=old_nodes)
        negative_edges_oon = generate_negative_edges(graph, new_edges_count['o-o-nobank'], edge_type='o-o-nobank', edgebank=all_edgebanks[i], old_nodes=old_nodes)
        negative_edges_on = generate_negative_edges(graph, new_edges_count['o-n'], edge_type='o-n', edgebank=all_edgebanks[i], old_nodes=old_nodes)
        negative_edges_nn = generate_negative_edges(graph, new_edges_count['n-n'], edge_type='n-n', edgebank=all_edgebanks[i], old_nodes=old_nodes)

        tmp_samples_oo = [torch.tensor([u, v]) for u, v in negative_edges_oo]
        tmp_samples_oon = [torch.tensor([u, v]) for u, v in negative_edges_oon]
        tmp_samples_on = [torch.tensor([u, v]) for u, v in negative_edges_on]
        tmp_samples_nn = [torch.tensor([u, v]) for u, v in negative_edges_nn]
        
        # Add to our samples
        sorted_samples['o-o-bank']['X'][i].extend(tmp_samples_oo)
        sorted_samples['o-o-bank']['y'][i].extend([0 for _ in range(len(negative_edges_oo))])
        sorted_samples['o-o-nobank']['X'][i].extend(tmp_samples_oon)
        sorted_samples['o-o-nobank']['y'][i].extend([0 for _ in range(len(negative_edges_oon))])
        sorted_samples['o-n']['X'][i].extend(tmp_samples_on)
        sorted_samples['o-n']['y'][i].extend([0 for _ in range(len(negative_edges_on))])
        sorted_samples['n-n']['X'][i].extend(tmp_samples_nn)
        sorted_samples['n-n']['y'][i].extend([0 for _ in range(len(negative_edges_nn))])
        
        old_nodes.update(graph.nodes())  # Add the old nodes
    
    print(len(sorted_samples['o-o-bank']['X']))
    print(len(sorted_samples['o-o-bank']['X'][0]))
    
    return sorted_samples, new_edges_count


def generate_training_data_cached(training_graphs, all_edgebanks, MAX_SAMPLES, dataset, seed, saved_data_file_path):
    cache_path = saved_data_file_path + "/" + dataset + "_" + str(seed)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            print(f"Loading training data from cache: {cache_path}")
            return pickle.load(f)

    # Generate the data
    data = generate_training_data(training_graphs, all_edgebanks)

    # Save it
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
        print(f"Saved training data to cache: {cache_path}")

    return data

def generate_validation_data(training_graphs, old_training_nodes, all_edgebanks, MAX_SAMPLES):
    sorted_samples = {
        'o-o-bank': {'X': [], 'y': []},
        'o-o-nobank': {'X': [], 'y': []},
        'o-n': {'X': [], 'y': []},
        'n-n': {'X': [], 'y': []},
        }  # A dict to sort embeddings for multiheaded MLP training
    
    # Generate embedding inputs and labels
    for i, graph in enumerate(training_graphs):  # Since we go one graph back for predictions  
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
                if u in old_training_nodes and v in old_training_nodes:
                    if v in all_edgebanks.get(u, []):
                        edge_type = 'o-o-bank'
                        new_edges_count[edge_type] += 1
                    else:
                        edge_type = 'o-o-nobank'
                        new_edges_count[edge_type] += 1
                elif (u in old_training_nodes and v not in old_training_nodes) or (u not in old_training_nodes and v in old_training_nodes):
                    edge_type = 'o-n'
                    new_edges_count[edge_type] += 1
                elif u not in old_training_nodes and v not in old_training_nodes:
                    edge_type = 'n-n'
                    new_edges_count[edge_type] += 1

                # Store the sample
                if edge_type != "any":
                    sorted_samples[edge_type]['X'][i].append(torch.tensor([u, v]))
                    sorted_samples[edge_type]['y'][i].append(1)

            except Exception as e:
                print(f"[FATAL] Unexpected failure at outer loop for edge ({u}, {v}): {type(e).__name__} - {e}")
            
        # Generate an equal amount of negative labels for each type of edge
        negative_edges_oo = generate_negative_edges(graph, new_edges_count['o-o-bank'], edge_type='o-o-bank', edgebank=all_edgebanks, old_nodes=old_training_nodes)
        negative_edges_oon = generate_negative_edges(graph, new_edges_count['o-o-nobank'], edge_type='o-o-nobank', edgebank=all_edgebanks, old_nodes=old_training_nodes)
        negative_edges_on = generate_negative_edges(graph, new_edges_count['o-n'], edge_type='o-n', edgebank=all_edgebanks, old_nodes=old_training_nodes)
        negative_edges_nn = generate_negative_edges(graph, new_edges_count['n-n'], edge_type='n-n', edgebank=all_edgebanks, old_nodes=old_training_nodes)

        tmp_samples_oo = [torch.tensor([u, v]) for u, v in negative_edges_oo]
        tmp_samples_oon = [torch.tensor([u, v]) for u, v in negative_edges_oon]
        tmp_samples_on = [torch.tensor([u, v]) for u, v in negative_edges_on]
        tmp_samples_nn = [torch.tensor([u, v]) for u, v in negative_edges_nn]
        
        # Add to our samples
        sorted_samples['o-o-bank']['X'][i].extend(tmp_samples_oo)
        sorted_samples['o-o-bank']['y'][i].extend([0 for _ in range(len(negative_edges_oo))])
        sorted_samples['o-o-nobank']['X'][i].extend(tmp_samples_oon)
        sorted_samples['o-o-nobank']['y'][i].extend([0 for _ in range(len(negative_edges_oon))])
        sorted_samples['o-n']['X'][i].extend(tmp_samples_on)
        sorted_samples['o-n']['y'][i].extend([0 for _ in range(len(negative_edges_on))])
        sorted_samples['n-n']['X'][i].extend(tmp_samples_nn)
        sorted_samples['n-n']['y'][i].extend([0 for _ in range(len(negative_edges_nn))])
    
    return sorted_samples, new_edges_count

def generate_validation_data_cached(training_graphs, old_training_nodes, all_edgebanks, MAX_SAMPLES, dataset, seed, type_data, saved_data_file_path):
    cache_path = saved_data_file_path + "/" + dataset + "_" + type_data + "_" + str(seed)
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