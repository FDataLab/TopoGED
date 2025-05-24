import numpy as np 
import networkx as nx
import pandas as pd 
import matplotlib.pyplot as plt 
import random
from sklearn.metrics import roc_auc_score,average_precision_score

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from GraphGeneration.utils.Evaluator import Evaluator

# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 'networkcentra', 'networkcindicator', 'networkcoindash', 'networkdgd', 'networkiconomi', 'Reddit_B'])
args = parser.parse_args()


def compute_reappearance_probabilities(nodes, t_curr, decay_factor=3.0, alpha=1.0, epsilon=1e-8):
    if not nodes:
        return {}

    max_degree = max(degree for _, (_, degree) in nodes.items())

    probs = {}
    for node_id, (last_seen, degree) in nodes.items():
        recency_score = np.exp(-max(0, t_curr - last_seen) / decay_factor)
        degree_score = (degree / max_degree) ** alpha if max_degree > 0 else epsilon
        raw_score = recency_score * degree_score
        probs[node_id] = max(raw_score, epsilon)  # Apply epsilon floor to avoid exact 0

    # Normalize to make a valid probability distribution
    total = sum(probs.values())
    for node in probs:
        probs[node] /= total

    return probs


def update_edgebank(graph, edgebank):
    """
    Update the edgebank based on the current graph
    
    Args:
        graph (nx.Graph): The current graph to update based on
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        
    Returns:
        edgebank (dict): The updated edgebank; updated in place
    """
    for u, v in graph.edges():
        edgebank.setdefault(u, []).append(v)
        
    return edgebank


def build_accumulating_filtration_sequence_with_edgebank(embedding, p_old_nodes, p_new_nodes, E_oo, E_nn, E_on, E_oon, graph_num, edgebank=None, existing_nodes=None, seed=42):
    """
    Our main driver function to build graphs, takes in various arguments to guide the graph construction
    Specifically, this version randomly adds edges, without taking much of the TopER embedding into account. It is purely greedy
    
    Args:
        embedding (list): The TopER embedding to guide construction of the graph, stores the number of nodes and edges to add to the graph
        p_old_nodes (int): The number of old nodes that we are going to see in this graph
        p_new_nodes (int): The number of new nodes that we are going to see in this graph
        E_oo (int): The number of edges type 'oo' to add (old edges from the edgebank)
        E_nn (int): The number of edges type 'nn' to add (new edges that involves two new nodes)
        E_on (int): The number of edges type 'on' to add (new edges between one new node and one old node (either direction))
        E_oon (int): The number of edges type 'oon' to add (new edges between two old nodes that was not in the edgebank)
        graph_num (int): The current graph number we are on
        edgebank (dict): The edgebank from previously constructed graphs
        existing_nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing reappearance probabilities
        seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
        
    Returns:
        filtration_graphs (list(nx.DiGraph)): A list of nx Graphs that we built up from our TopER embedding
        node_types (dict): A dictionary that stores 'old_nodes' and 'new_nodes' organized into lists
        existing_nodes (dict): The updated version of existing nodes passed into the function
        edge_type_map (dict): A dictionary that sorts the types of edges for later analysis
        edgebank (dict): The updated edgebank given the newly constructed graphs
    """
    
    random.seed(seed)
    np.random.seed(seed)

    if existing_nodes is None:
        existing_nodes = {}

    V_total = int(embedding[-1][0])
    E_total = int(embedding[-1][1])
    W_total = embedding[-1][2] 

    # Sample old nodes
    probs = compute_reappearance_probabilities(existing_nodes, graph_num)
    node_ids = list(probs.keys())
    weights = list(probs.values())

    if graph_num > 0:
        old_nodes = list(np.random.choice(node_ids, size=p_old_nodes, replace=False, p=np.array(weights)/np.sum(weights)))  # Makes sure that we select only unique nodes each time
    else:
        old_nodes = []
    
    # Create new node IDs
    if existing_nodes:
        max_id = max(existing_nodes.keys())
    else:
        max_id = 0

    new_nodes = list(range(max_id + 1, max_id + 1 + p_new_nodes))
        
    all_nodes = old_nodes + new_nodes

    edges = set()
    edge_type_map = {}  # For calculating AUC scores later 

    def sample_edges(src_list, dst_list, count, edge_type=None):
        sampled = set()
        attempts = 0

        if edge_type == "o-o-bank" and edgebank is not None:
            for u in src_list:
                if u in edgebank:
                    for v in edgebank[u]:
                        if v in dst_list and u != v and v in edgebank.get(u, []) and (u, v) not in edges:
                            sampled.add((u, v))
                            edge_type_map.setdefault(edge_type, []).append((u, v))
                            edges.add((u, v))
                            if len(sampled) >= count:
                                return list(sampled)

        elif edge_type == "o-o-nobank" and edgebank is not None:
            for u in src_list:
                if u in edgebank:
                    for v in edgebank[u]:
                        if u != v and v not in edgebank.get(u, []) and (u, v) not in edges:
                            sampled.add((u, v))
                            edge_type_map.setdefault(edge_type, []).append((u, v))
                            edges.add((u, v))
                            if len(sampled) >= count:
                                return list(sampled)

        # Random fallback
        else:
            while len(sampled) < count and attempts < count * 10:
                if not src_list or not dst_list:
                    break
                u = random.choice(src_list)
                v = random.choice(dst_list)
                
                # Allows for either o-n or n-o
                if edge_type == "o-n":
                    u, v = (u, v) if random.random() < 0.5 else (v, u)
                    
                if u != v and (u, v) not in edges:
                    sampled.add((u, v))
                    edge_type_map.setdefault(edge_type, []).append((u, v))
                    edges.add((u, v))
                attempts += 1
            return list(sampled)
        
        return list(sampled)

    # Use directly passed-in counts
    edge_pool = (
        sample_edges(old_nodes, old_nodes, E_oo, edge_type="o-o-bank")
        + sample_edges(old_nodes, new_nodes, E_on, edge_type="o-n")
        + sample_edges(new_nodes, new_nodes, E_nn, edge_type="n-n")
        + sample_edges(old_nodes, old_nodes, E_oon, edge_type="o-o-nobank")
    )

    weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
    edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

    G = nx.DiGraph()
    used_edges = set()
    filtration_graphs = []
    current_nodes = set(all_nodes)
    G.add_nodes_from(current_nodes)

    for i, (v_target, e_target, w_target) in enumerate(embedding):
        v_target = int(v_target)
        e_target = int(e_target)

        available_edges = [
            (u, v) for (u, v) in edge_pool
            if u in current_nodes and v in current_nodes and (u, v) not in used_edges
        ]

        needed = e_target - G.number_of_edges()
        selected_edges = available_edges[:needed]

        for (u, v) in selected_edges:
            G.add_edge(u, v, weight=edge_weight_map[(u, v)])
            used_edges.add((u, v))

        filtration_graphs.append(G.copy())

    node_types = {
        "old_nodes": old_nodes,
        "new_nodes": new_nodes
    }
    
    # Update existing nodes for the format
    for node in G.nodes(data=False):
        existing_nodes[node] = (graph_num, G.degree(node))
            
    edgebank = update_edgebank(filtration_graphs[-1], edgebank)

    return filtration_graphs, node_types, existing_nodes, edge_type_map, edgebank


def modifyGraphIds(graphs):
    '''
    For the target graphs, modify their ids to start at 0 for an instance of a node, then increment throughout the graphs
    
    Args:
        graphs (list(nx.Graph)): A list of graphs to modify
        
    Returns:
        graphs (list(nx.Graph)): The modified graphs (operations performed in-place)       
    '''
    # This dictionary will store the mapping of original node IDs to new node IDs
    node_mapping = {}
    new_id = 0
    updated_graphs = []

    # Iterate over all graphs in the list of lists (where each graph is a subgraph in the list)
    # First pass: assign a new ID to every unique node
    for graph_list in graphs:
        for graph in graph_list:
            for node in graph.nodes:
                if node not in node_mapping:
                    node_mapping[node] = new_id
                    new_id += 1

    # Second pass: relabel all graphs using the full mapping
    for timestep, graph_list in enumerate(graphs):
        updated_graphs.append([])
        for graph in graph_list:
            relabeled_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
            updated_graphs[timestep].append(relabeled_graph)

    return updated_graphs


def build_edgebanks_from_start(graphs):
    """
    Build the edgebanks for each graph in graphs, stores all edges from graph i-1 in each index i
    
    Args:
        graphs (list(nx.Graph)): A list of nx Graphs that we will build our edgebanks from
        
    Returns:
        edgebanks (list(dict)): A list of dictionary edgebanks that store all edges from the previous graphs in each index
    """
    edgebanks = [{}]  # Initialize an empty list for edgebanks

    # Loop over all graphs (starting from the second graph)
    for i in range(1, len(graphs)):
        curr_edgebank = {}

        # Add edges from all previous graphs (not the current graph)
        for j in range(i):  # Loop through all previous graphs (graphs 0 to i-1)
            for u, v in graphs[j][-1].edges():  # Accessing the graph directly
                curr_edgebank.setdefault(u, []).append(v)  # Add edge from u to v

        edgebanks.append(curr_edgebank)  # Append the current edgebank to the list

    return edgebanks


# Data Loading and Prep

dataset = args.dataset
my_loader = Loader()
my_evaluator = Evaluator()

# Construct csv
run_number = 1
structure_pred_file_path = f'GraphGeneration/output/results/structure/{dataset}/contids/structure_pred.csv'
structure_true_file_path = f'GraphGeneration/output/results/structure/{dataset}/contids/structure_true.csv'
structure_diff_file_path = f'GraphGeneration/output/results/structure/{dataset}/contids/structure_diff.csv'
kernel_pred_file_path = f'GraphGeneration/output/results/kernel/{dataset}/contids/kernel_pred.csv'
kernel_true_file_path = f'GraphGeneration/output/results/kernel/{dataset}/contids/kernel_true.csv'
edge_file_path = f'GraphGeneration/output/results/structure/{dataset}/contids/edge_analysis.csv'
topER_file_path = f'GraphGeneration/output/results/topER/{dataset}/contids/toper_diff.csv'
animation_path = f'GraphGeneration/output/results/animations/{dataset}/contids/pred_vs_true.mp4'


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

columns_edges = ['Graph Number', 'precision overall', 'recall overall', 'tp_overall', 'fp_overall','tn_overall','fn_overall', 'precision oo', 'recall oo', 'tp_oo', 'fp_oo','tn_oo','fn_oo', 'precision oon', 'recall oon', 'tp_oon', 'fp_oon','tn_oon','fn_oon',  'precision on', 'recall on', 'tp_on', 'fp_on','tn_on','fn_on', 'precision nn', 'recall nn', 'tp_nn', 'fp_nn','tn_nn','fn_nn', 
                     'Correct Node IDs', 'Correct Old Node IDs', 'Precision Old IDs', 'Recall Old IDs',  'Correct New Node IDs', 'Precision New IDs', 
                     'Recall New IDs', 'Correct Overall IDs', 'Precision Overall IDs', 'Recall Overall IDs']

# Write the header and empty content
pd.DataFrame(columns=columns_edges).to_csv(edge_file_path, index=False)

columns_kernel = ['Subgraph 1', 'Subgraph 2', 'Subgraph 3', 'Subgraph 4']

# Write the header and empty content
pd.DataFrame(columns=columns_kernel).to_csv(kernel_pred_file_path, index=False)
pd.DataFrame(columns=columns_kernel).to_csv(kernel_true_file_path, index=False)

# Load probabilities
probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities')  # Activation doesn't matter here
probabilities = probabilities_df.values.tolist()

# Load all features, thresholds, and target subgraphs
features, _ = my_loader.load_data(dataset, activation='Degree', type='features', include_weights=True)
thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=True)
target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)


# Initialize list for predicted graphs
pred_graphs = []

# Build the edgebanks for construction
tmp_target_graphs = modifyGraphIds(target_graphs)
all_edgebanks = build_edgebanks_from_start(tmp_target_graphs)
existing_nodes = {}  # The current nodes we have seen for continuous id implementation
old_nodes_true = set()
curr_edgebank_pred = {}


# Graph Creation

# Iterate through each graph in the dataset
for i in range(len(probabilities)):
    print('Constructing graph number: ', i + 1)
    count_old = probabilities[i][0]
    count_new = probabilities[i][1]
    p0 = probabilities[i][2]
    p1 = probabilities[i][3]
    p2 = probabilities[i][4]
    p3 = probabilities[i][5]

    # Get the embedding and reshape it for construction
    embedding = features[i]
    embedding = list(zip(embedding[0::3], embedding[1::3], embedding[2::3]))

    old_bank = curr_edgebank_pred.copy()  # Used for later evaluation, so we need to save it

    # Build the filtration sequence using the current parameters
    filtration_sequence, node_types, existing_nodes, edge_type_map, curr_edgebank_pred = build_accumulating_filtration_sequence_with_edgebank(
        embedding, p_old_nodes=count_old, p_new_nodes=count_new, E_oo=p0, E_nn=p1, E_on=p2, E_oon=p3, graph_num=i, edgebank=curr_edgebank_pred, existing_nodes=existing_nodes
    )
    
    # Evaluate the graphs
    results_diff_structure = my_evaluator.evaluateTwoStructure(filtration_sequence[-1], target_graphs[i][-1], graph_num=i)
    results_edges = my_evaluator.evaluateEdges(filtration_sequence[-1], target_graphs[i][-1], old_bank, all_edgebanks[i], graph_num=i)
    results_true_structure = my_evaluator.evaluateSingleStructure(target_graphs[i][-1], graph_num=i)
    results_pred_structure = my_evaluator.evaluateSingleStructure(filtration_sequence[-1], graph_num=i)
    pred_kernel, true_kernel, distance = my_evaluator.evaluateOrca(filtration_sequence[-1], target_graphs[i][-1])

    results_diff_structure['Kernel Distance'] = distance  # The kernel distance will be part of our structure evaluation

    # Store all results
    pd.DataFrame([results_diff_structure]).to_csv(structure_diff_file_path, mode='a', header=False, index=False)
    pd.DataFrame([results_edges]).to_csv(edge_file_path, mode='a', header=False, index=False)
    pd.DataFrame([results_true_structure]).to_csv(structure_true_file_path, mode='a', header=False, index=False)
    pd.DataFrame([results_pred_structure]).to_csv(structure_pred_file_path, mode='a', header=False, index=False)
    pd.DataFrame([pred_kernel]).to_csv(kernel_pred_file_path, mode='a', header=False, index=False)
    pd.DataFrame([true_kernel]).to_csv(kernel_true_file_path, mode='a', header=False, index=False)
    
    # Append the last graph from the filtration (assumed to be the "predicted" one)
    pred_graphs.append(filtration_sequence)


# TopER Comparison and G/S Eval

# Flatten the graphs to embed them, only take the last graphs so that we don't mess anything up
embedding_graphs_pred = [inner_list[-1] for inner_list in pred_graphs]
embedding_graphs_target = [inner_list[-1] for inner_list in target_graphs]
embedder = EmbedDegree(include_weights=False)

# Embed the graphs, we recompute the true embeddings just in case
all_embeddings, _, _ = embedder.process_graphs_for_embeddings(embedding_graphs_pred)
true_embeddings, _, _ = embedder.process_graphs_for_embeddings(embedding_graphs_target)

# The true labels for G/S task
_, labels = my_loader.load_data(dataset, 'Degree', include_weights=False)
labels = np.array(labels)

# Compute the Growth/Shrink labels for the predicted graphs
pred_gs_labels = [1]  # First graph is assumed to grow

# Generate predictions for G/S; based on the number of edges in the graph
# 1 if the current graph has more edges than the previous graph; 0 otherswise
for i in range(1, len(embedding_graphs_pred)):
    prev_edges = embedding_graphs_pred[i - 1].number_of_edges()
    curr_edges = embedding_graphs_pred[i].number_of_edges()
    pred_gs_labels.append(1 if curr_edges > prev_edges else 0)

predictions = np.array(pred_gs_labels)


# Compute metrics for Growth/Shrink task
aucroc = roc_auc_score(labels, predictions)
aucpr = average_precision_score(labels, predictions)

# Display results
print(f'G/S AUCROC: {aucroc}')
print(f'G/S AUCPR: {aucpr}')

# Used to store topER evaluation
columns = ['graph_num', 'l2_norm', 'cosine_similarity', 'g/s_pred_label', 'g/s_true_label']
for i in range(10):
    columns.append(f'node_diff_{i+1}')
    columns.append(f'edge_diff_{i+1}')

# Write the header and empty content
pd.DataFrame(columns=columns).to_csv(topER_file_path, index=False)


# Loop through all embeddings
for idx, (embedding, true_embedding) in enumerate(zip(all_embeddings, true_embeddings)):
    # Set graph_num based on index (1-based)
    graph_num = idx + 1
    pred_label = predictions[idx]
    true_label = labels[idx] 

    # Compare embeddings and get the result
    result = my_evaluator.evaluateTopER(embedding, true_embedding, pred_label=pred_label, true_label=true_label, graph_num=graph_num)

    # Append the result to the CSV
    pd.DataFrame([result]).to_csv(topER_file_path, mode='a', header=False, index=False)


# Animation Creation

from itertools import chain

# Flatten a list of lists into a single list of NetworkX graphs
predicted_flat = list(chain(*pred_graphs))
target_flat = list(chain(*target_graphs))

# Call the create_animation function with the flattened lists
#my_evaluator.create_animation(predicted_flat, target_flat, output_file=animation_path)