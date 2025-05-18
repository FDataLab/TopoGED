# New strat
# Keep the embeddings of old node:
    # One with a dict of {id: embedding}
    # Other with a dict of {degree: [available nodes]}
# Take the old node embeddings we expect to see in this new graph
# Cluster nodes together based on expected degree
# Take the average of their embeddings and use that to compute the expected embedding for new nodes
# Then plug into MLP
# Embed nodes at the end of every graph

from collections import defaultdict
import numpy as np 
import networkx as nx
import pandas as pd 
import random
from sklearn.metrics import roc_auc_score, average_precision_score
import copy
import math

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from GraphGeneration.utils.Evaluator import Evaluator


# Models in use
from GraphGeneration.models.MultiHeadedEdgePredictor import MultiHeadedEdgePredictor
from GraphGeneration.models.EdgePredictor import EdgePredictorMLP

import torch
import torch.nn as nn
from torch_geometric.utils import from_networkx
from itertools import product

# Import all embedding methods
from utils.embedding_methods.degree import EmbedDegree
from node2vec import Node2Vec

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkcindicator', 'networkiconomi', 'networkdgd'])
parser.add_argument("--strategy", type=str, required=True, choices=['MultiheadedMLP', 'SingleMLP'], help="The type of MLP NN to use")
parser.add_argument("--embedding", type=str, required=True, choices=['Position', 'NodeType', 'Position+NodeType', 'None'], help="Allows appending positional encodings or an integer node type onto the end of the embeddings")
parser.add_argument("--mlpEncoding", type=str, required=True, choices=['Concat', 'Product', 'Addition', 'Subtraction'], help="How you want to input node embeddings to the MLP")  # Product and addition lead to potential noise as we use directed graphs
parser.add_argument("--embedOld", type=str, required=True, choices=['True', 'False'], help="If you want to let the MLP predict edge type \'o-o-bank\', otherwise these edges are randomly added")
parser.add_argument("--oldDegree", type=str, required=True, choices=['True', 'False'], help="If you want reappearing nodes to reuse their most recent degree")
parser.add_argument("--embeddingType", type=str, required=True, choices=['Linear', 'Node2Vec'], help="How nodes should be embedded. Either with Node2Vec or with a Linear mutliplication of adjacency matrix by node feature matrix")

args = parser.parse_args()

# Set seeds
global_seed = 42
random.seed(global_seed)
np.random.seed(global_seed)


# Set up device
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
except Exception:
    device = torch.device("cpu")
    print("Using CPU")
    

# Node2Vec Parameters
node2vec_dimensions = 60  # We add features onto the end since Node2Vec doesn't embed features
node2vec_walk_length = 50  # Number of nodes visited per walk (Higher is more global, smaller is local)
node2vec_num_walks = 10  # Number of walks to start per node (Higher is more detailed and stable)
node2vec_p = 1.0  # Return parameter, the likelihood of revisiting a node(Higher is less backtracking)
node2vec_q = 1.0  # The walk bias for determining direction (Higher is more DFS-like; lower is BFS-like)
node2vec_window = 10  # The context size (Higher is broader learning)
node2vec_min_count = 1  # Minimum number of occurrences for a node to be considered (Higher will ignore more rare nodes)
node2vec_batch_words = 4  # The batch size for when Word2Vec is used (Higher will train faster; but with more memory)
node2vec_workers = 1  # Number of workers (threads)

# Each of these are (2 * node_embedding_size)
if args.embeddingType == 'Node2Vec':
    embedding_dim = 128
elif args.embeddingType == 'Linear':
    embedding_dim = 8  # Must be same as the number of features


def compute_linear_gnn_embeddings(G: nx.DiGraph):
    """
    An embedding method inspired by LinearGNNs from GraphAny where Z=AX given A is the adjacency matrix and X is the node feature matrix
    One of two available methods
    
    Args:
        G (nx.DiGraph): The graph to embed
        
    Returns:
        embeddings (dict): The constructed dictionary of {node: [embedding]} pairs
    """
    all_nodes = sorted(G.nodes())
    id_to_idx = {node_id: idx for idx, node_id in enumerate(all_nodes)}
    idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}
    
    A = nx.to_numpy_array(G, nodelist=all_nodes, dtype=np.float32)  # The matrix to operate on
    
    # Normalize A
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    A_normalized = A / row_sums
    
    X = np.array([list(G.nodes[node]['feat'].values()) for node in all_nodes], dtype=np.float32)
    
    A_normalized = torch.tensor(A_normalized, device=device)
    X = torch.tensor(X, device=device)
    Z = A_normalized @ X  # As computed in GraphAny
    
    embeddings = {idx_to_id[i]: Z[i] for i in range(Z.shape[0])}
    
    return embeddings


def compute_node2vec_embeddings(G: nx.DiGraph):
    """
    Use Node2Vec to embed nodes in the constructed graph. Appends node features onto the end since Node2Vec does not account for features
    One of two available methods
    
    Args:
        G (nx.DiGraph): The graph to embed
        
    Returns:
        embeddings (dict): The constructed dictionary of {node: [embedding]} pairs
    """
    node2vec = Node2Vec(
        G,
        dimensions=node2vec_dimensions,
        walk_length=node2vec_walk_length,
        num_walks=node2vec_num_walks,
        workers=node2vec_workers,
        p=node2vec_p,
        q=node2vec_q,
        quiet=True
    )
    
    model = node2vec.fit(
        window=node2vec_window, 
        min_count=node2vec_min_count, 
        batch_words=node2vec_batch_words
    )  # Perform Node2Vec

    # Used to generate an embedding for isolated nodes
    all_vectors = [model.wv[key] for key in model.wv.index_to_key]
    mean_vector = torch.tensor(np.mean(all_vectors, axis=0), dtype=torch.float32, device=device)

    # Get embeddings and concatenate the node features
    embeddings = {}
    for node in G.nodes():
        if node in model.wv:
            node2vec_emb = torch.tensor(model.wv[node], dtype=torch.float32)
        else:
            node2vec_emb = mean_vector
            
        # Add on features since Node2Vec doesn't account for features
        feat_dict = G.nodes[node]['feat']
        sorted_keys = sorted(feat_dict.keys())  # Sort the keys for consistency
        sorted_values = [feat_dict[k] for k in sorted_keys]
        node_feat = torch.tensor(sorted_values, dtype=torch.float32, device=device)  # Shape of (4,)
        combined = torch.cat([node2vec_emb, node_feat], dim=0)
        embeddings[node] = combined
    
    return embeddings


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
    max_attempts = 100000
    attempts = 0
    
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
        elif edge_type == 'n-n':
            if G.nodes[u]['feat']['type'] == 1 and G.nodes[v]['feat']['type'] == 1:
                if not G.has_edge(u, v):
                    negatives.add((u, v))
        elif edge_type == 'o-n':
            if (G.nodes[u]['feat']['type'], G.nodes[v]['feat']['type']) in [(0, 1), (1, 0)]:
                if not G.has_edge(u, v):
                    negatives.add((u, v))
                    
    # if attempts >= max_attempts:
    #     print(f'Max attempts for type {edge_type} reached')
                    
    return list(negatives)


def train_on_stage(G, mlp, optimizer, loss_fn, edge_type, edgebank=None, prev_embeddings=None, degree_clusters=None, graph_num=0):
    """
    Train the MLP on a certain state of the subgraph, whether it has no edges or some filled in from the old edge types
    We build in order of 'o-o-bank', 'o-o-nobank', 'o-n', 'n-n'
    
    Args:
        G (nx.DiGraph): Our current subgraph we are learning on for the MLP
        mlp (MLP NN): The MLP that we are using to predict edges given node embeddings
        optimizer : The optimizer we are using for learning weights
        loss_fn : The loss function for calculating train_loss
        edge_type (string): The type of edge that we are currently working on predicting
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges 
        prev_embeddings (dict): The embeddings for all nodes up to this point
        degree_clusters (dict): The degree cluster groupings for all nodes in the starter graph
        graph_num (int): The current graph number, used for positional embeddings for nodes if needed

    Returns:
        loss.item() (float): The model's loss from training it
        prev_embeddings (dict): Our updated node embeddings
        degree_clusters (dict): Our updated degree clusters
    """
    mlp.train()
    
    # Get the current embeddings for the graph
    curr_embeddings = {}

    # Mimicks how we assign new node ids later
    for node, data in G.nodes(data=True):
        if node in prev_embeddings:
            base_embedding = prev_embeddings[node].clone().detach().to(device).float()
        else:
            base_embedding = degree_clusters.get(data['feat']['maxDegree'], [])
            
            # Protects from crashes
            if base_embedding is None and len(base_embedding) == 0:
                base_embedding = [np.zeros(32)]
            
        # Convert to tensor for concatenation
        base_embedding = torch.tensor(base_embedding, dtype=torch.float32).to(device)

        additional_features = []  # If needed according to arguments

        if 'NodeType' in args.embedding:
            node_type_feat = torch.tensor([data['feat']['type']], dtype=torch.float32).to(device)  # Ensure 1D
            additional_features.append(node_type_feat)

        if 'Position' in args.embedding:
            pos_feat = torch.tensor([math.cos(graph_num)], dtype=torch.float32).to(device)  # Ensure 1D
            additional_features.append(pos_feat)

        if additional_features:
            base_embedding = torch.cat([base_embedding] + additional_features, dim=0)

        curr_embeddings[node] = base_embedding


    # Filter edges based on the stage type
    if edge_type == 'o-o-bank':
        pos_edges = [
            (u, v) for u, v in G.edges()
            if G.nodes[u]['feat']['type'] == 0 and G.nodes[v]['feat']['type'] == 0 and v in edgebank.get(u, [])
        ]
    elif edge_type == 'o-o-nobank':
        pos_edges = [
            (u, v) for u, v in G.edges()
            if G.nodes[u]['feat']['type'] == 0 and G.nodes[v]['feat']['type'] == 0 and v not in edgebank.get(u, [])
        ]
    elif edge_type == 'n-n':
        pos_edges = [
            (u, v) for u, v in G.edges()
            if G.nodes[u]['feat']['type'] == 1 and G.nodes[v]['feat']['type'] == 1
        ]
    elif edge_type == 'o-n':
        pos_edges = [
            (u, v) for u, v in G.edges()
            if (G.nodes[u]['feat']['type'], G.nodes[v]['feat']['type']) in [(0, 1), (1, 0)]
        ]

    # print('Getting negative edges and setup')
    if len(pos_edges) == 0:
        print('There are no edges')
        return 0.0  # skip training if there are no edges

    # Negative sampling
    neg_edges = generate_negative_edges(G, len(pos_edges), edge_type, edgebank)
    # print('Setup')
    all_edges = pos_edges + neg_edges
    labels = torch.tensor([1] * len(pos_edges) + [0] * len(neg_edges), dtype=torch.float32).to(device)

    src_embed = torch.stack([curr_embeddings[u] for u, v in all_edges]).to(device)
    dst_embed = torch.stack([curr_embeddings[v] for u, v in all_edges]).to(device)
    
    # print('Predicting')
    if args.strategy == 'SingleMLP':
        preds = mlp(src_embed, dst_embed).squeeze()
    elif args.strategy == 'MultiheadedMLP':
        preds = mlp(src_embed, dst_embed, edge_type).squeeze()
        
    # Embeddings depend on our strategy
    if args.embeddingType == 'Node2Vec':
        final_embeddings = compute_node2vec_embeddings(G)
    elif args.embeddingType == 'Linear':
        final_embeddings = compute_linear_gnn_embeddings(G)
        
    prev_embeddings.update(final_embeddings)  # Blindly overwrites the previously existing embeddings
    
    for node in G.nodes():        
        degree = G.nodes[node]['feat']['maxDegree']
        
        curr_embedding = embeddings[node]
        old_embedding = degree_clusters.get(degree, [])
        
        # Average the embeddings if both exist
        if old_embedding is not None and len(old_embedding) > 0:
            new_embedding = (np.array(curr_embedding) + np.array(old_embedding)) / 2
        else:
            new_embedding = curr_embedding
            
        degree_clusters[degree] = new_embedding  # Add the embedding
        
        
    # Update model parameters    
    loss = loss_fn(preds, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), prev_embeddings, degree_clusters


def generate_subgraphs(graph, edgebank: dict, prev_graph=None, thresholds=None):
    """
    Generates specific states of graph for training on, these are subgraphs that only contain certain edgetypes
    Helps simulate actual construction
    
    Args:
        graph (nx.DiGraph): The graph that we are attempting to generate subgraphs for
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges 
        prev_graph (nx.DiGraph): The previous graph that we use to determine if we are on the first graph or not
        thresholds (list): The thresholds according to TopER, assigns the maximum degree of nodes
    
    Returns: 
        res (list): Our list of subgraphs, may have None graphs in the middle if this is first graph
    """
    tmp_graph = nx.DiGraph()
    
    # Assign base features
    for node in graph.nodes():
        graph.nodes[node]['feat'] = {}  # Set up the dictionary
        graph.nodes[node]['feat']['id'] = node
        graph.nodes[node]['feat']['type'] = 0 if node in edgebank.keys() else 1
        
        graph.nodes[node]['feat']['currDegree'] = graph.degree(node)
        
        if np.any(thresholds):
            node_degree = graph.degree(node)
            graph.nodes[node]['feat']['maxDegree'] = next((t for t in thresholds if node_degree <= t), thresholds[-1])
        else:
            graph.nodes[node]['feat']['maxDegree'] = graph.degree(node)
            
        tmp_graph.add_node(node, feat=graph.nodes[node]['feat'])
        tmp_graph.nodes[node]['feat']['currDegree'] = 0
        
        
    if prev_graph is None:
        # All nodes are new
        if args.embedOld == 'True':
            return [None, None, None, graph]
        else:
            return [None, None, graph]
    
    else:
        res = []
        
        # o-o-bank as base
        for (u, v) in graph.edges():
            #print(f"Checking edge: u={u}, v={v}, u_type={graph.nodes[u]['feat']['type']}, v_type={graph.nodes[v]['feat']['type']}")
            if u in edgebank.keys() and v in edgebank.keys() and v in edgebank.get(u, []):
                #print('Adding edge')
                tmp_graph.add_edge(u, v)
                
        update_degrees(tmp_graph)

        if args.embedOld == 'True':
            res.append(tmp_graph)
        
        # o-o-nobank
        for (u, v) in graph.edges():
            #print(f"Checking edge: u={u}, v={v}, u_type={graph.nodes[u]['feat']['type']}, v_type={graph.nodes[v]['feat']['type']} for oon")
            if graph.nodes[u]['feat']['type'] == 0 and graph.nodes[v]['feat']['type'] == 0 and v not in edgebank.get(u, []):
                #print('Adding edge oon')
                tmp_graph.add_edge(u, v)
        update_degrees(tmp_graph)
        curr_graph = copy.deepcopy(tmp_graph)
        #print(f"After section oon: {tmp_graph.number_of_edges()} edges, {tmp_graph.number_of_nodes()} nodes")
        res.append(curr_graph)        
        
        # o-n
        for (u, v) in graph.edges():
            #print(f"Checking edge: u={u}, v={v}, u_type={graph.nodes[u]['feat']['type']}, v_type={graph.nodes[v]['feat']['type']}")
            if (graph.nodes[u]['feat']['type'] == 1 and graph.nodes[v]['feat']['type'] == 0) or (graph.nodes[u]['feat']['type'] == 0 and graph.nodes[v]['feat']['type'] == 1):
                tmp_graph.add_edge(u, v)
                #print('Adding edge on')
        update_degrees(tmp_graph)
        curr_graph = copy.deepcopy(tmp_graph)
        #print(f"After section on: {tmp_graph.number_of_edges()} edges, {tmp_graph.number_of_nodes()} nodes")
        res.append(curr_graph)  
        
        # n-n
        for (u, v) in graph.edges():
            #print(f"Checking edge: u={u}, v={v}, u_type={graph.nodes[u]['feat']['type']}, v_type={graph.nodes[v]['feat']['type']} for nn")
            if graph.nodes[u]['feat']['type'] == 1 and graph.nodes[v]['feat']['type'] == 1:
                tmp_graph.add_edge(u, v)
                #print('Adding edge nn')
        update_degrees(tmp_graph)
        curr_graph = copy.deepcopy(tmp_graph)
        #print(f"After section nn: {tmp_graph.number_of_edges()} edges, {tmp_graph.number_of_nodes()} nodes")
        res.append(curr_graph)  
        
        return res


def setupMLP():    
    """
    Set up the MLP based on the arguments provided in the command line starter
    
    Args:
        None
    
    Returns:
        None
    """
    input_dim = embedding_dim  # Starting input dimension depends on what method we are using
    
    # Input size changes if we are doing different methods, this keeps it consistent
    if 'Position' in args.embedding:
        input_dim += 2
    if 'NodeType' in args.embedding:
        input_dim += 2
        
    # Set up the MLPs according to arguments
    if args.strategy == 'SingleMLP':
        mlp = EdgePredictorMLP(in_channels=input_dim, hidden_channels=32, input_type=args.mlpEncoding)
        
    elif args.strategy == 'MultiheadedMLP':
        if args.embedOld == 'True':
            flags = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
        else:
            flags = ['o-o-nobank', 'o-n', 'n-n']
        mlp = MultiHeadedEdgePredictor(in_channels=input_dim, hidden_channels=32, edge_types=flags, input_type=args.mlpEncoding)
        
    mlp.to(device)
        
    return mlp


def train_models(graphs, epochs_per_stage=150, lr=0.001, edgebanks=None, thresholds=None, embeddings=None, degree_clusters=None):
    """
    Train the MLP for later construction over a set list of graphs
    
    Args:
        graphs (list): A list of networkx graphs that we are going to use for initial training
        epochs_per_stage (int): How many epochs to train for, for each edge type
        lr (float): The learning rate for the model
        edgebanks (list): A list of edgebanks; where edgebank is a dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        thresholds (list): The thresholds according to TopER, assigns the maximum degree of nodes
        embeddings (dict):  The embeddings for all nodes in the starter graph; updated as we go
        degree_clusters (dict): The degree cluster groupings for all nodes in the starter graph; updated as we go
        
    Returns:
        mlp (MLP NN): The MLP that we have trained
    """
    # Set up
    mlp = setupMLP()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for graph_num, graph in enumerate(graphs[1:], start=1):
        subgraphs = generate_subgraphs(graph, edgebanks[graph_num], graphs[graph_num - 1] if graph_num > 0 else None, thresholds)  # Get subgraphs for each edge type (assuming o-o-bank has been added)
        print(f"\n--- Training on Graph #{graph_num} ---")
        
        if args.embedOld == 'True':
            for i, (subgraph, flag) in enumerate(zip(subgraphs, ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n'])):
                # print(flag)
                # For the first graphs processing
                if subgraph is None or len(subgraph.edges()) == 0:
                    continue
                for epoch in range(epochs_per_stage):
                    loss, embeddings, degree_clusters = train_on_stage(subgraph, mlp, optimizer, loss_fn, edge_type=flag, edgebank=edgebanks[graph_num], prev_embeddings=embeddings, degree_clusters=degree_clusters, graph_num=graph_num)
                    if (epoch + 1) % 10 == 0: 
                        print(f"  Graph Number {graph_num} Stage {flag} - Epoch {epoch+1}/{epochs_per_stage} | Loss: {loss:.4f}")
                        
        elif args.embedOld == 'False':
            # print(flag)
            for i, (subgraph, flag) in enumerate(zip(subgraphs, ['o-o-nobank', 'o-n', 'n-n'])):
                # For the first graphs processing
                if subgraph is None or len(subgraph.edges()) == 0:
                    continue
                for epoch in range(epochs_per_stage):
                    loss, embeddings, degree_clusters = train_on_stage(subgraph, mlp, optimizer, loss_fn, edge_type=flag, edgebank=edgebanks[graph_num], prev_embeddings=embeddings, degree_clusters=degree_clusters, graph_num=graph_num)
                    if (epoch + 1) % 10 == 0: 
                        print(f"  Graph Number {graph_num} Stage {i} - Epoch {epoch+1}/{epochs_per_stage} | Loss: {loss:.4f}")

    return mlp
    
    
def generate_candidates(graph:nx.DiGraph, nodes_1, flag, nodes_2=None, edgebank=None):
    """
    Generate all possible edges that we could add (directed)
    
    Args:
        graph (nx.DiGraph): The current graph that we are constructing
        nodes_1 (list): The set of source node ids
        flag (string): The edge type that we are making candidates for
        nodes_2 (list): The set of destination node ids
        edgebank (dict):  A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
    
    Returns:
        candidates (list): A list of tuples for all possible edges that can be added given the nodes
    """
    candidates = []  # The edges that we could add
    
    # Different processing depending on available nodes for the edge
    if flag == 'o-n':
        candidates = [(node_1, node_2) for node_1, node_2 in product(nodes_1, nodes_2) if node_1 != node_2 and (node_1, node_2) not in graph.edges() and ((node_1 in edgebank.keys() and node_2 not in edgebank.keys()) or (node_1 not in edgebank.keys() and node_2 in edgebank.keys()))and (graph.degree(node_1) < graph.nodes[node_1]['feat']['maxDegree']) and (graph.degree(node_2) < graph.nodes[node_2]['feat']['maxDegree'])]
    elif flag == 'o-o-nobank':
        candidates = [(node_1, node_2) for node_1, node_2 in product(nodes_1, nodes_1) if node_1 != node_2 and (node_1, node_2) not in graph.edges() and node_2 not in edgebank.get(node_1, []) and (graph.degree(node_1) < graph.nodes[node_1]['feat']['maxDegree']) and (graph.degree(node_2) < graph.nodes[node_2]['feat']['maxDegree'])]
    elif flag == 'o-o-bank':
        candidates = [(node_1, node_2) for node_1, node_2 in product(nodes_1, nodes_1) if node_1 != node_2 and (node_1, node_2) not in graph.edges() and node_2 in edgebank.get(node_1, []) and (graph.degree(node_1) < graph.nodes[node_1]['feat']['maxDegree']) and (graph.degree(node_2) < graph.nodes[node_2]['feat']['maxDegree'])]
    elif flag == 'n-n':
        for u in nodes_1:
            for v in nodes_1:
                if u != v and not graph.has_edge(u, v) and (graph.degree(u) < graph.nodes[u]['feat']['maxDegree']) and (graph.degree(v) < graph.nodes[v]['feat']['maxDegree']):  # Skip edges that already exist in the graph
                    candidates.append((u, v))
    
    return candidates


def predict_edges(graph, edge_type, node_types, edgebank, mlp, embeddings, top_k, graph_num):
    """
    Predict what edges we will see in the graph, this is done by passing the node embeddings into the MLP and selecting the top_k most likely edges
    
    Args:
        graph (nx.DiGraph): The graph that we are currently constructing
        edge_type (string): The current edge type we are predicting edges for
        node_types (dict): A dictionary storing the old nodes and new nodes in ['old_nodes'] and ['new_nodes'] respectively
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        mlp (MLP NN): An MLP that predicts the probability of an edge occurring
        embeddings (dict): The embeddings of all old nodes we have seen up to this point
        top_k (int): How many edges we are going to select
        graph_num (int): Used for assigning a positional encoding onto the node embedding
    
    Returns:
        top_edges (list): The top_k edges that we have decided to add here
    """
    if edge_type == 'o-o-bank' or edge_type == 'o-o-nobank':
        available_nodes = node_types['old_nodes']
        candidate_edges = generate_candidates(graph, nodes_1=available_nodes, nodes_2=None, flag=edge_type, edgebank=edgebank)

    elif edge_type == 'n-n':
        available_nodes = node_types['new_nodes']
        candidate_edges = generate_candidates(graph, nodes_1=available_nodes, nodes_2=None, flag=edge_type, edgebank=edgebank)

    elif edge_type == 'o-n':
        nodes = node_types['old_nodes'] + node_types['new_nodes']  # Since all nodes are valid candidates
        candidate_edges = generate_candidates(graph, nodes_1=nodes, nodes_2=nodes, flag=edge_type, edgebank=edgebank)
    
    # Predict edge probabilities using the MLP
    edge_probs = []
    for u, v in candidate_edges:
        src_embed = embeddings[u]
        dst_embed = embeddings[v]

        # Convert to torch.Tensor if necessary
        if isinstance(src_embed, np.ndarray):
            src_embed = torch.tensor(src_embed, dtype=torch.float32, device=device)
        else:
            src_embed = src_embed.to(device)
        if isinstance(dst_embed, np.ndarray):
            dst_embed = torch.tensor(dst_embed, dtype=torch.float32, device=device)
        else:
            dst_embed = dst_embed.to(device)

        # Add batch dimension if needed
        if src_embed.dim() == 1:
            src_embed = src_embed.unsqueeze(0)
        if dst_embed.dim() == 1:
            dst_embed = dst_embed.unsqueeze(0)

        # Append onto the end
        if 'NodeType' in args.embedding:
            src_type = torch.tensor([[1.0]] if u in node_types['new_nodes'] else [[0.0]], device=device)
            dst_type = torch.tensor([[1.0]] if v in node_types['new_nodes'] else [[0.0]], device=device)
            src_embed = torch.cat([src_embed, src_type], dim=1)
            dst_embed = torch.cat([dst_embed, dst_type], dim=1)

        if 'Position' in args.embedding:
            cos_val = torch.tensor([[math.cos(graph_num)]], dtype=torch.float32, device=device)
            src_embed = torch.cat([src_embed, cos_val], dim=1)
            dst_embed = torch.cat([dst_embed, cos_val], dim=1)
            
        # Predict edge probability
        if args.strategy == 'SingleMLP':
            prob = mlp(src_embed, dst_embed)
        elif args.strategy == 'MultiheadedMLP':
            prob = mlp(src_embed, dst_embed, edge_type)
        
        edge_probs.append((u, v, prob.item()))

    # Sort and select top_k
    edge_probs.sort(key=lambda x: x[2], reverse=True)
    top_edges = [(u, v) for u, v, _ in edge_probs[:top_k]]

    return top_edges


def compute_reappearance_probabilities(nodes, t_curr, decay_factor=3.0, alpha=1.0, epsilon=1e-8):
    """
    Compute the probability for each node to reappear given how long ago it was seen and its latest degree
    Nodes of higher degree, and nodes seen more recently are preferred
    
    Args:
        nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing probabilities
        t_curr (int): The current graph number we are on, used to compute probabilities
        decay_factor (float): How quickly the recency of a node decays. Higher means that the nodes seen long ago decay slower
        alpha (float): Our decay constant, controls how influential degree is (alpha > 1 means that it prefers degree, alpha < 1 means that it matters less)
        epsilon (float): Prevents having 0 probabilities for a node, and thus prevents numpy errors later on
    
    Returns:
        probs (dict):  A dictionary of {node_id: percent probability} probabilities for each node in nodes
    """
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

        
def get_node_features(graph, thresholds, embedding, old_nodes, new_nodes):
    """
    Assign the maximum degree of a node, either using its last seen degree (if args.oldDegree == True) or randomly giving it one
    
    Args:
        graph (nx.DiGraph): The Graph we are attempting to construct, we assign node features here
        thresholds (list): The thresholds according to TopER, assigns the maximum degree of nodes
        embedding (list): The current TopER graph embedding vector used to figure out how many nodes have a degree
        old_nodes (list): The list of old nodes that are in the graph
        new_nodes (list): The list of new nodes that are in the graph
        
    Returns:
        None
    """
    degree_counts = [embedding[i][0] for i in range(0, len(embedding))]
    
    degree_dict = {thresholds[i]: degree_counts[i] for i in range(len(thresholds))}
    
    degree_assignment = []  # This will store the assigned degrees
    
    for degree, count in degree_dict.items():
        degree_assignment.extend([degree] * count)
        
    random.shuffle(degree_assignment)
    
    if args.oldDegree == 'True':
        for node in old_nodes:
            old_degree = existing_nodes[node][1]

            # Find the smallest degree in degree_assignment ≥ old_degree
            suitable_degrees = [d for d in degree_assignment if d >= old_degree]
            if not suitable_degrees:
                assigned_degree = degree_assignment.pop()

            assigned_degree = min(suitable_degrees)
            degree_assignment.remove(assigned_degree)
            
            graph.nodes[node]['feat']['currDegree'] = 0
            graph.nodes[node]['feat']['maxDegree'] = assigned_degree
        
        # Give the node a random new degree    
        for node in new_nodes:
            assigned_degree = degree_assignment.pop()

            graph.nodes[node]['feat']['currDegree'] = 0
            graph.nodes[node]['feat']['maxDegree'] = assigned_degree
            
    else:
        for i, node in enumerate(graph.nodes):        
            # Assign features to the node as an attribute
            graph.nodes[node]['feat']['currDegree'] = 0  # Starts at 0
            graph.nodes[node]['feat']['maxDegree'] = degree_assignment[i]
 

def update_degrees(graph: nx.DiGraph):
    """
    After updating the graph, between edge types, update the nodes current degree feature
    
    Args:
        graph (nx.DiGraph): The current graph in construction
        
    Returns:
        None
    """
    for node in graph.nodes(data=False):
        graph.nodes[node]['feat']['currDegree'] = graph.degree(node)
        
     
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


def build_accumulating_filtration_sequence_with_edgebank(embedding, graph_num, p_old_nodes, p_new_nodes, E_oo, E_nn, E_on, E_oon, thresholds, embeddings=None, degree_clusters=None, edgebank=None, existing_nodes=None, mlp=None, seed=42):
    """
    Our main driver function to build graphs, takes in various arguments to guide the graph construction
    Specifically, this version uses an MLP to assign edges to two nodes based on the probability of them forming an edge
    
    Args:
        embedding (list): The TopER embedding to guide construction of the graph, stores the number of nodes and edges to add to the graph
        graph_num (int): The current graph number we are on
        p_old_nodes (int): The number of old nodes that we are going to see in this graph
        p_new_nodes (int): The number of new nodes that we are going to see in this graph
        E_oo (int): The number of edges type 'oo' to add (old edges from the edgebank)
        E_nn (int): The number of edges type 'nn' to add (new edges that involves two new nodes)
        E_on (int): The number of edges type 'on' to add (new edges between one new node and one old node (either direction))
        E_oon (int): The number of edges type 'oon' to add (new edges between two old nodes that was not in the edgebank)
        thresholds (list): The thresholds for node degrees 'maxDegree' as dicted by TopER
        embeddings (dict): The embeddings of all old nodes we have seen up to this point
        degree_clusters (dict): A dictionary of {'degree': [created_embedding]} that we use to assign the embeddings for new nodes
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        existing_nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing reappearance probabilities
        mlp (MLP NN): An MLP that predicts the probability of an edge occurring
        seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
        
    Returns:
        filtration_graphs (list(nx.DiGraph)): A list of nx Graphs that we built up from our TopER embedding
        node_types (dict): A dictionary that stores 'old_nodes' and 'new_nodes' organized into lists
        existing_nodes (dict): The updated version of existing nodes passed into the function
        edge_type_map (dict): A dictionary that sorts the types of edges for later analysis
        edgebank (dict): The updated edgebank given the newly constructed graphs
        embeddings (dict): Our newly updated embeddings based on the constructed graph
        degree_clusters (dict): Our newly updated degree clusters
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
    tmp_graph = nx.DiGraph()  # A graph for computing node embeddings easily
     
    node_types = {
        "old_nodes": old_nodes,
        "new_nodes": new_nodes
    } 
    
     
    # Add the nodes to the graph
    for node in old_nodes:
        tmp_graph.add_node(node)
        feature_dict_old = {'id': node, 'type': 0}  
        tmp_graph.nodes[node]['feat'] = feature_dict_old
    for node in new_nodes:
        tmp_graph.add_node(node)
        feature_dict_new = {'id': node, 'type': 1}  
        tmp_graph.nodes[node]['feat'] = feature_dict_new
    
    get_node_features(tmp_graph, thresholds, embedding, old_nodes, new_nodes)  # Assign maximum degrees

    curr_embeddings = {}
    for node, data in tmp_graph.nodes(data=True):
        if node in embeddings:
            curr_embeddings[node] = embeddings[node]
        else:
            new_embedding = degree_clusters.get(data['feat']['maxDegree'], [])
            
            # Protects from crashes
            if new_embedding is None and len(new_embedding) == 0:
                new_embedding = [np.zeros(32)]

            curr_embeddings[node] = new_embedding


    def sample_edges(src_list, dst_list, count, edge_type=None):
        sampled = set()
        attempts = 0

        if args.embedOld == False and edge_type == "o-o-bank" and edgebank is not None:
            for u in src_list:
                if u in edgebank:
                    for v in edgebank[u]:
                        if v in dst_list and u != v and v in edgebank.get(u, []) and (u, v) not in edges:
                            sampled.add((u, v))
                            edge_type_map.setdefault(edge_type, []).append((u, v))
                            edges.add((u, v))
                            if len(sampled) >= count:
                                return list(sampled)

        else:
            if count > 0:
                sampled = predict_edges(tmp_graph, edge_type, node_types, edgebank, mlp, curr_embeddings, top_k=count, graph_num=graph_num)
            
        return list(sampled)

    # Get edges of each type
    oo_bank_edges = sample_edges(old_nodes, old_nodes, count=E_oo, edge_type="o-o-bank")
    tmp_graph.add_edges_from(oo_bank_edges)
    update_degrees(tmp_graph)
    
    oo_nobank_edges = sample_edges(old_nodes, old_nodes, count=E_oon, edge_type="o-o-nobank")
    tmp_graph.add_edges_from(oo_nobank_edges)
    update_degrees(tmp_graph)
    
    on_edges = sample_edges(old_nodes, new_nodes, count=E_on, edge_type="o-n")
    tmp_graph.add_edges_from(on_edges)
    update_degrees(tmp_graph)
    
    nn_edges = sample_edges(new_nodes, new_nodes, count=E_nn, edge_type="n-n")
    tmp_graph.add_edges_from(nn_edges)
    update_degrees(tmp_graph)
    
    
    edge_pool = (oo_bank_edges + oo_nobank_edges + on_edges + nn_edges)

    weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
    edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

    G = nx.DiGraph()
    used_edges = set()
    filtration_graphs = []

    for i, (v_target, e_target, w_target) in enumerate(embedding):
        v_target = int(v_target)
        e_target = int(e_target)

        current_nodes = set(all_nodes[:v_target])
        G.add_nodes_from(current_nodes)

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
    
    # Update existing nodes for the format
    for node in tmp_graph.nodes(data=False):
        if node in new_nodes:
            existing_nodes[node] = (graph_num, tmp_graph.degree(node))
            
    edgebank = update_edgebank(filtration_graphs[-1], edgebank)

    # Embeddings depend on our strategy
    if args.embeddingType == 'Node2Vec':
        final_embeddings = compute_node2vec_embeddings(tmp_graph)
    elif args.embeddingType == 'Linear':
        final_embeddings = compute_linear_gnn_embeddings(tmp_graph)
        
    
    embeddings.update(final_embeddings)  # Blindly overwrites the previously existing embeddings
    
    # Generate the embeddings we will assign new nodes by averaging
    for node in tmp_graph.nodes():
        degree = tmp_graph.nodes[node]['feat']['maxDegree']
        
        curr_embedding = embeddings[node]
        old_embedding = degree_clusters.get(degree, [])
        
        # Average the embeddings if both exist
        if old_embedding is not None and len(old_embedding) > 0:
            new_embedding = (np.array(curr_embedding) + np.array(old_embedding)) / 2
        else:
            new_embedding = curr_embedding
            
        degree_clusters[degree] = new_embedding  # Add the embedding

    return filtration_graphs, node_types, existing_nodes, edge_type_map, edgebank, embeddings, degree_clusters


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
    for i, graph_list in enumerate(graphs):
        updated_graphs.append([])
        for graph in graph_list:
            relabeled_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
            updated_graphs[i].append(relabeled_graph)

    return updated_graphs, len(node_mapping)



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
                u_key = u
                v_key = v
                curr_edgebank.setdefault(u_key, []).append(v_key)  # Add edge from u to v

        edgebanks.append(curr_edgebank)  # Append the current edgebank to the list

    return edgebanks


def process_starter_graph(graph: nx.DiGraph, thresholds):
    """
    Process our very first graph, this is our 'primer' used to construct the later graphs
    We do this since we need some node embeddings and features to start with
    
    Args:
        graph (nx.DiGraph): The first graph in the dataset, which we are embedding the nodes for
        thresholds (list): A list of integers, from TopER, used to assign the max degree of a node
    """
    # Assign base features
    for node in graph.nodes():
        graph.nodes[node]['feat'] = {}  # Set up the dictionary
        graph.nodes[node]['feat']['id'] = node
        graph.nodes[node]['feat']['type'] = 1
        node_degree = graph.degree(node)  # The current nodes degree
        
        if np.any(thresholds):
            graph.nodes[node]['feat']['currDegree'] = node_degree
            graph.nodes[node]['feat']['maxDegree'] = next((t for t in thresholds if node_degree <= t), thresholds[-1])
        else:
            graph.nodes[node]['feat']['currDegree'] = node_degree
            graph.nodes[node]['feat']['maxDegree'] = graph.degree(node)
        
    # Embeddings depend on our strategy
    if args.embeddingType == 'Node2Vec':
        final_embeddings = compute_node2vec_embeddings(graph)
    elif args.embeddingType == 'Linear':
        final_embeddings = compute_linear_gnn_embeddings(graph)
        
    
    existing_nodes = {}
    # process the nodes for old node evaluation
    for node in graph.nodes(data=False):
        existing_nodes[node] = (0, graph.degree(node))
            
    # Build the edgebank
    edgebank = {}
    for u, v in graph.edges():  # Accessing the graph directly
        edgebank.setdefault(u, []).append(v)
        
    # Process the degree clusters for generating the embeddings for new nodes
    degree_clusters = defaultdict(list)
    for node in graph.nodes():        
        degree = graph.nodes[node]['feat']['maxDegree']
        
        curr_embedding = final_embeddings[node]
        old_embedding = degree_clusters.get(degree, [])
        
        # Average the embeddings if both exist
        if old_embedding is not None and len(old_embedding) > 0:
            new_embedding = (np.array(curr_embedding) + np.array(old_embedding)) / 2
        else:
            new_embedding = curr_embedding
            
        degree_clusters[degree] = new_embedding  # Add the embedding
    
    
    return final_embeddings, degree_clusters, existing_nodes, edgebank


# Data Loading and Prep

dataset = args.dataset
my_loader = Loader()
my_evaluator = Evaluator()

# Construct csv
run_number = 1
structure_pred_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}/structure_pred.csv'
structure_true_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}/structure_true.csv'
structure_diff_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}/structure_diff.csv'
kernel_pred_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}/kernel_pred.csv'
kernel_true_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}/kernel_true.csv'
edge_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}/edge_analysis.csv'
topER_file_path = f'GraphGeneration/output/results/topER/{dataset}/model_gen_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}/toper_diff.csv'
animation_path = f'GraphGeneration/output/results/animations/{dataset}/model_gen_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}/pred_vs_true.mp4'

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
probabilities_df = pd.read_csv(f'ReinforcementLearning/output/probabilities/{dataset}_1back.csv').iloc[:, 1:]
probabilities = probabilities_df.values.tolist()

# Load all features, thresholds, and target subgraphs
features, _ = my_loader.load_data(dataset, activation='Degree', type='features', include_weights=True)
thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=True)
target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)

# Initialize list for predicted graphs
pred_graphs = []

# Build the edgebanks for construction
tmp_target_graphs, _ = modifyGraphIds(target_graphs)
all_edgebanks = build_edgebanks_from_start(tmp_target_graphs)

target_training_graphs = [tmp_target_graphs[i][-1] for i in range(int(len(tmp_target_graphs) * 0.50))]  # First 50% of the graphs

print('Starting training')
embeddings, degree_clusters, existing_nodes, curr_edgebank_pred = process_starter_graph(target_graphs[0][-1], thresholds)  # We need a graph to get things going

# Train the MLP for later construction
mlp = train_models(target_training_graphs, edgebanks=all_edgebanks, thresholds=thresholds, embeddings=embeddings, degree_clusters=degree_clusters)
old_nodes_true = set()  # The old nodes for the true graphs, used for evaluation
curr_edgebank_pred = {}  # The current edgebank for the predicted graphs; starts empty


# Graph Creation

# Iterate through each graph in the dataset
for i in range(1, len(probabilities)):
    print('Constructing graph number: ', i + 1)
    
    # Get the number of resources available for this graph
    count_old = probabilities[i][0]
    count_new = probabilities[i][1]
    p0 = probabilities[i][2]
    p1 = probabilities[i][3]
    p2 = probabilities[i][4]
    p3 = probabilities[i][5]

    # Get the embedding and reshape it
    embedding = features[i]
    embedding = list(zip(embedding[0::3], embedding[1::3], embedding[2::3]))

    # Build the filtration sequence using the current parameters
    filtration_sequence, node_types, existing_nodes, edge_type_map, curr_edgebank_pred, embeddings, degree_clusters = build_accumulating_filtration_sequence_with_edgebank(
        embedding, graph_num=i, p_old_nodes=count_old, p_new_nodes=count_new, E_oo=p0, E_nn=p1, E_on=p2, E_oon=p3, thresholds=thresholds, embeddings=embeddings, degree_clusters=degree_clusters, edgebank=curr_edgebank_pred, existing_nodes=existing_nodes, mlp=mlp
    )
    
    # Evaluate the graphs
    results_diff_structure = my_evaluator.evaluateTwoStructure(filtration_sequence[-1], target_graphs[i][-1], graph_num=i)
    results_edges = my_evaluator.evaluateEdges(filtration_sequence[-1], target_graphs[i][-1], curr_edgebank_pred, all_edgebanks[i], graph_num=i)
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

del target_graphs[0]

# Flatten the graphs to embed them, only take the last graphs so that we don't mess anything up
embedding_graphs = [inner_list[-1] for inner_list in pred_graphs]
embedder = EmbedDegree(include_weights=False)

# Load the TopER embeddings and labels for the G/S task and TopER comparisons
all_embeddings, _, _ = embedder.process_graphs_for_embeddings(embedding_graphs)
true_embeddings, labels = my_loader.load_data(dataset, 'Degree', include_weights=False)
labels = np.array(labels)


# Compute the Growth/Shrink labels for the predicted graphs
pred_gs_labels = [1]  # First graph is assumed to grow

# Generate predictions for G/S; based on the number of edges in the graph
# 1 if the current graph has more edges than the previous graph; 0 otherswise
for i in range(1, len(embedding_graphs)):
    prev_edges = embedding_graphs[i - 1].number_of_edges()
    curr_edges = embedding_graphs[i].number_of_edges()
    pred_gs_labels.append(1 if curr_edges > prev_edges else 0)

predictions = np.array(pred_gs_labels)

tmp_labels = labels[1:]  # Since we don't do the first graph


# Compute metrics for Growth/Shrink task
aucroc = roc_auc_score(tmp_labels, predictions)
aucpr = average_precision_score(tmp_labels, predictions)

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


true_embeddings = list(true_embeddings)[1:]

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

# Call the create_animation function with the flattened lists; it takes a while, so it is commented out for now
#my_evaluator.create_animation(predicted_flat, target_flat, output_file=animation_path)