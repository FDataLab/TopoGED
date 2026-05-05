import networkx as nx
import numpy as np
import torch
from itertools import product
import math


def generate_candidates(graph:nx.DiGraph, nodes_1, flag, nodes_2=None, edgebank=None, is_directed=False): 
    """
    Generate all possible edges that we could add (directed)
    
    Args:
        graph (nx.DiGraph): The current graph that we are constructing
        nodes_1 (list): The set of source node ids
        flag (string): The edge type that we are making candidates for
        nodes_2 (list): The set of destination node ids
        edgebank (dict):  A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        is_directed (bool): Whether or not the graph is directed the processing changes slightly
    
    Returns:
        candidates (list): A list of tuples for all possible edges that can be added given the nodes
    """
    active_1 = [n for n in nodes_1 if graph.degree(n) < graph.nodes[n]['feat']['maxDegree']]
    
    if flag == 'o-n':
        active_2 = [n for n in nodes_2 if graph.degree(n) < graph.nodes[n]['feat']['maxDegree']]
        # Use sets for O(1) edge lookup
        existing_edges = set(graph.edges())
        
        candidates = []
        # Check O -> N
        for u in active_1:
            for v in active_2:
                if u != v and (u, v) not in existing_edges:
                    candidates.append((u, v))
        
        # Check N -> O if directed
        if is_directed:
            for u in active_2:
                for v in active_1:
                    if u != v and (u, v) not in existing_edges:
                        candidates.append((u, v))
        return candidates

    elif flag in ['o-o-nobank', 'o-o-bank', 'n-n']:
        existing_edges = set(graph.edges())
        candidates = []
        seen_undirected = set()
        
        # Convert edgebank to set for O(1) lookup if it exists
        eb_lookup = edgebank if edgebank is not None else {}

        for i, u in enumerate(active_1):
            for v in active_1: # For bank/nobank, we usually check against same set
                if u == v or (u, v) in existing_edges:
                    continue
                
                # Bank constraints logic
                if flag == 'o-o-nobank' and v in eb_lookup.get(u, set()):
                    continue
                elif flag == 'o-o-bank' and v not in eb_lookup.get(u, set()):
                    continue
                
                if not is_directed:
                    if (v, u) in seen_undirected:
                        continue
                    seen_undirected.add((u, v))
                
                candidates.append((u, v))
        return candidates


def predict_edges(graph, edge_type, node_types, edgebank, link_prediction_decoder, old_node_embeddings, top_k, graph_num, device, is_directed=False, train=False):
    """
    Predict what edges we will see in the graph, this is done by passing the node embeddings into the MLP and selecting the top_k most likely edges
    
    Args:
        graph (nx.DiGraph): The graph that we are currently constructing
        edge_type (string): The current edge type we are predicting edges for
        node_types (dict): A dictionary storing the old nodes and new nodes in ['old_nodes'] and ['new_nodes'] respectively
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        link_prediction_decoder (MLP NN): An MLP that predicts the probability of an edge occurring
        old_node_embeddings (dict): The embeddings of all old nodes we have seen up to this point
        top_k (int): How many edges we are going to select
        graph_num (int): Used for assigning a positional encoding onto the node embedding
    
    Returns:
        top_edges (list): The top_k edges that we have decided to add here
    """
    if edge_type in ['o-o-bank', 'o-o-nobank']:
        candidate_edges = generate_candidates(graph, nodes_1=node_types['old_nodes'], nodes_2=None, flag=edge_type, edgebank=edgebank, is_directed=is_directed)
    elif edge_type == 'n-n':
        candidate_edges = generate_candidates(graph, nodes_1=node_types['new_nodes'], nodes_2=None, flag=edge_type, is_directed=is_directed)
    elif edge_type == 'o-n':
        candidate_edges = generate_candidates(graph, nodes_1=node_types['old_nodes'], nodes_2=node_types['new_nodes'], flag=edge_type, is_directed=is_directed)
    
    if not candidate_edges or top_k <= 0:
        return []

    # 2. Vectorized Probability Prediction
    # Convert list of tuples to tensors immediately
    candidates_arr = np.array(candidate_edges)
    u_indices = torch.tensor(candidates_arr[:, 0], dtype=torch.long, device=device)
    v_indices = torch.tensor(candidates_arr[:, 1], dtype=torch.long, device=device)

    # Perform bulk inference
    with torch.no_grad():
        # This assumes old_node_embeddings is the tensor we've been building
        src_embeds = old_node_embeddings[u_indices]
        dst_embeds = old_node_embeddings[v_indices]
        
        # Single forward pass for ALL candidates
        probs = link_prediction_decoder(src_embeds, dst_embeds, edge_type).view(-1)

    # 3. Sort by probability (descending)
    # We move to CPU here to handle the NetworkX degree logic which is CPU-bound
    probs_cpu = probs.cpu().numpy()
    sorted_idx = np.argsort(-probs_cpu) # Negative for descending
    
    sorted_candidates = candidates_arr[sorted_idx]
    sorted_probs = probs_cpu[sorted_idx]

    # 4. Degree Constraint Logic (CPU Optimized)
    top_edges = []
    rejected_edges = []
    
    # Pre-fetch degrees to avoid repeated dict lookups in the loop
    # Using a local cache for speed
    node_feats = {n: graph.nodes[n]['feat'] for n in np.unique(candidates_arr)}

    for i in range(len(sorted_candidates)):
        if len(top_edges) >= top_k:
            break
            
        u, v = sorted_candidates[i]
        u_feat = node_feats[u]
        v_feat = node_feats[v]
        
        if u_feat['currDegree'] < u_feat['maxDegree'] and v_feat['currDegree'] < v_feat['maxDegree']:
            top_edges.append((int(u), int(v)))
            u_feat['currDegree'] += 1
            v_feat['currDegree'] += 1
        else:
            rejected_edges.append((int(u), int(v)))

    # 5. Fill remaining if degree constraints were too strict
    if len(top_edges) < top_k and rejected_edges:
        needed = top_k - len(top_edges)
        top_edges.extend(rejected_edges[:needed])
        # Update degrees for the forced edges
        for u, v in rejected_edges[:needed]:
            node_feats[u]['currDegree'] += 1
            node_feats[v]['currDegree'] += 1

    # Likely to be an issue in the early graphs
    if top_k != len(top_edges):
        print(f'[WARNING] There was an incorrect amount of predicted edges for Graph #{graph_num} and edgetype: {edge_type}')
        print(f'[WARNING] There were {len(top_edges)} edges when there was supposed to be {top_k} edges with {len(candidate_edges)} options')

    return top_edges


def sample_edges(count, tmp_graph, node_types, link_prediction_decoder, 
                 curr_embeddings, graph_num, device, edgebank=None, edge_type=None):
    sampled = set()

    if count > 0:
        sampled = predict_edges(tmp_graph, edge_type, node_types, edgebank, link_prediction_decoder, 
                                curr_embeddings, top_k=count, graph_num=graph_num, device=device)
            
    return list(sampled)
