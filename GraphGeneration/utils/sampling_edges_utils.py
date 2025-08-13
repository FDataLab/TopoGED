import networkx as nx
import numpy as np
import torch
from itertools import product
import math
from GraphGeneration.models.temporal_gnn.script.config import args

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

    # Degree constraint checker
    def can_add_edge(u, v):
        return (
            u in graph.nodes and v in graph.nodes and
            u != v and
            not graph.has_edge(u, v) and
            graph.degree(u) < graph.nodes[u]['feat']['maxDegree'] and
            graph.degree(v) < graph.nodes[v]['feat']['maxDegree']
        )

    if flag == 'o-n':
        # Edge between one old and one new node, and only one of them is in edgebank
        candidates = [
            (u, v) for u, v in product(nodes_1, nodes_2)
            if can_add_edge(u, v) and (
                (u in edgebank and v not in edgebank) or (u not in edgebank and v in edgebank)
            )
        ]

    elif flag == 'o-o-nobank':
        # Edge between two old nodes that are not in edgebank
        candidates = [
            (u, v) for u, v in product(nodes_1, nodes_1)
            if can_add_edge(u, v) and v not in edgebank.get(u, [])
        ]

    elif flag == 'o-o-bank':
        # Edge between two old nodes that are in edgebank
        candidates = [
            (u, v) for u, v in product(nodes_1, nodes_1)
            if can_add_edge(u, v) and v in edgebank.get(u, [])
        ]

    elif flag == 'n-n':
        # Edge between two new nodes
        candidates = [
            (u, v) for u, v in product(nodes_1, nodes_1)
            if can_add_edge(u, v)
        ]
    
    return candidates


def predict_edges(graph, edge_type, node_types, edgebank, link_prediction_decoder, old_node_embeddings, top_k, graph_num, device):
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
    if edge_type == 'o-o-bank' or edge_type == 'o-o-nobank':
        available_nodes = node_types['old_nodes']
        candidate_edges = generate_candidates(graph, nodes_1=available_nodes, nodes_2=None, flag=edge_type, edgebank=edgebank)

    elif edge_type == 'n-n':
        available_nodes = node_types['new_nodes']
        candidate_edges = generate_candidates(graph, nodes_1=available_nodes, nodes_2=None, flag=edge_type, edgebank=edgebank)

    elif edge_type == 'o-n':
        nodes = set(node_types['old_nodes']).union(node_types['new_nodes'])  # Since all nodes are valid candidates
        candidate_edges = generate_candidates(graph, nodes_1=nodes, nodes_2=nodes, flag=edge_type, edgebank=edgebank) #TODO kha: check this
    
    # Predict edge probabilities using the MLP
    edge_probs = []
    any_node = next(iter(old_node_embeddings))
    embed_dim = len(old_node_embeddings[any_node])
    
    for u, v in candidate_edges:
        if u not in old_node_embeddings and edge_type in ['n-n', 'o-n']:
            old_node_embeddings[u] = torch.zeros(embed_dim, device=device)
        if v not in old_node_embeddings and edge_type in ['n-n', 'o-n']:
            old_node_embeddings[v] = torch.zeros(embed_dim, device=device)
            
        src_embed = old_node_embeddings[u]
        dst_embed = old_node_embeddings[v]

        # Convert to torch.Tensor if necessary
        if isinstance(src_embed, np.ndarray):
            src_embed = torch.tensor(src_embed, dtype=torch.float32).to(device)
        if isinstance(dst_embed, np.ndarray):
            dst_embed = torch.tensor(dst_embed, dtype=torch.float32).to(device)

        # Add batch dimension if needed
        if src_embed.dim() == 1:
            src_embed = src_embed.unsqueeze(0)
        if dst_embed.dim() == 1:
            dst_embed = dst_embed.unsqueeze(0)
            
        # Predict edge probability
        prob = link_prediction_decoder(src_embed, dst_embed, edge_type)
        
        edge_probs.append((u, v, prob.item()))

    # Sort and select top_k
    edge_probs.sort(key=lambda x: x[2], reverse=True)
    top_edges = [(u, v) for u, v, _ in edge_probs[:top_k]]

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
