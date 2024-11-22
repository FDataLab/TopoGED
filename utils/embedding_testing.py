import numpy as np


# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def degree_activation(degrees, threshold, graph):
    return {node for node, degree in degrees.items() if degree <= threshold}


# In-degree activation function: Activates nodes based on their in-degree
def in_degree_activation(degrees, threshold, graph):
    return {node for node in degrees if graph.in_degree(node) <= threshold}


# Out-degree activation function: Activates nodes based on their out-degree
def out_degree_activation(degrees, threshold, graph):
    return {node for node in degrees if graph.out_degree(node) <= threshold}


def betweenness_activation(betweenness, threshold, graph):
    pass 


def closeness_activation(closeness, threshold, graph):
    pass 


def weight_activation(weight, threshold, graph):
    pass 


def compute_degree_thresholds(graphs, num_buckets=10):
    all_degrees = []
    
    # Collect all degrees from all graphs
    for graph in graphs:
        degrees = dict(graph.degree()).values()
        all_degrees.extend(degrees)
    
    # Compute the thresholds based on percentiles of the degree distribution
    thresholds = np.percentile(all_degrees, np.linspace(0, 100, num_buckets + 1)[1:])
    
    return thresholds


def process_graphs_for_embeddings(graphs):
    all_embeddings = []
    
    # Compute global thresholds using the linear bucketing method
    thresholds = compute_degree_thresholds(graphs, num_buckets=10)
    
    for graph in graphs:
        metric = dict(graph.degree())  # Degree centrality as metric
        active_data = graph_filtration(graph, metric, out_degree_activation, thresholds)   
        all_embeddings.append(active_data)  # 20-dimensional embedding
    
    return all_embeddings


# Filtration function for generating embeddings (50-dimensional active data)
def graph_filtration(graph, metric, activation_func, thresholds):
    active_data = []

    for threshold in thresholds:
        # Get active node set based on the activation function
        active_node_set = activation_func(metric, threshold, graph)
        
        # Filter edges to include only those between active nodes
        active_edges = {edge for edge in graph.edges(data=False) 
                        if edge[0] in active_node_set and edge[1] in active_node_set}
        
        # Compute features:
        node_count = len(active_node_set)  # Number of active nodes
        
        # In-degree: count edges directed towards active nodes (edges where the target is active)
        in_degree = sum(1 for edge in active_edges if edge[1] in active_node_set and edge[0] in active_node_set)
        
        # Out-degree: count edges directed from active nodes (edges where the source is active)
        out_degree = sum(1 for edge in active_edges if edge[0] in active_node_set and edge[1] in active_node_set)
        
        # Weight calculations:
        # In-weight: sum the weights of edges directed towards active nodes (edges where target is active)
        in_weight = sum(graph[edge[0]][edge[1]].get('value', 1) for edge in active_edges if edge[1] in active_node_set and edge[0] in active_node_set)
        
        # Out-weight: sum the weights of edges directed from active nodes (edges where source is active)
        out_weight = sum(graph[edge[0]][edge[1]].get('value', 1) for edge in active_edges if edge[0] in active_node_set and edge[1] in active_node_set)
        
        # Append the features for this threshold
        active_data.extend([node_count, in_degree, out_degree, in_weight, out_weight])

    return active_data  # Returns a vector with features for all thresholds