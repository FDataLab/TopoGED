import numpy as np


# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci


def degree_activation(threshold, graph):
    degrees = dict(graph.degree())
    return {node for node, degree in degrees.items() if degree <= threshold}


def closeness_activation(threshold, graph):
    closeness = nx.closeness_centrality(graph)
    return {node for node, centrality in closeness.items() if centrality <= threshold}

def betweenness_activation(threshold, graph):
    betweenness = nx.betweenness_centrality(graph)
    return {node for node, centrality in betweenness.items() if centrality <= threshold}


def weight_activation(threshold, graph):
    return {node for node in graph.nodes if sum(data.get('value', 1) for _, _, data in graph.edges(node, data=True)) <= threshold} 


def degree_centrality_activation(threshold, graph):
    degree_centrality = nx.degree_centrality(graph)
    return {node for node, centrality in degree_centrality.items() if centrality <= threshold}


def forman_ricci_activation(threshold, graph):
    # Extract curvature values for nodes
    active_edges = {(u, v) for u, v, data in graph.edges(data=True) 
                    if data.get("formanCurvature", float('inf')) <= threshold}

    # Collect all nodes connected to these edges
    return {node for edge in active_edges for node in edge}


def ollivier_ricci_activation(threshold, graph):
    ollivier_ricci = OllivierRicci(graph)
    ollivier_ricci.compute_ricci_curvature()

    # Extract curvature values for nodes
    return {node for node, curvature in ollivier_ricci.G.nodes(data="ricciCurvature") if curvature >= threshold}


def compute_degree_thresholds(graphs, num_buckets=10):
    all_degrees = []
    
    # Collect all degrees from all graphs
    for graph in graphs:
        degrees = dict(graph.degree()).values()
        all_degrees.extend(degrees)
    
    # Compute the thresholds based on percentiles of the degree distribution
    thresholds = np.percentile(all_degrees, np.linspace(0, 100, num_buckets + 1)[1:])
    
    return thresholds


def compute_closeness_thresholds(graphs, num_buckets=10):
    all_closeness = []
    
    # Collect all closeness centrality values from all graphs
    for graph in graphs:
        closeness = nx.closeness_centrality(graph)
        all_closeness.extend(closeness.values())  # Collect centrality values as a list
    
    # Compute the thresholds based on percentiles of the closeness centrality distribution
    thresholds = np.percentile(all_closeness, np.linspace(0, 100, num_buckets + 1)[1:])
    
    return thresholds


def compute_betweenness_centrality_thresholds(graphs, num_buckets=10):
    all_betweenness_centralities = []
    
    # Collect all betweenness centrality values from all graphs
    for graph in graphs:
        # Compute betweenness centrality for all nodes in the graph
        betweenness = nx.betweenness_centrality(graph)
        
        # Collect the betweenness centrality values
        all_betweenness_centralities.extend(betweenness.values())
    
    # Compute the thresholds based on percentiles of the betweenness centrality distribution
    thresholds = np.percentile(all_betweenness_centralities, np.linspace(0, 100, num_buckets + 1)[1:])
    
    return thresholds


def compute_degree_centrality_thresholds(graphs, num_buckets=10):
    all_degree_centrality = []
    
    # Collect all degree centrality values from all graphs
    for graph in graphs:
        degree_centrality = nx.degree_centrality(graph)
        all_degree_centrality.extend(degree_centrality.values())  # Collect centrality values as a list
    
    # Compute the thresholds based on percentiles of the degree centrality distribution
    thresholds = np.percentile(all_degree_centrality, np.linspace(0, 100, num_buckets + 1)[1:])
    
    return thresholds


def compute_weight_thresholds(graphs, num_buckets=10):
    all_weighted_degrees = []
    
    # Collect all weighted degrees from all graphs
    for graph in graphs:
        weighted_degrees = []
        
        if graph.is_directed():
            # For directed graphs, sum incoming and outgoing edge values
            for node in graph.nodes():
                in_value = sum(value for _, _, value in graph.in_edges(node, data='value'))
                out_value = sum(value for _, _, value in graph.out_edges(node, data='value'))
                weighted_degrees.append(in_value + out_value)
        else:
            # For undirected graphs, sum values of all edges
            for node in graph.nodes():
                weighted_degree = sum(value for _, _, value in graph.edges(node, data='value'))
                weighted_degrees.append(weighted_degree)
        
        all_weighted_degrees.extend(weighted_degrees)
    
    # Compute the thresholds based on percentiles of the weighted degree distribution
    thresholds = np.percentile(all_weighted_degrees, np.linspace(0, 100, num_buckets + 1)[1:])
    
    return thresholds


def compute_forman_ricci_thresholds(graphs, num_buckets=10):
    all_forman_ricci_curvatures = []
    
    # Collect all Forman-Ricci curvatures from all graphs
    for graph in graphs:
        edWts = nx.get_edge_attributes(graph,'value')
        formanRicci = {}
        for ed in graph.edges():
            # See calc at   https://doi.org/10.1016/j.chaos.2018.11.031
            v1 = ed[0]
            v2 = ed[1]
            #Assuming all vertex weights are 1 since unspecified
            wtv1 = 1.0
            wtv2 = 1.0
            we = edWts[ed] # weight of edge

            sumV1I = 0.0
            sumV2O = 0.0
            for v1pred in graph.predecessors(v1):
                sumV1I += wtv1/np.sqrt(edWts[(v1pred,v1)] * we)
            for v2nxt in graph.successors(v2):
                sumV2O += wtv2/np.sqrt(edWts[(v2,v2nxt)] * we)
            # check if bidirection, then would need to consider V1 out and V2 int as well
            if any(temp == v2 for temp in graph.predecessors(v1)):
                for v1nxt in graph.successors(v1):
                    sumV1I += wtv1/np.sqrt(edWts[(v1,v1nxt)] * we)
                for v2pred in graph.predecessors(v2):
                    sumV2O += wtv2/np.sqrt(edWts[(v2pred,v2)] * we)
            formanRicci[ed] = np.round(we * (wtv1/we - sumV1I) + we * (wtv2/we - sumV2O), 3)
        
        nx.set_edge_attributes(graph, formanRicci, "formanCurvature")        
        
        all_forman_ricci_curvatures.extend(formanRicci.values())
    
    # Compute the thresholds based on percentiles of the Forman-Ricci curvature distribution
    thresholds = np.percentile(all_forman_ricci_curvatures, np.linspace(0, 100, num_buckets + 1)[1:])
    return thresholds



from collections import Counter
# Helper function
def wasserstein_distance(neighbors_u, neighbors_v):
    pass


def get_neighbors(graph, u, v):
    pass 


def distance(graph, u, v):
    pass


def compute_ollivier_ricci_curvature(graph, u, v):
    neighbors_u = None
    neighbors_v = None
    curvature = 1 - (wasserstein_distance(neighbors_u, neighbors_v) / distance(graph, u, v))
    return curvature


def compute_ollivier_ricci_thresholds(graphs, num_buckets=10):
    all_ollivier_ricci_curvatures = []
    
    # Collect all Ollivier-Ricci curvatures from all graphs
    for graph in graphs:
        ollivier_ricci_curvatures = []
        
        # For each edge in the graph, calculate the Ollivier-Ricci curvature
        for u, v in graph.edges():
            curvature = compute_ollivier_ricci_curvature(graph, u, v)
            graph[u][v]['ollivierRicci'] = curvature
            ollivier_ricci_curvatures.append(curvature)
        
        all_ollivier_ricci_curvatures.extend(ollivier_ricci_curvatures)
    
    # Compute the thresholds based on percentiles of the Ollivier-Ricci curvature distribution
    thresholds = np.percentile(all_ollivier_ricci_curvatures, np.linspace(0, 100, num_buckets + 1)[1:])
    
    return thresholds

def graph_to_undirected(graph):
    undirected_graph = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if undirected_graph.has_edge(u, v):
            # Combine the values, e.g., take the max
            undirected_graph[u][v]['value'] = max(undirected_graph[u][v]['value'], data['value'])
        else:
            undirected_graph.add_edge(u, v, **data)

    # Inspect the undirected graph
    return undirected_graph


def process_graphs_for_embeddings(graphs, thresholds_func, activation_func, num_buckets=10):
    all_embeddings = []
    
    # Compute global thresholds using the linear bucketing method
    thresholds = thresholds_func(graphs, num_buckets=num_buckets)
    
    for graph in graphs: 
        active_data = graph_filtration(graph,  activation_func, thresholds)   
        all_embeddings.append(active_data)  # 20-dimensional embedding
    
    return all_embeddings


# Filtration function for generating embeddings (50-dimensional active data)
def graph_filtration(graph, activation_func, thresholds):
    active_data = []

    for threshold in thresholds:
        # Get active node set based on the activation function
        active_node_set = activation_func(threshold, graph)
        
        # Filter edges to include only those between active nodes
        active_edges = {edge for edge in graph.edges(data=False) 
                        if edge[0] in active_node_set and edge[1] in active_node_set}
        
        # Compute features:
        node_count = len(active_node_set)  # Number of active nodes
        
        # In-degree: count edges directed towards active nodes (edges where the target is active)
        edges = sum(1 for edge in active_edges if edge[1] in active_node_set and edge[0] in active_node_set)
        
        # Weight calculations:
        # In-weight: sum the weights of edges directed towards active nodes (edges where target is active)
        weight = sum(graph[edge[0]][edge[1]].get('value', 1) for edge in active_edges if edge[1] in active_node_set and edge[0] in active_node_set)
        
        
        # Append the features for this threshold
        active_data.extend([node_count, edges, weight])

    return active_data  # Returns a vector with features for all thresholds