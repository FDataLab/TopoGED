import numpy as np

def degree_activation(degrees, threshold):
    return {node for node, degree in degrees.items() if degree <= threshold}

# Compute global degree thresholds based on the percentile distribution of degrees
def compute_degree_thresholds(graphs, num_thresholds=10):
    all_degrees = []
    
    # Collect all degrees from all graphs
    for graph in graphs:
        degrees = dict(graph.degree()).values()
        all_degrees.extend(degrees)
    
    # Compute the thresholds based on percentiles of the degree distribution
    thresholds = np.percentile(all_degrees, np.linspace(0, 100, num_thresholds + 1)[1:])
    
    return thresholds

# Filtration function for generating embeddings (20-dimensional active data)
def graph_filtration(graph, metric, activation_func, thresholds):
    active_data = []
    
    for threshold in thresholds:
        active_node_set = activation_func(metric, threshold)
        active_edges = {edge for edge in graph.edges() if edge[0] in active_node_set and edge[1] in active_node_set}
        # Each filtration gives 2 features: (number of active edges, number of active nodes)
        active_data.extend([len(active_node_set), len(active_edges)])  # 2 features per threshold
    
    return active_data  # 20-dimensional vector

# Process the graphs and generate embeddings (each embedding is 20-dimensional)
def process_graphs_for_embeddings(graphs):
    all_embeddings = []
    
    # Compute global thresholds based on all graphs
    thresholds = compute_degree_thresholds(graphs, num_thresholds=10)
    
    for graph in graphs:
        metric = dict(graph.degree())  # Degree centrality as metric
        active_data = graph_filtration(graph, metric, degree_activation, thresholds)   
        all_embeddings.append(active_data)  # 20-dimensional embedding
    
    return all_embeddings