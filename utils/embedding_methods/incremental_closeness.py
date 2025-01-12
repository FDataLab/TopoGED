import networkx as nx
import numpy as np
from collections import defaultdict


class EmbedIncrementalCloseness():
    def __init__(self, num_buckets=10):
        self.num_buckets = num_buckets
        self.closeness_cache = defaultdict(dict)  # Cache for incremental closeness updates
        
    def incremental_closeness_activation(self, threshold, graph):
        # Incremental computation of closeness centrality
        closeness = self.incremental_closeness_centrality(graph)
        return {node for node, centrality in closeness.items() if centrality <= threshold}
    
    def incremental_closeness_centrality(self, graph):
        """
        Calculate or retrieve closeness centrality incrementally.
        """
        if graph in self.closeness_cache:
            return self.closeness_cache[graph]
        
        # Otherwise, compute it from scratch for now
        closeness = nx.closeness_centrality(graph)
        self.closeness_cache[graph] = closeness
        return closeness

    def compute_closeness_thresholds(self, graphs):
        all_closeness = []
        
        # Collect all closeness centrality values from all graphs
        for graph in graphs:
            closeness = self.incremental_closeness_centrality(graph)
            all_closeness.extend(closeness.values())  # Collect centrality values as a list
        
        # Compute the thresholds based on percentiles of the closeness centrality distribution
        thresholds = np.percentile(all_closeness, np.linspace(0, 100, self.num_buckets + 1)[1:])
        
        return thresholds

    def process_graphs_for_embeddings(self, graphs):
        all_embeddings = []
        
        # Compute global thresholds using the linear bucketing method
        thresholds = self.compute_closeness_thresholds(graphs)
        
        for graph in graphs: 
            active_data = self.graph_filtration(graph, thresholds)   
            all_embeddings.append(active_data)  # 20-dimensional embedding
        
        return all_embeddings

    # Filtration function for generating embeddings (50-dimensional active data)
    def graph_filtration(self, graph, thresholds):
        active_data = []

        for threshold in thresholds:
            # Get active node set based on the activation function
            active_node_set = self.incremental_closeness_activation(threshold, graph)
            
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
