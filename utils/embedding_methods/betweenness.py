import networkx as nx
import numpy as np


class EmbedBetweenness():
    def __init__(self, num_buckets=10, include_weights=True):
        self.num_buckets = num_buckets
        self.weight_flag = include_weights


    def betweenness_activation(self, threshold, graph):
        betweenness = nx.betweenness_centrality(graph)
        return {node for node, centrality in betweenness.items() if centrality <= threshold}


    def compute_betweenness_centrality_thresholds(self, graphs):
        all_betweenness_centralities = []
        
        # Collect all betweenness centrality values from all graphs
        for graph in graphs:
            # Compute betweenness centrality for all nodes in the graph
            betweenness = nx.betweenness_centrality(graph)
            
            # Collect the betweenness centrality values
            all_betweenness_centralities.extend(betweenness.values())
        
        # Compute the thresholds based on percentiles of the betweenness centrality distribution
        thresholds = np.percentile(all_betweenness_centralities, np.linspace(0, 100, self.num_buckets + 1)[1:])
        
        return thresholds
    
    
    def process_graphs_for_embeddings(self, graphs):
        all_embeddings = []
        
        # Compute global thresholds using the linear bucketing method
        thresholds = self.compute_betweenness_centrality_thresholds(graphs)
        
        for graph in graphs: 
            active_data = self.graph_filtration(graph, thresholds)   
            all_embeddings.append(active_data)  # 20-dimensional embedding
        
        return all_embeddings


    # Filtration function for generating embeddings (50-dimensional active data)
    def graph_filtration(self, graph, thresholds):
        active_data = []

        for threshold in thresholds:
            # Get active node set based on the activation function
            active_node_set = self.betweenness_activation(threshold, graph)
            
            # Filter edges to include only those between active nodes
            active_edges = {edge for edge in graph.edges(data=False) 
                            if edge[0] in active_node_set and edge[1] in active_node_set}
            
            # Compute features:
            node_count = len(active_node_set)  # Number of active nodes
            
            # In-degree: count edges directed towards active nodes (edges where the target is active)
            edges = sum(1 for edge in active_edges if edge[1] in active_node_set and edge[0] in active_node_set)
            
            # If we decide to include weight in the vectors
            if self.weight_flag:
                # Weight calculations:
                # In-weight: sum the weights of edges directed towards active nodes (edges where target is active)
                weight = sum(graph[edge[0]][edge[1]].get('value', 1) for edge in active_edges if edge[1] in active_node_set and edge[0] in active_node_set)
                
                # Append the features for this threshold
                active_data.extend([node_count, edges, weight])
            
            else:
                active_data.extend([node_count, edges])

        return active_data  # Returns a vector with features for all thresholds