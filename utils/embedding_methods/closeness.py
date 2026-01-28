import networkx as nx
import numpy as np


class EmbedCloseness():
    def __init__(self, num_buckets=10, include_weights=True):
        self.num_buckets = num_buckets
        self.weight_flag = include_weights
        
            
    def closeness_activation(self, threshold, graph):
        closeness = nx.closeness_centrality(graph)
        return {node for node, centrality in closeness.items() if centrality <= threshold}


    def compute_closeness_thresholds(self, graphs):
        all_closeness = []
        
        # Collect all closeness centrality values from all graphs
        for graph in graphs:
            closeness = nx.closeness_centrality(graph)
            all_closeness.extend(closeness.values())  # Collect centrality values as a list
        
        # Compute the thresholds based on percentiles of the closeness centrality distribution
        thresholds = np.percentile(all_closeness, np.linspace(0, 100, self.num_buckets + 1)[1:])
        
        return thresholds


    def process_graphs_for_embeddings(self, graphs):
        all_embeddings = []
        all_subgraphs = []
        
        # Compute global thresholds using the linear bucketing method
        thresholds = self.compute_closeness_thresholds(graphs)
        
        for graph in graphs: 
            active_data, subgraphs = self.graph_filtration(graph, thresholds)   
            all_embeddings.append(active_data)  # 20-dimensional embedding
            all_subgraphs.append(subgraphs)  # Subgraphs at each threshold
        
        return all_embeddings, all_subgraphs, thresholds


    # Filtration function for generating embeddings (50-dimensional active data)
    def graph_filtration(self, graph, thresholds):
        active_data = []
        subgraphs = []  # The subgraphs for each threshold

        for threshold in thresholds:
            # Get active node set based on the activation function
            active_node_set = self.closeness_activation(threshold, graph)
            
            # Filter edges to include only those between active nodes
            active_edges = {edge for edge in graph.edges(data=False) 
                            if edge[0] in active_node_set and edge[1] in active_node_set}
            
            # Get a subgraph to add
            subgraph = graph.subgraph(active_node_set).copy()
            subgraphs.append(subgraph)
            
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

        return active_data, subgraphs  # Returns a vector with features for all thresholds