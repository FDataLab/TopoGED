import networkx as nx
import numpy as np

class EmbedForman:
    def __init__(self, num_buckets=10):
        self.num_buckets = num_buckets


    def forman_ricci_activation(self, threshold, graph):
        # Extract curvature values for nodes
        active_edges = {(u, v) for u, v, data in graph.edges(data=True) 
                        if data.get("formanCurvature", float('inf')) <= threshold}

        # Collect all nodes connected to these edges
        return {node for edge in active_edges for node in edge}


    def compute_forman_ricci_thresholds(self, graphs):
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
        thresholds = np.percentile(all_forman_ricci_curvatures, np.linspace(0, 100, self.num_buckets + 1)[1:])
        return thresholds


    def process_graphs_for_embeddings(self, graphs, is_directed=True):
        all_embeddings = []
        
        if not is_directed:
            new_graphs = []
            for graph in graphs:
                new_graphs.append(graph.to_directed())
            graphs = new_graphs
        
        # Compute global thresholds using the linear bucketing method
        thresholds = self.compute_forman_ricci_thresholds(graphs)
        
        for graph in graphs: 
            active_data = self.graph_filtration(graph, thresholds)   
            all_embeddings.append(active_data)  # 20-dimensional embedding
        
        # To account for making the directional edges
        if not is_directed:
            for j in range(len(all_embeddings)):
                for i in range(0, len(all_embeddings[0]) - 1, 3):
                    all_embeddings[j][i + 1] /= 2
                    all_embeddings[j][i + 2] /= 2
        
        return all_embeddings


    # Filtration function for generating embeddings (50-dimensional active data)
    def graph_filtration(self, graph, thresholds):
        active_data = []

        for threshold in thresholds:
            # Get active node set based on the activation function
            active_node_set = self.forman_ricci_activation(threshold, graph)
            
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