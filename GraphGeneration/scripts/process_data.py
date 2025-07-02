from collections import defaultdict
import networkx as nx
import numpy as np
import torch
from GraphGeneration.models.temporal_gnn.script.config import args
from GraphGeneration.scripts.compute_embedding import compute_linear_gnn_embeddings, compute_node2vec_embeddings, compute_node_embeddings_GCLSTM, compute_node_embeddings_HTGN, compute_node_embeddings_LSTM

def process_starter_graph(graphs: list, thresholds: list, encoder_model):
    """
    Process our very first graph, this is our 'primer' used to construct the later graphs
    We do this since we need some node embeddings and features to start with
    
    Args:
        graph (nx.DiGraph): The first graph in the dataset, which we are embedding the nodes for
        thresholds (list): A list of integers, from TopER, used to assign the max degree of a node
    
    Returns:
        final_embeddings (dict): Our embeddings for all seen nodes
        degree_clusters (dict): A dictionary of {'degree': [created_embedding]} that we use to assign the embeddings for new nodes
        existing_nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing reappearance probabilities
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
    """
    
    # Utility function for CUDA
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)
    
    # Our return values
    edgebank = {}
    existing_nodes = {}
    degree_clusters = defaultdict(list)
    final_embeddings = {}
    
    # Assign base features
    for graph_num, graph in enumerate(graphs):
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
            curr_embeddings = compute_node2vec_embeddings(graph)
        elif args.embeddingType == 'Linear':
            curr_embeddings = compute_linear_gnn_embeddings(graph)
        elif args.embeddingType == 'LSTM':       
            # graph_snapshots = [G_0, G_1, ..., G_T]  # each G must have node['feat']
            curr_embeddings = compute_node_embeddings_LSTM(graphs[:graph_num + 1], encoder_model)
        elif args.embeddingType == 'GCLSTM':
            curr_embeddings = compute_node_embeddings_GCLSTM(graphs[:graph_num + 1], encoder_model)
        elif args.embeddingType == 'HTGN':
            curr_embeddings = compute_node_embeddings_HTGN(graphs[:graph_num + 1], encoder_model)
        else: curr_embeddings = {}
        final_embeddings.update(curr_embeddings)  # Write the embeddings to return for later predictions
        
        # process the nodes for old node evaluation
        for node in graph.nodes(data=False):
            existing_nodes[node] = (graph_num, graph.degree(node))
                
        # Build the edgebank
        for u, v in graph.edges():  # Accessing the graph directly
            edgebank.setdefault(u, []).append(v)
        
        # Process the degree clusters for generating the embeddings for new nodes
        for node in graph.nodes():        
            degree = graph.nodes[node]['feat']['maxDegree']
            
            curr_embedding = final_embeddings[node]
            old_embedding = degree_clusters.get(degree, [])
            
            # Average the embeddings if both exist
            if old_embedding is not None and len(old_embedding) > 0:
                new_embedding = (to_numpy(curr_embedding) + to_numpy(old_embedding)) / 2
            else:
                new_embedding = curr_embedding
                
            degree_clusters[degree] = new_embedding  # Add the embedding
    
    return final_embeddings, degree_clusters, existing_nodes, edgebank

def modifyGraphIds(graphs, thresholds):
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
        updated_sublist = []
        for graph in graph_list:
            curr_mapping = {}  # Mapping applies to this specific graph

            for node in graph.nodes:
                # Ensure that 'feat' exists and is properly initialized
                if 'feat' not in graph.nodes[node]:
                    graph.nodes[node]['feat'] = {}

                # Mark the node as new or old
                if node not in node_mapping:
                    node_mapping[node] = new_id
                    new_id += 1
                    graph.nodes[node]['feat']['type'] = 1  # Node is new
                else:
                    graph.nodes[node]['feat']['type'] = 0  # Node is old

                # Map the node and update the ID in the feature dictionary
                curr_mapping[node] = node_mapping[node]
                graph.nodes[node]['feat']['id'] = node_mapping[node]

                node_degree = graph.degree(node)  # Current node's degree

                # If thresholds are available, calculate the max degree based on thresholds
                if np.any(thresholds):
                    graph.nodes[node]['feat']['currDegree'] = node_degree
                    graph.nodes[node]['feat']['maxDegree'] = next((t for t in thresholds if node_degree <= t), thresholds[-1])
                else:
                    # If no thresholds, use degree as maxDegree
                    graph.nodes[node]['feat']['currDegree'] = node_degree
                    graph.nodes[node]['feat']['maxDegree'] = node_degree

            # Relabel the graph nodes according to the new IDs
            relabeled_graph = nx.relabel_nodes(graph, curr_mapping, copy=True)

            # Preserve features for relabeled nodes
            for old_node, new_node in curr_mapping.items():
                relabeled_graph.nodes[new_node]['feat'] = graph.nodes[old_node]['feat'].copy()
                # print(f"Old Node: {old_node}, Features: {graph.nodes[old_node]['feat']}")
                # print(f"New Node: {new_node}, Features: {relabeled_graph.nodes[new_node]['feat']}")

            updated_sublist.append(relabeled_graph)
        updated_graphs.append(updated_sublist)

    return updated_graphs, len(node_mapping)

def build_edgebanks_from_start(graphs, days=5):
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
        for j in range(max(i - days, 0), i):  # Loop through all previous graphs (graphs 0 to i-1)
            for u, v in graphs[j][-1].edges():  # Accessing the graph directly
                u_key = u
                v_key = v
                curr_edgebank.setdefault(u_key, []).append(v_key)  # Add edge from u to v
                curr_edgebank.setdefault(v_key, []).append(u_key)  # Add edge from u to v

        edgebanks.append(curr_edgebank)  # Append the current edgebank to the list

    return edgebanks