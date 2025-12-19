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


def modifyGraphIds(graphs, thresholds, days_back=5):
    all_mappings = []
    updated_graphs = []
    global_node_counter = 0

    for i, graph_list in enumerate(graphs):
        curr_mapping = {}
        curr_sublist = []
        start = max(0, i - days_back)

        # Combine past mappings into one dict
        combined_past = {k: v for mapping in all_mappings[start:i] for k, v in mapping.items()}

        for subgraph in graph_list:
            for node in subgraph.nodes():
                if node not in curr_mapping:
                    if node in combined_past:
                        # Reuse old mapping
                        curr_mapping[node] = combined_past[node]
                    else:
                        # Assign new ID
                        curr_mapping[node] = global_node_counter
                        global_node_counter += 1

            # Relabel with new mapping
            relabeled_g = nx.relabel_nodes(subgraph, curr_mapping, copy=True)

            # Add node features
            for old_node, new_node in curr_mapping.items():
                relabeled_g.nodes[new_node].setdefault('feat', {})
                relabeled_g.nodes[new_node]['feat']['id'] = new_node
                # Mark as new (type=1) if node not seen in combined_past
                relabeled_g.nodes[new_node]['feat']['type'] = 1 if old_node not in combined_past else 0

                node_degree = subgraph.degree(old_node)
                if np.any(thresholds):
                    relabeled_g.nodes[new_node]['feat']['currDegree'] = node_degree
                    relabeled_g.nodes[new_node]['feat']['maxDegree'] = next(
                        (t for t in thresholds if node_degree <= t), thresholds[-1]
                    )
                else:
                    relabeled_g.nodes[new_node]['feat']['currDegree'] = node_degree
                    relabeled_g.nodes[new_node]['feat']['maxDegree'] = node_degree

            curr_sublist.append(relabeled_g)

        updated_graphs.append(curr_sublist)
        all_mappings.append(curr_mapping)

    return updated_graphs, global_node_counter - 1
                    

def build_edgebanks_from_start(graphs, is_directed, days_back=5):
    """
    Build the edgebanks for each graph in graphs, stores all edges from graph i-1 in each index i
    
    Args:
        graphs (list(nx.Graph)): A list of nx Graphs that we will build our edgebanks from
        is_directed (bool): A flag for representing if the graph is a DiGraph or not (True/False)
        days_back (int): How many days back we look to determine our edge bank; also known as our context window (default 5)
        
    Returns:
        edgebanks (list(dict)): A list of dictionary edgebanks that store all edges from the previous graphs in each index
    """
    edgebanks = [{}]  # Initialize an empty list for edgebanks

    # Loop over all graphs (starting from the second graph)
    for i in range(1, len(graphs)):
        curr_edgebank = {}

        # Add edges from all previous graphs (not the current graph)
        for j in range(max(i - days_back, 0), i):  # Loop through all previous graphs (graphs i - days to i-1)
            
            for u, v in graphs[j][-1].edges():  # Accessing the graph directly
                curr_edgebank.setdefault(u, set()).add(v)  # Add edge from u to v
                if not is_directed:
                    curr_edgebank.setdefault(v, set()).add(u)
        edgebanks.append(curr_edgebank)

    return edgebanks