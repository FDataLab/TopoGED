import random
import torch
import numpy as np 
import networkx as nx
from collections import defaultdict
from GraphGeneration.scripts.compute_embedding import compute_node2vec_embeddings

from GraphGeneration.encoders.TGN.utils.my_utils import build_graph_data, GraphData
from GraphGeneration.encoders.TGN.utils.utils import get_neighbor_finder


def compute_reappearance_probabilities(graphs, days_back, decay_factor=1.0, alpha=3.0, beta=5.0, epsilon=1e-8):
    """
    Compute the probability for each node to reappear given how long ago it was seen and its latest degree
    Nodes of higher degree, and nodes seen more recently are preferred
    
    Args:
        graphs: The snapshots we have observed up to the t_curr
        t_curr (int): The current graph number we are on, used to compute probabilities
        days_back (int)
        decay_factor (float): How quickly the recency of a node decays. Higher means that the nodes seen long ago decay slower
        alpha (float): Our decay constant, controls how influential degree is (alpha > 1 means that it prefers degree, alpha < 1 means that it matters less)
        epsilon (float): Prevents having 0 probabilities for a node, and thus prevents numpy errors later on
    
    Returns:
        probs (dict):  A dictionary of {node_id: percent probability} probabilities for each node in nodes
    """
    nodes = dict()
    frequency = defaultdict(int)
    # Create the nodes dict degree history
    # nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing probabilities
    
    for t, G in enumerate(graphs):
        for node in G.nodes():
            nodes[node] = (t, G.degree(node))
            frequency[node] += 1
    
    if not nodes:
        return {}
    
    max_degree = max(degree for _, (_, degree) in nodes.items())
    
    probs = {}
    t_curr = len(graphs)  # Makes the formula work best
    
    for node_id, (last_seen, degree) in nodes.items():
        recency_score = np.exp(-(t_curr - last_seen) / decay_factor) if decay_factor > 0 else 1.0
        degree_score = (degree / max_degree) ** alpha if max_degree > 0 else 1.0
        frequency_score = (frequency[node_id] ** beta) if beta > 0 else 1.0
        
        raw_score = recency_score * degree_score * frequency_score
        
        probs[node_id] = max(raw_score, epsilon)  # Apply epsilon floor to avoid exact 0

    # Normalize to make a valid probability distribution
    total = sum(probs.values())
    for node in probs:
        probs[node] /= total

    return probs
       
       
def get_node_features(constructing_graph, previous_graphs, thresholds, graph_description, old_nodes, new_nodes):
    """
    Assign the maximum degree of a node, either using its last seen degree (if args.oldDegree == True) or randomly giving it one
    
    Args:
        constructing_graph (nx.DiGraph): The Graph we are attempting to construct, we assign node features here
        thresholds (list): The thresholds according to TopER, assigns the maximum degree of nodes
        embedding (list): The current TopER graph embedding vector used to figure out how many nodes have a degree
        old_nodes (list): The list of old nodes that are in the graph
        new_nodes (list): The list of new nodes that are in the graph
        
    Returns:
        None
    """
    
    existing_nodes = dict()
    
    # Add the nodes to the graph
    for node in old_nodes:
        constructing_graph.add_node(node)
        feature_dict_old = {'id': node}  
        constructing_graph.nodes[node]['feat'] = feature_dict_old
    for node in new_nodes:
        constructing_graph.add_node(node)
        feature_dict_new = {'id': node}  
        constructing_graph.nodes[node]['feat'] = feature_dict_new
        
    # Create the nodes dict degree history
    # nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing probabilities
    for t, G in enumerate(previous_graphs):
        for node in G.nodes():
            degree = G.degree(node)
            existing_nodes[node] = (t, degree)
    
    degree_counts = [graph_description[i][0] for i in range(0, len(graph_description))]
    
    degree_dict = {thresholds[i]: degree_counts[i] for i in range(len(thresholds))}
    
    degree_assignment = []  # This will store the assigned degrees
    
    for degree, count in degree_dict.items():
        degree_assignment.extend([degree] * count)
    # print(degree_assignment)
    random.shuffle(degree_assignment)
    

    for node in old_nodes:
        old_degree = existing_nodes[node][1]

        # Find the smallest degree in degree_assignment ≥ old_degree
        suitable_degrees = [d for d in degree_assignment if d >= old_degree]
        if suitable_degrees:
            assigned_degree = min(suitable_degrees)
            degree_assignment.remove(assigned_degree)
        else:
            assigned_degree = degree_assignment.pop()
        
        constructing_graph.nodes[node]['feat']['currDegree'] = 0
        constructing_graph.nodes[node]['feat']['maxDegree'] = assigned_degree
    
    # Give the node a random new degree    
    for node in new_nodes:
        assigned_degree = degree_assignment.pop()

        constructing_graph.nodes[node]['feat']['currDegree'] = 0
        constructing_graph.nodes[node]['feat']['maxDegree'] = assigned_degree
 
    return constructing_graph
    
def update_degrees(graph: nx.DiGraph):
    """
    After updating the graph, between edge types, update the nodes current degree feature
    
    Args:
        graph (nx.DiGraph): The current graph in construction
        
    Returns:
        None
    """
    for node in graph.nodes(data=False):
        if 'feat' not in graph.nodes[node]:
            graph.nodes[node]['feat'] = {'id': node}  
            graph.nodes[node]['feat']['currDegree'] = 0
        else:
            graph.nodes[node]['feat']['currDegree'] = graph.degree(node)
     
     
def generate_tgcn_node_features(target_graphs, embedding_dim, feature_type='binary', device="cpu"):
    all_nodes = sorted(set(node for graphs in target_graphs for node in graphs[-1].nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    
    if feature_type == 'node2vec':
        node_features = torch.zeros((num_nodes, embedding_dim), device=device) # Current node features
        node_features_sorted = [node_features.clone()]  # We put the features of what we know up to this point (looking only one day ahead)

        for i in range(1, len(target_graphs)):
            curr_embeddings = compute_node2vec_embeddings(target_graphs[i - 1][-1], device=device, add_degree=False)

            for node, emb in curr_embeddings.items():
                if node in node_to_idx:
                    node_idx = node_to_idx[node]
                    node_features[node_idx] = emb.to(device)
            
            node_features_sorted.append(node_features.clone())
            
        print('Generated TGCN node features using Node2Vec embeddings.')
        return node_features_sorted
        
    elif feature_type == 'one_hot':
        return torch.eye(num_nodes).to(device)
    
    
    elif feature_type == 'binary':
        node_features = torch.zeros((num_nodes, embedding_dim), device=device) # Current node features
        
        for node_id in range(num_nodes):
            bin_str = format(node_id, f'0{embedding_dim}b')  # Make it binary
            node_features[node_id] = torch.tensor([float(b) for b in bin_str], dtype=torch.float32, device=device)
        
        return node_features
    

    elif feature_type == 'zeros':
        return torch.zeros((num_nodes, embedding_dim), device=device)
    
    
def generate_gnn_node_embeddings(embedder, feature_type, input_features, prev_graphs, days_back, embedding_dim, curr_nodes, thresholds=None, new_node_strategy='zeros', device='cpu', max_id=None):
    # For the last days_back graphs, compute embeddings
    history = prev_graphs[-days_back:] if len(prev_graphs) >= days_back else prev_graphs
    if curr_nodes is None or len(curr_nodes) == 0:
        curr_nodes = [10000]  # Simple fix while I find why the error happens
    if max_id is None:
        max_id = max(curr_nodes)
        
    all_embeddings = {id: torch.zeros(embedding_dim) for id in range(max_id + 1)}
    
    if feature_type == 'GAT' or feature_type == 'GCN':        
        for graphs in history:
            curr_graph = graphs[-1]  # Since our target_graphs are lists of graphs
            
            # Original node IDs in this graph
            nodes_in_graph = list(curr_graph.nodes())
            # Map original node IDs to 0..N-1 indices
            node_id_map = {nid: i for i, nid in enumerate(nodes_in_graph)}

            # Remap edges to local indices
            edge_index = torch.tensor(
                [[node_id_map[u], node_id_map[v]] for u, v in curr_graph.edges()],
                dtype=torch.long,
                device=device
            ).t().contiguous()

            # Get features for these nodes
            nodes_tensor = torch.tensor([nid for nid in nodes_in_graph], dtype=torch.long, device=device)
            curr_features = input_features[nodes_tensor]

            # Forward pass through the GNN
            if feature_type == 'GAT':
                out = embedder((curr_features, edge_index))
                curr_embeddings = out[0] if isinstance(out, tuple) else out
            else:  # GCN
                curr_embeddings = embedder(curr_features, edge_index)

            # Map back embeddings to original node IDs
            for nid, local_idx in node_id_map.items():
                all_embeddings[nid] = curr_embeddings[local_idx]

        # Stack embeddings for all current nodes in order
        max_nid = max(all_embeddings.keys())
        emb_matrix = torch.zeros(max_nid + 1, embedding_dim, device=device)

        for nid, emb in all_embeddings.items():
            emb_matrix[nid] = emb
            
        new_nodes_indices = [n for n in curr_nodes if n not in all_embeddings]
        
        if new_nodes_indices:
            new_nodes_tensor = torch.tensor(new_nodes_indices, dtype=torch.long, device=device)
            if new_node_strategy == 'zeros':
                pass # Already initialized to zero
            

            elif new_node_strategy == 'random':
                # Small guassian noise
                noise = torch.randn(len(new_nodes_indices), embedding_dim, device=device) * 0.1
                emb_matrix[new_nodes_tensor] = noise
                

            elif new_node_strategy == 'degree_average':
                last_graph = history[-1][-1] if history else None
                
                bucket_embeddings = defaultdict(list)
                
                if last_graph:
                    # Collect embeddings by degree bucket
                    for nid in last_graph.nodes():
                        if nid in all_embeddings:
                            deg = last_graph.degree(nid)
                            # Find bucket
                            bucket = 0
                            for t in thresholds:
                                if deg <= t:
                                    bucket = t
                                    break
                            if bucket == 0: bucket = float('inf') # > max threshold
                            
                            bucket_embeddings[bucket].append(all_embeddings[nid])
                
                # Compute means
                bucket_means = {}
                global_mean = torch.zeros(embedding_dim, device=device)
                count = 0
                
                for bucket, embs in bucket_embeddings.items():
                    if embs:
                        stacked = torch.stack(embs)
                        mean = torch.mean(stacked, dim=0)
                        bucket_means[bucket] = mean
                        global_mean += torch.sum(stacked, dim=0)
                        count += len(embs)
                
                if count > 0: global_mean /= count

                
                lowest_bucket = thresholds[0] if thresholds else 0
                
                # If we have a mean for small nodes, use it. Else global mean. Else random.
                fallback_emb = bucket_means.get(lowest_bucket, global_mean)
                emb_matrix[new_nodes_tensor] = fallback_emb
        
        return emb_matrix
    
    elif feature_type == 'GCLSTM':
        x_list = []
        edge_index_list = []

        global_N = input_features.shape[0]  # total number of global nodes

        for graphs in prev_graphs:
            curr_graph = graphs[-1]

            # ---- GLOBAL NODE FEATURES ----
            # Create full matrix (N × feat_dim)
            # nodes not in graph get zero features
            x_t = torch.zeros((global_N, input_features.shape[1]), device=device)
            
            nodes_in_graph = list(curr_graph.nodes())
            nodes_tensor = torch.tensor(nodes_in_graph, dtype=torch.long, device=device)

            # fill in the features for present nodes
            x_t[nodes_tensor] = input_features[nodes_tensor]

            # ---- GLOBAL EDGE INDEX ----
            # Use global node IDs directly
            edges = torch.tensor(list(curr_graph.edges()),
                                dtype=torch.long, device=device)

            # transpose to shape [2, num_edges]
            edge_index_t = edges.t().contiguous()

            x_list.append(x_t)
            edge_index_list.append(edge_index_t)

        # Required params
        node_id_list = list(curr_nodes)                     
        node_id_map = {nid: nid for nid in node_id_list}     

        all_embeddings = embedder(x_list, edge_index_list, node_id_list, node_id_map)

        # Now gather embeddings for the current graph's nodes
        curr_nodes = list(curr_nodes)
        global_N = input_features.shape[0]   # total number of global nodes
        embedding_dim = next(iter(all_embeddings.values())).shape[0]

        # Initialize full matrix (zeros for nodes never seen)
        emb_matrix = torch.zeros((global_N, embedding_dim), device=device)

        for nid, emb in all_embeddings.items():
            emb_matrix[nid] = emb

        return emb_matrix
        
    elif feature_type == 'TGN':
        print('Generating TGN node embeddings...')
        sources = []
        destinations = []
        times = []
        e_idxs = []
        curr_idx = 0
        
        for i, graphs in enumerate(prev_graphs):
            curr_graph = graphs[-1]
            edges = list(curr_graph.edges())
            
            for (u, v) in edges:
                sources.append(u)
                destinations.append(v)
                times.append(i)
                e_idxs.append(curr_idx)
                curr_idx += 1
        data = GraphData()
        sources = np.array(sources)
        destinations = np.array(destinations)
        e_idxs = np.array(e_idxs)
        times = np.array(times)
        data.sources = sources
        data.destinations = destinations
        data.timestamps = times
        data.edge_idxs = e_idxs   
        
        neighbor_finder = get_neighbor_finder(data, uniform=False)
        embedder.set_neighbor_finder(neighbor_finder)
             
        
        negative_nodes = destinations.copy() # Dummy negative nodes
        
        src_embs, dst_embs, neg_embs = embedder.compute_temporal_embeddings(
            sources, destinations, negative_nodes, times, e_idxs)
        
        for i, node_id in enumerate(sources):
            all_embeddings[node_id] = src_embs[i].detach().clone()
        for i, node_id in enumerate(destinations):
            all_embeddings[node_id] = dst_embs[i].detach().clone()
        
        max_node_id = max(all_embeddings.keys())
        emb_matrix = torch.stack([all_embeddings[nid] for nid in range(max_node_id + 1)])
        return emb_matrix