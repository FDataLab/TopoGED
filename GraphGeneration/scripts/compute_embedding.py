from collections import defaultdict
import numpy as np 
import networkx as nx
import torch
from node2vec import Node2Vec

# Set up device
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
except Exception:
    device = torch.device("cpu")
    print("Using CPU")

# Node2Vec Parameters
node2vec_dimensions = 60  # We add features onto the end since Node2Vec doesn't embed features 
node2vec_walk_length = 50  # Number of nodes visited per walk (Higher is more global, smaller is local)
node2vec_num_walks = 10  # Number of walks to start per node (Higher is more detailed and stable)
node2vec_p = 1.0  # Return parameter, the likelihood of revisiting a node (Higher is less backtracking)
node2vec_q = 1.0  # The walk bias for determining direction (Higher is more DFS-like; lower is BFS-like)
node2vec_window = 10  # The context size (Higher is broader learning)
node2vec_min_count = 1  # Minimum number of occurrences for a node to be considered (Higher will ignore more rare nodes)
node2vec_batch_words = 4  # The batch size for when Word2Vec is used (Higher will train faster; but with more memory)
node2vec_workers = 1  # Number of workers (threads)

def compute_linear_gnn_embeddings(G: nx.DiGraph):
    """
    An embedding method inspired by LinearGNNs from GraphAny where Z=AX given A is the adjacency matrix and X is the node feature matrix
    One of two available methods
    
    Args:
        G (nx.DiGraph): The graph to embed
        
    Returns:
        embeddings (dict): The constructed dictionary of {node: [embedding]} pairs
    """
    all_nodes = sorted(G.nodes())
    id_to_idx = {node_id: idx for idx, node_id in enumerate(all_nodes)}
    idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}
    
    A = nx.to_numpy_array(G, nodelist=all_nodes, dtype=np.float32)  # The matrix to operate on
    
    # Normalize A
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    A_normalized = A / row_sums
    
    X = np.array([list(G.nodes[node]['feat'].values()) for node in all_nodes], dtype=np.float32)
    
    A_normalized = torch.tensor(A_normalized, device=device)
    X = torch.tensor(X, device=device)
    Z = A_normalized @ X  # As computed in GraphAny
    
    embeddings = {idx_to_id[i]: Z[i] for i in range(Z.shape[0])}
    
    return embeddings

def compute_node2vec_embeddings(G: nx.DiGraph):
    """
    Use Node2Vec to embed nodes in the constructed graph. Appends node features onto the end since Node2Vec does not account for features
    One of two available methods
    
    Args:
        G (nx.DiGraph): The graph to embed
        
    Returns:
        embeddings (dict): The constructed dictionary of {node: [embedding]} pairs
    """
    node2vec = Node2Vec(
        G,
        dimensions=node2vec_dimensions,
        walk_length=node2vec_walk_length,
        num_walks=node2vec_num_walks,
        workers=node2vec_workers,
        p=node2vec_p,
        q=node2vec_q,
        quiet=True
    )
    
    model = node2vec.fit(
        window=node2vec_window, 
        min_count=node2vec_min_count, 
        batch_words=node2vec_batch_words
    )  # Perform Node2Vec

    # Used to generate an embedding for isolated nodes
    all_vectors = [model.wv[key] for key in model.wv.index_to_key]
    mean_vector = torch.tensor(np.mean(all_vectors, axis=0), dtype=torch.float32).to(device)

    # Get embeddings and concatenate the node features
    embeddings = {}
    for node in G.nodes():
        if node in model.wv:
            node2vec_emb = torch.tensor(model.wv[node], dtype=torch.float32).to(device)
        else:
            node2vec_emb = mean_vector
            
        # Add on features since Node2Vec doesn't account for features
        feat_dict = G.nodes[node]['feat']
        sorted_keys = sorted(feat_dict.keys())  # Sort the keys for consistency
        sorted_values = [feat_dict[k] for k in sorted_keys]
        node_feat = torch.tensor(sorted_values, dtype=torch.float32).to(device)  # Shape of (4,)
        combined = torch.cat([node2vec_emb, node_feat], dim=0)
        embeddings[node] = combined
    
    return embeddings

# LSTM embeddings  
def compute_node_embeddings_LSTM(graph_snapshots, lstm_model):
    """
    Args:
        graph_snapshots (list of nx.DiGraph): temporal graphs with node['feat'] ready
        lstm_model (SimpleNodeLSTM): LSTM model to compute temporal embeddings
    Returns:
        dict of {node_id: final temporal embedding}
    """
    # Step 1: Collect per-timestep node embeddings
    node_history = defaultdict(list)

    for G in graph_snapshots:
        snapshot_embeddings = compute_node2vec_embeddings(G)
        for node, emb in snapshot_embeddings.items():
            node_history[node].append(emb) # TODO: Check nodeId if the same for every snapshot

    # Step 2: Run LSTM on each node's time-series embedding
    final_node_embeddings = lstm_model(node_history)
    return final_node_embeddings

# GCN embeddings
def get_GCN_data(graph_snapshots):
    """
        Converts a sequence of NetworkX graph snapshots into PyTorch-ready inputs for GCN models.

        Args:
            graph_snapshots (list of nx.DiGraph): Temporal graph snapshots.
                Each node must have its 'feat' computed via node2vec or similar embedding method.

        Returns:
            x_list (List[Tensor]): List of node feature matrices (shape [num_nodes, F]) for each snapshot.
            edge_index_list (List[Tensor]): List of edge index tensors (shape [2, num_edges]) for each snapshot.
            node_list (List): Sorted list of unique global node IDs.
            node_id_map (Dict): Mapping from node ID to index in feature matrix.
    """
    # Step 1: Build global node set
    all_nodes = set()
    for G in graph_snapshots:
        all_nodes.update(G.nodes())
    node_list = sorted(list(all_nodes))  # fixed order
    node_id_map = {node: i for i, node in enumerate(node_list)}  # map node → index
    N = len(node_list)
    x_list = []
    edge_index_list = []

    F = node2vec_dimensions + 4 # number of features per node (change this if you want more features)

    for G in graph_snapshots:
        node2vec_embeddings = compute_node2vec_embeddings(G)
        x_t = torch.zeros(N, F)  # default: zero for all nodes
        for node in G.nodes():
            idx = node_id_map[node]
            x_t[idx] = node2vec_embeddings[node].clone().detach()

        edge_index = []
        for u, v in G.edges():
            src = node_id_map[u]
            tgt = node_id_map[v]
            edge_index.append([src, tgt])
        edge_index = torch.tensor(edge_index).t().contiguous()  # shape [2, E]

        x_list.append(x_t)
        edge_index_list.append(edge_index)
    return x_list, edge_index_list, node_list, node_id_map

def compute_node_embeddings_GCLSTM(graph_snapshots, gclstm_model):
    """
    Args:
        graph_snapshots (list of nx.DiGraph): temporal graphs with node['feat'] ready
        lstm_model (SimpleNodeLSTM): LSTM model to compute temporal embeddings
    Returns:
        dict of {node_id: final temporal embedding}
    """
    x_list, edge_index_list, node_id_list, node_id_map = get_GCN_data(graph_snapshots)
    final_node_embeddings = gclstm_model(x_list, edge_index_list, node_id_list, node_id_map)
    return final_node_embeddings

# HTGN for encoding
def compute_node_embeddings_HTGN(graph_snapshots, HTGN_model):
    """
    Args:
        graph_snapshots (list of nx.DiGraph): temporal graphs with node['feat'] ready
        lstm_model (SimpleNodeLSTM): LSTM model to compute temporal embeddings
    Returns:
        dict of {node_id: final temporal embedding}
    """
    x_list, edge_index_list, node_id_list, node_id_map = get_GCN_data(graph_snapshots)
    HTGN_model.init_hiddens()
    final_node_embeddings = HTGN_model(edge_index=edge_index_list[-1], x=None, node_id_list=node_id_list, node_id_map=node_id_map)
    return final_node_embeddings
