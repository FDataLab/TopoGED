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