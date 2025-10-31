from collections import defaultdict
import math
import numpy as np 
import networkx as nx
import torch
from node2vec import Node2Vec
from GraphGeneration.models.temporal_gnn.script.config import args
import yaml

# Load YAML config
with open("GraphGeneration/encoder.yaml", "r") as file:
    encoder_config = yaml.safe_load(file)

def compute_linear_gnn_embeddings(G: nx.DiGraph, device):
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


def compute_node2vec_embeddings(G, device, old_nodes_days=None):
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
        dimensions=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_dimensions"],
        walk_length=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_walk_length"],
        num_walks=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_num_walks"],
        workers=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_workers"],
        p=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_p"],
        q=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_q"],
        quiet=True
    )
    
    model = node2vec.fit(
        window=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_window"], 
        min_count=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_min_count"], 
        batch_words=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_batch_words"],
        workers=1 
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
            if old_nodes_days:
                node2vec_emb = mean_vector if node in old_nodes_days else torch.zeros_like(mean_vector)
            else:
                node2vec_emb = mean_vector
            
        # Add on features since Node2Vec doesn't account for features
        # feat_dict = G.nodes[node]['feat']
        # sorted_keys = sorted(feat_dict.keys())  # Sort the keys for consistency
        # sorted_values = [feat_dict[k] for k in sorted_keys]
        # node_feat = torch.tensor(sorted_values, dtype=torch.float32).to(device)  # Shape of (4,)
        # combined = torch.cat([node2vec_emb, node_feat], dim=0)
        embeddings[node] = node2vec_emb
    
    return embeddings


# LSTM embeddings  
def compute_node_embeddings_LSTM(graph_snapshots, lstm_model, device):
    """
    Args:
        graph_snapshots (list of nx.DiGraph): temporal graphs with node['feat'] ready
        lstm_model (SimpleNodeLSTM): LSTM model to compute temporal embeddings
    Returns:
        dict of {node_id: final temporal embedding}
    """
    # Collect per-timestep node embeddings
    node_history = defaultdict(list)
    old_nodes = set()
    null_embed = torch.tensor([0]*(encoder_config["encoder_model"]["node2vec_setup"]["node2vec_dimensions"]),
                              dtype=torch.float32).to(device)
    for G in graph_snapshots:
        snapshot_embeddings = compute_node2vec_embeddings(G, device)
        for node, emb in snapshot_embeddings.items():
            node_history[node].append(emb) # TODO: Check nodeId if the same for every snapshot
        
        for node in old_nodes:
            if node not in snapshot_embeddings:
                node_history[node].append(null_embed)
        old_nodes = old_nodes | set(G.nodes())
    
    # Run LSTM on each node's time-series embedding
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

    F = encoder_config["encoder_model"]["node2vec_setup"]["node2vec_dimensions"] # number of features per node (change this if you want more features)

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

def compute_embedding(embeddingType, graphs, device, encoder_model=None):
    if embeddingType == 'Node2Vec':
        final_embeddings = compute_node2vec_embeddings(graphs[-1], device)
    elif embeddingType == 'Linear':
        final_embeddings = compute_linear_gnn_embeddings(graphs[-1], device)
    elif embeddingType == 'LSTM':       
        # graph_snapshots = [G_0, G_1, ..., G_T]  # each G must have node['feat']
        final_embeddings = compute_node_embeddings_LSTM(graphs, encoder_model, device)
        cos_val = torch.tensor([[math.cos(len(graphs))]], dtype=torch.float32, device=device)
        for node in final_embeddings:
            if final_embeddings[node].dim() == 1:
                final_embeddings[node] = final_embeddings[node].unsqueeze(0) 

            final_embeddings[node] = torch.cat([final_embeddings[node], cos_val], dim=1) 
            final_embeddings[node] = final_embeddings[node].squeeze(0)

            
    elif embeddingType == 'GCLSTM':
        final_embeddings = compute_node_embeddings_GCLSTM(graphs, encoder_model)
    elif embeddingType == 'HTGN':
        final_embeddings = compute_node_embeddings_HTGN(graphs, encoder_model)
    
    return final_embeddings


def compute_temporal_node_embeddings(embedding_history, days_back, device, model_type, model=None, graph_snapshots=None):
    past_days = embedding_history[-days_back:]  # get the last days_back days
    num_missing = days_back - len(past_days)    # how many days we are short

    # Determine embedding size F from the first available snapshot
    F = len(next(iter(past_days[0].values())))

    null_embed = torch.zeros(F, device=device, dtype=torch.float32)

    # Prepend zero dicts for missing days
    if num_missing > 0:
        all_nodes_in_past = set().union(*[d.keys() for d in past_days])
        zero_dicts = [{node: null_embed.clone() for node in all_nodes_in_past} for _ in range(num_missing)]
        past_days = zero_dicts + past_days

    # Get all nodes across the full window
    all_nodes = set()
    for emb_dict in past_days:
        all_nodes.update(emb_dict.keys())

    # Build per-node sequences
    node_history = {node: [] for node in all_nodes}
    for emb_dict in past_days:
        for node in all_nodes:
            node_history[node].append(emb_dict.get(node, null_embed))

    if model_type == 'LSTM':
        if model is None:
            raise ValueError("Provide lstm_model for model_type='LSTM'")
        final_embeddings = model(node_history)

    elif model_type == 'GCLSTM':
        if model is None:
            raise ValueError("Provide gclstm_model for model_type='GCLSTM'")
        if graph_snapshots is None:
            raise ValueError("Provide graph_snapshots for GCLSTM to construct edge_index_list")

        if graph_snapshots is not None:
            num_missing = len(past_days) - len(graph_snapshots)
            if num_missing > 0:
                graph_snapshots = [None]*num_missing + graph_snapshots

        node_list = sorted(list(all_nodes))
        node_id_map = {node: i for i, node in enumerate(node_list)}
        x_list = []
        edge_index_list = []

        for i, emb_dict in enumerate(past_days):
            N = len(node_list)
            x_t = torch.zeros(N, F, device=device)
            for node, emb in emb_dict.items():
                idx = node_id_map[node]
                x_t[idx] = emb
            x_list.append(x_t)

            # Build edge_index_list from graph_snapshots if available
            G = graph_snapshots[i] if graph_snapshots is not None else None
            if G is None:
                edge_index = torch.empty((2,0), dtype=torch.long, device=device)
            else:
                edges = [[node_id_map[u], node_id_map[v]] for u,v in G.edges()]
                edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
            edge_index_list.append(edge_index)

        final_embeddings = model(x_list, edge_index_list, node_list, node_id_map)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return final_embeddings

def group_node2vec_embeddings(all_embeddings, old_nodes_days, days_back, use_ma):
    """
    Get the embeddings to use for constructing the current graph from the past days_back days
    If using moving average, only consider days the node existed
    
    params:
        all_embeddings (list[dict]): List of node2vec embeddings for all previous days
        old_nodes_days (list): The nodes we will see in the current graph (or expect to see in the prediction case)
        days_back (int): How many days back to consider
        use_ma (bool): Whether to use moving average or the most recent embedding for a node
        
    returns:
        node_embeddings (dict): The constructed dictionary of {node: [embedding]} pairs
    """
    if days_back > len(all_embeddings):
        recent_embeddings = all_embeddings
    else:
        recent_embeddings = all_embeddings[-days_back:]

    node_embeddings = {}

    for node in old_nodes_days:
        node_embs = []

        node_embs = [emb_dict[node] for emb_dict in recent_embeddings if node in emb_dict]
        d
        if use_ma:
            # Moving average only over days where the node exists
            node_embeddings[node] = torch.stack(node_embs, dim=0).mean(dim=0)
        else:
            # Just take the most recent embedding
            node_embeddings[node] = node_embs[-1]

    return node_embeddings