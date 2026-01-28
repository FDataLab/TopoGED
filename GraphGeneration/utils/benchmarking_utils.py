import random
# import dgl
import torch
import networkx as nx
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, negative_sampling, from_networkx
from torch_geometric.data import Data
from GraphGeneration.encoders.TGN.utils.utils import NeighborFinder
from GraphGeneration.encoders.TGN.model.TGNBatch import TGNBatch

class RandEdgeSampler:
    def __init__(self, src_list, dst_list, seed=None):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        if seed is not None:
            np.random.seed(seed)

    def sample(self, size):
        """
        Samples random nodes from the known universe of sources and destinations.
        Used for the contrastive loss: -log(pos) - Q * log(1-neg) 
        """
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

class BenchmarkerArgs:
    def __init__(self, encoder_config, model, device):
        # Create all arguments here
        self.device = device
        self.model = model

        # For roland
        if self.model == 'ROLAND':
            self.num_nodes = 0  # Assigned later
            self.nfeat = encoder_config['benchmarking']['roland']['nfeat']
            self.nhid = encoder_config['benchmarking']['roland']['nhid']
            self.dropout = encoder_config['benchmarking']['roland']['dropout']
        
        
        # For egcn
        elif self.model == 'EvolveGCN':
            self.nfeat = encoder_config['benchmarking']['evolve_gcn']['nfeat']
            self.nhid = encoder_config['benchmarking']['evolve_gcn']['nhid']
            self.egcn_type = 'EGCNH'
            self.use_gru = True
            self.nb_window = 1
        
        # For gclstm
        elif self.model == 'GCLSTM':
            self.nfeat = encoder_config['benchmarking']['gclstm']['nfeat']
            self.nhid = encoder_config['benchmarking']['gclstm']['nhid']
            self.chebyshev_filter = encoder_config['benchmarking']['gclstm']['chebyshev_filter']

        # For tgcn
        elif self.model == 'TGCN':
            self.nfeat = encoder_config['benchmarking']['tgcn']['nfeat']
            self.nhid = encoder_config['benchmarking']['tgcn']['nhid']
            
        # For wingnn
        elif self.model == 'WinGNN':
            self.nfeat = encoder_config['benchmarking']['wingnn']['nfeat']
            self.nhid = encoder_config['benchmarking']['wingnn']['nhid']
            
        elif self.model == 'VGAE':
            self.input_dim = encoder_config['benchmarking']['vgae']['input_dim']
            self.hidden1_dim = encoder_config['benchmarking']['vgae']['hidden1_dim']
            self.hidden2_dim = encoder_config['benchmarking']['vgae']['hidden2_dim']
            
        elif self.model == 'TGN':
            pass  # All arguments handled elsewhere


def get_binary_encoding(num_nodes, embedding_dim=64):
    """
    Creates a binary encoding matrix for node IDs.
    
    Args:
        num_nodes: Total number of nodes.
        embedding_dim: Dimension of the binary embedding.
    """
    
    # Create an array of indices [0, 1, 2, ..., num_nodes-1]
    indices = np.arange(num_nodes).reshape(-1, 1)
    
    # Generate bit masks: [2^0, 2^1, 2^2, ...]
    masks = 2**np.arange(embedding_dim)
    
    # Perform bitwise AND and convert to boolean/float
    # This creates a (num_nodes, embedding_dim) binary matrix
    binary_matrix = (indices & masks) > 0
    
    return torch.tensor(binary_matrix, dtype=torch.float)

def create_samples_roland(networkx_graphs, num_global_nodes, embedding_dim=64):
    snapshots = []
    
    # 1. Pre-generate the binary encoding (Identity)
    # This remains the same: unique ID for every node in the universe
    global_features = get_binary_encoding(num_global_nodes, embedding_dim - 1)

    for t in range(len(networkx_graphs) - 1):
        G_t = networkx_graphs[t]
        
        # 2. Convert G_t structure to PyG Data
        data = from_networkx(G_t)
        
        # --- FEATURE ENGINEERING BLOCK ---
        # 3a. Start with Binary Features
        x_base = global_features.clone() 

        # 3b. Calculate Structural Features (Degree)
        # We calculate this specifically for the current snapshot G_t
        # Use a dictionary to map node ID to degree, ensuring global node alignment
        node_degrees = dict(G_t.degree())
        deg_tensor = torch.zeros((num_global_nodes, 1), dtype=torch.float)
        
        for node_idx in range(num_global_nodes):
            # If node exists in current snapshot, get its degree, else 0
            deg_tensor[node_idx] = float(node_degrees.get(node_idx, 0))

        # 3c. Normalize Degrees (prevents gradient explosion)
        # Log-scaling is often better for power-law graphs (like social networks)
        deg_norm = torch.log(deg_tensor + 1.0)
        
        # 3d. Concatenate Identity + Structure
        # Your input_channel in GNN config must now be (embedding_dim + 1)
        data.x = torch.cat([x_base, deg_norm], dim=1) 
        # ---------------------------------

        # 4. Global universe metadata
        data.num_nodes = num_global_nodes
        
        # 5. Ground truth from next snapshot
        next_graph = networkx_graphs[t+1]
        if len(next_graph.edges) == 0:
            continue # Skip snapshots with no future activity
            
        pos_edge_index = torch.tensor(list(next_graph.edges), dtype=torch.long).t()
        
        # 6. Sample negatives
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_global_nodes, 
            num_neg_samples=pos_edge_index.size(1)
        )
        
        # 7. Labels
        data.edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        data.edge_label = torch.cat([
            torch.ones(pos_edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1))
        ], dim=0)
        
        data.num_current_edges = data.edge_index.size(1)
        snapshots.append(data)
        
    return snapshots

def calculate_mrr(pos_score, neg_scores):
    """
    Computes MRR for a single positive link against multiple negative samples.
    
    Args:
        pos_score (Tensor): Score for the true edge (shape: [1])
        neg_scores (Tensor): Scores for negative edges (shape: [num_neg_samples])
    
    Returns:
        float: Reciprocal Rank (1/rank)
    """
    # Combine scores and find the rank of the positive score
    # We use (scores > pos_score) to see how many negatives are ranked higher
    # +1 because ranks are 1-based.
    rank = (neg_scores > pos_score).sum().item() + 1
    return 1.0 / rank

@torch.no_grad()
def predict_edges_gclstm(model, data, H=None, C=None, threshold=0.5):
    """
    Args:
        model: GCLSTM instance
        data: Current snapshot PyG Data
        H: Previous hidden state
        C: Previous cell state
    """
    model.eval()
    
    # GCLSTM forward returns a tuple: (H_next, C_next)
    H_next, C_next = model(data.x, data.edge_index, getattr(data, 'edge_weight', None), H, C)
    
    z = H_next
    
    logits = torch.matmul(z, z.t())
    probs = torch.sigmoid(logits)
    
    mask = probs > threshold
    mask.fill_diagonal_(False)
    
    predicted_edges = mask.nonzero(as_tuple=False).t()
    
    return predicted_edges, H_next, C_next

def create_samples_gclstm(graphs, neg_ratio=1.0, embedding_dim=64):
    samples = []
    for i in range(len(graphs) - 1):
        curr_g = nx.convert_node_labels_to_integers(graphs[i])
        data = from_networkx(curr_g)
        
        if data.x is None:
            data.x = get_binary_encoding(data.num_nodes, embedding_dim)
        
        next_g = nx.convert_node_labels_to_integers(graphs[i+1])
        num_nodes = next_g.number_of_nodes()
        
        pos_edges = torch.tensor(list(next_g.edges), dtype=torch.long).t()
        neg_edges = negative_sampling(
            edge_index=pos_edges, 
            num_nodes=num_nodes, 
            num_neg_samples=int(pos_edges.size(1) * neg_ratio)
        )
        
        # --- CONCATENATE FOR THE TRAIN LOOP ---
        # Combine pos and neg edges into edge_label_index
        data.edge_label_index = torch.cat([pos_edges, neg_edges], dim=-1)
        
        # Create ground truth labels (1 for pos, 0 for neg)
        pos_y = torch.ones(pos_edges.size(1))
        neg_y = torch.zeros(neg_edges.size(1))
        data.edge_label = torch.cat([pos_y, neg_y], dim=0)
        
        samples.append(data)
        
    return samples


def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                        data.edge_idxs,
                                                        data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)


def tgn_setup(snapshots, num_nodes, embedding_dim=64, undirected=True):
    adj_list = [[] for _ in range(num_nodes)]
    
    # This count must match the global indexing logic in create_samples_tgn
    edge_count = 0
    for t_idx, data in enumerate(snapshots):
        # We use the sources/destinations we prepared in the TGNBatch
        # to ensure the NeighborFinder 'sees' exactly what the model sees
        src_nodes = data.sources
        dst_nodes = data.destinations
        
        for u, v in zip(src_nodes, dst_nodes):
            # 1. Forward edge
            adj_list[u].append((v, edge_count, float(t_idx)))
            edge_count += 1
            
            # 2. Backward edge (if undirected)
            if undirected:
                adj_list[v].append((u, edge_count, float(t_idx)))
                edge_count += 1
    
    # Initialize the TGN utility
    neighbor_finder = NeighborFinder(adj_list, uniform=False)
    
    # Feature generation
    node_features = get_binary_encoding(num_nodes, embedding_dim).cpu().numpy()
    
    # edge_features must have exactly 'edge_count' rows
    # This ensures no IndexError: index X is out of bounds
    edge_features = np.zeros((edge_count, 1), dtype=np.float32)
    
    return neighbor_finder, node_features, edge_features


def create_samples_tgn(graphs, neg_ratio=1.0, embedding_dim=64):
    samples = []
    # This counter ensures edge_idxs are continuous and stay within 
    # the bounds of the edge_features matrix.
    cumulative_edge_count = 0 
    
    for i in range(len(graphs) - 1):
        # Current graph (Snapshot T)
        curr_g = nx.convert_node_labels_to_integers(graphs[i])
        data = from_networkx(curr_g)
        
        if data.x is None:
            data.x = get_binary_encoding(data.num_nodes, embedding_dim)
        
        # Next graph (Snapshot T+1)
        next_g = nx.convert_node_labels_to_integers(graphs[i+1])
        num_nodes = next_g.number_of_nodes()
        
        pos_edges = torch.tensor(list(next_g.edges), dtype=torch.long).t()
        neg_edges = negative_sampling(
            edge_index=pos_edges, 
            num_nodes=num_nodes, 
            num_neg_samples=int(pos_edges.size(1) * neg_ratio)
        )

        src = pos_edges[0].cpu().numpy()
        dst = pos_edges[1].cpu().numpy()
        neg_dst = neg_edges[1].cpu().numpy()
        
        t_batch = np.full(src.shape, i, dtype=np.float32)

        num_pos_edges = len(src)
        edge_idxs = np.arange(cumulative_edge_count, cumulative_edge_count + num_pos_edges)
        cumulative_edge_count += num_pos_edges

        edge_label_index = torch.cat([pos_edges, neg_edges], dim=-1)
        edge_label = torch.cat([torch.ones(pos_edges.size(1)), 
                               torch.zeros(neg_edges.size(1))], dim=0)
        
        sample = TGNBatch(
            x=data.x,
            edge_index=data.edge_index,
            sources=src,            
            destinations=dst,       
            timestamps=t_batch,     
            edge_idxs=edge_idxs,    
            neg_dst=neg_dst,        
            t=t_batch,              
            edge_label_index=edge_label_index,
            edge_label=edge_label
        )
        
        samples.append(sample)
        
    return samples


def create_samples_vgae(graphs, neg_ratio=1.0, embedding_dim=64, num_nodes=None):
    """
    Prepares snapshots for VGAE with pre-computed Adjacency Matrices.
    Input (data.adj_t): Normalized Adjacency from Snapshot t
    Target (data.edge_label_index): Edges from Snapshot t+1
    """
    samples = []
    # If num_nodes isn't provided, we infer it from the largest graph
    if num_nodes is None:
        num_nodes = max(g.number_of_nodes() for g in graphs)

    for i in range(len(graphs) - 1):
        # 1. PROCESS CURRENT GRAPH (The Encoder Input)
        curr_g = nx.convert_node_labels_to_integers(graphs[i])
        data = from_networkx(curr_g)
        
        # Binary Node Features
        data.x = get_binary_encoding(num_nodes, embedding_dim)
        
        # --- PRE-COMPUTE NORMALIZED ADJACENCY (A_hat) ---
        # Add self-loops to the current edge index
        edge_index = data.edge_index
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Compute symmetric normalization: D^-0.5 * A * D^-0.5
        row, col = edge_index_with_self_loops
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Store as a Sparse Tensor attribute inside the 'data' object
        data.adj_t = torch.sparse_coo_tensor(
            edge_index_with_self_loops, 
            norm, 
            size=(num_nodes, num_nodes)
        ).coalesce()

        # 2. PROCESS NEXT GRAPH (The Prediction Target)
        next_g = nx.convert_node_labels_to_integers(graphs[i+1])
        pos_edges = torch.tensor(list(next_g.edges), dtype=torch.long).t()
        
        neg_edges = negative_sampling(
            edge_index=pos_edges, 
            num_nodes=num_nodes, 
            num_neg_samples=int(pos_edges.size(1) * neg_ratio)
        )
        
        # 3. PACKAGING
        data.edge_label_index = torch.cat([pos_edges, neg_edges], dim=-1)
        data.edge_label = torch.cat([
            torch.ones(pos_edges.size(1)), 
            torch.zeros(neg_edges.size(1))
        ], dim=0)
        
        samples.append(data)
        
    return samples


@torch.no_grad()
def predict_edges_tgcn(model, data, H=None, threshold=0.5):
    """
    Args:
        model: TGCN instance
        data: Current snapshot PyG Data
        H: Previous hidden state tensor
    """
    model.eval()
    
    # TGCN forward usually takes (X, edge_index, edge_weight, H)
    # Note: Some implementations expect H to be initialized if None
    H_next = model(data.x, data.edge_index, getattr(data, 'edge_weight', None), H)
    
    # The output H_next is the node embeddings for this time step
    z = H_next
    
    logits = torch.matmul(z, z.t())
    probs = torch.sigmoid(logits)
    
    mask = probs > threshold
    mask.fill_diagonal_(False)
    
    predicted_edges = mask.nonzero(as_tuple=False).t()
    
    return predicted_edges, H_next


def create_samples_tgcn(graphs, neg_ratio=1.0, embedding_dim=64):
    """
    Prepares snapshots for TGCN
    Identical to TGCN sample creation
    """
    samples = []
    for i in range(len(graphs) - 1):
        curr_g = nx.convert_node_labels_to_integers(graphs[i])
        data = from_networkx(curr_g)
        
        if data.x is None:
            data.x = get_binary_encoding(data.num_nodes, embedding_dim)
        
        next_g = nx.convert_node_labels_to_integers(graphs[i+1])
        num_nodes = next_g.number_of_nodes()
        
        pos_edges = torch.tensor(list(next_g.edges), dtype=torch.long).t()
        
        neg_edges = negative_sampling(
            edge_index=pos_edges, 
            num_nodes=num_nodes, 
            num_neg_samples=int(pos_edges.size(1) * neg_ratio)
        )
        
        data.pos_edge_label_index = pos_edges
        data.neg_edge_label_index = neg_edges
        samples.append(data)
        
    return samples


def create_samples_tgat(graphs, embedding_dim=64):
    """
    Groups continuous temporal events into snapshot-like samples for benchmarking.
    """
    samples = []
    
    # FIX: Find the maximum node index across ALL graphs to avoid IndexError
    # g.nodes() returns the IDs. We find the max of those.
    max_node_id = 0
    for g in graphs:
        if g.number_of_nodes() > 0:
            max_node_id = max(max_node_id, max(g.nodes()))
    
    # 1. Initialize Global Adjacency for the NeighborFinder
    # Size it using max_node_id + 1 so that any node ID is a valid index
    adj_list = [[] for _ in range(max_node_id + 1)]
    edge_counter = 0

    # 2. Pre-generate node features
    # Similarly, use max_node_id + 1 for the embedding table size
    global_x = get_binary_encoding(max_node_id + 1, embedding_dim)

    for t in range(len(graphs)):
        G_t = graphs[t]
        src_list, dst_list, ts_list, idx_list = [], [], [], []
        
        for u, v in G_t.edges():
            # Now these will never be out of range
            adj_list[u].append((v, edge_counter, float(t)))
            adj_list[v].append((u, edge_counter, float(t)))
            
            src_list.append(u)
            dst_list.append(v)
            ts_list.append(float(t))
            idx_list.append(edge_counter)
            edge_counter += 1

        # Standard processing continues...
        data = from_networkx(G_t)
        data.x = global_x
        data.sources = np.array(src_list)
        data.destinations = np.array(dst_list)
        data.timestamps = np.array(ts_list)
        data.edge_idxs = np.array(idx_list)
        data.edge_label = torch.ones(len(src_list)) 
        
        samples.append(data)
        
    return samples

@torch.no_grad()
def predict_edges_egcn(model, data, threshold=0.5):
    """
    Predicts edges for the next time step based on current node embeddings.
    
    Args:
        model: Trained EvolveGCN
        data: PyG Data object (current graph)
    """
    model.eval()
    
    # EvolveGCN-O requires: x, edge_index, edge_weight
    # If you don't have edge weights, pass None
    edge_weight = getattr(data, 'edge_weight', None)
    
    # 1. Get temporal embeddings
    z = model(data.x, data.edge_index, edge_weight)
    
    # 2. Calculate link probabilities (Global Similarity)
    # Using dot product + sigmoid
    logits = torch.matmul(z, z.t())
    probs = torch.sigmoid(logits)
    
    # 3. Apply threshold
    mask = probs > threshold
    mask.fill_diagonal_(False) # No self-loops
    
    predicted_edges = mask.nonzero(as_tuple=False).t()
    final_probs = probs[mask]
    
    return predicted_edges


def create_samples_egcn(graphs, neg_ratio=1.0, embedding_dim=64):
    """
    Converts a sequence of NetworkX graphs into PyG samples for EvolveGCN.
    
    Args:
        nx_graphs (list): List of NetworkX graphs [G_t1, G_t2, ... G_tn]
    Returns:
        samples (list): List of PyG Data objects with pos/neg edges for the NEXT step.
    """
    samples = []
    
    for i in range(len(graphs) - 1):
        # Prepare current graph (Input)
        curr_g = nx.convert_node_labels_to_integers(graphs[i])
        data = from_networkx(curr_g)
        
        if data.x is None:
            data.x = get_binary_encoding(data.num_nodes, embedding_dim)
        
        # Prepare next graph (Target)
        next_g = nx.convert_node_labels_to_integers(graphs[i+1])
        num_nodes = next_g.number_of_nodes()
        
        # Extract Positive Edges from the NEXT graph
        pos_edges = torch.tensor(list(next_g.edges), dtype=torch.long).t()
        
        # Use PyG negative_sampling instead of a manual while loop
        num_neg = int(pos_edges.size(1) * neg_ratio)
        neg_edges = negative_sampling(
            edge_index=pos_edges,      # Edges to avoid sampling
            num_nodes=num_nodes,       # Total node count
            num_neg_samples=num_neg,   # How many to sample
            method='sparse'            # 'sparse' is generally better for large graphs
        )
        
        data.edge_label_index = torch.cat([pos_edges, neg_edges], dim=-1)
        
        # Create ground truth labels (1 for pos, 0 for neg)
        pos_y = torch.ones(pos_edges.size(1))
        neg_y = torch.zeros(neg_edges.size(1))
        data.edge_label = torch.cat([pos_y, neg_y], dim=0)
        
        # Attach prediction targets to the current data object
        data.pos_edge_label_index = pos_edges
        data.neg_edge_label_index = neg_edges
        
        samples.append(data)
        
    return samples


# @torch.no_grad()
# def predict_edges_wingnn(model, node_features, threshold=0.5):
#     model.eval()
#     num_nodes = node_features.shape[0]

#     # Create empty graph with self-loops for the GCN pass
#     g = dgl.graph((torch.arange(num_nodes), torch.arange(num_nodes)))
    
#     # Get embeddings
#     z = model(g, node_features)
    
#     # Global similarity scoring
#     logits = torch.matmul(z, z.t())
#     probs = torch.sigmoid(logits)
    
#     mask = (probs > threshold)
#     mask.fill_diagonal_(False)
    
#     predicted_edges = mask.nonzero(as_tuple=False).t() 
#     final_probs = probs[mask]
    
#     return predicted_edges


# def create_samples_wingnn(graphs, neg_ratio=1.0, embedding_dim=64):
#     samples = []
#     for g in graphs:
#         g_nx = nx.convert_node_labels_to_integers(g)
#         dgl_g = dgl.from_networkx(g_nx)
        
#         # DGL uses ndata for features. We'll store it in 'h'
#         dgl_g.ndata['h'] = get_binary_encoding(dgl_g.num_nodes(), embedding_dim)
        
#         u, v = dgl_g.edges()
#         pos_edges = torch.stack([u, v], dim=0)

#         num_nodes = dgl_g.num_nodes()
#         num_neg = int(dgl_g.num_edges() * neg_ratio)
        
#         # (Using the PyG negative sampler for speed here even for DGL)
#         neg_edges = negative_sampling(pos_edges, num_nodes=num_nodes, num_neg_samples=num_neg)
        
#         samples.append({
#             'graph': dgl_g,
#             'pos_edges': pos_edges,
#             'neg_edges': neg_edges
#         })
#     return samples