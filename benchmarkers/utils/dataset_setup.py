import pandas as pd
import numpy as np 
import networkx as nx 
import os
import sys
import torch
import pickle
import scipy.sparse as sp
from torch_geometric.data import Data

# Ensure path is set for local imports if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ==========================================
# 1. Utility Functions
# ==========================================

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix: D^{-0.5} (A + I) D^{-0.5}"""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def get_sparse_identity(n):
    """Creates a torch sparse identity matrix of size n x n to save RAM."""
    indices = torch.arange(n).repeat(2, 1)
    values = torch.ones(n)
    return torch.sparse_coo_tensor(indices, values, torch.Size([n, n]))

def get_snapshot_duration(dataset):
    # Mapping for standard datasets
    mapping = {
        'Hypertext09': '10min', 'ia-contact': '10min',
        'enron': '11_BINS', 'Enron': '11_BINS',
        'radoslaw': '8H', 'fb-forum': '8H',
        'HepPH': '1M', 'HepTH': '1M', 'tgbl-review': '1M',
        'tgbl-coin': '1W'
    }
    return mapping.get(dataset, '1D')

def ensure_edgelist(dataset):
    """Restored: Standardizes raw data files into a consistent CSV edgelist."""
    output_path = f'data/input/raw/edgelist/{dataset}.txt'
    if os.path.exists(output_path):
        return output_path

    print(f"Generating edgelist for {dataset}...")
    # Map dataset names to their raw file locations
    paths = {
        'Hypertext09': 'benchmarkers/data/input/raw/ia-contacts_hypertext2009.edges',
        'fb-forum': 'benchmarkers/data/input/raw/fb-forum.edges',
        'HepTH': 'benchmarkers/data/input/raw/cit-HepTH.edges',
        'HepPH': 'benchmarkers/data/input/raw/cit-HepPH.edges',
        'ia-contact': 'benchmarkers/data/input/raw/ia-contact.edges',
        'radoslaw': 'benchmarkers/data/input/raw/ia-radoslaw-email.edges',
        'uci-message': 'benchmarkers/data/input/raw/out.opsahl-ucsocial',
        'bitcoinotc': 'benchmarkers/data/input/raw/soc-sign-bitcoinotc.edges',
        'bitcoinalpha': 'benchmarkers/data/input/raw/soc-sign-bitcoinalpha.edges'
    }
    
    input_path = paths.get(dataset)
    if not input_path or not os.path.exists(input_path):
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generic parsing based on common formats
    try:
        if dataset in ['Hypertext09', 'fb-forum']:
            df = pd.read_csv(input_path, sep=None, engine='python', names=['from', 'to', 'timestamp'])
            df = pd.DataFrame({'from': df['from'], 'to': df['to'], 'date': df['timestamp'], 'value': 1.0})
        elif dataset in ['HepTH', 'HepPH', 'radoslaw', 'uci-message', 'ia-contact']:
            df = pd.read_csv(input_path, sep=r'\s+', comment='%', names=['from', 'to', 'value_orig', 'timestamp'], engine='python')
            df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
            df['value'] = 1.0
        elif dataset in ['bitcoinotc', 'bitcoinalpha']:
            df = pd.read_csv(input_path, names=['source', 'target', 'rating', 'timestamp'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
            df = df[['source', 'target', 'date', 'rating']].rename(columns={'source':'from', 'target':'to', 'rating':'value'})
        
        df[['from', 'to', 'date', 'value']].to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error converting {dataset}: {e}")

    return output_path

def toNetworkx(edgelist, snapshot_duration, is_directed=True):
    """Parses edgelist and returns a list of unique NetworkX graphs."""
    if not os.path.exists(edgelist):
        raise FileNotFoundError(f"Edgelist not found at {edgelist}.")

    edgelist_df = pd.read_csv(edgelist)
    if 'value' not in edgelist_df.columns:
        edgelist_df['value'] = 1.0
        
    global_nodes = set(edgelist_df['from']).union(set(edgelist_df['to']))

    # Standardize Time
    if pd.api.types.is_numeric_dtype(edgelist_df['date']):
        edgelist_df['timestamp'] = pd.to_datetime(edgelist_df['date'], unit='s')
    else:
        edgelist_df['timestamp'] = pd.to_datetime(edgelist_df['date'])
    
    edgelist_df = edgelist_df.sort_values('timestamp').reset_index(drop=True)
    
    # Apply Time Grouping
    if snapshot_duration == '11_BINS':
        edgelist_df['snapshot_id'] = pd.qcut(edgelist_df.index, 11, labels=False)
    else:
        edgelist_df['snapshot_id'] = edgelist_df['timestamp'].dt.floor(snapshot_duration)
    
    # Aggregate duplicate edges in the same window
    edgelist_df = edgelist_df.groupby(['from', 'to', 'snapshot_id'])['value'].sum().reset_index()

    uniq_snapshots = sorted(edgelist_df['snapshot_id'].unique())
    graph_class = nx.DiGraph if is_directed else nx.Graph
    
    data = []
    print(f"[Checking Graph Integrity for {len(uniq_snapshots)} snapshots...]")
    
    for i, ts in enumerate(uniq_snapshots):
        ts_edges = edgelist_df[edgelist_df['snapshot_id'] == ts]
        
        # FIX: Fresh instance every time
        ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to', edge_attr=['value'], create_using=graph_class())
        ts_G.add_nodes_from(global_nodes)
        
        # Integrity Check: Compare with previous snapshot
        if i > 0:
            prev_G = data[-1]
            # Simple check for identical structure
            if ts_G.number_of_edges() == prev_G.number_of_edges() and \
               list(ts_G.edges()) == list(prev_G.edges()):
                print(f"  (!) Warning: Snapshot {i} is identical to Snapshot {i-1}")
         
        data.append(ts_G)
    
    print(f"[Done] Total Snapshots processed: {len(data)}\n")
    return data

def printStatistics(graphs, dataset, snapshot_duration):
    all_nodes = set()
    num_edges = 0
    for G in graphs:
        all_nodes.update(G.nodes())
        num_edges += G.number_of_edges()
    print(f'Statistics for {dataset}:\n\tNodes: {len(all_nodes)}\n\tEdges: {num_edges}\n\tSnaps: {len(graphs)}\n\tDur: {snapshot_duration}')

# ==========================================
# 2. Model-Specific Converters (Optimized)
# ==========================================

def toEvolveGCN(graphs, node_count, node_map):
    A_list = []
    sparse_x = get_sparse_identity(node_count)
    for G in graphs:
        G_rel = nx.relabel_nodes(G, node_map)
        G_rel.add_nodes_from(range(node_count))
        adj_norm = normalize_adj(nx.adjacency_matrix(G_rel))
        A_list.append(sparse_mx_to_torch_sparse_tensor(adj_norm))
    return {'A_list': A_list, 'Nodes_list': [sparse_x]*len(graphs), 'node_count': node_count, 'feature_dim': node_count}
    # TODO might switch this to use binary encoding for node features

def toStandardPyG(graphs, node_count, node_map):
    processed = []
    sparse_x = get_sparse_identity(node_count)
    for G in graphs:
        G_rel = nx.relabel_nodes(G, node_map)
        G_rel.add_nodes_from(range(node_count))
        # Ensure PyG format (bi-directional for undirected)
        G_pyg = nx.to_directed(G_rel) if not G_rel.is_directed() else G_rel
        edges = torch.tensor(list(G_pyg.edges()), dtype=torch.long).t().contiguous()
        processed.append(Data(x=sparse_x, edge_index=edges))
    return {'snapshots': processed, 'node_count': node_count, 'feature_dim': node_count}

# ==========================================
# 3. Dynamic Loading Interface
# ==========================================

def load_data(model_name, dataset_name, is_directed=True):
    path = ensure_edgelist(dataset_name)
    dur = get_snapshot_duration(dataset_name)
    graphs = toNetworkx(path, dur, is_directed=is_directed)
    printStatistics(graphs, dataset_name, dur)
    
    all_nodes = sorted(list(set().union(*(G.nodes() for G in graphs))))
    node_map = {node: i for i, node in enumerate(all_nodes)}
    node_count = len(all_nodes)
    
    if model_name.lower() == 'evolvegcn':
        return toEvolveGCN(graphs, node_count, node_map)
    return toStandardPyG(graphs, node_count, node_map)