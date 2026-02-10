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
    return torch.sparse.FloatTensor(indices, values, shape)

def toNetworkx(edgelist, snapshot_duration):
    """
    Parses edgelist and returns a list of NetworkX graphs (snapshots).
    Preserves 'value' attribute for weighted graphs.
    """
    data = []
    # Check if file exists
    if not os.path.exists(edgelist):
        raise FileNotFoundError(f"Edgelist not found at {edgelist}. Ensure data extraction ran.")

    edgelist_df = pd.read_csv(edgelist)
    
    # Standardize Time
    if pd.api.types.is_numeric_dtype(edgelist_df['date']):
        edgelist_df['timestamp'] = pd.to_datetime(edgelist_df['date'], unit='s')
    else:
        edgelist_df['timestamp'] = pd.to_datetime(edgelist_df['date'])
    
    # Determine Snapshot IDs
    if snapshot_duration in ['M', 'ME', '1M']:
        edgelist_df['snapshot_id'] = edgelist_df['timestamp'].dt.to_period('M')
    else:
        freq = f"{snapshot_duration}s" if isinstance(snapshot_duration, int) else snapshot_duration
        edgelist_df['snapshot_id'] = edgelist_df['timestamp'].dt.floor(freq)
    
    edgelist_df = edgelist_df.drop_duplicates(subset=["from", "to", "snapshot_id"])
    uniq_snapshots = sorted(edgelist_df['snapshot_id'].unique())

    for ts in uniq_snapshots:
        # Include 'value' in selection
        ts_edges = edgelist_df.loc[edgelist_df['snapshot_id'] == ts, ['from', 'to', 'value']]
        # Tell NetworkX to store 'value' as an edge attribute
        ts_G = nx.from_pandas_edgelist(ts_edges, 'from', 'to', edge_attr=['value'])
        data.append(ts_G)
    
    return data

def printStatistics(graphs, dataset, snapshot_duration):
    all_node_set = set()
    num_edges = 0
    for graph in graphs:
        all_node_set.update(graph.nodes())
        num_edges += len(graph.edges())
    
    print(f'Statistics for dataset: {dataset}')
    print(f'\tNum Nodes: {len(all_node_set)}')
    print(f'\tNum Edges: {num_edges}')
    print(f'\tNum Snapshots: {len(graphs)}')
    print(f'\tSnapshot Duration: {snapshot_duration}')

# ==========================================
# 2. Model-Specific Converters
# ==========================================

def toEvolveGCN(graphs, dataset_name, use_node_features=True, save=True):
    print(f"--- Preprocessing {dataset_name} for EvolveGCN ---")
    all_nodes = set()
    for G in graphs: all_nodes.update(G.nodes())
    sorted_nodes = sorted(list(all_nodes))
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    node_count = len(sorted_nodes)
    
    has_features = False
    if len(graphs) > 0 and use_node_features:
        first_node = list(graphs[0].nodes())[0]
        if 'feat' in graphs[0].nodes[first_node]: has_features = True

    A_list = []
    Nodes_list = []
    
    for G in graphs:
        G_relabeled = nx.relabel_nodes(G, node_map)
        G_relabeled.add_nodes_from(range(node_count))
        
        # Adjacency
        adj = nx.adjacency_matrix(G_relabeled)
        adj_norm = normalize_adj(adj)
        A_list.append(sparse_mx_to_torch_sparse_tensor(adj_norm))
        
        # Features
        if has_features:
            feat_list = [G_relabeled.nodes[n].get('feat', np.zeros(1)) for n in range(node_count)]
            x = torch.tensor(np.array(feat_list), dtype=torch.float)
        else:
            x = torch.eye(node_count, dtype=torch.float)
        Nodes_list.append(x)

    data_dict = {
        'A_list': A_list,
        'Nodes_list': Nodes_list,
        'node_count': node_count,
        'feature_dim': Nodes_list[0].size(1)
    }
    
    if save:
        save_path = f"benchmarkers/data/input/cached/evolvegcn/cached_{dataset_name}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(data_dict, f)
        print(f"Saved EvolveGCN data to {save_path}")
        
    return data_dict

def toGCLSTM(graphs, dataset, use_node_features=False, save=True):
    all_nodes = set()
    for G in graphs: all_nodes.update(G.nodes())
    sorted_nodes = sorted(list(all_nodes))
    node_count = len(sorted_nodes)
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    
    processed_data = []
    has_features = False # Simplified feature check logic
    
    for G in graphs:
        G_remapped = nx.relabel_nodes(G, node_map)
        G_remapped.add_nodes_from(range(node_count))
        
        # Features
        if use_node_features and has_features:
             # Add feature extraction logic if needed
             x = torch.eye(node_count) # Placeholder logic
        else:
             x = torch.eye(node_count)
             
        edge_index = torch.tensor(list(G_remapped.edges)).t().contiguous()
        if edge_index.shape[0] == 0: edge_index = torch.empty((2, 0), dtype=torch.long)
        
        processed_data.append(Data(x=x, edge_index=edge_index))
    
    data_dict = {'snapshots': processed_data, 'node_count': node_count, 'feature_dim': x.shape[1]}
    
    if save:
        save_path = f'benchmarkers/data/input/cached/gclstm/cached_{dataset}.pkl'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(data_dict, f)
        print(f"Saved GCLSTM data to {save_path}")
        
    return data_dict

def toHTGN(graphs, dataset_name, use_node_features=False, save=True):
    print(f"--- Preprocessing {dataset_name} for HTGN ---")
    all_nodes = set()
    for G in graphs: all_nodes.update(G.nodes())
    sorted_nodes = sorted(list(all_nodes))
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    node_count = len(sorted_nodes)
    
    processed_snapshots = []
    has_features = False
    if len(graphs) > 0 and use_node_features:
        first_node = list(graphs[0].nodes())[0]
        if 'feat' in graphs[0].nodes[first_node]: has_features = True

    for G in graphs:
        G_relabeled = nx.relabel_nodes(G, node_map)
        G_relabeled.add_nodes_from(range(node_count))
        
        edges = list(G_relabeled.edges())
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            if not nx.is_directed(G):
                row, col = edge_index
                mask = row < col
                edge_index = torch.cat([edge_index[:, mask], torch.stack([col[mask], row[mask]], dim=0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        if has_features:
            feat_list = [G_relabeled.nodes[n].get('feat', np.zeros(1)) for n in range(node_count)]
            x = torch.tensor(np.array(feat_list), dtype=torch.float)
        else:
            x = torch.eye(node_count, dtype=torch.float)
        
        processed_snapshots.append(Data(x=x, edge_index=edge_index))

    data_dict = {'snapshots': processed_snapshots, 'node_count': node_count, 'feature_dim': processed_snapshots[0].x.size(1)}
    
    if save:
        save_path = f"benchmarkers/data/input/cached/htgn/cached_{dataset_name}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(data_dict, f)
        print(f"Saved HTGN data to {save_path}")
        
    return data_dict

def toROLAND(graphs, dataset_name, use_node_features=True, save=True):
    print(f"--- Preprocessing {dataset_name} for ROLAND ---")
    all_nodes = set()
    for G in graphs: all_nodes.update(G.nodes())
    sorted_nodes = sorted(list(all_nodes))
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    node_count = len(sorted_nodes)
    
    processed_snapshots = []
    
    # Feature Detection
    has_node_feat = False
    has_edge_feat = False
    edge_feat_name = None
    edge_dim = 0
    
    if len(graphs) > 0:
        first_G = graphs[0]
        first_edge = list(first_G.edges())[0] if len(first_G.edges()) > 0 else None
        first_node = list(first_G.nodes())[0] if len(first_G.nodes()) > 0 else None
        
        if first_node is not None and 'feat' in first_G.nodes[first_node]: has_node_feat = True
        
        if first_edge is not None:
            if 'value' in first_G.edges[first_edge]:
                has_edge_feat = True; edge_feat_name = 'value'; edge_dim = 1
            elif 'weight' in first_G.edges[first_edge]:
                has_edge_feat = True; edge_feat_name = 'weight'; edge_dim = 1
            elif 'feat' in first_G.edges[first_edge]:
                has_edge_feat = True; edge_feat_name = 'feat'
                feat_val = first_G.edges[first_edge]['feat']
                edge_dim = len(feat_val) if isinstance(feat_val, (list, np.ndarray)) else 1

    for G in graphs:
        G_relabeled = nx.relabel_nodes(G, node_map)
        G_relabeled.add_nodes_from(range(node_count))
        
        edges = list(G_relabeled.edges())
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            if has_edge_feat:
                edge_attrs = []
                for u, v in edges:
                    val = G_relabeled.edges[u, v].get(edge_feat_name, 0.0)
                    if not isinstance(val, (list, np.ndarray)): val = [val]
                    edge_attrs.append(val)
                edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)
            else:
                edge_attr = None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long); edge_attr = None

        if has_node_feat and use_node_features:
            x_list = [G_relabeled.nodes[n].get('feat', np.zeros(1)) for n in range(node_count)]
            x = torch.tensor(np.array(x_list), dtype=torch.float)
        else:
            x = torch.eye(node_count, dtype=torch.float)

        processed_snapshots.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    data_dict = {
        'snapshots': processed_snapshots, 'node_count': node_count,
        'feature_dim': processed_snapshots[0].x.size(1), 'edge_dim': edge_dim
    }
    
    if save:
        save_path = f"benchmarkers/data/input/cached/roland/cached_{dataset_name}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(data_dict, f)
        print(f"Saved ROLAND data to {save_path}")
        
    return data_dict

def toTGCN(graphs, dataset, use_node_features=False, save=True):
    all_nodes = set()
    for G in graphs: all_nodes.update(G.nodes())
    sorted_nodes = sorted(list(all_nodes))
    node_count = len(sorted_nodes)
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    
    processed_data = []
    global_max = 1.0 # Normalization logic...
    
    for G in graphs:
        G_remapped = nx.relabel_nodes(G, node_map)
        G_remapped.add_nodes_from(range(node_count))
        x = torch.eye(node_count) # Simplified for brevity
        edge_index = torch.tensor(list(G_remapped.edges)).t().contiguous()
        if edge_index.shape[0] == 0: edge_index = torch.empty((2, 0), dtype=torch.long)
        processed_data.append(Data(x=x, edge_index=edge_index))
    
    data_dict = {'snapshots': processed_data, 'node_count': node_count, 'feature_dim': x.shape[1], 'max_value': global_max}
    if save:
        save_path = f'benchmarkers/data/input/cached/tgcn/cached_{dataset}.pkl'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(data_dict, f)
    return data_dict

def toVGRNN(graphs, dataset_name, save=True):
    print(f"--- Preprocessing {dataset_name} for VGRNN ---")
    all_nodes = set()
    for G in graphs: all_nodes.update(G.nodes())
    sorted_nodes = sorted(list(all_nodes))
    node_map = {node: i for i, node in enumerate(sorted_nodes)}
    node_count = len(sorted_nodes)
    
    processed_snapshots = []
    for G in graphs:
        G_relabeled = nx.relabel_nodes(G, node_map)
        G_relabeled.add_nodes_from(range(node_count))
        
        edges = list(G_relabeled.edges())
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            if not nx.is_directed(G):
                row, col = edge_index
                mask = row < col
                edge_index = torch.cat([edge_index[:, mask], torch.stack([col[mask], row[mask]], dim=0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        x = torch.eye(node_count, dtype=torch.float)
        processed_snapshots.append(Data(x=x, edge_index=edge_index))

    data_dict = {'snapshots': processed_snapshots, 'node_count': node_count, 'feature_dim': node_count}
    
    if save:
        save_path = f"benchmarkers/data/input/cached/vgrnn/cached_{dataset_name}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f: pickle.dump(data_dict, f)
        print(f"Saved VGRNN data to {save_path}")
        
    return data_dict

# ==========================================
# 3. Dynamic Loading Interface
# ==========================================

def get_snapshot_duration(dataset):
    if dataset in ['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion', 'networkaeternity', 'networkaragon', 
                   'networkbancor', 'networkcentra', 'networkcoindash', 'networkiconomi', 'networkcindicator', 
                   'networkdgd', 'Reddit_B', 'tgbl-wiki', 'bitcoinotc', 'bitcoinalpha', 'uci-message']:
        return '1D'
    elif dataset in ['Hypertext09', 'ia-contact']:
        return '10min'
    elif dataset in ['Enron']:
        return '3D'
    elif dataset in ['radoslaw', 'fb-forum']:
        return '8H'
    elif dataset in ['HepPH', 'HepTH']:
        return '1M'
    else:
        return '1D' # Default fallback
    

def ensure_edgelist(dataset):
    """
    Ensures raw data is converted to standardized edgelist.
    (This contains the logic previously in toRaw)
    """
    output_path = f'data/input/raw/edgelist/{dataset}.txt'
    if os.path.exists(output_path):
        return output_path

    print(f"Genering edgelist for {dataset}...")
    # Define input paths (relative to repo root)
    if dataset == 'Hypertext09':
        input_path = 'benchmarkers/data/input/raw/ia-contacts_hypertext2009.edges'
    elif dataset == 'fb-forum':
        input_path = 'benchmarkers/data/input/raw/fb-forum.edges'
    elif dataset in ['HepTH', 'HepPH']:
        input_path = f'benchmarkers/data/input/raw/cit-{dataset}.edges'
    elif dataset == 'ia-contact':
        input_path = f'benchmarkers/data/input/raw/{dataset}.edges'
    elif dataset == 'radoslaw':
        input_path = 'benchmarkers/data/input/raw/ia-radoslaw-email.edges'
    elif dataset == 'uci-message':
        input_path = 'benchmarkers/data/input/raw/out.opsahl-ucsocial'
    elif dataset in ['bitcoinotc', 'bitcoinalpha']:
        input_path = f'benchmarkers/data/input/raw/soc-sign-{dataset}.edges'
    else:
        return output_path

    # Conversion logic
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if dataset in ['Hypertext09', 'fb-forum']:
        try:
            df = pd.read_csv(input_path, sep=None, engine='python', names=['from', 'to', 'timestamp'])
            formatted_df = pd.DataFrame({'from': df['from'], 'to': df['to'], 'date': df['timestamp'], 'value': 1.0})
            formatted_df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Error converting {dataset}: {e}")

    elif dataset in ['HepTH', 'HepPH', 'radoslaw', 'uci-message', 'ia-contact']:
        df = pd.read_csv(input_path, sep=r'\s+', comment='%', names=['from', 'to', 'value_orig', 'timestamp'], engine='python')
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
        df['value'] = 1.0
        df[['from', 'to', 'date', 'value']].to_csv(output_path, index=False)

    elif dataset in ['bitcoinotc', 'bitcoinalpha']:
        df = pd.read_csv(input_path, names=['source', 'target', 'rating', 'timestamp'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
        df[['source', 'target', 'date', 'rating']].rename(columns={'source':'from', 'target':'to', 'rating':'value'}).to_csv(output_path, index=False)

    return output_path

def load_data(model_name, dataset_name):
    """
    Main Interface: Loads data for a specific model/dataset on-the-fly.
    Does NOT save to disk to save space.
    """
    edgelist_path = ensure_edgelist(dataset_name)
    duration = get_snapshot_duration(dataset_name)
    
    print(f"Loading {dataset_name} for {model_name} (Duration: {duration})...")
    graphs = toNetworkx(edgelist_path, duration)
    printStatistics(graphs, dataset_name, duration)
    
    model_name = model_name.lower()
    
    if model_name == 'evolvegcn':
        return toEvolveGCN(graphs, dataset_name, save=False)
    elif model_name == 'roland':
        return toROLAND(graphs, dataset_name, save=False)
    elif model_name == 'htgn':
        return toHTGN(graphs, dataset_name, save=False)
    elif model_name == 'gclstm':
        return toGCLSTM(graphs, dataset_name, save=False)
    elif model_name == 'tgcn':
        return toTGCN(graphs, dataset_name, save=False)
    elif model_name == 'vgrnn':
        return toVGRNN(graphs, dataset_name, save=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def toCached():
    """
    Batch processing, if you want to save to cache
    """
    
    datasets = ['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion', 'networkaeternity', 'networkaragon', 
        'networkbancor', 'networkcentra', 'networkcoindash', 'networkiconomi', 'networkcindicator', 
        'networkdgd', 'Reddit_B', 'tgbl-wiki', 'bitcoinotc', 'bitcoinalpha', 'uci-message', 'Hypertext09', 'ia-contact',
        'Enron', 'radoslaw', 'fb-forum', 'HepPH', 'HepTH']
    
    for dataset in datasets:
        try:
            # Generate graphs once per dataset
            edgelist_path = ensure_edgelist(dataset)
            duration = get_snapshot_duration(dataset)
            graphs = toNetworkx(edgelist_path, duration)
            printStatistics(graphs, dataset, duration)
            
            # Generate and SAVE all formats
            toEvolveGCN(graphs, dataset, save=True)
            toGCLSTM(graphs, dataset, save=True)
            toHTGN(graphs, dataset, save=True)
            toROLAND(graphs, dataset, save=True)
            toTGCN(graphs, dataset, save=True)
            toVGRNN(graphs, dataset, save=True)
            
        except Exception as e:
            print(f'Error processing {dataset}: {e}')
            # import traceback; traceback.print_exc()

if __name__ == "__main__":
    toCached()