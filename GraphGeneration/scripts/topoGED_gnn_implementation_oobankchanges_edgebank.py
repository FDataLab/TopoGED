import math
import numpy as np 
import networkx as nx
import random
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
import sys
sys.setrecursionlimit(5000)
import yaml
import pickle 
import time
import faulthandler
import copy
import time
import psutil
from torchmetrics.functional import auroc

faulthandler.enable()

#import line_profiler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.visualizer import Visualizer
from GraphGeneration.utils.Evaluator import Evaluator
from load_data import load_data, generate_training_data_cached, generate_validation_data_cached, generate_negative_edges
from GraphGeneration.utils.sampling_edges_utils import predict_edges
from GraphGeneration.utils.casting_type import to_tensor
from GraphGeneration.utils.graph_construction_utils import compute_reappearance_probabilities, generate_tgcn_node_features, get_node_features, update_degrees, generate_gnn_node_embeddings
from create_sub_graphs import create_nn_graph, create_on_graph
from GraphGeneration.utils.ablation_utils import ablationSetup

# Models in use
from GraphGeneration.models.model import setupMLP, load_encoder_model

# Import all node embedding methods
from compute_embedding import compute_node2vec_embeddings, compute_temporal_node_embeddings
from process_data import modifyGraphIds, build_edgebanks_from_start
from torch.utils.data import DataLoader

# Import Loss fn
from GraphGeneration.scripts.composite_graphlet_loss_fn import GraphletLoss
from GraphGeneration.utils.estimate_graphlet import run_graphlet_estimate   
# TODO Rename these ^^^

from utils.embedding_methods.degree import EmbedDegree


import warnings
from sklearn.exceptions import UndefinedMetricWarning
from GraphGeneration.utils.EdgeDataset import EdgeDataset

# Suppress only the specific AUC warning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Set up device
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
except Exception:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load YAML config
with open("GraphGeneration/encoder.yaml", "r") as file:
    encoder_config = yaml.safe_load(file)

# Set seeds
random.seed(encoder_config["seed"])
np.random.seed(encoder_config["seed"]) 

class Runner(object):
    def __init__(self):      
        self.seed = encoder_config["seed"]
        self.use_ma = encoder_config["use_moving_average"]  # Whether to use moving average for node2vec embeddings or not
        self.model_type = encoder_config["encoder_model"]["nodeEmbeddingType"]
        self.feature_type = encoder_config["encoder_model"]["other_models"]["feature_type"]  # Will be useful later
        self.edgebank_style = encoder_config["edgebank_style"]
        
        # Set up Evaluator
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()
        self.device = device
        self.process = psutil.Process(os.getpid())
        
        # Controls our window size and how we actually construct the graph (directed vs undirected)
        self.days_back = encoder_config["days_back"]
        self.is_directed = encoder_config["directed_flag"]
        
        if self.days_back <= 0:
            raise ValueError(f"days_back must be a positive integer (days_back > 0). Got {self.days_back}.")
        elif self.days_back >= 10000:
            days_back_val = 'all'
        else:
            days_back_val = str(self.days_back)
            
        # Some default file path
        self.file_visualization_path = "GraphGeneration/scripts/Visualize"
        self.saved_input = os.path.abspath(f'data/input/cached/{encoder_config["dataset"]}/saved_data_gnn_{self.model_type}{self.feature_type}_lr{encoder_config["training"]["lr"]}_{days_back_val}back_oobankchanges')
        self.saved_samples = os.path.join(self.saved_input, 'saved_samples.pkl')
        self.common_suffix = f'topoGED_embedding{encoder_config["encoder_model"]["addOnFeature"]}_mlpEncoding{encoder_config["decoder_model"]["encode_links"]}_embeddingType{encoder_config["encoder_model"]["nodeEmbeddingType"]}_lr{encoder_config["training"]["lr"]}_{days_back_val}back_oobankchanges_predvals{encoder_config["use_predicted_vals"]}'
        self.saved_graph_dir = f'data/output/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}_edgebank_{self.edgebank_style}'

        
        save_dir = os.path.join(self.file_visualization_path, encoder_config["dataset"], encoder_config["encoder_model"]["nodeEmbeddingType"])
        os.makedirs(save_dir, exist_ok=True)
        
        # Current target snapshot we want to predict
        self.starting_graph = encoder_config["starting_graph_idx"]
        self.current_target_snapshot = self.starting_graph
        
        # Controls our window size and how we actually construct the graph (directed vs undirected)
        self.days_back = encoder_config["days_back"]
        self.is_directed = encoder_config["directed_flag"]
        
        if self.days_back <= 0:
            raise ValueError(f"days_back must be a positive integer (days_back > 0). Got {self.days_back}.")
        
        # All the edge types
        self.all_edge_types = ['o-o-bank', 'n-n', 'o-n', 'o-o-nobank']
        self.best_validation_model_auc = 0
        
        days_back_val = 'all' 
        print('[INFO] USING ALL BACK FOR PROBABILITIES AS A TEST SINCE IM PRETTY SURE THAT ACTUALLY MAKES MORE SENSE')
        self.probabilities, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(encoder_config["dataset"], encoder_config["encoder_model"]["addOnFeature"], 
                                                                                                     encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"], days_back_val, encoder_config["use_predicted_vals"], encoder_config["num_toper_buckets"], use_test_style=encoder_config["use_test_style"])
        
        # Modify the graph ids to 1,2,3,...
        self.target_graphs, _ = modifyGraphIds(self.target_graphs, self.thresholds, 10000)
        
        # Exclusive to TGCN implementation; one hot encoding
        num_nodes = len(set(node for graphs in self.target_graphs for node in graphs[-1].nodes()))        
        
        feature_dim = encoder_config["encoder_model"]["other_models"]["feature_dim"]
        
        if self.feature_type == 'learnable':
            self.node_features = nn.Embedding(num_nodes, feature_dim).to(self.device)
            torch.nn.init.xavier_uniform_(self.node_features.weight)
        else:
            total_edges = sum(len(g[-1].edges()) for g in self.target_graphs)
            edge_features = np.zeros((total_edges, 0), dtype=np.float32)  # Since TGN needs features, provide dummy features
            self.node_features = generate_tgcn_node_features(self.target_graphs, feature_dim, feature_type=self.feature_type, device=self.device)
        
        self.embedding_dim = encoder_config["encoder_model"]["other_models"]["embedding_dim"]
        
        # Load the global encoder & decoder model
        self.encoder_model, self.input_dim = load_encoder_model(encoder_config, device=device, node2vec_dimensions=self.embedding_dim, 
                                                                hidden_dim=encoder_config["encoder_model"]["hidden_dim"], num_layers=encoder_config["encoder_model"]["other_models"]["num_layers"],
                                                                node_features=self.node_features, edge_features=edge_features)
                
        self.link_prediction_decoder = setupMLP(embedding_dim=self.input_dim*2, mlpEncoding=encoder_config["decoder_model"]["encode_links"])
        self.link_prediction_decoder.to(device)
        
        # Build the edgebanks for construction
        self.all_edgebanks = build_edgebanks_from_start(self.target_graphs, self.is_directed, self.days_back)        

        # Reshape the graph description
        # self.graph_descriptions = [list(zip(graph_description[0::3], graph_description[1::3], graph_description[2::3])) for graph_description in self.graph_descriptions]
        self.graph_descriptions = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in self.graph_descriptions]
        
        # Split training, validation, test graphs
        # Snapshots that we will use for traininng
        # Convert number of snapshots to integer
        self.num_snapshots = len(self.target_graphs)
        self.train_end = int(0.7 * self.num_snapshots)
        self.val_end = int(0.85 * self.num_snapshots)

        # Assign snapshots
        self.training_graphs = [self.target_graphs[i][-1] for i in range(self.train_end)]
        self.validation_graphs = [self.target_graphs[i][-1] for i in range(self.train_end, self.val_end)]
        self.test_graphs = [self.target_graphs[i][-1] for i in range(self.val_end, self.num_snapshots)]

        self.starting_graph = self.num_snapshots - len(self.test_graphs)

        self.max_node_id = 0  # The ID we will assign new node (incremented as we add nodes)


    # ======================= HELPER FUNCTIONS =======================
    def sample_old_nodes(self, prev_graphs):
        """
        Retrieve the old nodes that we want to add to the current graph
        We use compute_reappearance_probabilities to help figure out what nodes to add back
        
        Params:
            prev_graphs (list): The previous self.days_back graphs that we will take old nodes from
            
        Returns:
            (set(sampled_old_nodes)): The set of nodes that we will now add during construction
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Sample old nodes
        probs = compute_reappearance_probabilities(graphs=prev_graphs, days_back=self.days_back)
        node_ids = list(probs.keys())
        weights = list(probs.values())

        # Prevents failing if there are no old nodes
        if not node_ids:
            print(f'There are no old nodes to sample from')
            return set([])

        sampled_old_nodes = list(np.random.choice(node_ids, size=self.current_target_count_old_nodes, replace=False, p=np.array(weights)/np.sum(weights)))  # Makes sure that we select only unique nodes each time
        
        return set(sampled_old_nodes)
            

    def create_samples(self, graphs, days_back, all_edgebanks, is_directed=False):
        # Initialize with lists for raw numeric storage
        sorted_samples = {
            et: {'u': [], 'v': [], 'y': []} for et in self.all_edge_types
        }
                
        for i, graph in enumerate(graphs):
            #if i < self.starting_graph: continue 
            
            # Determine old nodes for this snapshot
            old_nodes_days = set().union(*[g.nodes() for g in graphs[max(i - days_back, 0): i]])
            
            # Categorize edges by type
            snapshot_edges = {et: [] for et in self.all_edge_types}
            for u, v in graph.edges():
                if u in old_nodes_days and v in old_nodes_days:
                    edge_type = 'o-o-bank' if v in all_edgebanks[i].get(u, set()) else 'o-o-nobank'
                elif (u in old_nodes_days) != (v in old_nodes_days): # XOR logic for O-N
                    edge_type = 'o-n'
                else:
                    edge_type = 'n-n'
                snapshot_edges[edge_type].append((u, v))

            # Generate Positive and Negative samples for this snapshot
            for edge_type in self.all_edge_types:
                pos_edges = snapshot_edges[edge_type]
                num_pos = len(pos_edges)
                
                # Get Negatives
                neg_edges = generate_negative_edges(
                    graph, num_samples=num_pos, edge_type=edge_type,
                    old_nodes=old_nodes_days, is_directed=is_directed, edgebank=all_edgebanks[i]
                )
                
                # Combine and append to graph-indexed list
                # We store as separate lists (u, v, y) for memory efficiency
                u_list = [e[0] for e in pos_edges] + [e[0] for e in neg_edges]
                v_list = [e[1] for e in pos_edges] + [e[1] for e in neg_edges]
                y_list = [1] * num_pos + [0] * len(neg_edges)

                sorted_samples[edge_type]['u'].append(np.array(u_list, dtype=np.int32))
                sorted_samples[edge_type]['v'].append(np.array(v_list, dtype=np.int32))
                sorted_samples[edge_type]['y'].append(np.array(y_list, dtype=np.float32))
                    
        return sorted_samples
    
    # ======================= TRAIN MODEL =======================
    def run_validation(self, batch_size, epoch, samples, snapshot_num):
        criterion = nn.BCELoss()
        results = {et: {'loss': [], 'auc': []} for et in self.all_edge_types}
        # Note: In this version, only 'o-o-nobank' was active, but we loop for safety
        active_edge_types = ['o-o-nobank', 'o-n', 'n-n']

        self.link_prediction_decoder.eval()
        self.encoder_model.eval()

        n_graphs = len(samples['o-o-nobank']['u'])

        with torch.no_grad():  
            for i in range(n_graphs):
                #feat_idx = i + snapshot_num + self.starting_graph
                feat_idx = i + snapshot_num
                features = self.node_features.weight if self.feature_type == 'learnable' else \
                           (self.node_features[feat_idx] if self.feature_type == 'node2vec' else self.node_features)

                # Generate Embeddings ONCE per snapshot
                snapshot_embeddings = generate_gnn_node_embeddings(
                    self.encoder_model, self.model_type, features,
                    self.target_graphs[:feat_idx], self.days_back, 
                    embedding_dim=self.embedding_dim,
                    curr_nodes=self.target_graphs[feat_idx][-1].nodes(),
                    thresholds=self.thresholds, new_node_strategy='zeros',
                    device=self.device
                )

                for et in active_edge_types:
                    u_t = samples[et].get('u_t', [None])[i]
                    v_t = samples[et].get('v_t', [None])[i]
                    y_t = samples[et].get('y_t', [None])[i]

                    # Safety check if samples are empty for this type/snapshot
                    if u_t is None or u_t.numel() == 0:
                        continue

                    # We use snapshot_embeddings[u_t] to gather vectors in parallel on the GPU
                    ub = snapshot_embeddings[u_t]
                    vb = snapshot_embeddings[v_t]
                    
                    preds = self.link_prediction_decoder(ub, vb, edge_type=et).view(-1)
                    
                    # 3. Metrics Calculation
                    loss = criterion(preds, y_t.view(-1))
                    results[et]['loss'].append(loss.item())
                    
                    try:
                        auc = auroc(preds, y_t.view(-1).long(), task="binary")
                        results[et]['auc'].append(auc.item())
                    except Exception:
                        results[et]['auc'].append(0.5)

        avg_results = {et: {
            'loss': np.nanmean(results[et]['loss']) if results[et]['loss'] else 0.0,
            'auc': np.nanmean(results[et]['auc']) if results[et]['auc'] else 0.5
        } for et in self.all_edge_types}
        
        print(f"\n--- Validation Epoch {epoch+1} ---")
        for et, metrics in avg_results.items():
            if et in active_edge_types:
                print(f"[{et}] Val Loss: {metrics['loss']:.4f} | Val AUC: {metrics['auc']:.4f}")

        return avg_results


    def train_multi_head(self, training_samples, val_samples, test_samples):
        lr = encoder_config["training"]["lr"]
        batch_size = encoder_config["training"]["batch_size"]
        epochs = encoder_config["training"]["epochs"]
        
        # Optimizer Setup
        params = list(self.link_prediction_decoder.parameters()) + list(self.encoder_model.parameters())
        if self.feature_type == 'learnable':
            params += list(self.node_features.parameters())
        
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        active_edge_types = ['o-o-nobank', 'o-n', 'n-n']
        n_graphs = len(training_samples['o-o-nobank']['u'])

        best_val_auc = 0.0
        patience, counter = 10, 0
        best_state = {"encoder": None, "decoder": None}

        for epoch in range(epochs):
            self.encoder_model.train()
            self.link_prediction_decoder.train()
            train_results = {et: {'loss': [], 'auc': []} for et in active_edge_types}

            for g in range(n_graphs):
                # feat_idx = g + self.starting_graph
                feat_idx = g
                features = self.node_features.weight if self.feature_type == 'learnable' else \
                           (self.node_features[g] if self.feature_type == 'node2vec' else self.node_features)

                # IDEA 1: Cache Embeddings
                embeddings = generate_gnn_node_embeddings(
                    self.encoder_model, self.model_type, features,
                    self.target_graphs[:feat_idx], self.days_back, 
                    embedding_dim=self.embedding_dim,
                    curr_nodes=self.target_graphs[feat_idx][-1].nodes(),
                    thresholds=self.thresholds, new_node_strategy='zeros',
                    device=self.device
                )

                optimizer.zero_grad()
                snapshot_loss = 0.0
                has_active_data = False

                # IDEA 2: Sequentially process heads for a single backward pass
                for et in active_edge_types:
                    # Access the pre-tensorized data
                    u_t = training_samples[et]['u_t'][g]
                    v_t = training_samples[et]['v_t'][g]
                    y_t = training_samples[et]['y_t'][g].view(-1)

                    if u_t.numel() == 0: continue
                    has_active_data = True
                    
                    # --- STEP 2: VECTORIZED FORWARD PASS ---
                    # Using bulk indexing instead of manual chunk loops
                    u_emb = embeddings[u_t]
                    v_emb = embeddings[v_t]
                    
                    preds = self.link_prediction_decoder(u_emb, v_emb, edge_type=et).view(-1)
                    
                    loss = criterion(preds, y_t)
                    snapshot_loss += loss
                    
                    # Logging
                    train_results[et]['loss'].append(loss.item())
                    try:
                        auc = auroc(preds.detach(), y_t.long(), task="binary")
                        train_results[et]['auc'].append(auc.item())
                    except Exception:
                        train_results[et]['auc'].append(0.5)

                # Backward pass once per snapshot for all accumulated edge type losses
                if has_active_data and isinstance(snapshot_loss, torch.Tensor):
                    snapshot_loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()

            # Display Training results
            print(f"\n--- Training Epoch {epoch+1} ---")
            for et in active_edge_types:
                print(f"[{et}] Loss: {np.mean(train_results[et]['loss']):.4f} | AUC: {np.mean(train_results[et]['auc']):.4f}")

            # Validation
            val_res = self.run_validation(batch_size, epoch, val_samples, self.train_end)
            avg_val_auc = np.mean([val_res[et]['auc'] for et in val_res if et in active_edge_types or et == 'o-o-nobank'])
            
            print(f"\n>> Summary Epoch {epoch+1} | Avg Val AUC: {avg_val_auc:.4f}")

            if avg_val_auc > best_val_auc:
                best_val_auc = avg_val_auc
                counter = 0
                best_state["encoder"] = copy.deepcopy(self.encoder_model.state_dict())
                best_state["decoder"] = copy.deepcopy(self.link_prediction_decoder.state_dict())
            else:
                counter += 1
                if epoch >= 20 and counter >= patience: break

        if best_state["encoder"]:
            self.encoder_model.load_state_dict(best_state["encoder"])
            self.link_prediction_decoder.load_state_dict(best_state["decoder"])
            
    
    def train_models(self):
        if os.path.exists(self.saved_samples):
            with open(self.saved_samples, "rb") as f:
                all_samples = pickle.load(f)
        else:
            curr_graphs = [inner[-1] for inner in self.target_graphs]
            all_samples = self.create_samples(curr_graphs, self.days_back, self.all_edgebanks, self.is_directed)
            os.makedirs(os.path.dirname(self.saved_samples), exist_ok=True)
            
            with open(self.saved_samples, "wb") as f:
                pickle.dump(all_samples, f, protocol=5)
            print(f"[INFO] Saved all_samples to {self.saved_samples}")

        num_graphs = len(next(iter(all_samples.values()))['u'])
        n_train = int(0.7 * num_graphs)
        n_val = int(0.15 * num_graphs)

        def slice_samples(start, end):
            return {
                et: {
                    'u': all_samples[et]['u'][start:end],
                    'v': all_samples[et]['v'][start:end],
                    'y': all_samples[et]['y'][start:end]
                } for et in all_samples.keys()
            }

        training_samples = slice_samples(0, n_train)
        val_samples = slice_samples(n_train, n_train + n_val)
        test_samples = slice_samples(n_train + n_val, num_graphs)
        
        # Move all datasets to the device at once to avoid repeated transfers during training
        for dataset in [training_samples, val_samples, test_samples]:
            for et in self.all_edge_types:
                dataset[et]['u_t'] = [torch.tensor(a, dtype=torch.long, device=self.device) for a in dataset[et]['u']]
                dataset[et]['v_t'] = [torch.tensor(a, dtype=torch.long, device=self.device) for a in dataset[et]['v']]
                dataset[et]['y_t'] = [torch.tensor(a, dtype=torch.float, device=self.device) for a in dataset[et]['y']]

        self.train_multi_head(training_samples, val_samples, test_samples)
        
            
    # ======================= BUILD GRAPH =======================
    def build_accumulating_filtration_sequence_with_edgebank(self, current_target_snapshot):
        """
        Our main driver function to build graphs, takes in various arguments to guide the graph construction
        Specifically, this version uses an MLP to assign edges to two nodes based on the probability of them forming an edge
        But, this version also creates a new MLP before each new graph construction. A process called "continual learning"
        
        Args:
            current_target_graph_description (list): The TopER current_target_graph_description to guide construction of the graph, stores the number of nodes and edges to add to the graph
            
        Returns:
            filtration_graphs (list(nx.DiGraph)): A list of nx Graphs that we built up from our TopER current_target_graph_description
            node_types (dict): A dictionary that stores 'old_nodes' and 'new_nodes' organized into lists
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Get the edgebank up to the current target snapshot
        edgebank = self.all_edgebanks[current_target_snapshot] 
        current_target_graph_description = self.graph_descriptions[current_target_snapshot]
        current_target_graph_description = [
            (int(round(float(n))), int(round(float(e)))) 
            for n, e in current_target_graph_description
        ]
        prev_graphs = [graph[-1] for graph in self.target_graphs[max(current_target_snapshot - self.days_back, 0) : current_target_snapshot]]
        

        # Select the most recent edges from the last few graphs to add to the edgebank
        curr_edges_oobank = []
        curr_old_nodes = set()
        
        target_oo_bank = self.current_target_count['o-o-bank']
        target_nodes = self.current_target_count_old_nodes
        
        
        if self.edgebank_style == "default":
            # 1. Primary Pass: Add edges and their associated nodes
            for graph in prev_graphs[::-1]:
                if len(curr_edges_oobank) >= target_oo_bank:
                    break
                    
                for u, v in graph.edges():
                    if len(curr_edges_oobank) >= target_oo_bank:
                        break
                    
                    # Calculate how many NEW nodes this specific edge would introduce
                    new_nodes_to_add = 0
                    if u not in curr_old_nodes: new_nodes_to_add += 1
                    if v not in curr_old_nodes: new_nodes_to_add += 1
                    
                    # Logic: Only add the edge if we have enough "node slots" left
                    if len(curr_old_nodes) + new_nodes_to_add <= target_nodes:
                        curr_old_nodes.add(u)
                        curr_old_nodes.add(v)
                        curr_edges_oobank.append((u, v))

            # 2. Fill remaining node slots if we didn't use them all
            if len(curr_old_nodes) < target_nodes:
                # Get all nodes from history that we haven't picked yet
                all_hist_nodes = set().union(*[g.nodes() for g in prev_graphs])
                remaining_candidates = list(all_hist_nodes - curr_old_nodes)
                
                needed = target_nodes - len(curr_old_nodes)
                if remaining_candidates:
                    # Prioritize high-degree nodes or just sample to hit the target
                    fill_nodes = random.sample(remaining_candidates, min(len(remaining_candidates), needed))
                    curr_old_nodes.update(fill_nodes)
                    
        elif self.edgebank_style == "frequency":
            edge_stats = {} # Key: edge, Value: {'count': int, 'last_seen': int}
            all_hist_nodes = set() #

            for i, graph in enumerate(prev_graphs):
                all_hist_nodes.update(graph.nodes())
                
                for u, v in graph.edges():
                    if not self.is_directed:
                        edge = tuple(sorted((u, v)))
                    else:
                        edge = (u, v)
                    
                    if edge not in edge_stats:
                        edge_stats[edge] = {'count': 0, 'last_seen': i}
                    
                    edge_stats[edge]['count'] += 1
                    edge_stats[edge]['last_seen'] = i # Always update to latest index

            # Sort edges by frequency and recency
            sorted_candidates = sorted(
                edge_stats.keys(), 
                key=lambda e: (edge_stats[e]['count'], edge_stats[e]['last_seen']), 
                reverse=True
            )

            # Add edges and nodes
            for u, v in sorted_candidates:
                if len(curr_edges_oobank) >= target_oo_bank:
                    break
                
                # Count number of nodes to add
                new_nodes_to_add = 0
                if u not in curr_old_nodes: new_nodes_to_add += 1
                if v not in curr_old_nodes: new_nodes_to_add += 1
                
                # Logic: Only add the edge if we have enough "node slots" left
                if len(curr_old_nodes) + new_nodes_to_add <= target_nodes:
                    curr_old_nodes.add(u)
                    curr_old_nodes.add(v)
                    curr_edges_oobank.append((u, v))

            if len(curr_old_nodes) < target_nodes:
                # Get all nodes from history that we haven't picked yet
                remaining_candidates = list(all_hist_nodes - curr_old_nodes)
                
                needed = target_nodes - len(curr_old_nodes)
                if remaining_candidates:
                    # Prioritize high-degree nodes or just sample to hit the target
                    fill_nodes = random.sample(remaining_candidates, min(len(remaining_candidates), needed))
                    curr_old_nodes.update(fill_nodes)
            
            
        
        elif self.edgebank_style == "shuffle":
            # 1. Collect all unique edges from history
            unique_history = set()
            all_hist_nodes = set()

            for graph in prev_graphs:
                all_hist_nodes.update(graph.nodes())
                for u, v in graph.edges():
                    if not self.is_directed:
                        edge = tuple(sorted((u, v)))
                    else:
                        edge = (u, v)
                    unique_history.add(edge)
            
            # Shuffle edges
            shuffled_candidates = list(unique_history)
            random.shuffle(shuffled_candidates)

            for u, v in shuffled_candidates:
                if len(curr_edges_oobank) >= target_oo_bank:
                    break
                
                new_nodes_to_add = 0
                if u not in curr_old_nodes: new_nodes_to_add += 1
                if v not in curr_old_nodes: new_nodes_to_add += 1
                
                if len(curr_old_nodes) + new_nodes_to_add <= target_nodes:
                    curr_old_nodes.add(u)
                    curr_old_nodes.add(v)
                    curr_edges_oobank.append((u, v))

            if len(curr_old_nodes) < target_nodes:
                remaining_candidates = list(all_hist_nodes - curr_old_nodes)
                needed = target_nodes - len(curr_old_nodes)
                
                if remaining_candidates:
                    fill_nodes = random.sample(remaining_candidates, min(len(remaining_candidates), needed))
                    curr_old_nodes.update(fill_nodes)

        # Finalize
        old_nodes = curr_old_nodes
        num_old_nodes = len(old_nodes)
        
        # Create new node IDs
        new_nodes = np.arange(self.max_node_id, self.max_node_id + self.current_target_count_new_nodes)
        self.max_node_id += self.current_target_count_new_nodes
        
        
        all_nodes = list(old_nodes) + list(new_nodes)

        constructing_graph = nx.DiGraph() if self.is_directed else nx.Graph()  # A graph for computing node embeddings easily
        
        node_types = {
            "old_nodes": old_nodes,
            "new_nodes": new_nodes
        } 
        
        if self.H is not None:
            self.H = self.H.detach()
        if self.feature_type == 'learnable':
            features = self.node_features.weight
        else:
            features = self.node_features[current_target_snapshot] if self.feature_type == 'node2vec' else self.node_features

        # FIX: Wrap in no_grad to prevent graph building during construction
        with torch.no_grad():
            curr_embeddings = generate_gnn_node_embeddings(
                self.encoder_model, self.model_type, features, 
                self.target_graphs[:current_target_snapshot], self.days_back, 
                embedding_dim=self.embedding_dim, curr_nodes=all_nodes, thresholds=self.thresholds, new_node_strategy='zeros', device=self.device
            )

        constructing_graph = get_node_features(constructing_graph, prev_graphs, self.thresholds, current_target_graph_description, old_nodes, new_nodes)  
        constructing_graph.add_edges_from(curr_edges_oobank)
            
        for flag in ['o-o-nobank', 'o-n', 'n-n']:
            sampled_edges = predict_edges(
                constructing_graph, edge_type=flag, node_types=node_types, 
                edgebank=edgebank, link_prediction_decoder=self.link_prediction_decoder, 
                old_node_embeddings=curr_embeddings, top_k=self.current_target_count[flag], 
                graph_num=current_target_snapshot, device=self.device # CONSISTENT DEVICE
            )
        
            constructing_graph.add_edges_from(sampled_edges)
            update_degrees(constructing_graph)
                    

        # ======== START GRAPH CONSTRUCTION ========
        G = nx.DiGraph() if self.is_directed else nx.Graph()
        used_edges = set()
        filtration_graphs = []
        degrees = dict(constructing_graph.degree())

        all_edges = list(constructing_graph.edges())

        for threshold in self.thresholds[:-1]:
            current_nodes = {
                node for node, deg in degrees.items() if deg <= threshold
            }

            sub_g = nx.DiGraph() if self.is_directed else nx.Graph()
            sub_g.add_nodes_from(current_nodes)

            # SAFE edge filtering (no recursion)
            sub_g.add_edges_from(
                (u, v) for (u, v) in all_edges
                if u in current_nodes and v in current_nodes
            )

            filtration_graphs.append(sub_g)

        # ======================= FINAL GRAPH =======================
        final_full_graph = nx.DiGraph() if self.is_directed else nx.Graph()
        final_full_graph.add_nodes_from(constructing_graph.nodes())
        final_full_graph.add_edges_from(all_edges)

        filtration_graphs.append(final_full_graph)

        return filtration_graphs, node_types  # Hopefully this fixes
        
        
    def run(self):        
        """
        Our main runner function
        
        Params:
            None
            
        Returns: 
            None
        """     
        start_time = time.time()
        print("INFO: Dataset: {}".format(encoder_config["dataset"]))
        self.learnable_path = os.path.join(self.saved_input, rf"saved_models/embeddings")
        self.encoder_model_path = os.path.join(self.saved_input, rf'saved_models/embedder_{encoder_config["learnable_embedder"]["setup"]["init_type"]}_{self.seed}')
        self.decoder_model_path = os.path.join(self.saved_input, rf"saved_models/decoder_MLP_{self.seed}")
        start_time = time.time()
        if os.path.exists(self.decoder_model_path) and os.path.exists(self.encoder_model_path):
            # self.learnable_path.load_state_dict(torch.load(self.learnable_path, map_location=device))            
            # self.learnable_path.to(device)
            # self.learnable_path.eval()
            self.link_prediction_decoder.load_state_dict(torch.load(self.decoder_model_path, map_location=device))            
            self.link_prediction_decoder.to(device)
            self.link_prediction_decoder.eval()
            self.encoder_model.load_state_dict(torch.load(self.encoder_model_path, map_location=device))            
            self.encoder_model.to(device)
            self.encoder_model.eval()
            print(f"Link Prediction Decoder loaded from: {self.decoder_model_path}")
        else:
            # Train the Decoder and Encoder model
            print('Training the Link Prediction Decoder and Embedder')
            
            self.train_models()
            
            os.makedirs(os.path.dirname(self.decoder_model_path), exist_ok=True)
            torch.save(self.link_prediction_decoder.state_dict(), self.decoder_model_path)
            torch.save(self.encoder_model.state_dict(), self.encoder_model_path)

            print("Models successfully saved.")
            print('Finished training the Link Prediction Decoder and Encoder; Start Graph Construction')
       
        times = {'train': time.time() - start_time}
        
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 ** 2)
        gpu_stats = ""
        if torch.cuda.is_available():
            curr_alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            gpu_stats = f" | GPU Allocated: {curr_alloc:.2f}MB | GPU Peak: {peak_alloc:.2f}MB"
        
        print(f"{encoder_config['dataset']} oobankchanges TRAIN TIME: {times['train']:.2f}s | RAM: {ram_mb:.2f}MB{gpu_stats}")
        
        # Reset peak stats for Construction phase monitoring
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        
        if encoder_config["ablation"]:
            print('PERFORMING ABLATION STUDY')
            ablation_modes = [0, 1, 2, 3, 4, 5, 6]
            ablation_modes = [7, 8, 9]
        else:
            ablation_modes = [0]
        if encoder_config["sensitivity_analysis"]:
            sensitivity_params = [10]
            print('PERFORMING SENSITIVITY ANALYSIS ON TOPER LENGTHS')
        else:
            sensitivity_params = [encoder_config["num_toper_buckets"]]
                        
        if encoder_config["ablation"] and encoder_config["sensitivity_analysis"]:
            raise(ValueError('Cannot perform both ablation study and sensitivity analysis at the same time'))                
            
        # Save these since we will modify them as we go
        base_toper = copy.deepcopy(self.graph_descriptions)
        base_probs = copy.deepcopy(self.probabilities)
        for ablation_mode in ablation_modes:
            for toper_len in sensitivity_params:
                if ablation_mode == 0 and encoder_config["ablation"]:
                    print('Not reconstructing base graphs in ablation')
                    continue
                if ablation_mode > 0:
                    # Change it since that makes more sense now
                    self.saved_graph_dir = f'data/output/ablation/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}_edgebank_{self.edgebank_style}_ablation{ablation_mode}'
                    os.makedirs(self.saved_graph_dir, exist_ok=True)
                    # This gets unchanged if not using ablation
                    if ablation_mode < 9:
                        self.graph_descriptions, self.probabilities = ablationSetup(base_toper, base_probs, setting=ablation_mode)
                    elif ablation_mode == 9:
                        use_predicted_vals = False
                        _, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(encoder_config["dataset"], encoder_config["encoder_model"]["addOnFeature"], 
                                encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"], 'all', use_predicted_vals, toper_len, use_test_style=encoder_config["use_test_style"])
                        self.target_graphs, _ = modifyGraphIds(self.target_graphs, self.thresholds, 10000)
                        self.graph_descriptions = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in self.graph_descriptions]
                elif encoder_config["sensitivity_analysis"]:
                    # Change the TopER length here
                    _, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(encoder_config["dataset"], encoder_config["encoder_model"]["addOnFeature"], 
                                                                encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"], 'all', encoder_config["use_predicted_vals"], toper_len, use_test_style=encoder_config["use_test_style"])
                    self.probabilities = base_probs.copy()  # Just in case
                    # We have to redo these steps now
                    # Modify the graph ids to 1,2,3,...
                    self.target_graphs, _ = modifyGraphIds(self.target_graphs, self.thresholds, 10000)
                    self.graph_descriptions = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in self.graph_descriptions]
                    self.saved_graph_dir = f'data/output/sensitivity_analysis/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}_edgebank_{self.edgebank_style}_len{toper_len}'
                    os.makedirs(self.saved_graph_dir, exist_ok=True)
                    
                
                output_filepath = os.path.join(self.saved_graph_dir, f"{encoder_config['encoder_model']['nodeEmbeddingType']}_constructed_graphs_{encoder_config['dataset']}.pkl")
                
                
                # Old graphs that we know up to now
                self.old_graphs = [self.target_graphs[x][-1] for x in range(self.starting_graph)]
                
                all_node_ids = [node for graphs in self.old_graphs for node in graphs.nodes()]
                
                self.max_node_id = max(all_node_ids) + 1 if all_node_ids else 0

                all_built_graphs = []
                all_target_graphs = []
                all_pred_nodes = []
                all_true_nodes = []
                
                self.H = None
                
                start_time = time.time()
                # To predict snapshot i, we use snapshot 0,...,i-1 to train
                for i in range(self.starting_graph, len(self.probabilities)): 
                    # print("INFO: >>> Temporal Graph Construction <<<")
                    # print("INFO: Predict snapshot: ", i)
                    # print("======================================")

                    self.current_target_snapshot = i
                    
                    # Get all old nodes in our context window
                    self.current_target_old_nodes = set().union(*[g.nodes() for g in self.old_graphs[0: i]])
                    
                    current_target_graph_description = self.graph_descriptions[self.current_target_snapshot]
                    current_target_graph_description = [
                        (int(round(float(n))), int(round(float(e)))) 
                        for n, e in current_target_graph_description
                    ]
                    # Used to convert probabilities
                    V_total = int(current_target_graph_description[-1][0])
                    E_total = int(current_target_graph_description[-1][1])
                    
                    # Get the true count of 4 edges type and number of new, old nodes of the target snapshot (probabilities are fed in as percents)
                    node_raw = [p * V_total for p in self.probabilities[i][:2]]
                    node_counts = [int(math.floor(r)) for r in node_raw]
                    node_diff = V_total - sum(node_counts)
                    node_idx = sorted(range(2), key=lambda k: node_raw[k] - node_counts[k], reverse=True)
                    for j in range(abs(node_diff)):
                        node_counts[node_idx[j]] += 1 if node_diff > 0 else -1
                    self.current_target_count_old_nodes, self.current_target_count_new_nodes = node_counts

                    edge_raw = [p * E_total for p in self.probabilities[i][2:]]
                    edge_counts = [int(math.floor(r)) for r in edge_raw]
                    edge_diff = E_total - sum(edge_counts)
                    edge_idx = sorted(range(len(edge_raw)), key=lambda k: edge_raw[k] - edge_counts[k], reverse=True)
                    for j in range(abs(edge_diff)):
                        edge_counts[edge_idx[j]] += 1 if edge_diff > 0 else -1
                    self.current_target_count = {et: edge_counts[j] for j, et in enumerate(self.all_edge_types)}
                            
                    # Debugging:
                    if self.current_target_count_old_nodes + self.current_target_count_new_nodes != V_total:
                        print(f'WARNING: THE NUMBER OF NODES FROM PROBABILITIES IS WRONG: {self.current_target_count_old_nodes + self.current_target_count_new_nodes} != {V_total}')
                    if sum(self.current_target_count.values()) != E_total:
                        print(f'WARNING: THE NUMBER OF NODES FROM PROBABILITIES IS WRONG: {sum(self.current_target_count.values())} != {E_total}')
                    
                    
                    # Build the filtration sequence using the current parameters
                    try:
                        filtration_sequence, node_types = self.build_accumulating_filtration_sequence_with_edgebank(current_target_snapshot=i)
                        print(f"DEBUG: Successfully returned snap {i}")
                    except Exception as e:
                        print(f"CRITICAL ERROR IN BUILD: {e}")
                        raise e
                    
                    # Add the graphs to a list to save later
                    built_graph = filtration_sequence[-1]
                    target_graph = self.target_graphs[i][-1]
                    all_built_graphs.append(built_graph)
                    all_target_graphs.append(target_graph)
                    all_pred_nodes.append(node_types)
                    
                    # Get the node types for the target graph
                    current_nodes = target_graph.nodes()
                    old_nodes = current_nodes & self.current_target_old_nodes
                    new_nodes = current_nodes - old_nodes
                    all_true_nodes.append({"old_nodes": old_nodes, "new_nodes": new_nodes})
                    
                    # Add to the old graphs
                    self.old_graphs.append(self.target_graphs[i][-1])
            
                    
                output_filepath = os.path.join(self.saved_graph_dir, f"{encoder_config['encoder_model']['nodeEmbeddingType']}_constructed_graphs_{encoder_config['dataset']}.pkl")
                os.makedirs(self.saved_graph_dir, exist_ok=True)

                data_to_save = (all_built_graphs, all_target_graphs, all_pred_nodes, all_true_nodes)

                print("\n======================================")
                print(f"INFO: Saving {len(all_built_graphs)} pairs of graphs to {output_filepath}")
                print("======================================")

                with open(output_filepath, "wb") as f:
                    pickle.dump(data_to_save, f, protocol=5) 
                    
                output_filepath_old_only = os.path.join(self.saved_graph_dir, f"{encoder_config['encoder_model']['nodeEmbeddingType']}_constructed_graphs_{encoder_config['dataset']}_old_only.pkl")
                
                # Take the graphs that are just old nodes (o-o-bank and o-o-nobank only)
                # So we will save the same data (including nodes, minus new nodes and edges involving new nodes)
                
                # Data to save
                all_pred_nodes_old_only = copy.deepcopy(all_pred_nodes)
                all_true_nodes_old_only = copy.deepcopy(all_true_nodes)
                all_built_graphs_old_only = []
                all_target_graphs_old_only = []
                
                for i, (true_list, pred_list) in enumerate(zip(all_true_nodes, all_pred_nodes)):
                    true_old_nodes = true_list['old_nodes']
                    pred_old_nodes = pred_list['old_nodes']
                    
                    built_graph = all_built_graphs[i]
                    target_graph = all_target_graphs[i]
                    
                    new_built_graph = built_graph.subgraph(pred_old_nodes).copy()
                    new_target_graph = target_graph.subgraph(true_old_nodes).copy()
                    
                    all_built_graphs_old_only.append(new_built_graph)
                    all_target_graphs_old_only.append(new_target_graph)
                    all_pred_nodes_old_only[i]['new_nodes'] = set()
                    all_true_nodes_old_only[i]['new_nodes'] = set()
                    
                # Save the old only data    
                data_to_save_old_only = (all_built_graphs_old_only, all_target_graphs_old_only, all_pred_nodes_old_only, all_true_nodes_old_only)    
                with open(output_filepath_old_only, "wb") as f:
                    pickle.dump(data_to_save_old_only, f, protocol=5)  
                    
                end_time = time.time()
                print(f"Total Time taken: {end_time - start_time} seconds")
                times['construction'] = end_time - start_time
                print(times)
                
                process = psutil.Process(os.getpid())
                ram_mb = process.memory_info().rss / (1024 ** 2)
                gpu_stats = ""
                if torch.cuda.is_available():
                    curr_alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    peak_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    gpu_stats = f" | GPU Allocated: {curr_alloc:.2f}MB | GPU Peak: {peak_alloc:.2f}MB"
                
                print(f"{encoder_config['dataset']} TopoGED CONSTRUCTION TIME: {times['construction']:.2f}s | RAM: {ram_mb:.2f}MB{gpu_stats}")
                print(f"Total Times: {times}")
                
if __name__ == '__main__':
    runner = Runner()
    runner.run()

# To run the script
# python GraphGeneration/scripts/topoGED_end_to_end.py 

"""
[ro214340@evuser1 Topological-Temporal-GFM]$ squeue -u 'ro214340'
    JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
585371   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585372   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585373   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585374   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585375   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585376   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585370   highgpu TopoGED_ ro214340 PD       0:00      1 (Resources)
585377   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585378   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585379   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585381   highgpu TopoGED_ ro214340 PD       0:00      1 (Priority)
585369   highgpu TopoGED_ ro214340  R      15:24      1 evc104
585368   highgpu TopoGED_ ro214340  R      15:29      1 evc103
585367   highgpu TopoGED_ ro214340  R      15:43      1 evc103
"""