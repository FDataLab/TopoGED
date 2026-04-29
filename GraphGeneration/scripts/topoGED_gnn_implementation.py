import math
import numpy as np 
import networkx as nx
import random
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
import os
import sys
import yaml
import pickle 
import time
import copy
from torchmetrics.functional import auroc
#import line_profiler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.visualizer import Visualizer
from GraphGeneration.utils.Evaluator import Evaluator
from load_data import load_data, generate_training_data_cached, generate_validation_data_cached, generate_negative_edges
from GraphGeneration.utils.sampling_edges_utils import predict_edges
from GraphGeneration.utils.casting_type import to_tensor
from GraphGeneration.utils.graph_construction_utils import compute_reappearance_probabilities, generate_tgcn_node_features, get_node_features, update_degrees, generate_gnn_node_embeddings
from create_sub_graphs import create_nn_graph, create_on_graph

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
from nn.custom_model import Decoder

import warnings
from sklearn.exceptions import UndefinedMetricWarning
from GraphGeneration.utils.EdgeDataset import EdgeDataset
from GraphGeneration.utils.ablation_utils import ablationSetup

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
  

# Load YAML config
with open("GraphGeneration/encoder.yaml", "r") as file:
    encoder_config = yaml.safe_load(file)

# Set seeds
random.seed(encoder_config["seed"])
np.random.seed(encoder_config["seed"]) 

# MOVE IF WORKS
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, dropout=0.0):
        super(SimpleMLP, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:  # only add ReLU + dropout for hidden layers
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class Runner(object):
    def __init__(self):      
        self.seed = encoder_config["seed"]
        self.use_ma = encoder_config["use_moving_average"]  # Whether to use moving average for node2vec embeddings or not
        self.model_type = encoder_config["encoder_model"]["nodeEmbeddingType"]
        self.feature_type = encoder_config["encoder_model"]["other_models"]["feature_type"]  # Will be useful later
        
        # Set up Evaluator
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()
        self.device = device
        
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
        self.saved_input = os.path.abspath(f'data/input/cached/{encoder_config["dataset"]}/saved_data_gnn_{self.model_type}_{self.feature_type}_lr{encoder_config["training"]["lr"]}_{days_back_val}back_learnedparams')
        self.saved_samples = os.path.join(self.saved_input, 'saved_samples.pkl')
        self.common_suffix = f'topoGED_embedding{encoder_config["encoder_model"]["addOnFeature"]}_mlpEncoding{encoder_config["decoder_model"]["encode_links"]}_embeddingType{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.feature_type}_lr{encoder_config["training"]["lr"]}_{days_back_val}back_learnedparams_predvals{encoder_config["use_predicted_vals"]}'
        self.edge_eval_dir = f'GraphGeneration/output/results/edges_evaluation/{encoder_config["dataset"]}/{self.common_suffix}'
        self.structure_dir = f'GraphGeneration/output/results/structure/{encoder_config["dataset"]}/{self.common_suffix}'
        self.kernel_dir = f'GraphGeneration/output/results/kernel/{encoder_config["dataset"]}/{self.common_suffix}'
        self.topER_dir = f'GraphGeneration/output/results/topER/{encoder_config["dataset"]}/{self.common_suffix}'
        self.saved_graph_dir = f'data/output/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}'
        self.training_plots_path = f'GraphGeneration/output/results/training_plots/{encoder_config["dataset"]}/{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.feature_type}_lr{encoder_config["training"]["lr"]}_{days_back_val}back_learnedparams'

        
        save_dir = os.path.join(self.file_visualization_path, encoder_config["dataset"], encoder_config["encoder_model"]["nodeEmbeddingType"])
        os.makedirs(save_dir, exist_ok=True)
        
        # Current target snapshot we want to predict
        self.starting_graph = encoder_config["starting_graph_idx"]
        self.current_target_snapshot = self.starting_graph
        
        # All the edge types
        self.all_edge_types = ['o-o-bank', 'n-n', 'o-n', 'o-o-nobank']
        self.best_validation_model_auc = 0
        
        # Load all the snapshot true data 
        days_back_val = 'all' 
        print('[INFO] USING ALL BACK FOR PROBABILITIES AS A TEST SINCE IM PRETTY SURE THAT ACTUALLY MAKES MORE SENSE')
        self.probabilities, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(encoder_config["dataset"], encoder_config["encoder_model"]["addOnFeature"], 
                                                                                                     encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"], days_back_val, encoder_config["use_predicted_vals"], encoder_config["num_toper_buckets"])
        
        # Modify the graph ids to 1,2,3,...
        self.target_graphs, _ = modifyGraphIds(self.target_graphs, self.thresholds, 10000)  # Fixed

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

        print(len(self.test_graphs)) # 28 for college
        
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
        #probs = compute_reappearance_probabilities(graphs=prev_graphs, days_back=self.days_back, decay_factor=0.25, alpha=5, beta=5)
        probs = compute_reappearance_probabilities(graphs=prev_graphs, days_back=self.days_back, decay_factor=encoder_config["sampling"]["decay_factor"], alpha=encoder_config["sampling"]["alpha"], beta=encoder_config["sampling"]["beta"])
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

    def run_validation(self, batch_size, epoch, samples, snapshot_num):
        criterion = nn.BCELoss()
        results = {et: {'loss': [], 'auc': []} for et in self.all_edge_types}
        
        self.link_prediction_decoder.eval()
        self.encoder_model.eval()

        n_graphs = len(samples[next(iter(samples))]['u'])

        # --- VECTORIZED VALIDATION PASS ---
        with torch.no_grad():
            for i in range(n_graphs):
                feat_idx = i + snapshot_num
                features = self.node_features.weight if self.feature_type == 'learnable' else \
                        (self.node_features[feat_idx] if self.feature_type == 'node2vec' else self.node_features)

                # 1. Generate node embeddings for the current snapshot
                snapshot_embeddings = generate_gnn_node_embeddings(
                    self.encoder_model, self.model_type, features,
                    self.target_graphs[:feat_idx], self.days_back, 
                    embedding_dim=self.embedding_dim,
                    curr_nodes=self.target_graphs[feat_idx][-1].nodes(),
                    device=self.device
                )

                for et in self.all_edge_types:
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

        # --- AGGREGATE RESULTS ---
        avg_results = {et: {
            'loss': np.nanmean(results[et]['loss']) if results[et]['loss'] else 0.0,
            'auc': np.nanmean(results[et]['auc']) if results[et]['auc'] else 0.5
        } for et in self.all_edge_types}
        
        print(f"\n--- Validation Epoch {epoch+1} ---")
        for et, metrics in avg_results.items():
            print(f"[{et}] Loss: {metrics['loss']:.4f} | AUC: {metrics['auc']:.4f}")
            
        return avg_results
        
        
    def train_multi_head(self, training_samples, val_samples, test_samples):
        lr = encoder_config["training"]["lr"]
        batch_size = encoder_config["training"]["batch_size"]
        epochs = encoder_config["training"]["epochs"]
        
        params = list(self.link_prediction_decoder.parameters()) + list(self.encoder_model.parameters())
        if self.feature_type == 'learnable': 
            params += list(self.node_features.parameters())
        
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        n_graphs = len(training_samples[next(iter(training_samples))]['u'])

        best_val_auc = 0.0
        patience, counter = 10, 0
        best_state = {"encoder": None, "decoder": None}


        for epoch in range(epochs):
            self.encoder_model.train()
            self.link_prediction_decoder.train()
            
            # Tracking metrics for training per edge type
            train_results = {et: {'loss': [], 'auc': []} for et in self.all_edge_types}
            
            for g in range(n_graphs):
                #feat_idx = g + self.starting_graph
                feat_idx = g
                features = self.node_features.weight if self.feature_type == 'learnable' else \
                           (self.node_features[g] if self.feature_type == 'node2vec' else self.node_features)

                embeddings = generate_gnn_node_embeddings(
                    self.encoder_model, self.model_type, features,
                    self.target_graphs[:feat_idx], self.days_back, 
                    embedding_dim=self.embedding_dim,
                    curr_nodes=self.target_graphs[feat_idx][-1].nodes(),
                    device=self.device
                )

                optimizer.zero_grad()
                snapshot_loss = 0.0
                has_active_data = False

                for et in self.all_edge_types:
                    u_t = training_samples[et]['u_t'][g]
                    v_t = training_samples[et]['v_t'][g]
                    y_t = training_samples[et]['y_t'][g].view(-1)

                    if u_t.numel() == 0: continue
                    has_active_data = True
                    
                    # --- STEP 2: VECTORIZED FORWARD PASS ---
                    # Gather all embeddings for this edge type in one go
                    # This is the "VGAE" logic: dot-product or decoder pass on the whole tensor
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
                    except:
                        train_results[et]['auc'].append(0.5)

                if has_active_data and isinstance(snapshot_loss, torch.Tensor):
                    snapshot_loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()

            # Display Training metrics per edge type
            print(f"\n--- Training Epoch {epoch+1} ---")
            for et in self.all_edge_types:
                avg_t_loss = np.nanmean(train_results[et]['loss']) if train_results[et]['loss'] else 0.0
                avg_t_auc = np.nanmean(train_results[et]['auc']) if train_results[et]['auc'] else 0.5
                print(f"[{et}] Loss: {avg_t_loss:.4f} | AUC: {avg_t_auc:.4f}")

            # Validation and Early Stopping
            val_res = self.run_validation(batch_size, epoch, val_samples, self.train_end)
            avg_val_auc = np.mean([val_res[et]['auc'] for et in self.all_edge_types])
            
            print(f"\n>> Epoch {epoch+1:02d} Summary | Avg Val AUC: {avg_val_auc:.4f}")

            if avg_val_auc > best_val_auc:
                best_val_auc = avg_val_auc
                counter = 0
                best_state["encoder"] = copy.deepcopy(self.encoder_model.state_dict())
                best_state["decoder"] = copy.deepcopy(self.link_prediction_decoder.state_dict())
            else:
                counter += 1
                if epoch >= 20 and counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if best_state["encoder"] is not None:
            self.encoder_model.load_state_dict(best_state["encoder"])
            self.link_prediction_decoder.load_state_dict(best_state["decoder"])
            print(f"\nRestored best model with Avg Val AUC: {best_val_auc:.4f}")
            
    
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
        prev_graphs = [graph[-1] for graph in self.target_graphs[max(current_target_snapshot - self.days_back, 0) : current_target_snapshot]]


        # Add graphs until there are enough nodes
        available_nodes = set().union(*[g.nodes() for g in prev_graphs])
        curr_back = max(current_target_snapshot - self.days_back, 0) - 1

        while len(available_nodes) < self.current_target_count_old_nodes and curr_back >= 0:
            new_graph_to_add = self.target_graphs[curr_back][-1]
            prev_graphs.insert(0, new_graph_to_add)
            available_nodes.update(new_graph_to_add.nodes()) # Update in-place is O(N) of new graph only
            curr_back -= 1

        old_nodes = self.sample_old_nodes(prev_graphs)  # Get the current old nodes we expect to see

           

        # Create new node IDs
        # Get the new node id
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
            # Fix: Ensure indexing is safe and on the correct device
            features = self.node_features[current_target_snapshot] if self.feature_type == 'node2vec' else self.node_features

        # 2. Generate Embeddings (Detached for inference)
        # We wrap this in no_grad to ensure we don't build a graph in memory during construction
        with torch.no_grad():
            curr_embeddings = generate_gnn_node_embeddings(
                self.encoder_model, 
                self.model_type, 
                features, 
                self.target_graphs[:current_target_snapshot], 
                self.days_back, 
                embedding_dim=self.embedding_dim, 
                curr_nodes=all_nodes, 
                device=self.device
            )

        # Assign maximum degrees / node features
        constructing_graph = get_node_features(
            constructing_graph, prev_graphs, self.thresholds, 
            current_target_graph_description, old_nodes, new_nodes
        )  

        # 3. SAMPLE EDGES STEP
        # Ensure we use the correct head for each type
        for flag in ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']:
            # Make sure predict_edges handles the 'flag' string to select the MLP head
            sampled_edges = predict_edges(
                constructing_graph, 
                edge_type=flag, 
                node_types=node_types, 
                edgebank=edgebank, 
                link_prediction_decoder=self.link_prediction_decoder,
                old_node_embeddings=curr_embeddings, 
                top_k=self.current_target_count[flag], 
                graph_num=current_target_snapshot, 
                device=self.device # Use self.device consistently
            )

            constructing_graph.add_edges_from(sampled_edges)


        # ======== START GRAPH CONSTRUCTION ========
        G = nx.DiGraph() if self.is_directed else nx.Graph()
        used_edges = set()

        filtration_graphs = []
        for i, threshold in enumerate(self.thresholds[0: len(self.thresholds) - 1]):
            current_nodes = [node for node, degree in constructing_graph.degree() if degree <= threshold]
            subgraph = constructing_graph.subgraph(current_nodes).copy()
            filtration_graphs.append(subgraph)

        filtration_graphs.append(constructing_graph.copy())  # The last graph is the full graph, we add it at the end regardless
        return filtration_graphs, node_types
        
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
        self.encoder_model_path = os.path.join(self.saved_input, rf'saved_models/embedder_{self.seed}')
        self.decoder_model_path = os.path.join(self.saved_input, rf"saved_models/decoder_MLP_{self.seed}")

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
       
       
               
        end_time = time.time()
        print(f'Training Time: {end_time - start_time}')
        times = {'train': end_time - start_time}
        start_time = time.time()
       
        if encoder_config["ablation"]:
            print('PERFORMING ABLATION STUDY')
            # Save these since we will modify them as we go
            base_toper = self.graph_descriptions.copy()
            base_probs = self.probabilities.copy()
            for ablation_mode in [0, 1, 2, 3, 4, 5, 6]:
                if ablation_mode > 0:
                    # Change it since that makes more sense now
                    self.saved_graph_dir = f'data/output/ablation/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}_ablation{ablation_mode}'
                    os.makedirs(self.saved_graph_dir, exist_ok=True)
                self.graph_descriptions, self.probabilities = ablationSetup(base_toper, base_probs, setting=ablation_mode)
            
                output_filepath = os.path.join(self.saved_graph_dir, f"{encoder_config["encoder_model"]["nodeEmbeddingType"]}_constructed_graphs_{encoder_config["dataset"]}.pkl")

            
                # Old graphs that we know up to now
                self.old_graphs = [self.target_graphs[x][-1] for x in range(self.starting_graph)]
                
                all_node_ids = [node for graphs in self.old_graphs for node in graphs.nodes()]
                
                self.max_node_id = max(all_node_ids) + 1 if all_node_ids else 0

                all_built_graphs = []
                all_target_graphs = []
                all_pred_nodes = []
                all_true_nodes = []
                
                self.H = None
                
                # To predict snapshot i, we use snapshot 0,...,i-1 to train
                for i in range(self.starting_graph, len(self.probabilities)): 
                    print("INFO: >>> Temporal Graph Construction <<<")
                    print("INFO: Predict snapshot: ", i)
                    print("======================================")

                    self.current_target_snapshot = i
                    
                    # Get all old nodes in our context window
                    self.current_target_old_nodes = set().union(*[g.nodes() for g in self.old_graphs[0: i]])
                    
                    current_target_graph_description = self.graph_descriptions[self.current_target_snapshot]
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
                        print(f'WARNING: THE NUMBER OF EDGES FROM PROBABILITIES IS WRONG: {sum(self.current_target_count.values())} != {E_total}')
                    
                    
                    # Build the filtration sequence using the current parameters
                    filtration_sequence, node_types = self.build_accumulating_filtration_sequence_with_edgebank(current_target_snapshot=i)
                    
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
                    
                    old_nodes_days = set().union(*[g.nodes() for g in self.old_graphs[max(i - self.days_back, 0): i]])   # Old nodes of days_back days before
                
                end_time = time.time()
                
                output_filepath = os.path.join(self.saved_graph_dir, f"{encoder_config["encoder_model"]["nodeEmbeddingType"]}_constructed_graphs_{encoder_config["dataset"]}.pkl")
                os.makedirs(self.saved_graph_dir, exist_ok=True)

                data_to_save = (all_built_graphs, all_target_graphs, all_pred_nodes, all_true_nodes)

                print("\n======================================")
                print(f"INFO: Saving {len(all_built_graphs)} pairs of graphs to {output_filepath}")
                print("======================================")

                with open(output_filepath, "wb") as f:
                    pickle.dump(data_to_save, f, protocol=5) 
                    
                
                output_filepath_old_only = os.path.join(self.saved_graph_dir, f"{encoder_config["encoder_model"]["nodeEmbeddingType"]}_constructed_graphs_{encoder_config["dataset"]}_old_only.pkl")
                
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
                    
                construction_time = time.time() - start_time
                times['construction'] = construction_time
                print(construction_time)
                print(times)
                    
            
if __name__ == '__main__':
    runner = Runner()
    runner.run()

# To run the script
# python GraphGeneration/scripts/topoGED_end_to_end.py 