import math
import time
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

from GraphGeneration.models.temporal_gnn.script.models.HTGN import HTGN
from GraphGeneration.models.temporal_gnn.script.hgcn.layers.layers import FermiDiracDecoder
import geoopt

class Runner(object):
    def __init__(self):      
        self.config = encoder_config
        self.seed = encoder_config["seed"]
        self.use_ma = encoder_config["use_moving_average"]  # Whether to use moving average for node2vec embeddings or not
        self.model_type = encoder_config["encoder_model"]["nodeEmbeddingType"]
        self.feature_type = encoder_config["encoder_model"]["other_models"]["feature_type"]  # Will be useful later
        
        # Set up Evaluator
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
        self.saved_input = os.path.abspath(f'data/input/cached/{encoder_config["dataset"]}/saved_data_htgn_lr{encoder_config["training"]["lr"]}')
        self.saved_samples = os.path.join(self.saved_input, 'saved_samples.pkl')
        self.common_suffix = f'topoGED_embedding{encoder_config["encoder_model"]["addOnFeature"]}_htgn'
        self.edge_eval_dir = f'GraphGeneration/output/results/edges_evaluation/{encoder_config["dataset"]}/{self.common_suffix}'
        self.structure_dir = f'GraphGeneration/output/results/structure/{encoder_config["dataset"]}/{self.common_suffix}'
        self.kernel_dir = f'GraphGeneration/output/results/kernel/{encoder_config["dataset"]}/{self.common_suffix}'
        self.topER_dir = f'GraphGeneration/output/results/topER/{encoder_config["dataset"]}/{self.common_suffix}'
        self.training_plots_path = f'GraphGeneration/output/results/training_plots/{encoder_config["dataset"]}/htgn_lr{encoder_config["training"]["lr"]}'

        
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
        self.probabilities, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(encoder_config["dataset"], encoder_config["encoder_model"]["addOnFeature"], 
                                                                                                     encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"], days_back_val, use_predicted=False)  # Doesn't actually matter here since construction doesn't use probabilities or toper
        
        # Modify the graph ids to 1,2,3,...
        self.target_graphs, _ = modifyGraphIds(self.target_graphs, self.thresholds, 10000)  # Fixed

        all_unique_nodes = set(node for graphs in self.target_graphs for node in graphs[-1].nodes())
        self.num_nodes = max(all_unique_nodes) + 500  # Just to add some padding
        num_nodes = self.num_nodes

        class HTGNArgs:
            def __init__(self):
                self.device = device
                self.num_nodes = num_nodes
                self.nfeat = encoder_config["encoder_model"]["other_models"]["feature_dim"]
                self.nhid = encoder_config["encoder_model"]["hidden_dim"]
                self.nout = encoder_config["encoder_model"]["other_models"]["embedding_dim"]
                self.dropout = 0.2            # Important!
                self.dropout1 = 0.2           # BaseModel uses this
                self.dropout2 = 0.2           # BaseModel uses this
                self.curvature = 1.0
                self.fixed_curvature = True
                self.manifold = 'PoincareBall'
                self.aggregation = 'att'  # Using Temporal Attention
                self.heads = 4
                self.nb_window = 3        # Lookback window for Temporal Attention
                self.use_hta = 1          # 1 to enable Temporal Attention
                self.use_gru = True
                self.model = 'GRU'
        
        self.htgn_args = HTGNArgs()

        self.htgn_model = HTGN(self.htgn_args).to(self.device)
        
        # Initialize the hidden states window
        self.htgn_model.init_hiddens()
        self.link_prediction_decoder = FermiDiracDecoder(r=2.0, t=1.0).to(self.device)
        
        
        self.embedding_dim = encoder_config["encoder_model"]["other_models"]["embedding_dim"]
        
        # Build the edgebanks for construction
        self.all_edgebanks = build_edgebanks_from_start(self.target_graphs, self.days_back)        

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

        self.new_node_id = 0  # The ID we will assign new node (incremented as we add nodes)
    
    
    # ======================= TRAIN MODEL =======================
    def create_samples(self, graphs, is_directed=False):
        snapshot_samples = []
        for i, graph in enumerate(graphs):            
            nodes = list(graph.nodes())
            edges = set(graph.edges())
            current_snapshot_X, current_snapshot_y = [], []

            for u, v in edges:
                current_snapshot_X.append({'u_id': u, 'v_id': v})
                current_snapshot_y.append(1)

            num_pos = len(edges)
            num_neg, attempts = 0, 0
            while num_neg < num_pos and attempts < num_pos * 3:
                attempts += 1
                u, v = random.sample(nodes, 2)
                if (u, v) not in edges:
                    if not is_directed and (v, u) in edges: continue
                    current_snapshot_X.append({'u_id': u, 'v_id': v})
                    current_snapshot_y.append(0)
                    num_neg += 1

            edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
            snapshot_samples.append({'X': current_snapshot_X, 'y': current_snapshot_y, 'edge_index': edge_index, 'nodes': nodes})
        return snapshot_samples
    

    def run_htgn_validation(self, val_snapshots):
        self.htgn_model.eval()
        self.link_prediction_decoder.eval()
        
        # FIX: Protect hidden state from validation mutation
        original_hiddens = [h.clone() for h in self.htgn_model.hiddens]
        
        criterion = nn.BCELoss()
        all_preds, all_y = [], []
        total_loss, total_count = 0.0, 0

        with torch.no_grad():
            for snapshot in val_snapshots:
                edge_index = snapshot['edge_index'].to(self.device)
                nodes = snapshot['nodes']
                node_id_map = {node_id: i for i, node_id in enumerate(nodes)}

                embeddings_dict, x_hyp = self.htgn_model(edge_index, nodes, node_id_map, x=None)
                
                self.htgn_model.hiddens.pop(0)
                self.htgn_model.hiddens.append(x_hyp)

                u_ids = [s['u_id'] for s in snapshot['X']]
                v_ids = [s['v_id'] for s in snapshot['X']]
                y_true = torch.as_tensor(snapshot['y'], dtype=torch.float32).to(self.device).view(-1)

                z_u = torch.stack([embeddings_dict[u] for u in u_ids])
                z_v = torch.stack([embeddings_dict[v] for v in v_ids])

                sq_dist = self.htgn_model.manifold.sqdist(z_u, z_v, c=self.htgn_model.c[2])
                probs = self.link_prediction_decoder(sq_dist).view(-1)

                loss = criterion(probs, y_true)
                total_loss += loss.item() * len(y_true)
                total_count += len(y_true)
                all_preds.append(probs)
                all_y.append(y_true)
            
            full_y = torch.cat(all_y)
            full_preds = torch.cat(all_preds)
            auc = auroc(full_preds, full_y.long(), task="binary").item()

        # FIX: Restore hidden state for next training epoch
        self.htgn_model.hiddens = original_hiddens
        
        avg_loss = total_loss / total_count if total_count > 0 else 0
        auc = roc_auc_score(torch.cat(all_y).numpy().ravel(), torch.cat(all_preds).numpy().ravel())
        return avg_loss, auc

    def train_htgn(self, training_snapshots, val_snapshots):
        os.makedirs(os.path.join(self.saved_input, "saved_models"), exist_ok=True)
        self.decoder_model_path = os.path.join(self.saved_input, f"saved_models/htgn_decoder_{self.seed}.pt")
        self.htgn_model_path = os.path.join(self.saved_input, f"saved_models/htgn_encoder_{self.seed}.pt")
    
        optimizer = geoopt.optim.RiemannianAdam(
            list(self.htgn_model.parameters()) + list(self.link_prediction_decoder.parameters()),
            lr=self.config["training"]["lr"]
        )
        criterion = nn.BCELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        train_losses, train_aucs, val_losses, val_aucs = [], [], [], []
        best_val_loss, patience, counter = float("inf"), 5, 0
        best_state = {"htgn": None, "decoder": None}

        for epoch in range(self.config["training"]["epochs"]):
            print(f"--- Epoch {epoch+1}/{self.config['training']['epochs']} ---")
            self.htgn_model.train()
            self.link_prediction_decoder.train()
            self.htgn_model.init_hiddens()
            
            epoch_loss, snapshot_aucs = 0, []

            for snapshot in training_snapshots:
                optimizer.zero_grad()
                edge_index = snapshot['edge_index'].to(self.device)
                nodes = snapshot['nodes']
                node_id_map = {node_id: i for i, node_id in enumerate(nodes)}

                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    embeddings_dict, x_hyp = self.htgn_model(edge_index, nodes, node_id_map, x=None)
                    
                    u_ids = [s['u_id'] for s in snapshot['X']]
                    v_ids = [s['v_id'] for s in snapshot['X']]
                    y_true = torch.as_tensor(snapshot['y'], dtype=torch.float32).to(self.device).view(-1)

                    z_u = torch.stack([embeddings_dict[u] for u in u_ids])
                    z_v = torch.stack([embeddings_dict[v] for v in v_ids])

                    sq_dist = self.htgn_model.manifold.sqdist(z_u, z_v, c=self.htgn_model.c[2])
                    probs = self.link_prediction_decoder(sq_dist).view(-1)
                    loss = criterion(probs, y_true)

                self.htgn_model.hiddens.pop(0)
                self.htgn_model.hiddens.append(x_hyp.detach())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                auc_val = auroc(probs, y_true.long(), task="binary")
                snapshot_aucs.append(auc_val.item())

            avg_val_loss, avg_val_auc = self.run_htgn_validation(val_snapshots)
            train_losses.append(epoch_loss / len(training_snapshots))
            train_aucs.append(np.mean(snapshot_aucs))
            val_losses.append(avg_val_loss)
            val_aucs.append(avg_val_auc)

            # print(f"Epoch {epoch:02d} | Train Loss: {train_losses[-1]:.4f} | Val AUC: {avg_val_auc:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_state = {
                    "htgn": {k: v.cpu().clone() for k, v in self.htgn_model.state_dict().items()},
                    "decoder": {k: v.cpu().clone() for k, v in self.link_prediction_decoder.state_dict().items()}
                }
                # Optional: Save checkpoint to disk every time we find a new best
                torch.save(best_state["htgn"], self.htgn_model_path)
                torch.save(best_state["decoder"], self.decoder_model_path)
            else:
                counter += 1
                if counter >= patience: 
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Restore best weights to the models in memory
        if best_state["htgn"]:
            self.htgn_model.load_state_dict(best_state["htgn"])
            self.link_prediction_decoder.load_state_dict(best_state["decoder"])

        os.makedirs(self.training_plots_path, exist_ok=True)
        self.visualizer.display_loss(train_losses, val_losses, len(train_losses), os.path.join(self.training_plots_path, 'loss.png'))
        
        print(f"Final best models saved to:\nEncoder: {self.htgn_model_path}\nDecoder: {self.decoder_model_path}")
        return self.link_prediction_decoder
            
    
    def train_models(self):
        if os.path.exists(self.saved_samples):
            with open(self.saved_samples, "rb") as f:
                snapshot_samples = pickle.load(f)
        else:
            snapshot_samples = self.create_samples([g[-1] for g in self.target_graphs], self.is_directed)
            os.makedirs(os.path.dirname(self.saved_samples), exist_ok=True)
            with open(self.saved_samples, "wb") as f:
                pickle.dump(snapshot_samples, f, protocol=5)

        training = snapshot_samples[:self.train_end]
        validation = snapshot_samples[self.train_end:self.val_end]
        test = snapshot_samples[self.val_end:]
        self.train_htgn(training, validation)
    
    
    def predict_next_edges(self, snapshot_idx, threshold=0.5):
        """
        Predicts edges for snapshot_idx based on the state of snapshot_idx - 1.
        """
        self.htgn_model.eval()
        self.link_prediction_decoder.eval()

        # Get the graph from the previous step to define 'old nodes'
        prev_graph = self.target_graphs[snapshot_idx - 1][-1]
        old_nodes = list(prev_graph.nodes())
        
        node_id_map = {node_id: idx for idx, node_id in enumerate(old_nodes)}
        edge_index = torch.tensor(list(prev_graph.edges()), dtype=torch.long).t().contiguous().to(self.device)

        with torch.no_grad():
            # This update ensures the HTGN is aware of the most recent history
            embeddings_dict, _ = self.htgn_model(edge_index, old_nodes, node_id_map, x=None)

            # --- FIX: Define Z by stacking the embeddings from the dictionary ---
            # We stack them in the order of 'old_nodes' so the indices match node_id_map
            Z = torch.stack([embeddings_dict[node_id] for node_id in old_nodes])

            # Candidate pairs: Every combination of existing nodes
            candidate_u_idx, candidate_v_idx = [], []
            num_nodes = len(old_nodes)
            for j in range(num_nodes):
                for k in range(j + 1, num_nodes):
                    candidate_u_idx.append(j)
                    candidate_v_idx.append(k)

            # Direct GPU indexing into the stacked matrix Z
            u_idx = torch.tensor(candidate_u_idx, device=self.device)
            v_idx = torch.tensor(candidate_v_idx, device=self.device)
            
            z_u = Z[u_idx]
            z_v = Z[v_idx]

            # Hyperbolic distance calculation using the HTGN manifold
            sq_dist = self.htgn_model.manifold.sqdist(z_u, z_v, c=self.htgn_model.c[2])
            
            # The FermiDiracDecoder acts as the fully connected layer to map 
            # distances to probabilities [cite: 186]
            probs = self.link_prediction_decoder(sq_dist).view(-1)

            mask = probs > threshold
            
            # Map indices back to original node IDs for the final output
            predicted_edges = [
                (old_nodes[candidate_u_idx[i]], old_nodes[candidate_v_idx[i]]) 
                for i, val in enumerate(mask) if val
            ]

        return predicted_edges

    
    def construct_predicted_graph(self, threshold):
        predicted_edges = self.predict_next_edges(snapshot_idx=self.current_target_snapshot, threshold=threshold)
        
        new_G = nx.Graph() # or nx.DiGraph()
        
        for u, v in predicted_edges:
            new_G.add_edge(u, v)
        
        # print(f"Constructed graph with {new_G.number_of_nodes()} nodes and {new_G.number_of_edges()} predicted edges.")
        return new_G
    
    
    def run(self):        
        """
        Our main runner function
        
        Params:
            None
            
        Returns: 
            None
        """     
        print("INFO: Dataset: {}".format(encoder_config["dataset"]))
        self.decoder_model_path = os.path.join(self.saved_input, f"saved_models/htgn_decoder_{self.seed}.pt")
        self.htgn_model_path = os.path.join(self.saved_input, f"saved_models/htgn_encoder_{self.seed}.pt")
        times = {}
        start_time = time.time()
        if os.path.exists(self.decoder_model_path) and os.path.exists(self.htgn_model_path):
            # Load HTGN Encoder
            self.htgn_model.load_state_dict(torch.load(self.htgn_model_path, map_location=device))
            self.htgn_model.to(device)
            self.htgn_model.eval()
            
            # Load Link Prediction Decoder
            self.link_prediction_decoder.load_state_dict(torch.load(self.decoder_model_path, map_location=device))            
            self.link_prediction_decoder.to(device)
            self.link_prediction_decoder.eval()
            
            print(f"INFO: HTGN Encoder loaded from: {self.htgn_model_path}")
            print(f"INFO: Link Prediction Decoder loaded from: {self.decoder_model_path}")
        else:
            # Train both models if any part is missing
            print('INFO: Models not found. Training the HTGN Encoder and Link Prediction Decoder...')
            
            # This will call train_htgn which now saves the best versions of both
            self.train_models()
            
            print("INFO: Models successfully trained and saved.")

        times['train'] = time.time() - start_time
        
        import psutil
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 ** 2)
        gpu_stats = ""
        if torch.cuda.is_available():
            curr_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
            peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            gpu_stats = f" | GPU Allocated: {curr_alloc:.2f}MB | GPU Peak: {peak_alloc:.2f}MB"
        
        print(f"{encoder_config["dataset"]} HTGN TRAIN TIME: {times['train']:.2f}s | RAM: {ram_mb:.2f}MB{gpu_stats}")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        
        for threshold in [0.7, 0.75]:
            # Data to save    
            all_built_graphs = []
            all_target_graphs = []
            all_pred_nodes = []
            all_true_nodes = []
            
            self.saved_graph_dir = f'data/output/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}_threshold{threshold}'
            output_filepath = os.path.join(self.saved_graph_dir, f"{encoder_config['encoder_model']['nodeEmbeddingType']}_constructed_graphs_{encoder_config['dataset']}.pkl")
            
            self.H = None
            start_time = time.time()
            # To predict snapshot i, we use snapshot 0,...,i-1 to train
            for i in range(self.starting_graph, len(self.probabilities)): 
                # print("INFO: >>> Temporal Graph Construction <<<")
                # print("INFO: Predict snapshot: ", i)
                # print("======================================")

                self.current_target_snapshot = i
                            
                self.htgn_model.init_hiddens()
                window_size = self.htgn_args.nb_window
                
                # Feed the model true graphs leading up to the one we want to predict
                # This ensures 'self.htgn_model.hiddens' is "primed" with real history
                start_prime = max(0, i - window_size)
                for prime_idx in range(start_prime, i):
                    prev_g = self.target_graphs[prime_idx][-1]
                    p_nodes = list(prev_g.nodes())
                    p_map = {node_id: idx for idx, node_id in enumerate(p_nodes)}
                    p_edges = torch.tensor(list(prev_g.edges()), dtype=torch.long).t().contiguous().to(self.device)
                    
                    with torch.no_grad():
                        # Forward pass updates the internal self.hiddens list
                        _, x_hyp = self.htgn_model(p_edges, p_nodes, p_map, x=None)
                        self.htgn_model.hiddens.pop(0)
                        self.htgn_model.hiddens.append(x_hyp)
                            
                # Build the filtration sequence using the current parameters
                built_graph = self.construct_predicted_graph(threshold)
                
        
                # Add the graphs to a list to save later
                target_graph = self.target_graphs[i][-1]
                all_built_graphs.append(built_graph)
                all_target_graphs.append(target_graph)
                all_pred_nodes.append({"old_nodes": built_graph.nodes(), "new_nodes": set()})  # By definition these are all old
                
                # Get the node types for the target graph
                current_nodes = target_graph.nodes()  # These are the old nodes
                all_true_nodes.append({"old_nodes": current_nodes, "new_nodes": set()})  # new nodes are by definition empty here
                
            self.saved_graph_dir = f'data/output/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}_threshold{threshold}'
            
            output_filepath = os.path.join(self.saved_graph_dir, f"{encoder_config['encoder_model']['nodeEmbeddingType']}_constructed_graphs_{encoder_config['dataset']}.pkl")
            os.makedirs(self.saved_graph_dir, exist_ok=True)

            data_to_save = (all_built_graphs, all_target_graphs, all_pred_nodes, all_true_nodes)

            # print("\n======================================")
            print(f"INFO: Saving {len(all_built_graphs)} pairs of graphs to {output_filepath}")
            # print("======================================")

            with open(output_filepath, "wb") as f:
                pickle.dump(data_to_save, f, protocol=5) 
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"INFO: Total execution time: {elapsed_time:.2f} seconds")
            ram_mb = process.memory_info().rss / (1024 ** 2)
            gpu_stats = ""
            if torch.cuda.is_available():
                curr_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                gpu_stats = f" | GPU Allocated: {curr_alloc:.2f}MB | GPU Peak: {peak_alloc:.2f}MB"
            
            print(f"CONSTRUCTION (Thr {threshold}): {elapsed_time:.2f}s | RAM: {ram_mb:.2f}MB{gpu_stats}")
            
            times[threshold] = elapsed_time
            
        print(times)
            
if __name__ == '__main__':
    runner = Runner()
    runner.run()

# To run the script
# python GraphGeneration/scripts/topoGED_end_to_end.py 