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
        self.seed = encoder_config["seed"]
        self.use_ma = encoder_config["use_moving_average"]  # Whether to use moving average for node2vec embeddings or not
        self.model_type = encoder_config["encoder_model"]["nodeEmbeddingType"]
        self.feature_type = encoder_config["encoder_model"]["other_models"]["feature_type"]  # Will be useful later
        self.predict_nodes = False  # Change to use encoder_config if it works
        
        if self.predict_nodes:
            self.node_predictor_path = 'GraphGeneration/output/results/old_node_optimization/best_GATmlp_model.pt'
            # Need to load variables from a path
            input_dim = 64
            output_dim = 1
            hidden_2 = 128
            num_layer = 3 
            combo = ['MLP']
            dropout = 0
            self.node_predictor = SimpleMLP(input_dim=64, hidden_dim=hidden_2, output_dim=1, num_layers=num_layer, dropout=0.2)
            self.node_predictor.load_state_dict(torch.load(self.node_predictor_path, map_location=device))            
            self.node_predictor.to(device)
            self.node_predictor.eval()
        
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
        self.saved_input = os.path.abspath(f'data/input/cached/{encoder_config["dataset"]}/saved_data_htgn_lr{encoder_config["training"]["lr"]}_{days_back_val}back_learnedparams')
        self.saved_samples = os.path.join(self.saved_input, 'saved_samples.pkl')
        self.common_suffix = f'topoGED_embedding{encoder_config["encoder_model"]["addOnFeature"]}_htgn_predictednodes{self.predict_nodes}_{days_back_val}back_learnedparams'
        self.edge_eval_dir = f'GraphGeneration/output/results/edges_evaluation/{encoder_config["dataset"]}/{self.common_suffix}'
        self.structure_dir = f'GraphGeneration/output/results/structure/{encoder_config["dataset"]}/{self.common_suffix}'
        self.kernel_dir = f'GraphGeneration/output/results/kernel/{encoder_config["dataset"]}/{self.common_suffix}'
        self.topER_dir = f'GraphGeneration/output/results/topER/{encoder_config["dataset"]}/{self.common_suffix}'
        self.saved_graph_dir = f'data/output/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}'
        self.training_plots_path = f'GraphGeneration/output/results/training_plots/{encoder_config["dataset"]}/htgn_lr{encoder_config["training"]["lr"]}_predictednodes{self.predict_nodes}_{days_back_val}back_learnedparams'

        
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
                                                                                                     encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"], days_back_val, encoder_config["use_predicted_vals"])
        
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

        self.new_node_id = 0  # The ID we will assign new node (incremented as we add nodes)


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
    
    
    # ======================= TRAIN MODEL =======================
    def create_samples(self, graphs, is_directed=False):
        snapshot_samples = []
        for i, graph in enumerate(graphs):
            if i < self.starting_graph: continue
            
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
                y_true = torch.tensor(snapshot['y'], dtype=torch.float32, device=self.device).view(-1)

                z_u = torch.stack([embeddings_dict[u] for u in u_ids])
                z_v = torch.stack([embeddings_dict[v] for v in v_ids])

                sq_dist = self.htgn_model.manifold.sqdist(z_u, z_v, c=self.htgn_model.c[2])
                probs = self.link_prediction_decoder(sq_dist).view(-1)

                loss = criterion(probs, y_true)
                total_loss += loss.item() * len(y_true)
                total_count += len(y_true)
                all_preds.append(probs.cpu())
                all_y.append(y_true.cpu())

        # FIX: Restore hidden state for next training epoch
        self.htgn_model.hiddens = original_hiddens
        
        avg_loss = total_loss / total_count if total_count > 0 else 0
        auc = roc_auc_score(torch.cat(all_y).numpy().ravel(), torch.cat(all_preds).numpy().ravel())
        return avg_loss, auc

    def train_htgn(self, training_snapshots, val_snapshots):
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
                    y_true = torch.tensor(snapshot['y'], dtype=torch.float32, device=self.device).view(-1)

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
                snapshot_aucs.append(roc_auc_score(y_true.cpu().numpy().ravel(), probs.detach().cpu().numpy().ravel()))

            avg_val_loss, avg_val_auc = self.run_htgn_validation(val_snapshots)
            train_losses.append(epoch_loss / len(training_snapshots))
            train_aucs.append(np.mean(snapshot_aucs))
            val_losses.append(avg_val_loss)
            val_aucs.append(avg_val_auc)

            print(f"Epoch {epoch:02d} | Train Loss: {train_losses[-1]:.4f} | Val AUC: {avg_val_auc:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_state = {
                    "htgn": {k: v.cpu().clone() for k, v in self.htgn_model.state_dict().items()},
                    "decoder": {k: v.cpu().clone() for k, v in self.link_prediction_decoder.state_dict().items()}
                }
            else:
                counter += 1
                if counter >= patience: break

        if best_state["htgn"]:
            self.htgn_model.load_state_dict(best_state["htgn"])
            self.link_prediction_decoder.load_state_dict(best_state["decoder"])

        os.makedirs(self.training_plots_path, exist_ok=True)
        self.visualizer.display_loss(train_losses, val_losses, len(train_losses), os.path.join(self.training_plots_path, 'loss.png'))
        return self.link_prediction_decoder
            
    
    def train_models(self):
        if os.path.exists(self.saved_samples):
            with open(self.saved_samples, "rb") as f:
                snapshot_samples = pickle.load(f)
        else:
            snapshot_samples = self.create_samples([g[-1] for g in self.target_graphs], self.is_directed)
            os.makedirs(os.path.dirname(self.saved_samples), exist_ok=True)
            with open(self.saved_samples, "wb") as f:
                pickle.dump(snapshot_samples, f)

        training = snapshot_samples[:self.train_end]
        validation = snapshot_samples[self.train_end:self.val_end]
        test = snapshot_samples[self.val_end:]
        self.train_htgn(training, validation)
        
            
    # ======================= BUILD GRAPH =======================
    def build_accumulating_filtration_sequence_with_edgebank(self, current_target_snapshot):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # 1. Context Prep
        edgebank = self.all_edgebanks[current_target_snapshot] 
        current_target_graph_description = self.graph_descriptions[current_target_snapshot]
        prev_graphs = [graph[-1] for graph in self.target_graphs[max(current_target_snapshot - self.days_back, 0) : current_target_snapshot]]
        old_nodes_days = set().union(*[g.nodes() for g in prev_graphs])

        # Target Counts
        V_total = int(current_target_graph_description[-1][0])
        E_total = int(current_target_graph_description[-1][1])

        # 2. Node Selection (Prediction of which nodes appear)
        if self.predict_nodes:
            # Use the previous hidden states window to find the most likely 'active' old nodes
            # Note: For HTGN, we use the learnable feat projected to the manifold
            self.htgn_model.eval()
            with torch.no_grad():
                # We use the internal embeddings to rank old nodes
                # Since x=None uses self.feat, we get global embeddings
                _, x_hyp = self.htgn_model(
                    edge_index=torch.zeros((2, 0), dtype=torch.long).to(self.device), # Dummy index
                    node_id_list=list(old_nodes_days),
                    node_id_map={n: i for i, n in enumerate(old_nodes_days)},
                    x=None
                )
                # Map back to Euclidean to pass through your node_predictor MLP
                X_tan = self.htgn_model.toTangentX(x_hyp, self.htgn_model.c[2])
                probs = self.node_predictor(X_tan).squeeze(-1).sigmoid()
                topk_probs, topk_indices = torch.topk(probs, self.current_target_count_old_nodes)
                old_nodes = set([list(old_nodes_days)[i] for i in topk_indices.cpu().tolist()])
        else:
            # Add graphs until there are enough nodes
            available_nodes = set().union(*[g.nodes() for g in prev_graphs])
            curr_back = self.days_back - 1
            while(len(available_nodes) < self.current_target_count_old_nodes):
                prev_graphs.insert(0, (self.target_graphs[curr_back][-1]))
                available_nodes = set().union(*[g.nodes() for g in prev_graphs])
                curr_back -= 1
            old_nodes = self.sample_old_nodes(prev_graphs)
            
        # Get the new node id
        tmp_graphs = [graph[-1] for graph in self.target_graphs[0 : current_target_snapshot]]  # Get the next new node id (the next available number)
        self.new_node_id = max([node for graphs in tmp_graphs for node in graphs.nodes()]) + 1 
        new_nodes = np.arange(self.new_node_id, self.new_node_id + self.current_target_count_new_nodes)
        
        all_nodes_list = list(old_nodes) + list(new_nodes)

        # 3. HTGN Inference (Generate current hyperbolic coordinates)
        self.htgn_model.eval()
        with torch.no_grad():
            # Use a dummy edge_index or the edges from the previous snapshot to 'prime' the HGCN
            prev_snapshot_edges = torch.tensor(list(prev_graphs[-1].edges()), dtype=torch.long).t().contiguous().to(self.device)
            
            node_id_map = {node_id: i for i, node_id in enumerate(all_nodes_list)}
            embeddings_dict, x_hyp = self.htgn_model(
                edge_index=prev_snapshot_edges, 
                node_id_list=all_nodes_list, 
                node_id_map=node_id_map, 
                x=None
            )
            
            # Update the temporal memory window with this inference result
            self.htgn_model.hiddens.pop(0)
            self.htgn_model.hiddens.append(x_hyp.detach())

        # 4. Hyperbolic Edge Construction (All-Pairs Distance)
        # We calculate the distance between all nodes to find the most likely links
        z = torch.stack([embeddings_dict[node] for node in all_nodes_list])
        
        # Calculate pairwise distances in the Poincare Ball
        # dists size: (N, N)
        dists = self.htgn_model.manifold.sqdist(z.unsqueeze(1), z.unsqueeze(0), c=self.htgn_model.c[2])
        
        # Convert distances to probabilities using the Fermi-Dirac decoder
        probs_matrix = self.link_prediction_decoder(dists)
        
        # Remove self-loops (set diagonal to 0)
        probs_matrix.fill_diagonal_(0)
        
        # 5. Select Top E_total edges
        # Flatten and get top indices
        flat_probs = probs_matrix.view(-1)
        top_probs, top_indices = torch.topk(flat_probs, k=min(E_total, flat_probs.size(0)))
        
        # Convert indices back to node pairs
        rows = top_indices // len(all_nodes_list)
        cols = top_indices % len(all_nodes_list)
        
        edge_list = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            edge_list.append((all_nodes_list[r], all_nodes_list[c]))

        # 6. Build the Filtration sequence for evaluation
        constructing_graph = nx.DiGraph() if self.is_directed else nx.Graph()
        constructing_graph.add_nodes_from(all_nodes_list)
        constructing_graph.add_edges_from(edge_list)
        
        # Set node features (degrees, etc) for evaluator compatibility
        constructing_graph = get_node_features(constructing_graph, prev_graphs, self.thresholds, 
                                              current_target_graph_description, old_nodes, new_nodes)
        
        filtration_graphs = []
        for i, threshold in enumerate(self.thresholds[0: len(self.thresholds) - 1]):
            current_nodes = [node for node, degree in constructing_graph.degree() if degree <= threshold]
            subgraph = constructing_graph.subgraph(current_nodes).copy()
            filtration_graphs.append(subgraph)

        filtration_graphs.append(constructing_graph.copy())

        return filtration_graphs, {"old_nodes": old_nodes, "new_nodes": new_nodes}
        
        
    def run(self):        
        """
        Our main runner function
        
        Params:
            None
            
        Returns: 
            None
        """     
        print("INFO: Dataset: {}".format(encoder_config["dataset"]))
        self.decoder_model_path = os.path.join(self.saved_input, rf"saved_models/decoder_MLP_{self.seed}")

        if os.path.exists(self.decoder_model_path):
            self.link_prediction_decoder.load_state_dict(torch.load(self.decoder_model_path, map_location=device))            
            self.link_prediction_decoder.to(device)
            self.link_prediction_decoder.eval()
            print(f"Link Prediction Decoder loaded from: {self.decoder_model_path}")
        else:
            # Train the Decoder and Encoder model
            print('Training the Link Prediction Decoder and Embedder')
            
            self.train_models()
            
            os.makedirs(os.path.dirname(self.decoder_model_path), exist_ok=True)
            torch.save(self.link_prediction_decoder.state_dict(), self.decoder_model_path)

            print("Models successfully saved.")
            print('Finished training the Link Prediction Decoder and Encoder; Start Graph Construction')
       
        # Old graphs that we know up to now
        self.old_graphs = [self.target_graphs[x][-1] for x in range(self.starting_graph)]
        
        all_node_ids = [node for graphs in self.old_graphs for node in graphs.nodes()]
        
        self.new_node_id = max(all_node_ids) + 1 if all_node_ids else 0

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
            self.current_target_count_old_nodes = int(round(self.probabilities[self.current_target_snapshot][0] * V_total))
            self.current_target_count_new_nodes = int(round(self.probabilities[self.current_target_snapshot][1] * V_total))
            self.current_target_count = {
                    edge_type: int(round(self.probabilities[self.current_target_snapshot][j + 2] * E_total))
                    for j, edge_type in enumerate(self.all_edge_types)
                }
            
            # Debugging:
            if self.current_target_count_old_nodes + self.current_target_count_new_nodes != V_total:
                print(f'WARNING: THE NUMBER OF NODES FROM PROBABILITIES IS WRONG: {self.current_target_count_old_nodes + self.current_target_count_new_nodes} != {V_total}')
            if sum(self.current_target_count.values()) != E_total:
                print(f'WARNING: THE NUMBER OF NODES FROM PROBABILITIES IS WRONG: {sum(self.current_target_count.values())} != {E_total}')
            
            
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
        
        output_filepath = os.path.join(self.saved_graph_dir, f"{encoder_config["encoder_model"]["nodeEmbeddingType"]}_constructed_graphs_{encoder_config["dataset"]}.pkl")
        os.makedirs(self.saved_graph_dir, exist_ok=True)

        data_to_save = (all_built_graphs, all_target_graphs, all_pred_nodes, all_true_nodes)

        print("\n======================================")
        print(f"INFO: Saving {len(all_built_graphs)} pairs of graphs to {output_filepath}")
        print("======================================")

        with open(output_filepath, "wb") as f:
            pickle.dump(data_to_save, f) 
            
            
if __name__ == '__main__':
    runner = Runner()
    runner.run()

# To run the script
# python GraphGeneration/scripts/topoGED_end_to_end.py 