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


class Runner(object):
    def __init__(self):      
        self.seed = encoder_config["seed"]
        self.use_ma = encoder_config["use_moving_average"]  # Whether to use moving average for node2vec embeddings or not
        self.model_type = encoder_config["encoder_model"]["nodeEmbeddingType"]
        self.feature_type = encoder_config["encoder_model"]["other_models"]["feature_type"]  # Will be useful later
        self.predict_nodes = False  # Change to use encoder_config if it works
        
        
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
        self.saved_input = os.path.abspath(f'data/input/cached/{encoder_config["dataset"]}/saved_data_gnn_{self.model_type}_{self.feature_type}_lr{encoder_config["training"]["lr"]}_{days_back_val}back_existanceweighted')
        self.saved_samples = os.path.join(self.saved_input, 'saved_samples.pkl')
        self.common_suffix = f'topoGED_embedding{encoder_config["encoder_model"]["addOnFeature"]}_mlpEncoding{encoder_config["decoder_model"]["encode_links"]}_embeddingType{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.feature_type}_lr{encoder_config["training"]["lr"]}_{days_back_val}back_existanceweighted'
        self.edge_eval_dir = f'GraphGeneration/output/results/edges_evaluation/{encoder_config["dataset"]}/{self.common_suffix}'
        self.structure_dir = f'GraphGeneration/output/results/structure/{encoder_config["dataset"]}/{self.common_suffix}'
        self.structure_dir = f'GraphGeneration/output/results/structure/{encoder_config["dataset"]}/{self.common_suffix}'
        self.kernel_dir = f'GraphGeneration/output/results/kernel/{encoder_config["dataset"]}/{self.common_suffix}'
        self.topER_dir = f'GraphGeneration/output/results/topER/{encoder_config["dataset"]}/{self.common_suffix}'
        self.saved_graph_dir = f'data/output/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}'
        self.training_plots_path = f'GraphGeneration/output/results/training_plots/{encoder_config["dataset"]}/{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.feature_type}_lr{encoder_config["training"]["lr"]}_{days_back_val}back_existanceweighted'

        
        save_dir = os.path.join(self.file_visualization_path, encoder_config["dataset"], encoder_config["encoder_model"]["nodeEmbeddingType"])
        os.makedirs(save_dir, exist_ok=True)
        
        # Current target snapshot we want to predict
        self.starting_graph = encoder_config["starting_graph_idx"]
        self.current_target_snapshot = self.starting_graph
        
        
        
        if self.days_back <= 0:
            raise ValueError(f"days_back must be a positive integer (days_back > 0). Got {self.days_back}.")
        
        # All the edge types
        self.all_edge_types = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
        self.best_validation_model_auc = 0
        
        # Load all the snapshot true data 
        self.probabilities, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(encoder_config["dataset"], encoder_config["encoder_model"]["addOnFeature"], 
                                                                                                     encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"], days_back_val, encoder_config["use_predicted_vals"])
        
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
                
        self.link_prediction_decoder = setupMLP(embedding_dim=self.input_dim*2, mlpEncoding=encoder_config["decoder_model"]["encode_links"], model_type='exist')
        self.link_prediction_decoder.to(device)

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
    def run_validation(self, batch_size, epoch, samples, snapshot_num):
        """
        Validate multi-head model (edge heads + optional 'node' task) on samples.
        samples: dict with keys for each edge type and optionally 'node', each value is {'X': [per-graph], 'y':[per-graph]}
        snapshot_num: offset into target_graphs to pick the right graph snapshots for embedding generation
        Returns: avg_results dict keyed by task with {'loss', 'auc'}
        """
        use_cuda = (self.device.type == "cuda")
        criterion = nn.BCELoss(reduction="mean")

        # initialize results for every edge type + node task (if present)
        task_keys = list(samples.keys())
        results = {k: {'loss': [], 'auc': []} for k in task_keys}

        self.link_prediction_decoder.eval()
        self.encoder_model.eval()

        # number of graph snapshots to validate over: take max across tasks to be safe
        n_graphs = max((len(v['X']) for v in samples.values()), default=0)

        with torch.no_grad():
            for i in range(n_graphs):
                # compute features & embeddings once per graph (use same indexing logic as in training)
                if self.feature_type == 'learnable':
                    features = self.node_features.weight
                else:
                    # use node2vec snapshot when configured; otherwise use shared node_features
                    # note: this mirrors training indexing
                    idx = i + snapshot_num + self.starting_graph
                    features = (self.node_features[idx] if self.feature_type == 'node2vec' else self.node_features)

                embeddings = generate_gnn_node_embeddings(
                    self.encoder_model,
                    self.model_type,
                    features,
                    self.target_graphs[: i + snapshot_num + self.starting_graph],
                    self.days_back,
                    embedding_dim=self.embedding_dim,
                    curr_nodes=self.target_graphs[i + snapshot_num + self.starting_graph][-1].nodes(),
                    device=self.device
                )

                # ---------- EDGE TASKS ----------
                for edge_type in [k for k in task_keys if k != 'node']:
                    # skip if this task lacks this graph index
                    if i >= len(samples[edge_type]['X']):
                        continue

                    X_samples = samples[edge_type]['X'][i]
                    y_samples = samples[edge_type]['y'][i]
                    if not X_samples:
                        continue

                    dataset = list(zip(X_samples, y_samples))
                    total_loss_sum = 0.0
                    total_count = 0
                    all_preds, all_y = [], []

                    for b_start in range(0, len(dataset), batch_size):
                        batch = dataset[b_start:b_start+batch_size]
                        u_ids = [x['u_id'] for x, _ in batch]
                        v_ids = [x['v_id'] for x, _ in batch]
                        yb = torch.tensor([y for _, y in batch], dtype=torch.float32, device=self.device).view(-1, 1)

                        ub = torch.stack([embeddings[u] for u in u_ids]).contiguous()
                        vb = torch.stack([embeddings[v] for v in v_ids]).contiguous()

                        with torch.cuda.amp.autocast(enabled=use_cuda):
                            preds = self.link_prediction_decoder(ub, vb, edge_type=edge_type).view(-1, 1)
                            bce_loss = criterion(preds, yb)

                        b_n = len(batch)
                        total_loss_sum += float(bce_loss.item()) * b_n
                        total_count += b_n

                        all_preds.append(preds.cpu())
                        all_y.append(yb.cpu())

                    if total_count == 0:
                        continue

                    avg_loss = total_loss_sum / total_count
                    all_preds_flat = torch.cat(all_preds).numpy().flatten()
                    y_np = torch.cat(all_y).numpy().flatten()

                    try:
                        auc = roc_auc_score(y_np, all_preds_flat)
                    except ValueError:
                        auc = float('nan')

                    results[edge_type]['loss'].append(avg_loss)
                    if not np.isnan(auc):
                        results[edge_type]['auc'].append(auc)

                # ---------- NODE EXISTENCE TASK (optional) ----------
                if 'node' in task_keys and i < len(samples['node']['X']):
                    node_X = samples['node']['X'][i]
                    node_y = samples['node']['y'][i]
                    if node_X:
                        node_dataset = list(zip(node_X, node_y))
                        total_loss_sum = 0.0
                        total_count = 0
                        all_preds, all_y = [], []

                        for b_start in range(0, len(node_dataset), batch_size):
                            batch = node_dataset[b_start:b_start+batch_size]
                            node_ids = [x['node_id'] for x, _ in batch]
                            yb = torch.tensor([y for _, y in batch], dtype=torch.float32, device=self.device).view(-1, 1)

                            nb = torch.stack([embeddings[n] for n in node_ids]).contiguous()

                            with torch.cuda.amp.autocast(enabled=use_cuda):
                                preds = self.link_prediction_decoder.predict_node_exists(nb).view(-1, 1)
                                bce_loss = criterion(preds, yb)

                            b_n = len(batch)
                            total_loss_sum += float(bce_loss.item()) * b_n
                            total_count += b_n

                            all_preds.append(preds.cpu())
                            all_y.append(yb.cpu())

                        if total_count > 0:
                            avg_loss = total_loss_sum / total_count
                            all_preds_flat = torch.cat(all_preds).numpy().flatten()
                            y_np = torch.cat(all_y).numpy().flatten()

                            try:
                                auc = roc_auc_score(y_np, all_preds_flat)
                            except ValueError:
                                auc = float('nan')

                            results['node']['loss'].append(avg_loss)
                            if not np.isnan(auc):
                                results['node']['auc'].append(auc)

        # Aggregate metrics over graphs for each task
        avg_results = {
            flag: {
                'loss': np.nanmean(results[flag]['loss']) if results[flag]['loss'] else 0.0,
                'auc' : np.nanmean(results[flag]['auc'])  if results[flag]['auc']  else 0.0
            }
            for flag in task_keys
        }

        # Logging
        output_dir = os.path.join(
            self.file_visualization_path,
            encoder_config["dataset"],
            encoder_config["encoder_model"]["nodeEmbeddingType"]
        )
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"multiheadMLP_val_performance_{self.seed}.txt")

        for flag in avg_results:
            msg = f"Epoch: {epoch+1:02d} | Task: {flag} | Val Loss: {avg_results[flag]['loss']:.4f} | Val AUCROC: {avg_results[flag]['auc']:.4f}"
            print(msg, flush=True)
            with open(file_path, "a") as f:
                f.write(msg + "\n")
                f.flush()

        return avg_results
            

    def create_samples(self, graphs, days_back, all_edgebanks, is_directed=False):
        """
        MOVE THIS TO A SEPARATE FILE; CURRENTLY HERE FOR REFERENCE WHILE REDOING CODE
        """
        # Prepare the sorted samples for each edge type, both positive and negative edges 
        # We will then send them to a pkl file and use them for training the model
        # Just generate all samples then shuffling and splitting can happen later
        # Need to organize the edges, then create the graphs (4 subgraphs per graph), then encode the nodes to make samples
        sorted_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
            'o-n': {'X': [], 'y': []},
            'n-n': {'X': [], 'y': []},
            }  # A dict to sort embeddings for multiheaded MLP training
                
        sorted_nodes = {
            'X': [],
            'y': []
        }
                
        # Organize the edges
        for i, graph in enumerate(graphs):
            old_nodes_days = set().union(*[g.nodes() for g in graphs[max(i - days_back, 0): i]])   # Old nodes of days_back days before
            if i < self.starting_graph:
                continue 

            new_edges_count = {
                'o-o-bank': 0,
                'o-o-nobank': 0,
                'o-n': 0,
                'n-n': 0,
            }
            
            sorted_edges = {
                'o-o-bank': [],
                'o-o-nobank': [],
                'o-n': [],
                'n-n': [],
            }
            
            for u, v in graph.edges():
                if u in old_nodes_days and v in old_nodes_days:
                    if v in all_edgebanks[i].get(u, set()):
                        edge_type = 'o-o-bank'
                    else:
                        edge_type = 'o-o-nobank'
                elif (u in old_nodes_days and v not in old_nodes_days) or (u not in old_nodes_days and v in old_nodes_days):
                    edge_type = 'o-n'
                elif u not in old_nodes_days and v not in old_nodes_days:   
                    edge_type = 'n-n'
                else:
                    print(f'[WARNING] Unknown edge type found in create_samples()')
                    continue 

                sorted_edges[edge_type].append((u, v))  # Add the edge to sorted samples
            
            # Figure out how many edges we added for generating an equal amount of negative samples
            for edge_type in sorted_edges:
                new_edges_count[edge_type] = len(sorted_edges[edge_type])
    
            constructing_graph = nx.DiGraph() if is_directed else nx.Graph()  # We will add samples here for encoder to use
            
            # I'm not sure if i want to make the o-o-bank and o-o-nobank edges with the old 
            # We will let o-o-bank, o-o-nobank, and o-n be formed from the old node embeddings
            # Before creating n-n, we will embed the graph again
            for edge_type in ['o-o-bank', 'o-o-nobank', 'o-n']:
                sorted_samples[edge_type]['X'].append([])
                sorted_samples[edge_type]['y'].append([])
                
                # Since we don't have data for new nodes yet for edge type o-n, we will assign a vector of 0's
                for u, v in sorted_edges[edge_type]:
                    sample = {
                        'u_id': u,
                        'v_id': v,
                    }
                    sorted_samples[edge_type]['X'][-1].append(sample)
                    sorted_samples[edge_type]['y'][-1].append(1)
                
                # Now get negative samples
                negative_edges = generate_negative_edges(
                    graph,
                    num_samples=new_edges_count[edge_type],
                    edge_type=edge_type,
                    old_nodes=old_nodes_days,
                    is_directed=is_directed,
                    edgebank=all_edgebanks[i]
                )
                
                for u, v in negative_edges:
                    sample = {
                        'u_id': u,
                        'v_id': v,
                    }
                    sorted_samples[edge_type]['X'][-1].append(sample)
                    sorted_samples[edge_type]['y'][-1].append(0)
                
                constructing_graph.add_edges_from(sorted_edges[edge_type])  # For embedding to get new node information later
            
            edge_type = 'n-n'
            sorted_samples[edge_type]['X'].append([])
            sorted_samples[edge_type]['y'].append([])
            
            # Generate positive samples
            for u, v in sorted_edges[edge_type]:
                sample = {
                    'u_id': u,
                    'v_id': v,
                }
                sorted_samples[edge_type]['X'][-1].append(sample)
                sorted_samples[edge_type]['y'][-1].append(1)
            
            # Now get negative samples
            negative_edges = generate_negative_edges(
                graph,
                num_samples=new_edges_count[edge_type],
                edge_type=edge_type,
                old_nodes=old_nodes_days,
                is_directed=is_directed,
                edgebank=all_edgebanks[i]
            )
            
            for u, v in negative_edges:
                sample = {
                    'u_id': u,
                    'v_id': v,
                }
                sorted_samples[edge_type]['X'][-1].append(sample)
                sorted_samples[edge_type]['y'][-1].append(0)
                    
            # ========= NODE EXISTENCE TASK =========
            # Potential old nodes to reappear
            all_seen_nodes = set().union(*[g.nodes() for g in graphs[i - self.days_back : i]])

            # Nodes that exist today
            current_nodes = set(graph.nodes())
            positives = list(current_nodes)

            # negative nodes = sample from nodes that have existed before but are absent today
            negatives_pool = list(all_seen_nodes - current_nodes)

            # match number of positives
            k = len(positives)

            # sample k negative nodes
            negatives = random.sample(negatives_pool, k) if len(negatives_pool) >= k else negatives_pool

            sorted_nodes['X'].append([])
            sorted_nodes['y'].append([])

            for node in positives:
                sorted_nodes['X'][-1].append({'node_id': node})
                sorted_nodes['y'][-1].append(1)

            # add negatives
            for node in negatives:
                sorted_nodes['X'][-1].append({'node_id': node})
                sorted_nodes['y'][-1].append(0)
             
        return sorted_samples, sorted_nodes
    

    def train_multi_head(self, training_samples, val_samples, test_samples):
        lr = encoder_config["training"]["lr"]
        batch_size = encoder_config["training"]["batch_size"]
        epochs = encoder_config["training"]["epochs"]
        use_cuda = (self.device.type == "cuda")

        self.link_prediction_decoder.train()
        self.encoder_model.train()
        self.H = None

        # Set optimizer
        if self.feature_type != 'learnable':
            optimizer = torch.optim.Adam(
                list(self.link_prediction_decoder.parameters()) + list(self.encoder_model.parameters()),
                lr=lr
            )
        else:
            optimizer = torch.optim.Adam(
                list(self.link_prediction_decoder.parameters()) +
                list(self.encoder_model.parameters()) +
                list(self.node_features.parameters()),
                lr=lr
            )

        criterion = nn.BCELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

        task_keys = list(training_samples.keys())
        n_graphs = len(next(iter(training_samples.values()))['X'])

        lambda_edge = 1.0 
        lambda_node = 0.05

        # For storing losses and AUCs
        train_losses_all = {key: [] for key in task_keys}
        train_aucs_all = {key: [] for key in task_keys}
        val_losses_all = {key: [] for key in task_keys}
        val_aucs_all = {key: [] for key in task_keys}
        
        counter = 0
        patience = 7
        best_val_loss_sum = float("inf")
        best_state = {
            "encoder": None,
            "decoder": None
        }

        for epoch in range(epochs):
            self.link_prediction_decoder.train()
            self.encoder_model.train()
            
            epoch_losses = {k: [] for k in task_keys}
            epoch_aucs = {k: [] for k in task_keys}

            self.link_prediction_decoder.train()
            self.encoder_model.train()

            for g in range(n_graphs):
                print(f'Epoch: {epoch}; Graph #{g}')
                # Generate embeddings once per graph
                if self.feature_type == 'learnable':
                    features = self.node_features.weight
                else:
                    features = self.node_features[g] if self.feature_type == 'node2vec' else self.node_features

                embeddings = generate_gnn_node_embeddings(
                    self.encoder_model,
                    self.model_type,
                    features,
                    self.target_graphs[: g + self.starting_graph],
                    self.days_back,
                    embedding_dim=self.embedding_dim,
                    curr_nodes=self.target_graphs[g + self.starting_graph][-1].nodes(),
                    device=self.device
                )

                total_loss_graph = 0.0  # Accumulate loss for this graph

                # ---------------- EDGE TASKS ----------------
                for edge_type in [t for t in task_keys if t != 'node']:
                    Xg = training_samples[edge_type]['X'][g]
                    yg = training_samples[edge_type]['y'][g]
                    if not Xg:
                        continue

                    dataset = list(zip(Xg, yg))
                    random.shuffle(dataset)

                    for b in range(0, len(dataset), batch_size):
                        batch = dataset[b:b+batch_size]
                        u_ids = [x['u_id'] for x,_ in batch]
                        v_ids = [x['v_id'] for x,_ in batch]
                        yb = torch.tensor([y for _,y in batch], dtype=torch.float32, device=self.device).view(-1,1)

                        ub = torch.stack([embeddings[u] for u in u_ids]).contiguous()
                        vb = torch.stack([embeddings[v] for v in v_ids]).contiguous()

                        with torch.cuda.amp.autocast(enabled=use_cuda):
                            preds = self.link_prediction_decoder(ub, vb, edge_type=edge_type).view(-1,1)
                            loss = criterion(preds, yb)
                            total_loss_graph += loss * lambda_edge

                            # AUC (detached)
                            try:
                                auc = roc_auc_score(yb.cpu().numpy(), preds.detach().cpu().numpy())
                                epoch_aucs[edge_type].append(auc)
                            except:
                                epoch_aucs[edge_type].append(np.nan)

                        epoch_losses[edge_type].append(loss.item())

                # ---------------- NODE TASK ----------------
                if 'node' in task_keys:
                    Xg = training_samples['node']['X'][g]
                    yg = training_samples['node']['y'][g]

                    dataset = list(zip(Xg, yg))
                    random.shuffle(dataset)

                    for b in range(0, len(dataset), batch_size):
                        batch = dataset[b:b+batch_size]
                        node_ids = [x['node_id'] for x,_ in batch]
                        yb = torch.tensor([y for _,y in batch], dtype=torch.float32, device=self.device).view(-1,1)

                        nb = torch.stack([embeddings[n] for n in node_ids]).contiguous()

                        with torch.cuda.amp.autocast(enabled=use_cuda):
                            preds = self.link_prediction_decoder.predict_node_exists(nb).view(-1,1)
                            loss = criterion(preds, yb)
                            total_loss_graph += loss * lambda_node

                            # AUC
                            try:
                                auc = roc_auc_score(yb.cpu().numpy(), preds.detach().cpu().numpy())
                                epoch_aucs['node'].append(auc)
                            except:
                                epoch_aucs['node'].append(np.nan)

                        epoch_losses['node'].append(loss.item())

                # Single backward pass per graph
                optimizer.zero_grad()
                scaler.scale(total_loss_graph).backward()
                scaler.step(optimizer)
                scaler.update()

        
            # Validation
            val_results = self.run_validation(batch_size, epoch, val_samples, self.train_end)
            for t in task_keys:
                val_losses_all[t].append(val_results[t]['loss'])
                val_aucs_all[t].append(val_results[t]['auc'])
                avg_train_loss = np.nanmean(epoch_losses[t]) if epoch_losses[t] else 0.0
                avg_train_auc = np.nanmean(epoch_aucs[t]) if epoch_aucs[t] else 0.0
                
                # 2. Append the single average value to the final list
                train_losses_all[t].append(avg_train_loss)
                train_aucs_all[t].append(avg_train_auc)
                print(f"Epoch: {epoch+1:02d} | Task: {t} | Train Loss: {train_losses_all[t][-1]:.4f} | Train AUCROC: {train_aucs_all[t][-1]:.4f}")
                
            if epoch >= 25:
                val_loss_sum = sum(val_results[et]['loss'] for et in task_keys)
                
                if val_loss_sum < best_val_loss_sum:
                    best_val_loss_sum = val_loss_sum
                    counter = 0
                    
                    # ---- SAVE BEST MODEL ----
                    best_state["encoder"] = {
                        k: v.detach().cpu().clone()
                        for k, v in self.encoder_model.state_dict().items()
                    }
                    best_state["decoder"] = {
                        k: v.detach().cpu().clone()
                        for k, v in self.link_prediction_decoder.state_dict().items()
                    }

                else:
                    counter += 1
                    if counter >= patience:
                        print('Early stoping triggered')
                        break
                
        # --------- Plot results ----------
        
        os.makedirs(self.training_plots_path, exist_ok=True)
        for key in task_keys:
            print('Sending to visualization dir')
            loss_path = os.path.join(self.training_plots_path, f'loss_{key}.png')
            aucroc_path = os.path.join(self.training_plots_path, f'aucroc_{key}.png')

            self.visualizer.display_loss(
                train_loss=train_losses_all[key],
                valid_loss=val_losses_all[key],
                num_epochs=epochs,
                save_path=loss_path,
                edge_type=key
            )

            self.visualizer.display_aucroc(
                train_aucroc=train_aucs_all[key],
                valid_aucroc=val_aucs_all[key],
                num_epochs=epochs,
                save_path=aucroc_path,
                edge_type=key
            )

        if best_state["encoder"] is not None:
            self.encoder_model.load_state_dict(best_state["encoder"])
            self.link_prediction_decoder.load_state_dict(best_state["decoder"])
            print("Best model restored from early stopping")

        return self.link_prediction_decoder
                
    
    def train_models(self):
        if os.path.exists(self.saved_samples):
            print(f"[INFO] Loading all_samples from {self.saved_samples}")
            with open(self.saved_samples, "rb") as f:
                edge_samples, node_samples = pickle.load(f)
        else:
            print("[INFO] Creating all_samples...")
            curr_graphs = [inner[-1] for inner in self.target_graphs]
            edge_samples, node_samples = self.create_samples(curr_graphs, self.days_back, self.all_edgebanks, self.is_directed)
            os.makedirs(os.path.dirname(self.saved_samples), exist_ok=True)
            with open(self.saved_samples, "wb") as f:
                pickle.dump((edge_samples, node_samples), f)
            print(f"[INFO] Saved all_samples to {self.saved_samples}")
        
        # Split samples 80%/10%/10%
        edge_types = edge_samples.keys()
        sample_tasks = list(edge_types) + ['node']
        num_graphs = len(next(iter(edge_samples.values()))['X'])  # Number of graphs
        
        n_train = int(0.7 * num_graphs)
        n_val = int(0.15 * num_graphs)
        n_test = num_graphs - n_train - n_val
        
        training_samples = {key: {'X': [], 'y': []} for key in sample_tasks}
        val_samples = {key: {'X': [], 'y': []} for key in sample_tasks}
        test_samples = {key: {'X': [], 'y': []} for key in sample_tasks}
        
        for edge_type in edge_types:
            for idx, (graph_X, graph_y) in enumerate(zip(edge_samples[edge_type]['X'], edge_samples[edge_type]['y'])):
                # Shuffle edges within the graph
                combined = list(zip(graph_X, graph_y))
                random.shuffle(combined)
                graph_X, graph_y = zip(*combined) if combined else ([], [])
                
                # Assign graph to correct split
                if idx < n_train:
                    training_samples[edge_type]['X'].append(list(graph_X))
                    training_samples[edge_type]['y'].append(list(graph_y))
                elif idx < n_train + n_val:
                    val_samples[edge_type]['X'].append(list(graph_X))
                    val_samples[edge_type]['y'].append(list(graph_y))
                else:
                    test_samples[edge_type]['X'].append(list(graph_X))
                    test_samples[edge_type]['y'].append(list(graph_y))
                    
        node_X_all = node_samples.get('X', [])
        node_y_all = node_samples.get('y', [])
        for idx in range(len(node_X_all)):
            graph_X = node_X_all[idx]
            graph_y = node_y_all[idx]
            combined = list(zip(graph_X, graph_y))
            random.shuffle(combined)
            if combined:
                graph_X, graph_y = zip(*combined)
                graph_X, graph_y = list(graph_X), list(graph_y)
            else:
                graph_X, graph_y = [], []

            if idx < n_train:
                training_samples['node']['X'].append(graph_X)
                training_samples['node']['y'].append(graph_y)
            elif idx < n_train + n_val:
                val_samples['node']['X'].append(graph_X)
                val_samples['node']['y'].append(graph_y)
            else:
                test_samples['node']['X'].append(graph_X)
                test_samples['node']['y'].append(graph_y)            
        
        
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
        
        # For TGCN
        prev_edges = torch.tensor(list(self.target_graphs[current_target_snapshot - 1][-1].edges()), dtype=torch.long).t().contiguous().to(self.device)

        old_nodes_days = set().union(*[g.nodes() for g in prev_graphs])  # Get all nodes over the past days_back days

        # How many nodes and edges we are expecting to see        
        V_total = int(current_target_graph_description[-1][0])
        E_total = int(current_target_graph_description[-1][1])

        # Predict which old nodes we will see
        
        if self.feature_type == 'learnable':
            features = self.node_features.weight
        else:
            features = self.node_features[current_target_snapshot] if self.feature_type == 'node2vec' else self.node_features
        curr_embeddings = generate_gnn_node_embeddings(self.encoder_model, self.model_type, features, self.target_graphs[:current_target_snapshot], self.days_back, embedding_dim=self.embedding_dim, curr_nodes=list(old_nodes_days), device=self.device)
        node_ids = []
        X = []
        
        curr_back = self.days_back - 1
        while(len(old_nodes_days) < self.current_target_count_old_nodes):
            prev_graphs.insert(0, (self.target_graphs[curr_back][-1]))
            old_nodes_days = set().union(*[g.nodes() for g in prev_graphs])
            curr_back -= 1
        
        for node in old_nodes_days:
            X.append(curr_embeddings[node].detach().cpu().numpy())
            node_ids.append(node)
        X = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.link_prediction_decoder.predict_node_exists(X).squeeze(-1).sigmoid()
        topk_probs, topk_indices = torch.topk(probs, self.current_target_count_old_nodes)
        old_nodes = [node_ids[i] for i in topk_indices.cpu().tolist()]
        old_nodes = set(old_nodes)
        
        
        # Create new node IDs
        tmp_graphs = [graph[-1] for graph in self.target_graphs[0 : current_target_snapshot]]  # Get the next new node id (the next available number)
        self.new_node_id = max([node for graphs in tmp_graphs for node in graphs.nodes()]) + 1 
        new_nodes = np.arange(self.new_node_id, self.new_node_id + self.current_target_count_new_nodes)
        
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
        curr_embeddings = generate_gnn_node_embeddings(self.encoder_model, self.model_type, features, self.target_graphs[:current_target_snapshot], self.days_back, embedding_dim=self.embedding_dim, curr_nodes=all_nodes, device=self.device)
        
        # Assign maximum degrees
        constructing_graph = get_node_features(constructing_graph, prev_graphs, self.thresholds, current_target_graph_description, old_nodes, new_nodes)  
        
            
        # SAMPLE EDGES STEP
        # Get edges of each type
        edge_pool = []
        
        # We use the old embeddings for each of these edge types
        for flag in ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']:
            sampled_edges = predict_edges(constructing_graph, edge_type=flag, node_types=node_types, edgebank=edgebank, link_prediction_decoder=self.link_prediction_decoder, 
                                old_node_embeddings=curr_embeddings, top_k=self.current_target_count[flag], graph_num=current_target_snapshot, device=device)
        
            constructing_graph.add_edges_from(sampled_edges)
            update_degrees(constructing_graph)  # REDUNDANT I THINK
        
            edge_pool = edge_pool + sampled_edges
            
            
        # weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
        # edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

        # ======== START GRAPH CONSTRUCTION ========
        G = nx.DiGraph() if self.is_directed else nx.Graph()
        used_edges = set()
        filtration_graphs = []

        # Trying some new logic here, this should capture the nodes properly and quicker. We will rely on TopER more heavily earlier
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
            pickle.dump(data_to_save_old_only, f)  
            
            
if __name__ == '__main__':
    runner = Runner()
    runner.run()

# To run the script
# python GraphGeneration/scripts/topoGED_end_to_end.py 