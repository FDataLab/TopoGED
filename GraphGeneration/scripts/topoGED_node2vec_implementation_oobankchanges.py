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

from GraphGeneration.utils.Evaluator import Evaluator
from utils.visualizer import Visualizer
from load_data import load_data, generate_training_data_cached, generate_validation_data_cached, generate_negative_edges
from GraphGeneration.utils.sampling_edges_utils import predict_edges
from GraphGeneration.utils.casting_type import to_tensor
from GraphGeneration.utils.graph_construction_utils import compute_reappearance_probabilities, get_node_features, update_degrees
from create_sub_graphs import create_nn_graph, create_on_graph

# Models in use
from GraphGeneration.models.model import setupMLP, load_encoder_model

# Import all node embedding methods
from compute_embedding import compute_node2vec_embeddings, group_node2vec_embeddings
from process_data import modifyGraphIds, build_edgebanks_from_start
from torch.utils.data import DataLoader

# Import Loss fn
from GraphGeneration.scripts.composite_graphlet_loss_fn import GraphletLoss
from GraphGeneration.utils.estimate_graphlet import run_graphlet_estimate   
# TODO Rename these ^^^

from utils.embedding_methods.degree import EmbedDegree


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

class Runner(object):
    def __init__(self):      
        self.seed = encoder_config["seed"]
        self.use_ma = encoder_config["use_moving_average"]  # Whether to use moving average for node2vec embeddings or not
        
        # Set up Evaluator
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()
        self.device = device
        # Some default file path
        self.file_visualization_path = "GraphGeneration/scripts/Visualize"
        self.saved_input = os.path.abspath(f'data/input/cached/{encoder_config["dataset"]}/saved_data_node2vec_{self.use_ma}_oobankchanges')
        self.saved_samples = os.path.join(self.saved_input, 'saved_samples.pkl')
        self.common_suffix = f'topoGED_embedding{encoder_config["encoder_model"]["addOnFeature"]}_mlpEncoding{encoder_config["decoder_model"]["encode_links"]}_embeddingType{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.use_ma}_oobankchanges'
        self.edge_eval_dir = f'GraphGeneration/output/results/edges_evaluation/{encoder_config["dataset"]}/{self.common_suffix}'
        self.structure_dir = f'GraphGeneration/output/results/structure/{encoder_config["dataset"]}/{self.common_suffix}'
        self.kernel_dir = f'GraphGeneration/output/results/kernel/{encoder_config["dataset"]}/{self.common_suffix}'
        self.topER_dir = f'GraphGeneration/output/results/topER/{encoder_config["dataset"]}/{self.common_suffix}'
        self.saved_graph_dir =  f'data/output/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}'
        self.training_plots_path = f'GraphGeneration/output/results/training_plots/{encoder_config["dataset"]}/{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.use_ma}_oobankchanges'
        
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
        self.all_edge_types = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
        self.best_validation_model_auc = 0
        
        # Load the global encoder & decoder model
        self.encoder_model, self.input_dim = load_encoder_model(encoder_config, device=device, node2vec_dimensions=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_dimensions"], 
                                                           hidden_dim=encoder_config["encoder_model"]["hidden_dim"])
        
        # Check if there is any add-on features we will plug at the end of encoder embedding
        self.add_degree = False
        if encoder_config["encoder_model"]["addOnFeature"] in ['NodeType', 'Position', 'Degree']:
            self.input_dim += 1
            if encoder_config["encoder_model"]["addOnFeature"] == 'Degree':
                self.add_degree = True
        
        self.link_prediction_decoder = setupMLP(embedding_dim=self.input_dim*2, mlpEncoding=encoder_config["decoder_model"]["encode_links"])
        self.link_prediction_decoder.to(device)
        
        # Load all the snapshot true data 
        self.probabilities, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(encoder_config["dataset"], encoder_config["encoder_model"]["addOnFeature"], 
                                                                                                     encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"], str(encoder_config["days_back"]), encoder_config["use_predicted_vals"])
        
        # Modify the graph ids to 1,2,3,...
        self.target_graphs, _ = modifyGraphIds(self.target_graphs, self.thresholds, self.days_back)

        # Build the edgebanks for construction
        self.all_edgebanks = build_edgebanks_from_start(self.target_graphs, self.days_back)        

        # Reshape the graph description
        # self.graph_descriptions = [list(zip(graph_description[0::3], graph_description[1::3], graph_description[2::3])) for graph_description in self.graph_descriptions]
        self.graph_descriptions = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in self.graph_descriptions]
        
        # Split training, validation, test graphs
        # Snapshots that we will use for traininng
        # Convert number of snapshots to integer
        self.num_snapshots = len(self.target_graphs)
        self.train_end = int(0.8 * self.num_snapshots)
        self.val_end = int(0.9 * self.num_snapshots)

        # Assign snapshots
        self.training_graphs = [self.target_graphs[i][-1] for i in range(self.train_end)]
        self.validation_graphs = [self.target_graphs[i][-1] for i in range(self.train_end, self.val_end)]
        self.test_graphs = [self.target_graphs[i][-1] for i in range(self.val_end, self.num_snapshots)]

        self.new_node_id = 0  # The ID we will assign new node (incremented as we add nodes)

        # Exclusive to this iteration of topoGED (with Node2Vec)
        self.node_embedding_history = [compute_node2vec_embeddings(self.target_graphs[i][-1], device, add_degree=self.add_degree) for i in range(0, self.starting_graph)]  # Store the history of node embeddings for Node2Vec
        

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
    
    
    # ======================= TRAIN MODEL =======================
    def run_validation(self, validation_samples, batch_size, epoch):
        use_cuda = (self.device.type == "cuda")
        criterion = nn.BCELoss()
        embedder = EmbedDegree(include_weights=False)
        lambda_toper = 0  # Weight for TopER structural loss
        toper_loss_fn = GraphletLoss()

        results = {edge_type: {'loss': [], 'auc': []} for edge_type in validation_samples.keys()}
        num_snapshots = len(next(iter(validation_samples.values()))['X'])

        # Optional: Keep previously predicted graphs for continuity
        validation_graphs = []

        for snapshot in range(num_snapshots):
            print(f"\nINFO: Validation on snapshot {snapshot}")
            for edge_type in [f for f in self.all_edge_types if f != 'o-o-bank']:
                X_list = validation_samples[edge_type]['X'][snapshot]
                y_list = validation_samples[edge_type]['y'][snapshot]
                if not X_list:
                    print(f"[WARNING] No validation samples for edge type: {edge_type}")
                    continue

                u_embs_np = np.stack([np.array(x['u_embedding'], dtype=np.float32) for x in X_list])
                v_embs_np = np.stack([np.array(x['v_embedding'], dtype=np.float32) for x in X_list])
                u_ids = [x['u_id'] for x in X_list]
                v_ids = [x['v_id'] for x in X_list]
                y_tensor = torch.tensor(y_list, dtype=torch.float32).view(-1, 1).to(self.device)

                dataset = TensorDataset(
                    torch.tensor(u_embs_np, dtype=torch.float32),
                    torch.tensor(v_embs_np, dtype=torch.float32),
                    y_tensor
                )
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                all_preds = []
                total_loss = 0.0
                all_ids = []

                constructing_graph = nx.DiGraph() if self.is_directed else nx.Graph()
                true_graph = nx.DiGraph() if self.is_directed else nx.Graph()
                positive_edges = [(int(u_ids[i]), int(v_ids[i])) for i in range(len(y_list)) if y_list[i] == 1]
                true_graph.add_edges_from(positive_edges)

                # Forward pass (no gradients)
                self.link_prediction_decoder.eval()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
                    for batch_idx, (ub, vb, yb) in enumerate(loader):
                        preds = self.link_prediction_decoder(ub, vb, edge_type=edge_type).view(-1, 1)
                        bce_loss = criterion(preds, yb)
                        total_loss += bce_loss.item() * ub.size(0)
                        all_preds.append(preds.detach().cpu())
                        for i in range(ub.size(0)):
                            idx_global = batch_idx * batch_size + i
                            all_ids.append((u_ids[idx_global], v_ids[idx_global]))

                # Average BCE loss
                total_loss /= len(X_list)

                # Compute TopER loss (structural similarity)
                all_preds_flat = torch.cat(all_preds).numpy().flatten()
                k = min(len(positive_edges), len(all_preds_flat))
                top_k_indices = np.argsort(all_preds_flat)[-k:]
                top_k_edges = [all_ids[i] for i in top_k_indices]
                constructing_graph.add_edges_from(top_k_edges)

                pred_embedding, _, _ = embedder.process_graphs_for_embeddings([constructing_graph])
                pred_embedding = pred_embedding[0]
                true_embedding, _, _ = embedder.process_graphs_for_embeddings([true_graph])
                true_embedding = true_embedding[0]

                toper_loss = toper_loss_fn(
                    to_tensor(pred_embedding, device=self.device).unsqueeze(0),
                    to_tensor(true_embedding, device=self.device).unsqueeze(0)
                )

                # Combine losses
                total_loss = float(total_loss + lambda_toper * toper_loss)
                auc = roc_auc_score(y_tensor.cpu().numpy().flatten(), all_preds_flat)

                results[edge_type]['loss'].append(total_loss)
                if not np.isnan(auc):
                    results[edge_type]['auc'].append(auc)

                # Optional: keep predicted graph for temporal continuity
                validation_graphs.append(constructing_graph)

            torch.cuda.empty_cache()

        # Average metrics
        avg_results = {
            edge_type: {
                'loss': np.mean(results[edge_type]['loss']) if results[edge_type]['loss'] else 0.0,
                'auc': np.mean(results[edge_type]['auc']) if results[edge_type]['auc'] else 0.0
            }
            for edge_type in results.keys()
        }

        # Aggregate and possibly save best model
        avg_auc = np.mean([avg_results[e]['auc'] for e in avg_results])
        if avg_auc >= getattr(self, "best_validation_model_auc", 0):
            self.best_validation_model_auc = avg_auc
            print("INFO: Saving best validation model...")
            os.makedirs(os.path.dirname(self.decoder_model_path), exist_ok=True)
            torch.save(self.link_prediction_decoder.state_dict(), self.decoder_model_path)
            #torch.save(self.encoder_model.state_dict(), self.encoder_model_path)
            print("INFO: Validation model saved successfully.")

        # Logging
        for flag in avg_results:
            msg = (
                f"Epoch: {epoch+1:02d} | Edge Type: {flag} | "
                f"Val Loss: {avg_results[flag]['loss']:.4f} | "
                f"Val AUCROC: {avg_results[flag]['auc']:.4f}"
            )
            print(msg, flush=True)

            # Construct the output directory
            output_dir = os.path.join(
                self.file_visualization_path,
                encoder_config["dataset"],
                encoder_config["encoder_model"]["nodeEmbeddingType"]
            )
            os.makedirs(output_dir, exist_ok=True)  # ✅ create directories if missing

            # Construct file path
            file_path = os.path.join(
                output_dir,
                f"multiheadMLP_val_performance_{self.seed}.txt"
            )

            # Write message to file
            with open(file_path, "a") as f:
                f.write(msg + "\n")
                f.flush()

        return avg_results
            

    def train_multi_head(self, training_samples, validation_samples, test_samples):
        """
        Params:
            training_samples ():
            validation_samples (): 
            
        Returns:

        """
        # For storing losses and aucs
        train_losses_all = {et: [] for et in self.all_edge_types}
        train_aucs_all   = {et: [] for et in self.all_edge_types}
        val_losses_all   = {et: [] for et in self.all_edge_types}
        val_aucs_all     = {et: [] for et in self.all_edge_types}
        
        
        lr = encoder_config["training"]["lr"]
        batch_size = encoder_config["training"]["batch_size"]
        epochs = encoder_config["training"]["epochs"]
        use_cuda = (self.device.type == "cuda")

        # choose workers; start safe at 0 on cluster, bump later to 2–4
        dl_num_workers = 0

        self.link_prediction_decoder.train()
        optimizer = torch.optim.Adam(
            self.link_prediction_decoder.parameters(),
            lr=lr
        )
        criterion = nn.BCELoss()  # Switching to BCELoss from BCEWithLogitsLoss
        toper_loss_fn = GraphletLoss()  # Rename TODO
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        lambda_toper = 0  # weight for the Graphlet loss term

        num_snapshots = len(next(iter(training_samples.values()))['X'])
        embedder = EmbedDegree(include_weights=False)

        # My code
        for epoch in range(encoder_config["training"]["epochs"]):
            self.link_prediction_decoder.train()  
            epoch_losses = {k: [] for k in self.all_edge_types if k != 'o-o-bank'}
            epoch_aucs   = {k: [] for k in self.all_edge_types if k != 'o-o-bank'}
            
            # Samples are all already generated
            for snapshot in range(num_snapshots):
                # The graph we will use to make toper loss
                constructing_graph = nx.DiGraph() if self.is_directed else nx.Graph()
                true_graph = nx.DiGraph() if self.is_directed else nx.Graph()
                
                for flag in [f for f in self.all_edge_types if f != 'o-o-bank']:
                    X_list = training_samples[flag]['X'][snapshot]
                    y_list = training_samples[flag]['y'][snapshot]
                    
                    if len(X_list) == 0:
                        continue

                    # Unpack u/v embeddings and IDs from dicts
                    u_embs_np = np.stack([np.array(x['u_embedding'], dtype=np.float32) for x in X_list])
                    v_embs_np = np.stack([np.array(x['v_embedding'], dtype=np.float32) for x in X_list])
                    u_ids = [x['u_id'] for x in X_list]
                    v_ids = [x['v_id'] for x in X_list]
                    y = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)

                    # True graph for TopER
                    positive_edges = [(int(u_ids[i]), int(v_ids[i])) for i in range(len(y_list)) if y_list[i] == 1]
                    true_graph = nx.DiGraph() if self.is_directed else nx.Graph()
                    true_graph.add_edges_from(positive_edges)

                    # DataLoader for batching BCE
                    dataset = TensorDataset(torch.tensor(u_embs_np), torch.tensor(v_embs_np), y)
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                    all_preds = []
                    all_ids = []

                    # Loop over batches to get predictions and BCE loss
                    bce_loss_total = 0.0
                    for batch_idx, (ub, vb, yb) in enumerate(loader):
                        ub = ub.to(self.device)
                        vb = vb.to(self.device)
                        yb = yb.to(self.device)

                        with torch.cuda.amp.autocast(enabled=use_cuda):
                            preds = self.link_prediction_decoder(ub, vb, edge_type=flag).view(-1, 1)
                            bce_loss = criterion(preds, yb)
                            # <-- accumulate as tensor, not float
                            if batch_idx == 0:
                                bce_loss_total = bce_loss * ub.size(0)  # weight by batch size
                            else:
                                bce_loss_total += bce_loss * ub.size(0)

                        all_preds.append(preds.detach().cpu())
                        for i in range(ub.size(0)):
                            idx_global = batch_idx * batch_size + i
                            all_ids.append((u_ids[idx_global], v_ids[idx_global]))

                    # Average BCE over all samples (still a tensor)
                    bce_loss_total /= len(X_list)

                    # Compute TopER loss
                    if len(constructing_graph.nodes()) > 0:
                        pred_embedding, _, _ = embedder.process_graphs_for_embeddings([constructing_graph])
                        pred_embedding = pred_embedding[0]
                    else:
                        pred_embedding = torch.zeros((20), device=self.device)
                    true_embedding, _, _ = embedder.process_graphs_for_embeddings([true_graph])
                    true_embedding = true_embedding[0]
                    toper_loss = toper_loss_fn(
                        to_tensor(pred_embedding, device=self.device).unsqueeze(0),
                        to_tensor(true_embedding, device=self.device).unsqueeze(0)
                    )

                    
                    # Combine losses and backprop
                    total_loss = bce_loss_total + lambda_toper * toper_loss
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Flatten predictions for Top-K selection
                    all_preds_flat = torch.cat(all_preds).numpy().flatten()
                    k = min(len(positive_edges), len(all_preds_flat))
                    top_k_indices = np.argsort(all_preds_flat)[-k:]
                    constructing_graph = nx.DiGraph() if self.is_directed else nx.Graph()
                    top_k_edges = [all_ids[i] for i in top_k_indices]
                    constructing_graph.add_edges_from(top_k_edges)

                    # Record metrics
                    preds_cpu = all_preds_flat
                    y_cpu = y.numpy().flatten()
                    auc = roc_auc_score(y_cpu, preds_cpu)
                    if not np.isnan(auc):
                        epoch_aucs[flag].append(auc)
                    epoch_losses[flag].append(total_loss.item())                                 

            # Get results for this epoch
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else 0
            for flag in [f for f in self.all_edge_types if f != 'o-o-bank']:
                msg = (
                    f"Epoch: {epoch+1:02d} | Edge Type: {flag} | "
                    f"Train Loss: {np.mean(epoch_losses[flag]):.4f} | "
                    f"Train AUCROC: {np.mean(epoch_aucs[flag]):.4f} | "
                    f"GPU: {gpu_mem_alloc:.1f}MiB"
                )
                print(msg, flush=True)
                with open(rf'{self.file_visualization_path}/{encoder_config["dataset"]}/{encoder_config["encoder_model"]["nodeEmbeddingType"]}/multiheadMLP_performance_{self.seed}.txt', "a") as f:
                    f.write(msg + "\n")
                    f.flush()
                    
            # Run validation
            val_results = self.run_validation(validation_samples, batch_size, epoch)        
            
            for et in [f for f in self.all_edge_types if f != 'o-o-bank']:
                # Training metrics
                train_losses_all[et].append(np.mean(epoch_losses[et]) if len(epoch_losses[et]) else 0)
                train_aucs_all[et].append(np.mean(epoch_aucs[et]) if len(epoch_aucs[et]) else 0)
                
                # Validation metrics
                val_losses_all[et].append(val_results[et]['loss'])
                val_aucs_all[et].append(val_results[et]['auc'])
                
                
        os.makedirs(self.training_plots_path, exist_ok=True)
        for et in [f for f in self.all_edge_types if f != 'o-o-bank']:
            loss_path = os.path.join(self.training_plots_path, f'loss_{et}.png')
            aucroc_path = os.path.join(self.training_plots_path, f'aucroc_{et}.png')
            
            # Display Loss curves
            self.visualizer.display_loss(
                train_loss=train_losses_all[et],
                valid_loss=val_losses_all[et],
                num_epochs=encoder_config["training"]["epochs"],
                save_path=loss_path,
                edge_type=et
            )

            # Display AUC curves
            self.visualizer.display_aucroc(
                train_aucroc=train_aucs_all[et],
                valid_aucroc=val_aucs_all[et],
                num_epochs=encoder_config["training"]["epochs"],
                save_path=aucroc_path,
                edge_type=et
            )
                        
        return self.link_prediction_decoder
    
    
    def create_samples(self, graphs, days_back, all_edgebanks, is_directed=False):
        """
        MOVE THIS TO A SEPARATE FILE; CURRENTLY HERE FOR REFERENCE WHILE REDOING CODE
        """
        # Prepare the sorted samples for each edge type, both positive and negative edges 
        # We will then send them to a pkl file and use them for training the model
        # Just generate all samples then shuffling and splitting can happen later
        # Also just use Node2Vec for right now and I can switch it to self.encoder_model later
        # Need to organize the edges, then create the graphs (4 subgraphs per graph), then encode the nodes to make samples
        sorted_samples = {
            'o-o-nobank': {'X': [], 'y': []},
            'o-n': {'X': [], 'y': []},
            'n-n': {'X': [], 'y': []},
            }  # A dict to sort embeddings for multiheaded MLP training
        
        
        all_embeddings = []  # Store the embeddings for each snapshot here (completed graphs only)
        
        # Organize the edges
        for i, graph in enumerate(graphs):
            old_nodes_days = set().union(*[g.nodes() for g in graphs[max(i - days_back, 0): i]])   # Old nodes of days_back days before
            if i < self.starting_graph:
                all_embeddings.append(compute_node2vec_embeddings(graph, device, old_nodes_days=old_nodes_days, add_degree=self.add_degree))  # Store the embedding of this graph's nodes for later use
                continue 

            old_node_embeddings = group_node2vec_embeddings(all_embeddings, old_nodes_days, days_back, self.use_ma)

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
            
            oobank_edges = []
            
            for u, v in graph.edges():
                if u in old_nodes_days and v in old_nodes_days:
                    if v in all_edgebanks[i].get(u, set()):
                        edge_type = 'o-o-bank'
                        oobank_edges.append((u, v))
                        continue
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
            
            constructing_graph.add_edges_from(oobank_edges)
            # old_node_embeddings = compute_node2vec_embeddings(graph, device, old_nodes_days=old_nodes_days, add_degree=self.add_degree)
            # TODO Try this
            
            # I'm not sure if i want to make the o-o-bank and o-o-nobank edges with the old 
            # We will let o-o-bank, o-o-nobank, and o-n be formed from the old node embeddings
            # Before creating n-n, we will embed the graph again
            for edge_type in ['o-o-nobank', 'o-n']:
                sorted_samples[edge_type]['X'].append([])
                sorted_samples[edge_type]['y'].append([])
                
                
                # Since we don't have data for new nodes yet for edge type o-n, we will assign a vector of 0's
                for u, v in sorted_edges[edge_type]:
                    u_embedding = old_node_embeddings.get(u, torch.zeros(int(self.input_dim), device=device))
                    v_embedding = old_node_embeddings.get(v, torch.zeros(int(self.input_dim), device=device))
                    # Cast in case
                    u_embedding = torch.tensor(u_embedding, device=device, dtype=torch.float32)
                    v_embedding = torch.tensor(v_embedding, device=device, dtype=torch.float32)
                    sample = {
                        'u_id': u,
                        'v_id': v,
                        'u_embedding': u_embedding,
                        'v_embedding': v_embedding
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
                    u_embedding = old_node_embeddings.get(u, torch.zeros(int(self.input_dim), device=device))
                    v_embedding = old_node_embeddings.get(v, torch.zeros(int(self.input_dim), device=device))
                    # Cast in case
                    u_embedding = torch.tensor(u_embedding, device=device, dtype=torch.float32)
                    v_embedding = torch.tensor(v_embedding, device=device, dtype=torch.float32)
                    sample = {
                        'u_id': u,
                        'v_id': v,
                        'u_embedding': u_embedding,
                        'v_embedding': v_embedding
                    }
                    sorted_samples[edge_type]['X'][-1].append(sample)
                    sorted_samples[edge_type]['y'][-1].append(0)
                
                
                constructing_graph.add_edges_from(sorted_edges[edge_type])  # For embedding to get new node information later
                
            # Embed graph here before adding n-n edges because we have some information now
            if constructing_graph.number_of_edges() <= 0:
                all_embeddings.append({})
                continue
            curr_embeddings = compute_node2vec_embeddings(constructing_graph, device, old_nodes_days=old_nodes_days, add_degree=self.add_degree)  # Get the current embeddings (handles empty nodes)
                
            edge_type = 'n-n'
            sorted_samples[edge_type]['X'].append([])
            sorted_samples[edge_type]['y'].append([])
            
            # Generate positive samples
            for u, v in sorted_edges[edge_type]:
                u_embedding = curr_embeddings.get(u, torch.zeros(int(self.input_dim), device=device))  # Fallback, probably unecessary
                v_embedding = curr_embeddings.get(v, torch.zeros(int(self.input_dim), device=device))  # Fallback, probably unecessary
                # Cast in case
                u_embedding = torch.tensor(u_embedding, device=device, dtype=torch.float32)
                v_embedding = torch.tensor(v_embedding, device=device, dtype=torch.float32)
                sample = {
                    'u_id': u,
                    'v_id': v,
                    'u_embedding': u_embedding,
                    'v_embedding': v_embedding
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
                u_embedding = curr_embeddings.get(u, torch.zeros(int(self.input_dim), device=device))  # Fallback, probably unecessary
                v_embedding = curr_embeddings.get(v, torch.zeros(int(self.input_dim), device=device))  # Fallback, probably unecessary
                # Cast in case
                u_embedding = torch.tensor(u_embedding, device=device, dtype=torch.float32)
                v_embedding = torch.tensor(v_embedding, device=device, dtype=torch.float32)
                sample = {
                    'u_id': u,
                    'v_id': v,
                    'u_embedding': u_embedding,
                    'v_embedding': v_embedding
                }
                sorted_samples[edge_type]['X'][-1].append(sample)
                sorted_samples[edge_type]['y'][-1].append(0)
        
            
            # Embed the graph before moving to the next graph (for referencing old nodes)
            all_embeddings.append(compute_node2vec_embeddings(graph, device, old_nodes_days=old_nodes_days, add_degree=self.add_degree))  # Store the embedding of this graph's nodes for later use
            emb_dict = all_embeddings[-1]
            if isinstance(emb_dict, dict) and len(emb_dict) > 0:
                embs = list(emb_dict.values())
                if len(embs) > 1:
                    embs_tensor = torch.stack([e.detach().cpu() for e in embs])

                    # Pairwise distance and uniqueness
                    pairwise_diff = torch.cdist(embs_tensor, embs_tensor)
                    mean_dist = pairwise_diff.mean().item()
                    unique_rows = len(torch.unique(embs_tensor, dim=0))

                    # Zero embeddings
                    zero_embs = sum(torch.allclose(e, torch.zeros_like(e)) for e in embs)

                    # Variance stats
                    variances = embs_tensor.var(dim=0)
                    avg_var = variances.mean().item()
                    min_var = variances.min().item()

                    print(
                        f"[DEBUG] Graph {len(all_embeddings)-1}: "
                        f"{unique_rows}/{len(embs)} unique | "
                        f"mean_dist={mean_dist:.6f} | "
                        f"zero={zero_embs} | "
                        f"avg_var={avg_var:.6f}, min_var={min_var:.6f}"
                    )
        from torch.nn.functional import cosine_similarity

        threshold = 0.999  # cosine similarity threshold to consider "identical"
        conflicts = []

        for edge_type, data in sorted_samples.items():
            X_batches = data['X']
            y_batches = data['y']
            for batch_i, (X_list, y_list) in enumerate(zip(X_batches, y_batches)):
                if not X_list:
                    continue

                embeddings = []
                labels = []
                ids = []

                for i, sample in enumerate(X_list):
                    emb = torch.cat([sample['u_embedding'], sample['v_embedding']]).detach().cpu()
                    embeddings.append(emb)
                    labels.append(int(y_list[i]))
                    ids.append((sample['u_id'], sample['v_id']))

                embeddings = torch.stack(embeddings)
                labels = torch.tensor(labels)

                # Compare each pair within this batch
                for i in range(len(embeddings)):
                    sims = cosine_similarity(embeddings[i].unsqueeze(0), embeddings).flatten()
                    for j in range(i + 1, len(embeddings)):
                        if sims[j] >= threshold and labels[i] != labels[j]:
                            conflicts.append({
                                'edge_type': edge_type,
                                'batch': batch_i,
                                'pair': (ids[i], ids[j]),
                                'similarity': sims[j].item(),
                                'labels': (labels[i].item(), labels[j].item())
                            })

        if conflicts:
            print(f"[WARNING] Found {len(conflicts)} conflicting samples with near-identical embeddings:")
            for c in conflicts[:10]:  # show first few
                print(f"  [{c['edge_type']}] {c['pair']} | sim={c['similarity']:.4f} | labels={c['labels']}")
        else:
            print("[INFO] No conflicting near-duplicate samples detected.")  
             
        return sorted_samples
        
    
    def train_models(self):
        if os.path.exists(self.saved_samples):
            print(f"[INFO] Loading all_samples from {self.saved_samples}")
            with open(self.saved_samples, "rb") as f:
                all_samples = pickle.load(f)
        else:
            print("[INFO] Creating all_samples...")
            curr_graphs = [inner[-1] for inner in self.target_graphs]
            all_samples = self.create_samples(curr_graphs, self.days_back, self.all_edgebanks, self.is_directed)
            os.makedirs(os.path.dirname(self.saved_samples), exist_ok=True)
            with open(self.saved_samples, "wb") as f:
                pickle.dump(all_samples, f)
            print(f"[INFO] Saved all_samples to {self.saved_samples}")
        
        print(f"[INFO] Edge types: {list(all_samples.keys())}")
        for edge_type, data in all_samples.items():
            print(f"[INFO] {edge_type}: {len(data['X'])} graphs total")
            non_empty = sum(1 for x in data['X'] if x)
            print(f"        {non_empty} graphs with non-empty X lists")
                
        # Split samples 80%/10%/10%
        edge_types = all_samples.keys()
        num_graphs = len(next(iter(all_samples.values()))['X'])  # Number of graphs
        
        n_train = int(0.8 * num_graphs)
        n_val = int(0.1 * num_graphs)
        n_test = num_graphs - n_train - n_val
        
        training_samples = {edge_type: {'X': [], 'y': []} for edge_type in edge_types}
        val_samples = {edge_type: {'X': [], 'y': []} for edge_type in edge_types}
        test_samples = {edge_type: {'X': [], 'y': []} for edge_type in edge_types}
        
        for edge_type in ['o-o-nobank', 'o-n', 'n-n']:  # Exclude o-o-bank for training
            for idx, (graph_X, graph_y) in enumerate(zip(all_samples[edge_type]['X'], all_samples[edge_type]['y'])):
                if not graph_X:  # skip empty sample sets
                    continue

                combined = list(zip(graph_X, graph_y))
                random.shuffle(combined)
                graph_X, graph_y = zip(*combined) if combined else ([], [])

                if idx < n_train:
                    training_samples[edge_type]['X'].append(list(graph_X))
                    training_samples[edge_type]['y'].append(list(graph_y))
                elif idx < n_train + n_val:
                    val_samples[edge_type]['X'].append(list(graph_X))
                    val_samples[edge_type]['y'].append(list(graph_y))
                else:
                    test_samples[edge_type]['X'].append(list(graph_X))
                    test_samples[edge_type]['y'].append(list(graph_y))
        
        print(training_samples)
        
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
        
        old_nodes_days = set().union(*[g.nodes() for g in prev_graphs])  # Get all nodes over the past days_back days
        curr_embeddings = group_node2vec_embeddings(self.node_embedding_history, old_nodes_days, self.days_back, self.use_ma)  # Get the embeddings for the old nodes
        
        # How many nodes and edges we are expecting to see        
        V_total = int(current_target_graph_description[-1][0])
        E_total = int(current_target_graph_description[-1][1])

        # Select the most recent edges from the last few graphs to add to the edgebank
        curr_edges_oobank = []
        num_old_nodes = 0
        curr_old_nodes = set()
        
        # Look at more recent edges first to add
        for graph in prev_graphs[::-1]:
            if len(curr_edges_oobank) >= self.current_target_count['o-o-bank']:
                break
            for u, v in graph.edges():
                if len(curr_edges_oobank) >= self.current_target_count['o-o-bank']:
                    break
                needed_node_slots = 0
                if u not in curr_old_nodes:
                    needed_node_slots += 1
                if v not in curr_old_nodes:
                    needed_node_slots += 1
                # We have room to add both nodes
                if num_old_nodes + needed_node_slots <= self.current_target_count_old_nodes:
                    num_old_nodes += needed_node_slots
                    curr_old_nodes.add(u)
                    curr_old_nodes.add(v)
                    curr_edges_oobank.append((u, v))
                else:
                    continue
                    

        # No longer sampling CHANGED
        #old_nodes = self.sample_old_nodes(prev_graphs)  # Get the current old nodes we expect to see
        old_nodes = curr_old_nodes

        # Create new node IDs
        new_nodes = np.arange(self.new_node_id, self.new_node_id + self.current_target_count_new_nodes)
        self.new_node_id += len(new_nodes)  # Get the next new node id (the next available number)
        all_nodes = list(old_nodes) + list(new_nodes)

        constructing_graph = nx.DiGraph() if self.is_directed else nx.Graph()  # A graph for computing node embeddings easily
        
        node_types = {
            "old_nodes": old_nodes,
            "new_nodes": new_nodes
        } 
        
        # Assign maximum degrees
        constructing_graph = get_node_features(constructing_graph, prev_graphs, self.thresholds, current_target_graph_description, old_nodes, new_nodes)  
        
        constructing_graph.add_edges_from(curr_edges_oobank)
        
        # Assign zero vector for new nodes
        key0 = next(iter(curr_embeddings))
        for new_node in new_nodes:  # Doesn't matter what embedding we use here for len
            curr_embeddings[new_node] = torch.zeros(len(curr_embeddings[key0]), device=device, dtype=torch.float32)

        
        # TODO Try later
        # Embed the graph before adding o-o-bank and such
        # curr_embeddings = compute_node2vec_embeddings(constructing_graph, device, add_degree=self.add_degree)
        
        
        # SAMPLE EDGES STEP
        # Get edges of each type
        edge_pool = []
        edge_pool.extend(curr_edges_oobank)
        
        # We use the old embeddings for each of these edge types
        for flag in ['o-o-nobank', 'o-n']:
            sampled_edges = predict_edges(constructing_graph, edge_type=flag, node_types=node_types, edgebank=edgebank, link_prediction_decoder=self.link_prediction_decoder, 
                                old_node_embeddings=curr_embeddings, top_k=self.current_target_count[flag], graph_num=current_target_snapshot, device=device)
        
            constructing_graph.add_edges_from(sampled_edges)
            update_degrees(constructing_graph)  # REDUNDANT I THINK
        
            edge_pool = edge_pool + sampled_edges
            
        flag = 'n-n'  # Embed the graph before computing this edge type now that we have info for new nodes
        curr_embeddings = compute_node2vec_embeddings(constructing_graph, device, add_degree=self.add_degree)  # Re-embed the graph to get new node embeddings
            
        # Get the edges
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
        #self.encoder_model_path = os.path.join(self.saved_input, rf'saved_models/encoder_{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.seed}')
        self.decoder_model_path = os.path.join(self.saved_input, rf"saved_data/decoder_MLP_{self.seed}")

        if os.path.exists(self.decoder_model_path):
            self.link_prediction_decoder.load_state_dict(torch.load(self.decoder_model_path, map_location=device))            
            self.link_prediction_decoder.to(device)
            self.link_prediction_decoder.eval()
            print(f"Link Prediction Decoder loaded from: {self.decoder_model_path}")
        else:
            # Train the Decoder and Encoder model
            print('Training the Link Prediction Decoder and Encoder')
            self.train_models()
            print('Finished training the Link Prediction Decoder and Encoder; Start Graph Construction')
       
        # Old graphs that we know up to now
        self.old_graphs = [self.target_graphs[x][-1] for x in range(self.starting_graph)]
        
        
        all_node_ids = set(node for graph in self.old_graphs for node in graph.nodes())        
        self.new_node_id = max(all_node_ids) + 1 if all_node_ids else 0

        

        all_built_graphs = []
        all_target_graphs = []
        all_pred_nodes = []
        all_true_nodes = []


        # To predict snapshot i, we use snapshot 0,...,i-1 to train
        for i in range(self.starting_graph, len(self.probabilities)): 
            print("INFO: >>> Temporal Graph Construction <<<")
            print("INFO: Predict snapshot: ", i)
            print("======================================")

            self.current_target_snapshot = i
            
            # Get all old nodes in our context window
            self.current_target_old_nodes = set().union(*[g.nodes() for g in self.old_graphs[max(i - self.days_back, 0): i]])
            
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
            self.node_embedding_history.append(compute_node2vec_embeddings(self.target_graphs[i][-1], device, old_nodes_days=old_nodes_days, add_degree=self.add_degree))
        
        output_filepath = os.path.join(self.saved_graph_dir, f"Node2Vec_constructed_graphs_{encoder_config["dataset"]}.pkl")
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


"""
[WARNING] Found 4270286 conflicting samples with near-identical embeddings:
  [o-o-bank] ((1492, 3628), (1492, 1624)) | sim=0.9997 | labels=(1, 0)
  [o-o-bank] ((1492, 3628), (1492, 1623)) | sim=0.9996 | labels=(1, 0)
  [o-o-bank] ((1492, 3628), (1492, 1693)) | sim=1.0000 | labels=(1, 0)
  [o-o-bank] ((1492, 3628), (1492, 1576)) | sim=0.9999 | labels=(1, 0)
  [o-o-bank] ((1492, 3628), (1492, 5128)) | sim=0.9999 | labels=(1, 0)
  [o-o-bank] ((1492, 5130), (1492, 1624)) | sim=0.9996 | labels=(1, 0)
  [o-o-bank] ((1492, 5130), (1492, 1623)) | sim=0.9995 | labels=(1, 0)
  [o-o-bank] ((1492, 5130), (1492, 1693)) | sim=0.9999 | labels=(1, 0)
  [o-o-bank] ((1492, 5130), (1492, 1576)) | sim=0.9999 | labels=(1, 0)
  [o-o-bank] ((1492, 5130), (1492, 5128)) | sim=1.0000 | labels=(1, 0)
  """