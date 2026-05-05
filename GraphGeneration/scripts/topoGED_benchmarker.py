"""
File to benchmark 
- EGCN
- GCLSTM
- ROLAND
- VGAE
- TGN
"""

import argparse
from ast import arg
import math
import time
import numpy as np 
import networkx as nx
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
import os
import sys
import yaml
import copy
import pickle 
import torch.nn.functional as F
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

# For this implementation
from GraphGeneration.models.temporal_gnn.script.models.load_model import load_model
from GraphGeneration.utils.benchmarking_utils import *
from GraphGeneration.encoders.TGN.model.TGNBatch import TGNBatch


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


# TODO MOVE WHEN WORKS
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, input_type='Concat'):
        super().__init__()

        self.input_type = input_type
        self.heads = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
           )

    def forward(self, src_embed, dst_embed):
        if self.input_type == 'Concat':
            edge_input = torch.cat([src_embed, dst_embed], dim=1)
        elif self.input_type == 'Addition':
            edge_input = src_embed + dst_embed
        elif self.input_type == 'Subtraction':
            edge_input = src_embed - dst_embed
        elif self.input_type == 'ElementwiseProduct':
            edge_input = src_embed * dst_embed
            
        return self.heads(edge_input).squeeze()



class Runner(object):
    def __init__(self, model_type):      
        self.config = encoder_config
        self.seed = encoder_config["seed"]
        self.use_ma = encoder_config["use_moving_average"]  # Whether to use moving average for node2vec embeddings or not
        self.model_type = model_type
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
        self.saved_input = os.path.abspath(f'data/input/cached/{encoder_config["dataset"]}/benchmarking/saved_data_{self.model_type}_lr{encoder_config["training"]["lr"]}')
        self.saved_samples = os.path.join(self.saved_input, 'saved_samples.pkl')
        self.common_suffix = f'topoGED_embedding{encoder_config["encoder_model"]["addOnFeature"]}_{self.model_type}_lr{encoder_config["training"]["lr"]}'
        self.training_plots_path = f'GraphGeneration/output/results/training_plots/{encoder_config["dataset"]}/benchmarking/htgn_lr{encoder_config["training"]["lr"]}'

        
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
        self.num_nodes = len(all_unique_nodes)  # Just to add some padding
        num_nodes = self.num_nodes

        # Split training, validation, test graphs
        self.num_snapshots = len(self.target_graphs)
        self.train_end = int(0.7 * self.num_snapshots)
        self.val_end = int(0.85 * self.num_snapshots)

        # Assign snapshots
        self.training_graphs = [self.target_graphs[i][-1] for i in range(self.train_end)]
        self.validation_graphs = [self.target_graphs[i][-1] for i in range(self.train_end, self.val_end)]
        self.test_graphs = [self.target_graphs[i][-1] for i in range(self.val_end, self.num_snapshots)]
        self.target_graphs_samples = [self.target_graphs[i][-1] for i in range(self.num_snapshots)]

        self.starting_graph = self.num_snapshots - len(self.test_graphs)

        # Load the model for link prediction
        args = BenchmarkerArgs(encoder_config, self.model_type, device)
        args.num_nodes = len(all_unique_nodes)
        
        if self.model_type == 'TGN' or self.model_type == 'TGAT':
            # Inefficient, but I recalculate here for simpler logic
            train_num_nodes = len(set(node for graphs in self.training_graphs for node in graphs.nodes()))
            target_graphs_samples = [self.target_graphs[i][-1] for i in range(len(self.target_graphs))]
            samples = create_samples_tgn(target_graphs_samples)
            
            training = samples[:self.train_end]
            validation = samples[self.train_end:self.val_end]
            test = samples[self.val_end:]
            
            def get_flattened_tgn_data(samples):
                class Container: pass
                c = Container()
                c.sources = np.concatenate([s.sources for s in samples])
                c.destinations = np.concatenate([s.destinations for s in samples])
                c.timestamps = np.concatenate([s.timestamps for s in samples])
                c.edge_idxs = np.concatenate([s.edge_idxs for s in samples])
                return c

            train_data_flat = get_flattened_tgn_data(training)
            all_data_flat = get_flattened_tgn_data(samples)
            
            self.train_ngh_finder = get_neighbor_finder(train_data_flat, uniform=False, max_node_idx=train_num_nodes)
            self.full_ngh_finder = get_neighbor_finder(all_data_flat, uniform=False, max_node_idx=self.num_nodes)
            
            args.neighbor_finder, args.node_features, args.edge_features = tgn_setup(samples, args.num_nodes)
            args.neighbor_finder = self.train_ngh_finder
            
        if self.model_type == 'TGAT':
            all_nodes = set()
            total_edges = 0

            for g in self.target_graphs_samples:
                all_nodes.update(g.nodes())
                total_edges += g.number_of_edges()

            num_nodes = max(all_nodes) # Highest index
            embedding_dim = encoder_config["encoder_model"]["other_models"]["embedding_dim"]

            # Create the zero-initialized inputs the model expects
            args.n_feat = np.zeros((num_nodes + 1, embedding_dim))
            args.e_feat = np.zeros((total_edges + 1, embedding_dim))

        
        
        self.model = load_model(args).to(self.device)
        self.args = args
        
        self.embedding_dim = encoder_config["encoder_model"]["other_models"]["embedding_dim"]
        
        self.decoder = MLP(in_channels=(self.embedding_dim * 2), hidden_channels=32, input_type='Concat').to(self.device)
        
        # Build the edgebanks for construction
        self.all_edgebanks = build_edgebanks_from_start(self.target_graphs, self.days_back)        

        # Reshape the graph description
        # self.graph_descriptions = [list(zip(graph_description[0::3], graph_description[1::3], graph_description[2::3])) for graph_description in self.graph_descriptions]
        self.graph_descriptions = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in self.graph_descriptions]

        self.new_node_id = 0  # The ID we will assign new node (incremented as we add nodes)
    
    
    @torch.no_grad()
    def evaluate_temporal_benchmarker(self, snapshots, threshold=0.5):
        self.model.eval()
        if self.model_type == 'GCLSTM' or self.model_type == 'EvolveGCN' or self.model_type == 'TGN' or self.model_type == 'ROLAND' or self.model_type == 'TGCN':
            self.decoder.eval()
        total_loss = 0
        all_snapshot_aucs = []
        criterion = nn.BCELoss()
        h_state = None 

        if self.model_type == 'TGN':
            self.model.set_neighbor_finder(self.full_ngh_finder)

        for data in snapshots:
            data = data.to(self.device)
            
            # --- 1. CONSISTENT PADDING ---
            current_num_nodes = data.x.size(0)
            x_input = data.x
            if current_num_nodes < self.args.num_nodes:
                pad_size = self.args.num_nodes - current_num_nodes
                x_input = F.pad(data.x, (0, 0, 0, pad_size), "constant", 0)
                
            # --- 2. FORWARD PASS ---
            if self.model_type == 'GCLSTM':
                h_prev, c_prev = h_state if h_state else (None, None)
                # Encoder returns spatio-temporal embeddings
                H, C = self.model(x_input, data.edge_index, None, h_prev, c_prev)
                z = H
                h_state = (H, C) # Maintain state for temporal continuity
                
                probs = self.decoder(z[data.edge_label_index[0]], z[data.edge_label_index[1]])
                labels = data.edge_label.float()
                loss = criterion(probs, labels)
                
            
            elif self.model_type == 'TGCN':
                h_prev = h_state if h_state is not None else None
                
                H = self.model(x_input, data.edge_index, None, h_prev)
                
                z = H
                h_state = H  # Maintain state for temporal continuity
                
                probs = self.decoder(z[data.edge_label_index[0]], z[data.edge_label_index[1]])
                labels = data.edge_label.float()
                loss = criterion(probs, labels)
                

            elif self.model_type == 'EvolveGCN':
                # EvolveGCN (Option 2) expects snapshots as a list
                z_list = self.model([data.edge_index], [x_input])
                z = z_list[-1]
                
                probs = self.decoder(z[data.edge_label_index[0]], z[data.edge_label_index[1]])
                labels = data.edge_label.float()
                loss = criterion(probs, labels)

            elif self.model_type == 'VGAE':
                # Inject the pre-computed normalized adjacency from Snapshot T
                self.model.base_gcn.adj = data.adj_t
                self.model.gcn_mean.adj = data.adj_t
                self.model.gcn_logstddev.adj = data.adj_t

                # Use encode() for consistency with reparameterization/latent space
                z = self.model.encode(x_input) 
                
                if torch.isnan(z).any() or torch.isinf(z).any():
                    # This usually means the Learning Rate is too high or Input is unnormalized
                    z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
            
                # Inner product decoder logic
                logits = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).sum(dim=-1)
                logits = torch.clamp(logits, min=-10, max=10)
                probs = torch.sigmoid(logits)
                
                labels = data.edge_label.float()
                
                # Binary Cross Entropy (clamping is necessary for some trials)
                eps = 1e-7
                probs = torch.clamp(probs, min=eps, max=1.0 - eps)
                recon_loss = criterion(probs, labels) * 0.1
                
                # KL Divergence term
                kl_loss = -0.5 / x_input.size(0) * torch.mean(torch.sum(
                1 + 2 * self.model.logstd - self.model.mean**2 - torch.exp(self.model.logstd)**2, dim=1))

                if encoder_config["dataset"] == 'networkadex':
                    kl_loss = 0  # Fixes errors with networkadex
                loss = recon_loss + (kl_loss)

            elif self.model_type == 'TGN':
                src_embed, dst_embed, neg_embed = self.model.compute_temporal_embeddings(
                    source_nodes=data.sources,
                    destination_nodes=data.destinations,
                    negative_nodes=data.neg_dst,
                    edge_times=data.t,
                    edge_idxs=data.edge_idxs,
                    n_neighbors=20
                )

                # Concatenate positive and negative pairs to pass through the decoder
                # Format: [src, src] and [dst, neg_dst]
                sources = torch.cat([src_embed, src_embed], dim=0)
                destinations = torch.cat([dst_embed, neg_embed], dim=0)

                probs = self.decoder(sources, destinations)
                labels = data.edge_label.to(self.device).float()
                
                loss = criterion(probs, labels)
            
            elif self.model_type == 'ROLAND':
                # Use the global state carried over from the previous snapshot
                h_new = self.model(
                    data.x, 
                    data.edge_index, 
                    previous_embeddings=h_state if h_state else [
                        torch.zeros(self.args.num_nodes, self.args.nhid, device=self.device),
                        torch.zeros(self.args.num_nodes, self.args.nhid, device=self.device)
                    ]
                )
                z = h_new[-1]
                # Update h_state for the next evaluation snapshot
                h_state = [h.detach() for h in h_new] 
                
                probs = self.decoder(z[data.edge_label_index[0]], z[data.edge_label_index[1]])
                labels = data.edge_label.float()
                loss = criterion(probs, labels)
            

            # calculate metrics
            total_loss += loss.item()
            # VGAE and others use AUC for link prediction evaluation [cite: 12, 48]
            auc = auroc(probs, labels.long(), task="binary")
            all_snapshot_aucs.append(auc.item())

        avg_loss = total_loss / len(snapshots) if snapshots else 0
        avg_auc = np.mean(all_snapshot_aucs) if all_snapshot_aucs else 0.5
        
        return avg_loss, avg_auc
    

    def train_temporal_benchmarker(self, train_samples, val_samples):
        os.makedirs(os.path.join(self.saved_input, "saved_models"), exist_ok=True)
        model_path = os.path.join(self.saved_input, f"saved_models/{self.model_type}_{self.seed}.pt")
        decoder_path = os.path.join(self.saved_input, f"saved_models/decoder_{self.seed}.pt")
        
        if self.model_type == 'GCLSTM' or self.model_type == 'EvolveGCN' or self.model_type == 'TGN' or self.model_type == 'ROLAND' or self.model_type == 'TGCN':
            params = list(self.model.parameters()) + list(self.decoder.parameters())
        else:
            params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.config["training"]["lr"], weight_decay=self.config["training"]["weight_decay"] if self.model_type=='VGAE' else 0)
        criterion = nn.BCELoss() 
        
        best_val_loss = float("inf")
        patience, counter = 5, 0
        num_prev_edges = 0  # Useful for ROLAND
        
        best_model_weights = None
        best_decoder_weights = None

        for epoch in range(self.config["training"]["epochs"]):
            self.model.train()
            self.decoder.train()
            epoch_loss = 0
            h_state = None 
            
            if self.model_type == 'TGN':
                self.model.set_neighbor_finder(self.train_ngh_finder)
            elif self.model_type == 'TGAT':
                self.model.set_neighbor_finder(self.train_ngh_finder)

            for data in train_samples:
                data = data.to(self.device)
                optimizer.zero_grad()
                
                # --- PADDING ---
                if self.model_type == 'GCLSTM' or self.model_type == 'EvolveGCN' or self.model_type == 'VGAE' or self.model_type == 'TGCN':
                    x_input = data.x
                    if data.x.size(0) < self.args.num_nodes:
                        pad_size = self.args.num_nodes - data.x.size(0)
                        x_input = F.pad(data.x, (0, 0, 0, pad_size), "constant", 0)
                
                # --- FORWARD PASS ---
                if self.model_type == 'GCLSTM':
                    h_prev, c_prev = h_state if h_state else (None, None)
                    H, C = self.model(x_input, data.edge_index, None, h_prev, c_prev)
                    
                    # Select embeddings for the pairs we want to predict data.edge_label_index shape: [2, num_samples]
                    src_embed = H[data.edge_label_index[0]]
                    dst_embed = H[data.edge_label_index[1]]
                    
                    probs = self.decoder(src_embed, dst_embed)
                    
                    # Labels must be float for BCELoss
                    labels = data.edge_label.float()
                    loss = criterion(probs, labels)
                    h_state = (H.detach(), C.detach())
                    
                elif self.model_type == 'TGCN':
                    # T-GCN uses a GRU cell, so it only has a hidden state (H), no cell state (C) [cite: 6, 213]
                    h_prev = h_state if h_state is not None else None
                    
                    # Forward pass: returns current hidden state [cite: 214]
                    # In T-GCN, GCN captures spatial features and GRU captures temporal features [cite: 7, 85, 86]
                    H = self.model(x_input, data.edge_index, None, h_prev)
                    
                    # Select spatio-temporal embeddings for the candidate pairs [cite: 184, 185]
                    src_embed = H[data.edge_label_index[0]]
                    dst_embed = H[data.edge_label_index[1]]
                    
                    # Obtain prediction probabilities through the fully connected layer (decoder) 
                    probs = self.decoder(src_embed, dst_embed)
                    
                    labels = data.edge_label.float()
                    loss = criterion(probs, labels)
                    
                    # Maintain the single hidden state for temporal continuity [cite: 185, 215]
                    h_state = H.detach()
                    
                elif self.model_type == 'TGN':
                    # Model will use a decoder with concatenation
                    src_z, dst_z, neg_z = self.model.compute_temporal_embeddings(
                        source_nodes=data.sources,
                        destination_nodes=data.destinations,
                        negative_nodes=data.neg_dst,
                        edge_times=data.t,
                        edge_idxs=data.edge_idxs,
                        n_neighbors=20
                    )

                    sources = torch.cat([src_z, src_z], dim=0)
                    destinations = torch.cat([dst_z, neg_z], dim=0)

                    probs = self.decoder(sources, destinations)
                    labels = data.edge_label.to(self.device).float()
                    loss = criterion(probs, labels) 
                
                elif self.model_type == 'EvolveGCN':
                    # Model will concatenate embedding u and v and use an MLP for link prediction
                    z_list = self.model([data.edge_index], [x_input])

                    # Take the embedding from the most recent time step
                    z_final = z_list[-1] 

                    # Index pairs from the final embedding
                    src_z = z_final[data.edge_label_index[0]]
                    dst_z = z_final[data.edge_label_index[1]]

                    probs = self.decoder(src_z, dst_z)
                    labels = data.edge_label.float()
                    loss = criterion(probs, labels)
                 
                 
                elif self.model_type == 'VGAE':
                    # Inject the pre-computed normalized adjacency from Snapshot T
                    # This allows the GCN to aggregate over the actual previous structure
                    self.model.base_gcn.adj = data.adj_t
                    self.model.gcn_mean.adj = data.adj_t
                    self.model.gcn_logstddev.adj = data.adj_t

                    
                    z = self.model.encode(x_input) 
                    
                    if torch.isnan(z).any() or torch.isinf(z).any():
                        # This usually means the Learning Rate is too high or Input is unnormalized
                        z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
                    
                    # Calculate dot product between source and destination embeddings
                    logits = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).sum(dim=-1)
                    logits = torch.clamp(logits, min=-10, max=10)  # Prevents some errors
                    probs = torch.sigmoid(logits)
                    labels = data.edge_label.float()
                    # Binary Cross Entropy (clamping is necessary for some trials)
                    eps = 1e-5
                    probs = torch.clamp(probs, min=eps, max=1.0 - eps)
                    
                    
                    # 3. LOSS CALCULATION
                    try:
                        recon_loss = criterion(probs, labels) * 0.1
                    except Exception as e:
                        # print("\n--- CRASH DIAGNOSTICS ---")
                        # print(f"Probs: min={probs.min().item():.8f}, max={probs.max().item():.8f}")
                        # print(f"Labels: min={labels.min().item()}, max={labels.max().item()}")
                        # print(f"Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
                        raise e
                    # recon_loss = criterion(probs, data.edge_label.float())
                    
                    # KL Divergence regularizer 
                    kl_loss = -0.5 / x_input.size(0) * torch.mean(torch.sum(
                    1 + 2 * self.model.logstd - self.model.mean**2 - torch.exp(self.model.logstd)**2, dim=1))
                    
                    if encoder_config["dataset"] == 'networkadex':
                        kl_loss = 0  # Fixes errors with networkadex
                    loss = recon_loss + (kl_loss)
                    # loss = recon_loss 
                
                elif self.model_type == 'ROLAND':
                    # 1. Initialize hidden state if it's the first snapshot of the epoch
                    if h_state is None:
                        h_state = [
                            torch.zeros(self.args.num_nodes, self.args.nhid, device=self.device),
                            torch.zeros(self.args.num_nodes, self.args.nhid, device=self.device)
                        ]
                    
                    # 2. Forward pass using the detached state from the previous snapshot
                    h_new = self.model(
                        data.x, 
                        data.edge_index, 
                        previous_embeddings=h_state,
                        num_current_edges=data.num_current_edges,
                        num_previous_edges=num_prev_edges # You'll need to track this
                    )
                    
                    z_train = h_new[-1]
                    probs = self.decoder(z_train[data.edge_label_index[0]], z_train[data.edge_label_index[1]])
                    labels = data.edge_label.float()
                    loss = criterion(probs, labels)
                    
                    # 3. Update state for the next snapshot in the loop
                    # Detaching is critical here to avoid OOM and Autograd errors
                    h_state = [h.detach() for h in h_new]
                    num_prev_edges = data.num_current_edges
                                    
                                    
                elif self.model_type == 'TGAT':
                    # 1. Sample 'fake' destinations for negative contrast
                    # Using the sampler logic from the reference code
                    size = len(data.sources)
                    _, dst_fake = self.sampler.sample(size)
                    
                    # 2. Forward Pass: returns probabilities after dot-product + sigmoid
                    # contrast() implements Equation 8 from the paper 
                    pos_prob, neg_prob = self.model.contrast(
                        data.sources, data.destinations, dst_fake, data.timestamps, NUM_NEIGHBORS
                    )
                    
                    # 3. Equation 8 Loss: -log(pos) - Q * log(1 - neg) 
                    # We use standard criterion (BCELoss) to achieve this log-likelihood
                    loss = criterion(pos_prob, torch.ones_like(pos_prob)) + \
                           criterion(neg_prob, torch.zeros_like(neg_prob))
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            avg_val_loss, avg_val_auc = self.evaluate_temporal_benchmarker(val_samples)
            print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_loss/len(train_samples):.4f} | Val AUC: {avg_val_auc:.4f}")

            # Early Stopping & Saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if self.model_type == 'GCLSTM' or self.model_type == 'EvolveGCN' or self.model_type == 'ROLAND' or self.model_type == 'TGCN':
                    best_decoder_weights = {k: v.cpu().clone() for k, v in self.decoder.state_dict().items()}
            elif epoch >= 25:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            torch.save(best_model_weights, model_path)
            
            if best_decoder_weights is not None:
                self.decoder.load_state_dict(best_decoder_weights)
                torch.save(best_decoder_weights, decoder_path)
                    
        return self.model
       
    
    def train_models(self):
        if os.path.exists(self.saved_samples):
            with open(self.saved_samples, "rb") as f:
                snapshot_samples = pickle.load(f)
        else:
            if self.model_type == 'ROLAND':
                snapshot_samples = create_samples_roland(self.target_graphs_samples, num_global_nodes=self.args.num_nodes)
            elif self.model_type == 'GCLSTM' or self.model_type == 'TGCN':
                snapshot_samples = create_samples_gclstm(self.target_graphs_samples, neg_ratio=1.0)
            elif self.model_type == 'TGN':
                snapshot_samples = create_samples_tgn(self.target_graphs_samples, neg_ratio=1.0)
            elif self.model_type == 'EvolveGCN':
                snapshot_samples = create_samples_egcn(self.target_graphs_samples, neg_ratio=1.0)
            elif self.model_type == 'VGAE':
                snapshot_samples = create_samples_vgae(self.target_graphs_samples, neg_ratio=1.0, embedding_dim=self.embedding_dim, num_nodes=self.num_nodes)  # See if this works for VGAE
            elif self.model_type == 'TGAT':
                snapshot_samples = create_samples_tgat(self.target_graphs_samples, embedding_dim=self.embedding_dim)
            
            # elif self.model_type == 'WinGNN':
            #     snapshot_samples = create_samples_wingnn(self.target_graphs, neg_ratio=1.0)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # print(f'[INFO] Successfully created snapshot samples for model type: {self.model_type}')
            
            os.makedirs(os.path.dirname(self.saved_samples), exist_ok=True)
            with open(self.saved_samples, "wb") as f:
                pickle.dump(snapshot_samples, f, protocol=5)

        training = snapshot_samples[:self.train_end]
        validation = snapshot_samples[self.train_end:self.val_end]
        test = snapshot_samples[self.val_end:]
        
        # if self.model_type == 'ROLAND':
        #     self.train_roland(snapshot_samples)
        # else:
        #     self.train_temporal_benchmarker(training, validation)
        self.train_temporal_benchmarker(training, validation)
    
    
    def init_hidden(self, snapshot_idx):
        self.h_state = None
        
        # Warm up: Process all snapshots leading up to the target
        # This ensures the LSTM 'memory' is filled with the history of the graph
        with torch.no_grad():
            for i in range(snapshot_idx):
                data = self.dataset[i].to(self.device)
                x_input = data.x
                if data.x.size(0) < self.args.num_nodes:
                    pad_size = self.args.num_nodes - data.x.size(0)
                    x_input = F.pad(data.x, (0, 0, 0, pad_size), "constant", 0)
                
                if self.model_type == 'GCLSTM':
                    h_prev, c_prev = self.h_state if self.h_state else (None, None)
                    H, C = self.model(x_input, data.edge_index, None, h_prev, c_prev)
                    self.h_state = (H, C)
                    
                elif self.model_type == 'TGCN':
                    h_prev = self.h_state if self.h_state is not None else None
                    H = self.model(x_input, data.edge_index, None, h_prev)
                    self.h_state = H.detach()
                    
                elif self.model_type == 'ROLAND':
                    # Forward pass updates the hierarchical states
                    h_new = self.model(
                        x_input, 
                        data.edge_index, 
                        previous_embeddings=self.h_state,
                    )
                    
                    # Update persistent memory for the next iteration in warmup
                    self.h_state = [h.detach() for h in h_new]
                    

    def predict_next_edges(self, snapshot_idx, threshold=0.5):
        self.model.eval()
        self.decoder.eval()
        data = self.dataset[snapshot_idx].to(self.device)
        
        if self.model_type == 'GCLSTM' or self.model_type == 'EvolveGCN' or self.model_type == 'VGAE' or self.model_type == 'TGCN':
            x_input = data.x
            if data.x.size(0) < self.args.num_nodes:
                pad_size = self.args.num_nodes - data.x.size(0)
                x_input = F.pad(data.x, (0, 0, 0, pad_size), "constant", 0)
                

        with torch.no_grad():
            if self.model_type == 'GCLSTM':
                h_prev, c_prev = self.h_state if self.h_state is not None else (None, None)
                H, C = self.model(x_input, data.edge_index, None, h_prev, c_prev)
                
                # Select embeddings for the pairs we want to predict data.edge_label_index shape: [2, num_samples]
                src_embed = H[data.edge_label_index[0]]
                dst_embed = H[data.edge_label_index[1]]
                
                probs = self.decoder(src_embed, dst_embed)
                
                src_nodes = data.edge_label_index[0]
                dst_nodes = data.edge_label_index[1]

                # Take valid edges (no self loops)
                mask = (probs >= threshold) & (src_nodes != dst_nodes)
                
                # Apply mask
                final_src = src_nodes[mask]
                final_dst = dst_nodes[mask]
                
                final_src_cpu = final_src.cpu().numpy()
                final_dst_cpu = final_dst.cpu().numpy()
                predicted_edges = np.column_stack((final_src_cpu, final_dst_cpu))
                
                # Update persistent hidden state for temporal continuity
                self.h_state = (H.detach(), C.detach())
                
            
            elif self.model_type == 'TGCN':
                # T-GCN uses GRU, which only has a single hidden state (H), no cell state (C)
                h_prev = self.h_state if self.h_state is not None else None
                
                # Forward pass: GCN extracts spatial features and GRU processes temporal sequence
                # self.model returns the current hidden state H
                H = self.model(x_input, data.edge_index, None, h_prev)
                
                # Select spatio-temporal embeddings for the candidate pairs
                # data.edge_label_index shape: [2, num_samples]
                src_embed = H[data.edge_label_index[0]]
                dst_embed = H[data.edge_label_index[1]]
                
                # The final mapping from embeddings to probabilities is done through the decoder
                probs = self.decoder(src_embed, dst_embed)
                
                src_nodes = data.edge_label_index[0]
                dst_nodes = data.edge_label_index[1]

                # Take valid edges (no self loops) based on the prediction threshold
                mask = (probs >= threshold) & (src_nodes != dst_nodes)
                
                # Apply mask to identify predicted links
                final_src = src_nodes[mask]
                final_dst = dst_nodes[mask]
                
                final_src_cpu = final_src.cpu().numpy()
                final_dst_cpu = final_dst.cpu().numpy()
                predicted_edges = np.column_stack((final_src_cpu, final_dst_cpu))
                
                # Update persistent hidden state for temporal continuity
                self.h_state = H.detach()
                
                
            elif self.model_type == 'EvolveGCN':
                z_list = self.model([data.edge_index], [x_input])
                z = z_list[-1]
                
                probs = self.decoder(z[data.edge_label_index[0]], z[data.edge_label_index[1]])
                
                src_nodes = data.edge_label_index[0]
                dst_nodes = data.edge_label_index[1]

                # Take valid edges (no self loops)
                mask = (probs >= threshold) & (src_nodes != dst_nodes)
                
                # Apply mask
                final_src = src_nodes[mask]
                final_dst = dst_nodes[mask]
                
                final_src_cpu = final_src.cpu().numpy()
                final_dst_cpu = final_dst.cpu().numpy()
                predicted_edges = np.column_stack((final_src_cpu, final_dst_cpu))
                
            elif self.model_type == 'VGAE':
                self.model.base_gcn.adj = data.adj_t
                self.model.gcn_mean.adj = data.adj_t
                self.model.gcn_logstddev.adj = data.adj_t

                # Use encode() for consistency with reparameterization/latent space
                z = self.model.encode(x_input) 
                
                # Inner product decoder logic
                logits = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).sum(dim=-1)
                probs = torch.sigmoid(logits)
                
                src_nodes = data.edge_label_index[0]
                dst_nodes = data.edge_label_index[1]

                # Take valid edges (no self loops)
                mask = (probs >= threshold) & (src_nodes != dst_nodes)
                
                # Apply mask
                final_src = src_nodes[mask]
                final_dst = dst_nodes[mask]
                
                final_src_cpu = final_src.cpu().numpy()
                final_dst_cpu = final_dst.cpu().numpy()
                predicted_edges = np.column_stack((final_src_cpu, final_dst_cpu))
                
            elif self.model_type == 'TGN':
                src_embed, dst_embed, neg_embed = self.model.compute_temporal_embeddings(
                    source_nodes=data.sources,
                    destination_nodes=data.destinations,
                    negative_nodes=data.neg_dst,
                    edge_times=data.t,
                    edge_idxs=data.edge_idxs,
                    n_neighbors=20
                )

                # Concatenate positive and negative pairs to pass through the decoder
                # Format: [src, src] and [dst, neg_dst]
                sources = torch.cat([src_embed, src_embed], dim=0)
                destinations = torch.cat([dst_embed, neg_embed], dim=0)

                probs = self.decoder(sources, destinations)
                
                src_nodes = data.edge_label_index[0]
                dst_nodes = data.edge_label_index[1]

                # Take valid edges (no self loops)
                mask = (probs >= threshold) & (src_nodes != dst_nodes)
                
                # Apply mask
                final_src = src_nodes[mask]
                final_dst = dst_nodes[mask]
                
                final_src_cpu = final_src.cpu().numpy()
                final_dst_cpu = final_dst.cpu().numpy()
                predicted_edges = np.column_stack((final_src_cpu, final_dst_cpu))


            elif self.model_type == 'ROLAND':
                # 1. Prepare node list and embeddings
                node_list = sorted(list(self.known_nodes))
                num_known = len(node_list)
                
                h_new = self.model(
                    data.x, 
                    data.edge_index, 
                    previous_embeddings=self.h_state,
                )
                z = h_new[-1]
                
                predicted_edges = []
                batch_size = 50000 
                
                for i in range(num_known):
                    source_node_idx = i
                    target_indices = torch.arange(i + 1, num_known, device=self.device)
                    
                    if target_indices.size(0) == 0:
                        continue

                    for chunk in torch.split(target_indices, batch_size):
                        src_tensor = torch.full((chunk.size(0),), source_node_idx, 
                                              dtype=torch.long, device=self.device)
                        
                        logits = self.decoder(z[src_tensor], z[chunk])
                        probs = torch.sigmoid(logits)
                        
                        mask = probs >= threshold
                        if mask.any():
                            # Move indices to CPU
                            # .cpu().numpy() creates an array of integers
                            # .flatten() ensures we don't have nested dimensions
                            src_indices_passed = src_tensor[mask].cpu().numpy().flatten()
                            dst_indices_passed = chunk[mask].cpu().numpy().flatten()
                            
                            # MAP TO GLOBAL IDs:
                            # We iterate through the flattened numpy array 
                            # Each element s_idx is now a numpy scalar that Python treats as an int index
                            for s_idx, d_idx in zip(src_indices_passed, dst_indices_passed):
                                u = node_list[int(s_idx)] # Explicitly cast to int for list safety
                                v = node_list[int(d_idx)]
                                predicted_edges.append((u, v))

                self.h_state = [h.detach() for h in h_new]

            else:
                raise(ValueError())
                
            
            
        return predicted_edges

    def construct_predicted_graph(self, threshold=0.5):
        # This now calls the updated prediction logic
        predicted_edges = self.predict_next_edges(snapshot_idx=self.current_target_snapshot, threshold=threshold)
        
        new_G = nx.Graph() 
        # Add all nodes first to ensure isolated nodes are preserved
        # new_G.add_nodes_from(range(self.num_nodes)) 
        
        for edge in predicted_edges:
            new_G.add_edge(edge[0], edge[1])
        
        # print(f"Constructed {self.model_type} graph: {new_G.number_of_edges()} edges.")
        return new_G
    
    
    def run(self):        
        """
        Our main runner function
        
        Params:
            None
            
        Returns: 
            None
        """     
        # print("INFO: Dataset: {}".format(encoder_config["dataset"]))
        self.encoder_model_path = os.path.join(self.saved_input, f"saved_models/{self.model_type}_{self.seed}.pt")
        self.decoder_model_path = os.path.join(self.saved_input, f"saved_models/decoder_{self.seed}.pt")
        start_time = time.perf_counter()
        if os.path.exists(self.encoder_model_path):
            # Load Link Prediction Decoder
            self.model.load_state_dict(torch.load(self.encoder_model_path, map_location=device))            
            self.model.to(device)
            self.model.eval()
            
            # print(f"INFO: Benchmark model  loaded from: {self.encoder_model_path}")
            
        else:
            # Train both models if any part is missing
            # print('INFO: Models not found. Training the newest model...')
            
            self.train_models()
            
            print("INFO: Model successfully trained and saved.")
            
        # Load best models
        if os.path.exists(self.encoder_model_path):
            # Load Link Prediction Decoder
            self.model.load_state_dict(torch.load(self.encoder_model_path, map_location=device))            
            self.model.to(device)
            self.model.eval()
            
            # print(f"INFO: Benchmark model loaded from: {self.encoder_model_path}")
            
        if os.path.exists(self.decoder_model_path):
            # Load Link Prediction Decoder
            self.decoder.load_state_dict(torch.load(self.decoder_model_path, map_location=device))            
            self.decoder.to(device)
            self.decoder.eval()
            
            # print(f"INFO: Assisting decoder loaded from: {self.decoder_model_path}")

        if os.path.exists(self.saved_samples):
            try:
                with open(self.saved_samples, "rb") as f:
                    self.dataset = pickle.load(f)
            except Exception as e:
                if self.model_type == 'ROLAND':
                    snapshot_samples = create_samples_roland(self.target_graphs_samples, num_global_nodes=self.args.num_nodes)
                elif self.model_type == 'GCLSTM' or self.model_type == 'TGCN':
                    snapshot_samples = create_samples_gclstm(self.target_graphs_samples, neg_ratio=1.0)
                elif self.model_type == 'TGN':
                    snapshot_samples = create_samples_tgn(self.target_graphs_samples, neg_ratio=1.0)
                elif self.model_type == 'EvolveGCN':
                    snapshot_samples = create_samples_egcn(self.target_graphs_samples, neg_ratio=1.0)
                elif self.model_type == 'VGAE':
                    snapshot_samples = create_samples_vgae(self.target_graphs_samples, neg_ratio=1.0, embedding_dim=self.embedding_dim, num_nodes=self.num_nodes)  # See if this works for VGAE
                elif self.model_type == 'TGAT':
                    snapshot_samples = create_samples_tgat(self.target_graphs_samples, embedding_dim=self.embedding_dim)
                
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
                
                
                os.makedirs(os.path.dirname(self.saved_samples), exist_ok=True)
                with open(self.saved_samples, "wb") as f:
                    pickle.dump(snapshot_samples, f, protocol=5)
                    
                with open(self.saved_samples, "rb") as f:
                    self.dataset = pickle.load(f)

        train_end_time = time.perf_counter()
        elapsed_train_time = train_end_time - start_time
        print(f"INFO: Training time: {elapsed_train_time:.2f} seconds")

        times = {"train": elapsed_train_time}
        
        import psutil
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 ** 2)
        gpu_stats = ""
        if torch.cuda.is_available():
            curr_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
            peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            gpu_stats = f" | GPU Allocated: {curr_alloc:.2f}MB | GPU Peak: {peak_alloc:.2f}MB"
        
        print(f"{encoder_config["dataset"]} ({self.model_type} TRAIN TIME: {times['train']:.2f}s | RAM: {ram_mb:.2f}MB{gpu_stats}")
        
        # Reset peak stats for Construction phase monitoring
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            
        thresholds = [0.7, 0.75]

         # Loop through different global thresholds  
        for global_threshold in thresholds:
            start_time = time.perf_counter()
            self.saved_graph_dir = f'data/output/constructed_graphs/benchmarking/{encoder_config["dataset"]}_{self.common_suffix}_threshold{global_threshold}'
            output_filepath = os.path.join(self.saved_graph_dir, f"{encoder_config['encoder_model']['nodeEmbeddingType']}_constructed_graphs_{encoder_config['dataset']}.pkl")
            
            
            # To predict snapshot i, we use snapshot 0,...,i-1 to train
            self.H = None
            self.C = None 
            
            if self.model_type == 'TGN':
                self.model.set_neighbor_finder(self.full_ngh_finder)
                
            elif self.model_type == 'TGAT':
                self.model.ngh_finder = self.full_ngh_finder
            
            all_built_graphs = []
            all_target_graphs = []
            all_pred_nodes = []
            all_true_nodes = []
            
            self.known_nodes = set()
            for i in range(0, self.starting_graph):
                for graph_list in self.target_graphs[i]:
                    self.known_nodes.update(graph_list.nodes())
            
            if self.model_type == 'GCLSTM' or self.model_type == 'ROLAND' or self.model_type == 'TGCN':
                self.init_hidden(self.starting_graph)
            
            # Loop through snapshots sequentially
            for i in range(self.starting_graph - 1, len(self.dataset)): 
                # print(f"INFO: Predict snapshot: {i} using {self.model_type}")
                self.current_target_snapshot = i
                
                # This triggers the rolling state update (H/C)
                built_graph = self.construct_predicted_graph(global_threshold)
                
                # Storage logic
                target_graph = self.target_graphs[i][-1]
                all_built_graphs.append(built_graph)
                all_target_graphs.append(target_graph)
                all_pred_nodes.append({"old_nodes": built_graph.nodes(), "new_nodes": set()})  # By definition these are all old
                
                # Get the node types for the target graph
                current_nodes = target_graph.nodes()  # These are the old nodes
                all_true_nodes.append({"old_nodes": current_nodes, "new_nodes": set()})  # new nodes are by definition empty here
                
                self.known_nodes.update(current_nodes)  # Necessary for ROLAND
                
            
            output_filepath = os.path.join(self.saved_graph_dir, f"{encoder_config['encoder_model']['nodeEmbeddingType']}_constructed_graphs_{encoder_config['dataset']}.pkl")
            os.makedirs(self.saved_graph_dir, exist_ok=True)

            data_to_save = (all_built_graphs, all_target_graphs, all_pred_nodes, all_true_nodes)

            # print("\n======================================")
            print(f"INFO: Saving {len(all_built_graphs)} pairs of graphs to {output_filepath}")
            # print("======================================")

            with open(output_filepath, "wb") as f:
                pickle.dump(data_to_save, f, protocol=5) 
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"INFO: Total execution time: {elapsed_time:.2f} seconds")
            times[str(global_threshold)] = elapsed_time
            
            ram_mb = process.memory_info().rss / (1024 ** 2)
            gpu_stats = ""
            if torch.cuda.is_available():
                curr_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                gpu_stats = f" | GPU Allocated: {curr_alloc:.2f}MB | GPU Peak: {peak_alloc:.2f}MB"
            
            print(f"CONSTRUCTION (Thr {global_threshold}): {elapsed_time:.2f}s | RAM: {ram_mb:.2f}MB{gpu_stats}")
            
        print(times)  # For record keeping
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model to benchmark: roland, egcn, gclstm, tgcn, wingnn', required=True)
    args = parser.parse_args()
    runner = Runner(args.model)
    runner.run()

# To run the script
# python GraphGeneration/scripts/topoGED_end_to_end.py 