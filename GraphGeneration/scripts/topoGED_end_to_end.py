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
import line_profiler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from GraphGeneration.utils.Evaluator import Evaluator
from load_data import load_data, generate_training_data_cached, generate_validation_data_cached
from GraphGeneration.utils.sampling_edges_utils import predict_edges
from GraphGeneration.utils.casting_type import to_tensor
from GraphGeneration.utils.graph_construction_utils import compute_reappearance_probabilities, get_node_features, update_degrees
from create_sub_graphs import create_nn_graph, create_on_graph

# Models in use
from GraphGeneration.models.model import setupMLP, load_encoder_model

# Import all node embedding methods
from compute_embedding import compute_embedding
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
        
        # Set up Evaluator
        self.evaluator = Evaluator()
        
        # Some default file path
        self.file_visualization_path = "GraphGeneration/scripts/Visualize"
        self.saved_input = os.path.abspath(f'data/input/cached/{encoder_config["dataset"]}/saved_data')
        self.common_suffix = f'topoGED_embedding{encoder_config["encoder_model"]["addOnFeature"]}_mlpEncoding{encoder_config["decoder_model"]["encode_links"]}_embeddingType{encoder_config["encoder_model"]["nodeEmbeddingType"]}'
        self.edge_eval_dir = f'GraphGeneration/output/results/edges_evaluation/{encoder_config["dataset"]}/{self.common_suffix}'
        self.structure_dir = f'GraphGeneration/output/results/structure/{encoder_config["dataset"]}/{self.common_suffix}'
        self.kernel_dir = f'GraphGeneration/output/results/kernel/{encoder_config["dataset"]}/{self.common_suffix}'
        self.topER_dir = f'GraphGeneration/output/results/topER/{encoder_config["dataset"]}/{self.common_suffix}'
        self.saved_graph_dir = f'data/output/constructed_graphs/{encoder_config["dataset"]}_{self.common_suffix}'
        
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
        #TODO have this reference the right things from encoder.yaml
        self.encoder_model, self.input_dim = load_encoder_model(encoder_config, device=device, node2vec_dimensions=encoder_config["encoder_model"]["node2vec_setup"]["node2vec_dimensions"], 
                                                           hidden_dim=encoder_config["encoder_model"]["hidden_dim"])
        
        # Check if there is any add-on features we will plug at the end of encoder embedding
        if encoder_config["encoder_model"]["addOnFeature"] in ['NodeType', 'Position']:
            self.input_dim += 1
        
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
        train_auc = {
                'o-o-bank': [],
                'o-o-nobank': [],
                'o-n': [],
                'n-n': [],
            }
        # For computing AUC Scores
        train_preds = []
        train_labels = []

        for i in range(self.val_end - self.train_end):
            snapshot = self.train_end + i
            self.encoder_model.eval()
            self.link_prediction_decoder.eval()
            with torch.no_grad():
                print("INFO: Validation on snapshot", snapshot)
                
                # Prepare current target graph count
                current_target_graph_description = self.graph_descriptions[snapshot]
                # Used to convert probabilities
                V_total = int(current_target_graph_description[-1][0])
                E_total = int(current_target_graph_description[-1][1])
                
                # Prepare the probability counts
                self.current_target_count_old_nodes = int(round(self.probabilities[snapshot][0] * V_total))
                self.current_target_count_new_nodes = int(round(self.probabilities[snapshot][1] * V_total))
                self.current_target_count = {
                        edge_type: int(round(self.probabilities[snapshot][j + 2] * E_total))
                        for j, edge_type in enumerate(self.all_edge_types)
                    }
                
                node_types = { 
                    "old_nodes": self.sample_old_nodes(self.training_graphs[max(0, snapshot - self.days_back): snapshot]),
                    "new_nodes": set()
                } 
                
                constructing_graph = nx.DiGraph() # Graph we try to predict
                    
                # Adding old nodes to constructing_graph
                constructing_graph.add_nodes_from(node_types['old_nodes'])
                
                for flag in self.all_edge_types:
                    curr_X_train = validation_samples[flag]['X'][i]
                    curr_y_train = validation_samples[flag]['y'][i]
                    
                    if len(curr_X_train) == 0 or len(curr_y_train) == 0:
                        print(f'No samples for edge type: {flag}')
                        continue
                    
                    curr_X_train = [x.cpu().detach().numpy() if torch.is_tensor(x) else x for x in curr_X_train]
                    curr_X_train = np.array(curr_X_train)
                    curr_y_train = np.array(curr_y_train)

                    temp_X_train = torch.tensor(curr_X_train, dtype=torch.float32).to(device)
                    temp_y_train = torch.tensor(curr_y_train, dtype=torch.float32).to(device)
                    train_loader = DataLoader(TensorDataset(temp_X_train, temp_y_train), batch_size=batch_size, shuffle=True)
                    
                    # Training graphs for predicting current snapshot
                    validation_graphs = self.training_graphs + self.validation_graphs[:i]
                    
                    for (x, y) in train_loader:
                        node_embeddings = compute_embedding(embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"], graphs=validation_graphs, encoder_model=self.encoder_model, device=device)
                        
                        # Get current embeddings
                        src_nodes = [int(n) for n in x[:, 0].tolist()]                
                        dst_nodes = [int(n) for n in x[:, 1].tolist()]
                        
                        # Add new nodes to the node_types
                        for n in src_nodes:
                            if n not in node_embeddings and flag in ['o-n', 'n-n']:
                                node_types["new_nodes"].add(n)
                                constructing_graph.add_node(n)
                                node_embeddings[n] = torch.zeros(self.input_dim, device=device)
                                
                        for n in dst_nodes:
                            if n not in node_embeddings and flag in ['o-n', 'n-n']:
                                node_types["new_nodes"].add(n)
                                constructing_graph.add_node(n)
                                node_embeddings[n] = torch.zeros(self.input_dim, device=device)
                        
                        src_embed = torch.stack([
                            node_embeddings[n] for n in src_nodes
                        ])

                        dst_embed = torch.stack([
                            node_embeddings[n] for n in dst_nodes
                        ])

                        if src_embed.dim() == 1:
                            src_embed = src_embed.unsqueeze(1)  
                        if dst_embed.dim() == 1:
                            dst_embed = dst_embed.unsqueeze(1) 
                        
                        preds = self.link_prediction_decoder(src_embed=src_embed, dst_embed=dst_embed, edge_type=flag)
                        
                        if preds.dim() == 0:
                            preds = preds.unsqueeze(0)
                        if y.dim() == 0:  # scalar value like torch.tensor(0.5)
                            y = y.unsqueeze(0)  # make it [1]
                        elif y.dim() == 2 and y.size(1) == 1:  # shape [batch_size, 1]
                            y = y.view(-1)
                                                
                        # Add to our labels for evaluation
                        train_preds.extend(preds.detach().cpu().numpy())
                        train_labels.extend(y.detach().cpu().numpy())

                    # Assign embeddings for all the training_nodes
                    curr_embeddings = compute_embedding(embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"], graphs=validation_graphs, encoder_model=self.encoder_model, device=device)
                    constructing_graph = get_node_features(constructing_graph.copy(), self.training_graphs + self.validation_graphs[:i], self.thresholds, self.graph_descriptions[snapshot], node_types["old_nodes"], node_types["new_nodes"])
                    sampled_edges = predict_edges(constructing_graph, edge_type=flag, node_types=node_types, edgebank=self.all_edgebanks[snapshot], link_prediction_decoder=self.link_prediction_decoder, 
                                old_node_embeddings=curr_embeddings, top_k=self.current_target_count[flag], graph_num=snapshot, device=device, train=False)
                    constructing_graph.add_edges_from(list(sampled_edges))
                    update_degrees(constructing_graph)
                    
                    # Update the training_graphs to involve with the constructing graph
                    if flag == self.all_edge_types[0]:
                        validation_graphs.append(constructing_graph)
                    else:
                        validation_graphs[-1] = constructing_graph 
                    
                    if len(np.unique(train_labels)) < 2:
                        train_auc[flag].append(0)
                    else:
                        train_auc[flag].append(roc_auc_score(train_labels, train_preds))  # Calculate scores
        
        # Record the Training Loss, AUC 
        current_model_auc = 0 #we take average of all edge types
        
        for flag in self.all_edge_types:
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            epochMessage = f"Epoch {epoch+1:02d} | Edge Type: {flag}  | Validation AUCROC {np.mean(train_auc[flag]):.4f} | GPU: {gpu_mem_alloc:.1f}MiB"
            current_model_auc += np.mean(train_auc[flag])
            print(epochMessage)
            with open(rf'{self.file_visualization_path}/{encoder_config["dataset"]}/{encoder_config["encoder_model"]["nodeEmbeddingType"]}/multiheadMLP_performance.txt', "a") as f:
                f.write(epochMessage + "\n")
                f.flush()
                
        # We check and cache if it has the best auc
        if current_model_auc/4 >= self.best_validation_model_auc:
            self.best_validation_model_auc = current_model_auc/4
            
            print("INFO: Saving the model...")
            torch.save(self.link_prediction_decoder.state_dict(), self.decoder_model_path)
            torch.save(self.encoder_model.state_dict(), self.encoder_model_path)
            print("INFO: The model is saved. Done.")
            

    def train_multi_head(self, training_samples, validation_samples):
        """
        Params:
            training_samples ():
            validation_samples (): 
            
        Returns:

        """
        lr = encoder_config["training"]["lr"]
        use_cuda = (device.type == "cuda")

        # choose workers; start safe at 0 on cluster, bump later to 2–4
        dl_num_workers = 0

        self.link_prediction_decoder.train()
        optimizer = torch.optim.Adam(
            list(self.encoder_model.parameters()) +
            list(self.link_prediction_decoder.parameters()),
            lr=lr
        )
        loss_fn = nn.BCEWithLogitsLoss()
        toper_loss_fn = GraphletLoss()  # Rename TODO
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

        for epoch in range(encoder_config["training"]["epochs"]):
            epoch_losses = {k: [] for k in self.all_edge_types}
            epoch_aucs   = {k: [] for k in self.all_edge_types}

            for snapshot in range(self.starting_graph, self.train_end):
                current_target_graph_description = self.graph_descriptions[snapshot]
                # Used to convert probabilities
                V_total = int(current_target_graph_description[-1][0])
                E_total = int(current_target_graph_description[-1][1])
                self.current_target_count_old_nodes = int(round(self.probabilities[snapshot][0] * V_total))
                self.current_target_count_new_nodes = int(round(self.probabilities[snapshot][1] * V_total))
                self.current_target_count = {
                        edge_type: int(round(self.probabilities[snapshot][j + 2] * E_total))
                        for j, edge_type in enumerate(self.all_edge_types)
                    }

                # Why are we sampling here, we don't penalize the model off of choosing the wrong old nodes
                node_types = {
                    "old_nodes": self.sample_old_nodes(
                        self.training_graphs[max(0, snapshot - self.days_back): snapshot],
                    ),
                    "new_nodes": set()
                }

                window_graphs = self.training_graphs[max(0, snapshot - self.days_back): snapshot]  # Get the graphs from the past days_back days

                # PyTorch <=1.9: no device_type kw
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    with torch.no_grad():
                        # TODO This part could be more efficient
                        # Compute embeddings for the past graphs in our context window
                        base_embeddings = compute_embedding(
                            embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"],
                            graphs=window_graphs,
                            encoder_model=self.encoder_model,
                            device=device
                        )

                constructing_graph = nx.DiGraph() if self.is_directed else nx.Graph()
                constructing_graph.add_nodes_from(node_types["old_nodes"])

                for flag in self.all_edge_types:
                    X_np = np.array([
                        (x.cpu().numpy() if torch.is_tensor(x) else x)
                        for x in training_samples[flag]['X'][snapshot]
                    ])
                    y_np = np.array(training_samples[flag]['y'][snapshot])
                    if len(X_np) == 0:
                        continue

                    X_np, y_np = shuffle(X_np, y_np, random_state=self.seed)

                    # keep on CPU for DataLoader workers
                    X = torch.tensor(X_np, dtype=torch.float32)
                    y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

                    # --------- CONDITIONAL DATALOADER ARGS ----------
                    dl_kwargs = dict(
                        batch_size=encoder_config["training"]["batch_size"],
                        shuffle=True,
                        num_workers=dl_num_workers,
                        pin_memory=use_cuda,
                        drop_last=True,
                    )
                    if dl_num_workers > 0:
                        # only valid when multiprocessing is enabled
                        dl_kwargs.update(
                            persistent_workers=False,  # set True later if stable
                            prefetch_factor=2
                        )
                    loader = DataLoader(TensorDataset(X, y), **dl_kwargs)
                    # ------------------------------------------------

                    flag_logits = []
                    flag_targets = []

                    for xb, yb in loader:
                        optimizer.zero_grad(set_to_none=True)

                        # move to GPU here (main process)
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)

                        src_nodes = xb[:, 0].long().tolist()
                        dst_nodes = xb[:, 1].long().tolist()

                        new_nodes_to_add = set(src_nodes + dst_nodes) - base_embeddings.keys()
                        for n in new_nodes_to_add:
                            n_int = int(n)
                            node_types["new_nodes"].add(n_int)
                            base_embeddings[n_int] = torch.zeros(self.input_dim, device=device)

                        src_embed = torch.stack([base_embeddings[int(n)] for n in src_nodes])
                        dst_embed = torch.stack([base_embeddings[int(n)] for n in dst_nodes])

                        with torch.cuda.amp.autocast(enabled=use_cuda):
                            logits = self.link_prediction_decoder(
                                src_embed=src_embed, dst_embed=dst_embed, edge_type=flag
                            ).view(-1, 1)
                            bce = loss_fn(logits, yb)
                            loss = bce
                            
                        embedder = EmbedDegree(include_weights=False)

                        # Make the TopER embedding
                        if len(node_types["old_nodes"]) > 0:
                            pred_embedding, _, _ = embedder.process_graphs_for_embeddings([constructing_graph])
                            pred_embedding = pred_embedding[0]
                            true_embedding, _, _ = embedder.process_graphs_for_embeddings([self.training_graphs[snapshot]])
                            true_embedding = true_embedding[0]
                        else:
                            pred_embedding = np.zeros(20)  # Don't hardcode
                            true_embedding = np.zeros(20)

                        toper_loss = toper_loss_fn(to_tensor(pred_embedding, device=device).unsqueeze(0), to_tensor(true_embedding, device=device).unsqueeze(0))

                        total_loss = 0.8 * loss + 0.2 * toper_loss  # Play with the weights a bit

                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        flag_logits.append(logits.detach().cpu())
                        flag_targets.append(yb.detach().cpu())
                        epoch_losses[flag].append(loss.item())
                    if len(flag_logits) == 0:
                        # No batches produced (tiny dataset). Avoid cat() on empty list.
                        epoch_aucs[flag].append(0.0)
                        if not epoch_losses[flag]:
                            epoch_losses[flag].append(0.0)   # keep logging sane
                    else:
                        L = torch.cat(flag_logits, dim=0).sigmoid().numpy()
                        T = torch.cat(flag_targets, dim=0).numpy()
                        epoch_aucs[flag].append(roc_auc_score(T, L) if len(np.unique(T)) > 1 else 0.0)

                    curr_embeds = base_embeddings
                    constructing_graph = get_node_features(
                        constructing_graph.copy(), self.training_graphs[:snapshot],
                        self.thresholds, self.graph_descriptions[snapshot],
                        node_types["old_nodes"], node_types["new_nodes"]
                    )
                    sampled_edges = predict_edges(
                        constructing_graph, edge_type=flag, node_types=node_types,
                        edgebank=self.all_edgebanks[snapshot],
                        link_prediction_decoder=self.link_prediction_decoder,
                        old_node_embeddings=curr_embeds,
                        top_k=self.current_target_count[flag], graph_num=snapshot, device=device, train=True
                    )
                    constructing_graph.add_edges_from(list(sampled_edges))
                    update_degrees(constructing_graph)


            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else 0
            for flag in self.all_edge_types:
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

            self.run_validation(
                validation_samples=validation_samples,
                batch_size=encoder_config["training"]["batch_size"],
                epoch=epoch
            )

        return self.link_prediction_decoder, self.encoder_model
    
    
    def run_validation_retry(self, samples):
        pass 
    
    
    def train_multi_head_retry(self, training_samples, validation_samples, test_samples):
        pass
        # We aren't going to use a trainable encoder since we have nothing to train node embeddings on
        # Therefore, just use something like Node2Vec to encode the nodes in create_samples_retry() and loop over those
        # Basically, we just do binary classification here. We can still add nodes to a graph and compute TopER loss or Graphlet Loss but that isn't as computationally bad
    
    
    def create_samples_retry(self, graphs, days_back, all_edgebanks, is_directed=False):
        """
        MOVE THIS TO A SEPARATE FILE; CURRENTLY HERE FOR REFERENCE WHILE REDOING CODE
        """
        # Prepare the sorted samples for each edge type, both positive and negative edges 
        # We will then send them to a pkl file and use them for training the model
        # Just generate all samples then shuffling and splitting can happen later
        # Also just use Node2Vec for right now and I can switch it to self.encoder_model later
        # Need to organize the edges, then create the graphs (4 subgraphs per graph), then encode the nodes to make samples
        sorted_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
            'o-n': {'X': [], 'y': []},
            'n-n': {'X': [], 'y': []},
            }  # A dict to sort embeddings for multiheaded MLP training
        
        
        all_embeddings = []  # Store the embeddings for each snapshot here (completed graphs only)
        
        # Organize the edges
        for i, graph in enumerate(graphs):
            
            if i < self.starting_graph_idx:
                continue 
            # Old nodes of 'days_back' days before 
            old_nodes_days = set().union(*[g.nodes() for g in graphs[max(i - days_back, 0): i]]) 
            old_node_embeddings = {}  # Use encoder_model to embed these nodes first
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
            
            constructing_graph = nx.DiGraph() if is_directed else nx.Graph()  # We will add samples here for encoder to use
            
            # I'm not sure if i want to make the o-o-bank and o-o-nobank edges with the old 
            # We will let o-o-bank, o-o-nobank, and o-n be formed from the old node embeddings
            # Before creating n-n, we will embed the graph again
            for edge_type in ['o-o-bank', 'o-o-nobank', 'o-n']:
                sorted_samples[edge_type]['X'].append([])
                sorted_samples[edge_type]['y'].append([])
                
                
                # Since we don't have data for new nodes yet for edge type o-n, we will assign a vector of 0's
                for u, v in sorted_edges[edge_type]:
                    u_embedding = old_node_embeddings.get(u, np.zeros(self.input_dim / 2))
                    v_embedding = old_node_embeddings.get(v, np.zeros(self.input_dim / 2))
                    sample = u_embedding + v_embedding
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
                    u_embedding = old_node_embeddings.get(u, np.zeros(self.input_dim / 2))
                    v_embedding = old_node_embeddings.get(v, np.zeros(self.input_dim / 2))
                    sample = u_embedding + v_embedding
                    sorted_samples[edge_type]['X'][-1].append(sample)
                    sorted_samples[edge_type]['y'][-1].append(0)
                
                
                constructing_graph.add_edges_from(sorted_edges[edge_type])  # For embedding to get new node information later
                
            # Embed graph here    
                
            # We will embed the graph before generating these samples
            edge_type = 'n-n'
            sorted_samples[edge_type]['X'].append([])
            sorted_samples[edge_type]['y'].append([])
            
        # Generate embedding inputs and labels
        for i, graph in enumerate(graphs): 
            # We don't generate samples for graphs we won't generate for (our starting history)
            if i < self.starting_graph_idx:
                continue
            
            
            # Generate negative samples for each edge type
            for edge_type in ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']:
                negative_edges = generate_negative_edges(
                    graph,
                    num_samples=new_edges_count[edge_type],
                    edge_type=edge_type,
                    old_nodes=old_nodes_days,
                    is_directed=is_directed,
                    edgebank=all_edgebanks[i]
                )

                tmp_samples = [torch.tensor([u, v]) for u, v in negative_edges]

                # Add negative samples to the corresponding lists
                sorted_samples[edge_type]['X'][-1].extend(tmp_samples)
                sorted_samples[edge_type]['y'][-1].extend([0] * len(negative_edges))

        
        return sorted_samples
        
    
    def train_models_retry(self):
        all_samples = self.create_samples_retry(self.target_graphs, self.days_back, self.all_edgebanks, self.is_directed)
        
        # Split samples 80%/10%/10%
        training_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
            'o-n': {'X': [], 'y': []},
            'n-n': {'X': [], 'y': []},
            }
        val_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
            'o-n': {'X': [], 'y': []},
            'n-n': {'X': [], 'y': []},
            }
        test_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
            'o-n': {'X': [], 'y': []},
            'n-n': {'X': [], 'y': []},
            }
        
        self.train_multi_head_retry(training_samples, val_samples, test_samples)
    

    def train_models(self):
        """
        Create and train the models used for graph construction, these will be used for later graph construction
        
        Params:
            None
            
        Returns:
            link_prediction_decoder (MLP NN): The trained MLP, either single or multiheaded
            encoder_model (): The Encoder we will use to get node features later
        """
        MAX_SAMPLES = 1000000  # 1 Million
        
        # Prepare training data
        training_sorted_samples = generate_training_data_cached(training_graphs=self.training_graphs,
                                                all_edgebanks=self.all_edgebanks[:self.train_end], days_back=self.days_back, MAX_SAMPLES=MAX_SAMPLES, dataset=encoder_config["dataset"], seed=self.seed, is_directed=self.is_directed, saved_data_file_path=self.saved_input)

        # Prepare validation data
        val_graphs_combined = self.training_graphs[-self.days_back:] + self.validation_graphs  # We pass in the last days of training so that we can get the old edges and nodes from it
        # This could also likely be done through the edgebank, but there would be a logic issue where if a node has no outgoing edge in the directed case, it fails
        
        val_edgebanks = self.all_edgebanks[self.train_end + 1 : self.val_end]

        # If fewer edgebanks than graphs, pad the beginning with the last few training edgebanks
        if len(val_edgebanks) < len(val_graphs_combined):
            num_missing = len(val_graphs_combined) - len(val_edgebanks)
            padding = self.all_edgebanks[self.train_end - num_missing + 1 : self.train_end + 1]
            val_edgebanks = padding + val_edgebanks
        
        # We pass all_edgebanks of the training snapshots edgebanks
        validation_sorted_samples = generate_validation_data_cached(training_graphs=val_graphs_combined, days_back=self.days_back, 
                                               all_edgebanks=val_edgebanks, MAX_SAMPLES=MAX_SAMPLES, dataset=encoder_config["dataset"], seed=self.seed, is_directed=self.is_directed, type_data="validation", saved_data_file_path=self.saved_input)
                
        # Prepare test data
        num_needed = self.days_back
        prev_val_graphs = self.validation_graphs[-num_needed:]
        if len(prev_val_graphs) < num_needed:
            num_missing = num_needed - len(prev_val_graphs)
            prev_train_graphs = self.training_graphs[-num_missing:]
        else:
            prev_train_graphs = []

        test_graphs_combined = prev_train_graphs + prev_val_graphs + self.test_graphs

        # Slice edgebanks for test
        test_edgebanks = self.all_edgebanks[self.val_end:]

        # Pad to match
        if len(test_edgebanks) < len(test_graphs_combined):
            num_missing = len(test_graphs_combined) - len(test_edgebanks)
            padding = self.all_edgebanks[self.val_end - num_missing : self.val_end]
            test_edgebanks = padding + test_edgebanks
        
        # We pass all_edgebanks of the training snapshots edgebanks
        test_sorted_samples = generate_validation_data_cached(training_graphs=test_graphs_combined, days_back=self.days_back, 
                                                all_edgebanks=test_edgebanks, MAX_SAMPLES=MAX_SAMPLES, dataset=encoder_config["dataset"], seed=self.seed, is_directed=self.is_directed, type_data="test", saved_data_file_path=self.saved_input)
        
        print('Training') 
    
        self.link_prediction_decoder, self.encoder_model = self.train_multi_head(training_samples=training_sorted_samples, validation_samples=validation_sorted_samples)
        
        return self.link_prediction_decoder, self.encoder_model
            
            
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
        prev_graphs = [graph[-1] for graph in self.target_graphs[max(current_target_snapshot - encoder_config["training"]["day"], 0):current_target_snapshot]]
        
        # How many nodes and edges we are expecting to see        
        V_total = int(current_target_graph_description[-1][0])
        E_total = int(current_target_graph_description[-1][1])

        old_nodes = self.sample_old_nodes(prev_graphs)

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

        # Assign embeddings for all the training_nodes
        curr_embeddings = compute_embedding(embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"], graphs=prev_graphs, encoder_model=self.encoder_model, device=device)
        
        # Assign zero vector for new nodes
        for new_node in new_nodes:
            curr_embeddings[new_node] = torch.zeros(len(curr_embeddings[old_nodes[0]]), device=device, dtype=torch.float32)

            
        # SAMPLE EDGES STEP
        # Get edges of each type
        edge_pool = []
        
        # Sample edges 4 phases
        for flag in self.all_edge_types:
            sampled_edges = predict_edges(constructing_graph, edge_type=flag, node_types=node_types, edgebank=edgebank, link_prediction_decoder=self.link_prediction_decoder, 
                                old_node_embeddings=curr_embeddings, top_k=self.current_target_count[flag], graph_num=current_target_snapshot, device=device)
        
            constructing_graph.add_edges_from(sampled_edges)
            update_degrees(constructing_graph)  # REDUNDANT
            new_embeddings = compute_embedding(embeddingType=encoder_config["encoder_model"]["nodeEmbeddingType"], graphs=prev_graphs + [constructing_graph], encoder_model=self.encoder_model, device=device)
            curr_embeddings.update(new_embeddings)  # Recompute old node embeddings
        
            edge_pool = edge_pool + sampled_edges
            
        # weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
        # edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

        # ======== START GRAPH CONSTRUCTION ========
        G = nx.DiGraph() if self.is_directed else nx.Graph()
        used_edges = set()
        filtration_graphs = []

        # for i, (v_target, e_target) in enumerate(current_target_graph_description):
        #     v_target = int(v_target)
        #     e_target = int(e_target)

        #     current_nodes = {node for node, degree in constructing_graph.degree() if degree <= self.thresholds[i]}
        #     G.add_nodes_from(current_nodes)

        #     available_edges = [
        #         (u, v) for (u, v) in edge_pool
        #         if u in current_nodes and v in current_nodes and (u, v) not in used_edges
        #     ]

        #     needed = e_target - G.number_of_edges()
        #     selected_edges = available_edges[:needed]

        #     for (u, v) in selected_edges:
        #         #G.add_edge(u, v, weight=edge_weight_map[(u, v)])
        #         G.add_edge(u, v)
        #         used_edges.add((u, v))

        #     filtration_graphs.append(G.copy())

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
        self.encoder_model_path = os.path.join(self.saved_input, rf'saved_models/encoder_{encoder_config["encoder_model"]["nodeEmbeddingType"]}_{self.seed}')
        self.decoder_model_path = os.path.join(self.saved_input, rf"saved_data/decoder_MLP_{self.seed}")

        if os.path.exists(self.encoder_model_path) and os.path.exists(self.decoder_model_path):
            self.link_prediction_decoder.load_state_dict(torch.load(self.decoder_model_path, map_location=device))
            self.encoder_model.load_state_dict(torch.load(self.encoder_model_path, map_location=device))
            
            self.link_prediction_decoder.to(device)
            self.encoder_model.to(device)
            
            self.link_prediction_decoder.eval()
            self.encoder_model.eval()
            print(f"Link Prediction Decoder loaded from: {self.decoder_model_path}")
            print(f"Encoder loaded from: {self.encoder_model_path}")
        else:
            # Train the Decoder and Encoder model
            print('Training the Link Prediction Decoder and Encoder')
            self.link_prediction_decoder, self.encoder_model = self.train_models()
            print('Finished training the Link Prediction Decoder and Encoder; Start Graph Construction')
       
        # Old graphs that we know up to now
        self.old_graphs = [self.target_graphs[x] for x in range(self.starting_graph)]
        
        all_node_ids = [node for g in self.old_graphs for node in g.nodes()]
        self.new_node_id = max(all_node_ids) + 1 if all_node_ids else 0

        all_built_graphs = []
        all_target_graphs = []

        # To predict snapshot i, we use snapshot 0,...,i-1 to train
        for i in range(self.starting_graph, len(self.probabilities)): 
            print("INFO: >>> Temporal Graph Construction <<<")
            print("INFO: Predict snapshot: ", i)
            print("======================================")

            self.current_target_snapshot = i
            
            # Get all old nodes in our context window
            self.current_target_old_nodes = set().union(*[g[-1].nodes() for g in self.old_graphs[max(i - self.days_back, 0): i]])
            
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
            target_graph = self.target_graphs[i]
            all_built_graphs.append(built_graph)
            all_target_graphs.append(target_graph)
            
            # Add to the old graphs
            self.old_graphs.append(self.target_graphs[i])
           
        
        output_filepath = os.path.join(self.saved_graph_dir, f"constructed_graphs_{encoder_config["dataset"]}.pkl")
        os.makedirs(self.saved_graph_dir, exist_ok=True)

        data_to_save = (all_built_graphs, all_target_graphs)

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