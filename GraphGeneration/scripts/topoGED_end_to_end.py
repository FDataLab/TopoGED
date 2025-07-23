import numpy as np 
import networkx as nx
import pandas as pd 
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import timeit
import statistics as stat

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from GraphGeneration.utils.Evaluator import Evaluator
from GraphGeneration.models.temporal_gnn.script.config import args
from load_data import load_data, generate_training_data
from GraphGeneration.utils.casting_type import to_numpy, to_tensor
from GraphGeneration.utils.sampling_edges_utils import sample_edges

# Models in use
from GraphGeneration.models.model import setupMLP, load_encoder_model
from itertools import product

# Import all node embedding methods
from compute_embedding import node2vec_dimensions, compute_embedding
from process_data import modifyGraphIds, build_edgebanks_from_start, process_starter_graph
from create_sub_graphs import create_nn_graph, create_on_graph, create_onn_with_hops_graph
from torch.utils.data import DataLoader

# Import Loss fn
from GraphGeneration.scripts.composite_graphlet_loss_fn import GraphletLoss

# Set seeds
global_seed = args.seed
random.seed(global_seed)
np.random.seed(global_seed) 

# Set up device
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
except Exception:
    device = torch.device("cpu")
    print("Using CPU")  

class Runner(object):
    def __init__(self):      
        # Set up Loader and Evaluator
        self.loader = Loader()
        self.evaluator = Evaluator()
        
        # Some default file path
        self.file_visualization_path = "GraphGeneration/scripts/Visualize"
        
        # Current target snapshot we want to predict
        self.current_target_snapshot = 2
        
        # Load the global encoder & decoder model
        self.encoder_model = load_encoder_model(args, device=device, node2vec_dimensions=node2vec_dimensions)
        self.link_prediction_decoder = setupMLP(args.nfeat, strategy=args.strategy, embedding=args.embedding, mlpEncoding=args.mlpEncoding, embedOld=args.embedOld)
        self.encoder_model.to_device(device)
        self.link_prediction_decoder.to_device(device)
        
        # Load all the snapshot true data 
        data = self.loader(dataset=args.dataset, neg_sample=args.neg_sample)
        self.probabilities, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(args.dataset, args.strategy, args.embedding, args.mlpEncoding, args.embedOld, args.trainingStyle, args.embeddingType)

        # Modify the graph ids to 1,2,3,...
        self.target_graphs, _ = modifyGraphIds(self.target_graphs, self.thresholds)

        # Build the edgebanks for construction
        self.all_edgebanks = build_edgebanks_from_start(self.target_graphs)
        
        # Split training, validation, test graphs
        # Snapshots that we will use for traininng
        # Convert number of snapshots to integer
        self.num_snapshots = len(self.probabilities)
        train_end = int(0.8 * self.num_snapshots)
        val_end = int(0.9 * self.num_snapshots)

        # Assign snapshots
        self.training_graphs = [self.target_graphs[i][-1] for i in range(train_end)]
        self.validation_graphs = [self.target_graphs[i][-1] for i in range(train_end, val_end)]
        self.test_graphs = [self.target_graphs[i][-1] for i in range(val_end, self.num_snapshots)]

        
    # ======================= TRAIN LINK PREDICTION MODEL =======================
    def train_multi_head(self, target_snapshot_graph_description, edge_type, X_train, y_train, X_val=None, y_val=None, lr=1e-3, epochs=250, batch_size=64, top_k=0):
        """
        Train a MultiHeaded MLP Neural Network for use in edge predictions
        
        Args:
            model (MultiheadedMLP): The Multiheaded MLP to train now
            edge_type (string): The type of edge we are training on, dictates what head to train
            X_train (np.array): The training features. A tuple of two node embeddings
            y_train (np.array): The training labels (aiming for a mix of positive and negative labels)
            X_val (np.array): The validation features for training verification
            y_val (np.array): The validation labels for training verification
            lr (float): The learning rate to use for the model
            epochs (int): The number of epochs to train for
            batch_size (int): The batch size to use for the training data
            
        Returns:
            link_prediction_decoder (Multiheaded MLP): The trained MLP
        """
        self.link_prediction_decoder.train()
        optimizer = torch.optim.Adam(list(self.encoder_model.parameters()) + list(self.link_prediction_decoder.parameters()), lr=lr)
        loss_fn = nn.BCELoss()
        graphlet_loss_fn = GraphletLoss()
        
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)

        # Can choose to use validation split, typically I don't
        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        
        # Train
        for epoch in range(epochs):
            train_loss = []
            # For computing AUC Scores
            train_preds = []
            train_labels = []
            
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                node_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=self.training_graphs[:i], encoder_model=self.encoder_model)
                
                # Get current embeddings
                half = x.shape[1] // 2  
                src_embed = node_embeddings[x[:, :half]]
                dst_embed =  node_embeddings[x[:, half:]] 

                if src_embed.dim() == 1:
                    src_embed = src_embed.unsqueeze(1)  
                if dst_embed.dim() == 1:
                    dst_embed = dst_embed.unsqueeze(1) 
                    
                preds = self.link_prediction_decoder(src_embed=src_embed, dst_embed=dst_embed, edge_type=edge_type)
                
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                if y.dim() == 0:  # scalar value like torch.tensor(0.5)
                    y = y.unsqueeze(0)  # make it [1]
                elif y.dim() == 2 and y.size(1) == 1:  # shape [batch_size, 1]
                    y = y.view(-1)
                
                # Construct graph based on models
                if (epoch + 1) % 10 == 0:  # Run graphlet loss every 10 epochs
                    pred_graph = self.build_graph_edge_type_with_edgebank(target_snapshot_graph_description, edge_type=edge_type, graph_num=self.current_target_snapshot, 
                                                                          p_old_nodes=self.current_target_count_old_nodes, top_k=top_k, thresholds=self.thresholds, embeddings=node_embeddings, 
                                                                          edgebank=self.all_edgebanks[i], mlp=self.link_prediction_decoder)
                    pred_graph = pred_graph[-1]
                    old_nodes = set().union(*[g.nodes() for g in self.training_graphs[:-1]])
                    
                    # Get all previous edges
                    previous_edges = set()
                    for g in self.training_graphs[:-1]:
                        previous_edges.update(g.edges())
                        
                    if edge_type == "o-o-bank":
                        # Get edges of the current graph
                        current_graph = self.training_graphs[-1]
                        current_edges = set(current_graph.edges())

                        # Get only new edges (those not in previous self.training_graphs)
                        oo_bank = current_edges & previous_edges
                        target_graph = nx.DiGraph()
                        target_graph.add_edges_from(oo_bank)
                    elif edge_type == "o-o-nobank":
                        current_graph = self.training_graphs[-1]
                        current_edges = set(current_graph.edges())

                        # Get only new edges (those not in previous self.training_graphs)
                        new_edges = current_edges - previous_edges
                        oo_nobank = set()
                        for u, v in new_edges:
                            if u in old_nodes and v in old_nodes:
                                oo_nobank.add((u, v))
                        target_graph = nx.DiGraph()
                        target_graph.add_edges_from(oo_nobank)
            
                    pred_kernel, true_kernel, distance = self.evaluator.evaluateOrca(pred_graph, target_graph)
                    graphlet_loss = graphlet_loss_fn(to_tensor(pred_kernel, device).unsqueeze(0), to_tensor(true_kernel, device).unsqueeze(0))
                else:
                    graphlet_loss = torch.tensor(0.0, device=device)
                    
                loss = 0.5*loss_fn(preds, y) + 0.5*graphlet_loss
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                
                # Add to our labels for evaluation
                train_preds.extend(preds.detach().cpu().numpy())
                train_labels.extend(y.detach().cpu().numpy())

            if len(np.unique(train_labels)) < 2:
                train_aucroc = float('inf')
            else:
                train_aucroc = roc_auc_score(train_labels, train_preds)  # Calculate scores

            if X_val is not None and y_val is not None:
                self.link_prediction_decoder.eval()
                with torch.no_grad():
                    half = X_val.shape[2] // 2 
                    
                    src_embed = X_val[:, :, :half] 
                    dst_embed = X_val[:, :, half:]  

                    if src_embed.dim() == 2:  
                        src_embed = src_embed.unsqueeze(1)
                    if dst_embed.dim() == 2:
                        dst_embed = dst_embed.unsqueeze(1)

                    preds_val = self.link_prediction_decoder(src_embed, dst_embed)
                    if preds_val.dim() == 0:
                        preds_val = preds_val.unsqueeze(0)
                    if y_val.dim() == 0:  # scalar value like torch.tensor(0.5)
                        y_val = y_val.unsqueeze(0)  # make it [1]

                    elif y_val.dim() == 2 and y_val.size(1) == 1:  # shape [batch_size, 1]
                        y_val = y_val.view(-1)

                    # Calculate the loss and accuracy
                    val_loss = loss_fn(preds_val, y_val).item()
                    if len(np.unique(y_val)) < 2:
                        val_aucroc = float('inf')
                    else:
                        val_aucroc = roc_auc_score(y_val.cpu().numpy(), preds_val.cpu().numpy())  # Calculate scores
                    
                self.link_prediction_decoder.train()
                
                if (epoch + 1) % 100 == 0:
                    epochMessage = f"Epoch {epoch+1:02d} | Edge Type: {edge_type} | Train Loss: {np.mean(train_loss):.4f} | Train AUCROC {train_aucroc:.4f} | Val Loss: {val_loss:.4f} | Val AUCROC: {val_aucroc:.4f}"
                    print(epochMessage)
                    with open(rf"{self.file_visualization_path}\{args.dataset}\{args.embeddingType}\multiheadMLP_performance.txt", "a") as f:
                        f.write(epochMessage + "\n")
            else:
                if (epoch + 1) % 100 == 0:
                    epochMessage = f"Epoch {epoch+1:02d} | Edge Type: {edge_type} | Train Loss: {np.mean(train_loss):.4f} | Train AUCROC {train_aucroc:.4f}"
                    print(epochMessage)
                    with open(rf"{self.file_visualization_path}\{args.dataset}\{args.embeddingType}\multiheadMLP_performance.txt", "a") as f:
                        f.write(epochMessage + "\n")
        return self.link_prediction_decoder

    def train_models(self, target_snapshot_graph_description, lr=0.001, seed=1024):
        """
        Create and train the models used for graph construction, these will be used for later graph construction
        
        Args:
            lr (float): The learning rate to use for the model
            seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
            
        Returns:
            link_prediction_decoder (MLP NN): The trained MLP, either single or multiheaded
        """
        MAX_SAMPLES = 1000000  # 1 Million

        old_nodes = set(self.training_graphs[0].nodes())  # A set of old nodes used to differentiate node types

        sorted_samples, new_edges_count = generate_training_data(training_graphs=self.training_graphs, old_nodes=old_nodes, 
                                                all_edgebanks=self.all_edgebanks, MAX_SAMPLES=MAX_SAMPLES, device=device)

        # Set up the training data (optional validation split)
        VALID_PERCENT = 0.0  # Constant
        flags = []  # Only used in Multiheaded MLP
        
        print('Data setup')
        
        # Set up the training and validation data
        training_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
        }
        
        valid_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
        }
           
        # Sort all necessary data
        for flag in ['o-o-bank', 'o-o-nobank']:
            curr_X = sorted_samples[flag]['X']
            curr_y = sorted_samples[flag]['y']
            
            if len(curr_X) == 0 or len(curr_y) == 0:
                continue
            
            # Numpy for sklearn
            curr_X = [x.cpu().detach().numpy() if torch.is_tensor(x) else x for x in curr_X]
            curr_X = np.array(curr_X)
            curr_y = np.array(curr_y)
    
            if VALID_PERCENT > 0.0:
                X_train_curr, X_val_curr, y_train_curr, y_val_curr = train_test_split(curr_X, curr_y, test_size=VALID_PERCENT, random_state=seed, shuffle=True)
            else:
                X_train_curr, y_train_curr = shuffle(curr_X, curr_y, random_state=seed)
                X_val_curr = None
                y_val_curr = None
                
            training_samples[flag]['X'] = X_train_curr
            training_samples[flag]['y'] = y_train_curr
            valid_samples[flag]['X'] = X_val_curr
            valid_samples[flag]['y'] = y_val_curr
        
        print('Training') 
        for flag in flags[:2]:
            X_train = training_samples[flag]['X']
            y_train = training_samples[flag]['y']
            X_val = valid_samples[flag]['X']
            y_val = valid_samples[flag]['y']
    
            if len(X_train) == 0 or len(y_train) == 0:
                print(len(X_train), len(y_train))
                print(f'No samples for edge type: {flag}')
                continue
    
            self.link_prediction_decoder = self.train_multi_head(target_snapshot_graph_description=target_snapshot_graph_description, edge_type=flag, 
                                                                    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, lr=lr, epochs=500, batch_size=64, 
                                                                    top_k=new_edges_count[flag])
        
        return self.link_prediction_decoder
            
    # ======================= BUILD GRAPH =======================
    def build_accumulating_filtration_sequence_with_edgebank(self, current_target_graph_description, prev_graphs, graph_num, 
                                    p_old_nodes, p_new_nodes, E_oo, E_nn, E_on, E_oon, thresholds, embeddings=None, 
                                    degree_clusters=None, edgebank=None, existing_nodes=None, seed=42):
        """
        Our main driver function to build graphs, takes in various arguments to guide the graph construction
        Specifically, this version uses an MLP to assign edges to two nodes based on the probability of them forming an edge
        But, this version also creates a new MLP before each new graph construction. A process called "continual learning"
        
        Args:
            graphs (list): list of graph from 0 to current
            current_target_graph_description (list): The TopER current_target_graph_description to guide construction of the graph, stores the number of nodes and edges to add to the graph
            graph_num (int): The current graph number we are on
            p_old_nodes (int): The number of old nodes that we are going to see in this graph
            p_new_nodes (int): The number of new nodes that we are going to see in this graph
            E_oo (int): The number of edges type 'oo' to add (old edges from the edgebank)
            E_nn (int): The number of edges type 'nn' to add (new edges that involves two new nodes)
            E_on (int): The number of edges type 'on' to add (new edges between one new node and one old node (either direction))
            E_oon (int): The number of edges type 'oon' to add (new edges between two old nodes that was not in the edgebank)
            thresholds (list): The thresholds for node degrees 'maxDegree' as dicted by TopER
            embeddings (dict): The embeddings of all old nodes we have seen up to this point
            degree_clusters (dict): A dictionary of {'degree': [created_embedding]} that we use to assign the embeddings for new nodes
            edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
            existing_nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing reappearance probabilities
            seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
            
        Returns:
            filtration_graphs (list(nx.DiGraph)): A list of nx Graphs that we built up from our TopER current_target_graph_description
            node_types (dict): A dictionary that stores 'old_nodes' and 'new_nodes' organized into lists
            existing_nodes (dict): The updated version of existing nodes passed into the function
            edge_type_map (dict): A dictionary that sorts the types of edges for later analysis
            edgebank (dict): The updated edgebank given the newly constructed graphs
            embeddings (dict): Our newly updated embeddings based on the constructed graph
            degree_clusters (dict): Our newly updated degree clusters
        """
        random.seed(seed)
        np.random.seed(seed)


        if existing_nodes is None:
            existing_nodes = {}

        V_total = int(current_target_graph_description[-1][0])
        E_total = int(current_target_graph_description[-1][1])
        W_total = current_target_graph_description[-1][2] 

        # Sample old nodes
        probs = compute_reappearance_probabilities(existing_nodes, graph_num)
        node_ids = list(probs.keys())
        weights = list(probs.values())

        if graph_num > 0:
            old_nodes = list(np.random.choice(node_ids, size=p_old_nodes, replace=False, p=np.array(weights)/np.sum(weights)))  # Makes sure that we select only unique nodes each time
        else:
            old_nodes = []
            
        # Create new node IDs
        if existing_nodes:
            max_id = max(existing_nodes.keys())
        else:
            max_id = 0

        new_nodes = list(range(max_id + 1, max_id + 1 + p_new_nodes))
        
        all_nodes = old_nodes + new_nodes

        edges = set()
        tmp_graph = nx.DiGraph()  # A graph for computing node embeddings easily
        
        node_types = {
            "old_nodes": old_nodes,
            "new_nodes": new_nodes
        } 
        
        # Add the nodes to the graph
        for node in old_nodes:
            tmp_graph.add_node(node)
            feature_dict_old = {'id': node, 'type': 0}  
            tmp_graph.nodes[node]['feat'] = feature_dict_old
        for node in new_nodes:
            tmp_graph.add_node(node)
            feature_dict_new = {'id': node, 'type': 1}  
            tmp_graph.nodes[node]['feat'] = feature_dict_new
        
        get_node_features(tmp_graph, thresholds, current_target_graph_description, old_nodes, new_nodes)  # Assign maximum degrees

        curr_embeddings = {}
        for node, data in tmp_graph.nodes(data=True):
            if node in embeddings:
                curr_embeddings[node] = embeddings[node]
            else:
                new_embedding = degree_clusters.get(data['feat']['maxDegree'], [])
                
                # Protects from crashes
                if new_embedding is None or len(new_embedding) == 0:
                    new_embedding = np.zeros(64) if args.embeddingType == 'Node2Vec' else np.zeros(4)

                curr_embeddings[node] = new_embedding

        # Get edges of each type
        oo_bank_edges = []
        oo_nobank_edges = []
        
        # Phase 1: o-o-bank
        oo_bank_edges = sample_edges(src_list=old_nodes, dst_list=old_nodes, count=E_oo, edgebank=edgebank, edges=edges, 
                 tmp_graph=tmp_graph, node_types=node_types, link_prediction_decoder=self.link_prediction_decoder, 
                 curr_embeddings=curr_embeddings, graph_num=self.current_target_snapshot, device=device, edge_type="o-o-bank")
        tmp_graph.add_edges_from(oo_bank_edges)
        update_degrees(tmp_graph)
        new_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=prev_graphs + [tmp_graph], encoder_model=self.sencoder_model)
        embeddings.update(new_embeddings)  # Recompute old node embeddings

        # Phase 2: o-o-nobank
        oo_nobank_edges = sample_edges(src_list=old_nodes, dst_list=old_nodes, count=E_oon, edgebank=edgebank, edges=edges, 
                 tmp_graph=tmp_graph, node_types=node_types, link_prediction_decoder=self.link_prediction_decoder, 
                 curr_embeddings=curr_embeddings, graph_num=self.current_target_snapshot, device=device, edge_type="o-o-nobank")
        tmp_graph.add_edges_from(oo_nobank_edges)
        update_degrees(tmp_graph)
        new_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=prev_graphs + [tmp_graph], encoder_model=self.encoder_model)
        embeddings.update(new_embeddings)  # Recompute old node embeddings

        # Phase 3: o-n
        on_edges = sample_edges(src_list=old_nodes, dst_list=new_nodes, count=E_on, edgebank=edgebank, edges=edges, 
                 tmp_graph=tmp_graph, node_types=node_types, link_prediction_decoder=self.link_prediction_decoder, 
                 curr_embeddings=curr_embeddings, graph_num=self.current_target_snapshot, device=device, edge_type="o-n")
        tmp_graph.add_edges_from(on_edges)
        update_degrees(tmp_graph)
        new_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=prev_graphs + [tmp_graph], encoder_model=self.encoder_model)
        embeddings.update(new_embeddings)  # Recompute old and new node embeddings

        # Phase 4: n-n
        nn_edges = sample_edges(src_list=new_nodes, dst_list=new_nodes, count=E_nn, edgebank=edgebank, edges=edges, 
                 tmp_graph=tmp_graph, node_types=node_types, link_prediction_decoder=self.link_prediction_decoder, 
                 curr_embeddings=curr_embeddings, graph_num=self.current_target_snapshot, device=device, edge_type="n-n")
        tmp_graph.add_edges_from(nn_edges)
        update_degrees(tmp_graph)
        new_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=prev_graphs + [tmp_graph], encoder_model=self.encoder_model)
        embeddings.update(new_embeddings)  # Final update  
        
        edge_pool = (oo_bank_edges + oo_nobank_edges + on_edges + nn_edges)
        weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
        edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

        G = nx.DiGraph()
        used_edges = set()
        filtration_graphs = []

        for i, (v_target, e_target, w_target) in enumerate(current_target_graph_description):
            v_target = int(v_target)
            e_target = int(e_target)

            current_nodes = set(all_nodes[:v_target])
            G.add_nodes_from(current_nodes)

            available_edges = [
                (u, v) for (u, v) in edge_pool
                if u in current_nodes and v in current_nodes and (u, v) not in used_edges
            ]

            needed = e_target - G.number_of_edges()
            selected_edges = available_edges[:needed]

            for (u, v) in selected_edges:
                G.add_edge(u, v, weight=edge_weight_map[(u, v)])
                used_edges.add((u, v))

            filtration_graphs.append(G.copy())
        
        # Update existing nodes for the format
        for node in G.nodes(data=False):
            if node in new_nodes:
                existing_nodes[node] = (graph_num, G.degree(node))
                
        edgebank = update_edgebank(filtration_graphs[-1], edgebank)
        graphs = prev_graphs + [tmp_graph]
        # Embeddings depend on our strategy
        final_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=graphs, encoder_model=self.encoder_model)
        
        embeddings.update(final_embeddings)  # Blindly overwrites the previously existing embeddings
        
        for node in tmp_graph.nodes():
            degree = tmp_graph.nodes[node]['feat']['maxDegree']
            
            curr_embedding = embeddings[node]
            old_embedding = degree_clusters.get(degree, [])
            
            # Average the embeddings if both exist
            if old_embedding is not None and len(old_embedding) > 0:
                new_embedding = (to_numpy(curr_embedding) + to_numpy(old_embedding)) / 2
            else:
                new_embedding = curr_embedding
                
            degree_clusters[degree] = new_embedding  # Add the embedding

        return filtration_graphs, node_types, existing_nodes, edge_type_map, edgebank, embeddings, degree_clusters

    def build_graph_edge_type_with_edgebank(self, current_target_graph_description, edge_type, graph_num, p_old_nodes, top_k, thresholds, embeddings=None, edgebank=None, existing_nodes=None, seed=42):
        """
        Our main driver function to build graphs, takes in various arguments to guide the graph construction
        Specifically, this version uses an MLP to assign edges to two nodes based on the probability of them forming an edge
        But, this version also creates a new MLP before each new graph construction. A process called "continual learning"
        
        Args:
            embedding (list): The TopER embedding to guide construction of the graph, stores the number of nodes and edges to add to the graph
            graph_num (int): The current graph number we are on
            thresholds (list): The thresholds for node degrees 'maxDegree' as dicted by TopER
            embeddings (dict): The embeddings of all old nodes we have seen up to this point
            degree_clusters (dict): A dictionary of {'degree': [created_embedding]} that we use to assign the embeddings for new nodes
            edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
            existing_nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing reappearance probabilities
            mlp (MLP NN): An MLP that predicts the probability of an edge occurring
            seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
            
        Returns:
            old_graphs (list(nx.DiGraph)): A list of nx Graphs that we built up from our TopER embedding
        """
        random.seed(seed)
        np.random.seed(seed)


        if existing_nodes is None:
            existing_nodes = {}

        W_total = current_target_graph_description[-1][2] 

        # Sample old nodes
        probs = compute_reappearance_probabilities(existing_nodes, graph_num)
        node_ids = list(probs.keys())
        weights = list(probs.values())

        if graph_num > 0:
            old_nodes = list(np.random.choice(node_ids, size=p_old_nodes, replace=False, p=np.array(weights)/np.sum(weights)))  # Makes sure that we select only unique nodes each time
        else:
            old_nodes = []
            
        all_nodes = old_nodes

        edges = set()
        tmp_graph = nx.DiGraph()  # A graph for computing node embeddings easily
        
        node_types = {
            "old_nodes": old_nodes,
            "new_nodes": []
        } 
        
        
        # Add the nodes to the graph
        for node in old_nodes:
            tmp_graph.add_node(node)
            feature_dict_old = {'id': node, 'type': 0}  
            tmp_graph.nodes[node]['feat'] = feature_dict_old
        
        get_node_features(tmp_graph, thresholds, current_target_graph_description, old_nodes, [])  # Assign maximum degrees

        curr_embeddings = {}
        for node, data in tmp_graph.nodes(data=True):
            if node in embeddings:
                curr_embeddings[node] = embeddings[node]
            else:
                new_embedding = np.zeros(64) if args.embeddingType == 'Node2Vec' else np.zeros(4)
                curr_embeddings[node] = new_embedding

        # Get edges of each type
        bank_edges = sample_edges(src_list=old_nodes, dst_list=old_nodes, count=top_k, edgebank=edgebank, edges=edges, 
                 tmp_graph=tmp_graph, node_types=node_types, link_prediction_decoder=self.link_prediction_decoder, 
                 curr_embeddings=curr_embeddings, graph_num=self.current_target_snapshot, device=device, edge_type=edge_type)
        tmp_graph.add_edges_from(bank_edges)
        
        edge_pool = (bank_edges)
        weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
        edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

        G = nx.DiGraph()
        used_edges = set()
        edge_type_graphs = []

        for i, (v_target, e_target, w_target) in enumerate(current_target_graph_description):
            v_target = int(v_target)
            e_target = int(e_target)

            current_nodes = set(all_nodes[:v_target])
            G.add_nodes_from(current_nodes)

            available_edges = [
                (u, v) for (u, v) in edge_pool
                if u in current_nodes and v in current_nodes and (u, v) not in used_edges
            ]

            needed = e_target - G.number_of_edges()
            selected_edges = available_edges[:needed]

            for (u, v) in selected_edges:
                G.add_edge(u, v, weight=edge_weight_map[(u, v)])
                used_edges.add((u, v))

            edge_type_graphs.append(G.copy())  
                    
        return edge_type_graphs

    def run(self): 
        # To predict snapshot i, we use snapshot 0,...,i-1 to train
        for i in range(2, len(self.probabilities)): 
            print("INFO: >>> Temporal Graph Construction <<<")
            print("INFO: Dataset: {}".format(args.dataset))
            print("INFO: Predict snapshot: ", i)
            print("INFO: Args: ", args)
            print("======================================")
            print("INFO: Encoder: {}".format(args.embeddingType))
            print('Constructing graph number: ', i)

            self.current_target_snapshot = i
            
            # Get all old nodes up to snapshot i - 1
            self.current_target_old_nodes = set().union(*[g[-1].nodes() for g in self.target_graphs[:i]]) 

            # Get the true count of 4 edges type and number of new, old nodes of the target snapshot
            self.current_target_count_old_nodes = self.probabilities[i][0]
            self.current_target_count_new_nodes = self.probabilities[i][1]
            self.current_target_count_oo = self.probabilities[i][2]
            self.current_target_count_nn = self.probabilities[i][3]
            self.current_target_count_on = self.probabilities[i][4]
            self.current_target_count_oon = self.probabilities[i][5]
        
            # Get and reshape the graph description
            graph_description = self.graph_descriptions[i]
            graph_description = list(zip(graph_description[0::3], graph_description[1::3], graph_description[2::3]))
        
            print('Training the Link Prediction Decoder')
            self.link_prediction_decoder = self.train_models(graph_description, lr=args.lr, seed=global_seed)
            print('Finished training the Link Prediction Decoder; Start Graph Construction')
            
            # Add to our training graphs depending on args.trainingStyle
            self.training_graphs.append(self.target_graphs[i][-1])
            
if __name__ == '__main__':
    print("INFO: Dataset: {}".format(args.dataset))
    runner = Runner()
    runner.run()
    