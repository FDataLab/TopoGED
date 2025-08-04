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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from GraphGeneration.utils.Evaluator import Evaluator
from GraphGeneration.models.temporal_gnn.script.config import args
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
        self.seed = global_seed
        self.best_validation_model_auc = 0
        
        # Set up Evaluator
        self.evaluator = Evaluator()
        
        # Some default file path
        self.file_visualization_path = "GraphGeneration/scripts/Visualize"
        self.saved_input = os.path.abspath(f'data/input/cached/{args.dataset}/saved_data')
        common_suffix = f"multiMLP_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embeddingType{args.embeddingType}"
        self.structure_dir = f"GraphGeneration/output/results/structure/{args.dataset}/{common_suffix}"
        self.kernel_dir = f"GraphGeneration/output/results/kernel/{args.dataset}/{common_suffix}"
        self.topER_dir = f"GraphGeneration/output/results/topER/{args.dataset}/{common_suffix}"

        # Current target snapshot we want to predict
        self.current_target_snapshot = 2
        
        # All the edge types
        self.all_edge_types = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
        
        # Load the global encoder & decoder model
        self.encoder_model, input_dim = load_encoder_model(args, device=device, node2vec_dimensions=args.nfeat, hidden_dim=64)
        
        # Check if there is any add-on features we will plug at the end of encoder embedding
        if args.embedding in ['NodeType', 'Position']:
            input_dim += 1
            args.nfeat += 1
             
        self.link_prediction_decoder = setupMLP(embedding_dim=input_dim*2, embedding=args.embedding, mlpEncoding=args.mlpEncoding, embedOld=args.embedOld)
        self.link_prediction_decoder.to(device)
        
        # Load all the snapshot true data 
        self.probabilities, self.graph_descriptions, self.thresholds, self.target_graphs = load_data(args.dataset, args.strategy, args.embedding, args.mlpEncoding, args.embedOld, args.trainingStyle, args.embeddingType)
        
        # Modify the graph ids to 1,2,3,...
        self.target_graphs, _ = modifyGraphIds(self.target_graphs, self.thresholds)

        # Build the edgebanks for construction
        self.all_edgebanks = build_edgebanks_from_start(self.target_graphs)        
        
        # Reshape the graph description
        self.graph_descriptions = [list(zip(graph_description[0::3], graph_description[1::3], graph_description[2::3])) for graph_description in self.graph_descriptions]
        
        # Split training, validation, test graphs
        # Snapshots that we will use for traininng
        # Convert number of snapshots to integer
        self.num_snapshots = len(self.probabilities)
        self.train_end = int(0.8 * self.num_snapshots)
        self.val_end = int(0.9 * self.num_snapshots)

        # Assign snapshots
        self.training_graphs = [self.target_graphs[i][-1] for i in range(self.train_end)]
        self.validation_graphs = [self.target_graphs[i][-1] for i in range(self.train_end, self.val_end)]
        self.test_graphs = [self.target_graphs[i][-1] for i in range(self.val_end, self.num_snapshots)]

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
        
        for i in range(1):
            snapshot = i + len(self.training_graphs) + 1
            self.encoder_model.eval()
            self.link_prediction_decoder.eval()
            with torch.no_grad():
                print("INFO: Validation on snapshot", snapshot)
                
                node_types = { 
                    "old_nodes": set().union(*(graph.nodes() for graph in self.training_graphs)),
                    "new_nodes": set()
                } 
                
                # Prepare current target graph count
                self.current_target_count_old_nodes = self.probabilities[snapshot][0]
                self.current_target_count_new_nodes = self.probabilities[snapshot][1]
                self.current_target_count = {
                    edge_type: self.probabilities[snapshot][j + 2]
                    for j, edge_type in enumerate(self.all_edge_types)
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

                    X_train_curr, curr_y_train = shuffle(curr_X_train, curr_y_train, random_state=self.seed)
                    temp_X_train = torch.tensor(X_train_curr, dtype=torch.float32).to(device)
                    temp_y_train = torch.tensor(curr_y_train, dtype=torch.float32).to(device)
                    train_loader = DataLoader(TensorDataset(temp_X_train, temp_y_train), batch_size=batch_size, shuffle=True)
                    
                    # Training graphs for predicting current snapshot
                    validation_graphs = self.training_graphs
                    
                    for (x, y) in train_loader:
                        node_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=validation_graphs, encoder_model=self.encoder_model, device=device)
                        
                        # Get current embeddings
                        src_nodes = [int(n) for n in x[:, 0].tolist()]                
                        dst_nodes = [int(n) for n in x[:, 1].tolist()]
                        
                        # Add new nodes to the node_types
                        for n in src_nodes:
                            if n not in node_embeddings and flag in ['o-n', 'n-n']:
                                node_types["new_nodes"].add(n)
                                constructing_graph.add_node(n)
                                node_embeddings[n] = torch.zeros(args.nfeat, device=device)
                                
                        for n in dst_nodes:
                            if n not in node_embeddings and flag in ['o-n', 'n-n']:
                                node_types["new_nodes"].add(n)
                                constructing_graph.add_node(n)
                                node_embeddings[n] = torch.zeros(args.nfeat, device=device)
                        
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
                    curr_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=validation_graphs, encoder_model=self.encoder_model, device=device)
                    constructing_graph = get_node_features(constructing_graph.copy(), self.training_graphs, self.thresholds, self.graph_descriptions[snapshot], node_types["old_nodes"], node_types["new_nodes"])
                    sampled_edges = predict_edges(constructing_graph, edge_type=flag, node_types=node_types, edgebank=self.all_edgebanks[snapshot], link_prediction_decoder=self.link_prediction_decoder, 
                                old_node_embeddings=curr_embeddings, top_k=self.current_target_count[flag], graph_num=snapshot, device=device)
                    constructing_graph.add_edges_from(list(sampled_edges))
                    update_degrees(constructing_graph)
                    
                    # Update the training_graphs to involve with the constructing graph
                    if flag == 'o-o-nobank':
                        validation_graphs.append(constructing_graph)
                    else:
                        validation_graphs[-1] = constructing_graph 
                    
                    if len(np.unique(train_labels)) < 2:
                        train_auc.append(0)
                    else:
                        train_auc[flag].append(roc_auc_score(train_labels, train_preds))  # Calculate scores
        
        # Record the Training Loss, AUC 
        current_model_auc = 0 #we take average of all edge types
        
        for flag in self.all_edge_types:
            epochMessage = f"Epoch {epoch+1:02d} | Edge Type: {flag}  | Validation AUCROC {np.mean(train_auc[flag]):.4f}"
            current_model_auc += np.mean(train_auc[flag])
            print(epochMessage)
            with open(rf"{self.file_visualization_path}\{args.dataset}\{args.embeddingType}\multiheadMLP_performance.txt", "a") as f:
                f.write(epochMessage + "\n")
                
        # We check and cache if it has the best auc
        if current_model_auc/4 >= self.best_validation_model_auc:
            self.best_validation_model_auc = current_model_auc
            
            print("INFO: Saving the model...")
            torch.save(self.link_prediction_decoder.state_dict(), self.model_path)
            torch.save(self.encoder_model.state_dict(), self.model_path)
            print("INFO: The model is saved. Done.")
            
    
    def train_multi_head(self, training_samples, validation_samples, epochs=250, batch_size=64, training_new_edges_count=0):
        """
        Train a MultiHeaded MLP Neural Network for use in edge predictions
        
        Args:
            model (MultiheadedMLP): The Multiheaded MLP to train now
            training_samples: The dictionary store the pos, neg edges of each snapshot, using for training
            validation_samples: The dictionary store the pos, neg edges of each snapshot, using for validation
            epochs (int): The number of epochs to train for
            batch_size (int): The batch size to use for the training data
        Returns:
            link_prediction_decoder (Multiheaded MLP): The trained MLP
        """
        lr = args.lr
        self.link_prediction_decoder.train()
        optimizer = torch.optim.Adam(list(self.encoder_model.parameters()) + list(self.link_prediction_decoder.parameters()), lr=lr)
        loss_fn = nn.BCELoss()
        graphlet_loss_fn = GraphletLoss()
        
        # Train
        for epoch in range(epochs):
            train_loss = {
                    'o-o-bank': [],
                    'o-o-nobank': [],
                    'o-n': [],
                    'n-n': [],
                }
            train_auc = {
                    'o-o-bank': [],
                    'o-o-nobank': [],
                    'o-n': [],
                    'n-n': [],
                }
            # For computing AUC Scores
            train_preds = []
            train_labels = []
            
            for snapshot in range(2, 3):
                print("INFO: Training on snapshot", snapshot)
                
                node_types = { 
                    "old_nodes": set().union(*(graph.nodes() for graph in self.training_graphs[max(0, snapshot - 5):snapshot])),
                    "new_nodes": set()
                } 
                
                # Prepare current target graph count
                self.current_target_count_old_nodes = self.probabilities[snapshot][0]
                self.current_target_count_new_nodes = self.probabilities[snapshot][1]
                self.current_target_count = {
                    edge_type: self.probabilities[snapshot][j + 2]
                    for j, edge_type in enumerate(self.all_edge_types)
                }
                
                constructing_graph = nx.DiGraph() # Graph we try to predict
                    
                # Adding old nodes to constructing_graph
                constructing_graph.add_nodes_from(node_types['old_nodes'])
                
                for flag in self.all_edge_types:
                    curr_X_train = training_samples[flag]['X'][snapshot]
                    curr_y_train = training_samples[flag]['y'][snapshot]
                    
                    if len(curr_X_train) == 0 or len(curr_y_train) == 0:
                        print(f'No samples for edge type: {flag}')
                        continue
                    
                    curr_X_train = [x.cpu().detach().numpy() if torch.is_tensor(x) else x for x in curr_X_train]
                    curr_X_train = np.array(curr_X_train)
                    curr_y_train = np.array(curr_y_train)

                    X_train_curr, curr_y_train = shuffle(curr_X_train, curr_y_train, random_state=self.seed)
                    temp_X_train = torch.tensor(X_train_curr, dtype=torch.float32).to(device)
                    temp_y_train = torch.tensor(curr_y_train, dtype=torch.float32).to(device)
                    train_loader = DataLoader(TensorDataset(temp_X_train, temp_y_train), batch_size=batch_size, shuffle=True)
                    
                    # Training graphs for predicting current snapshot
                    training_graphs = self.training_graphs[max(0, snapshot - 5):snapshot]
                    
                    for (x, y) in train_loader:
                        optimizer.zero_grad()
                        node_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=training_graphs, encoder_model=self.encoder_model, device=device)
                        
                        # Get current embeddings
                        src_nodes = [int(n) for n in x[:, 0].tolist()]                
                        dst_nodes = [int(n) for n in x[:, 1].tolist()]
                        
                        any_node = next(iter(node_embeddings))
                        embed_dim = len(node_embeddings[any_node])
                        
                        # Add new nodes to the node_types
                        for n in src_nodes:
                            if n not in node_embeddings and flag in ['o-n', 'n-n']:
                                node_types["new_nodes"].add(n)
                                constructing_graph.add_node(n)
                                node_embeddings[n] = torch.zeros(embed_dim, device=node_embeddings[any_node].device)
                                
                        for n in dst_nodes:
                            if n not in node_embeddings and flag in ['o-n', 'n-n']:
                                node_types["new_nodes"].add(n)
                                constructing_graph.add_node(n)
                                node_embeddings[n] = torch.zeros(embed_dim, device=node_embeddings[any_node].device)
                        
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
                            
                        # Constructing target graph
                        pred_graph, _ = self.build_accumulating_filtration_sequence_with_edgebank(current_target_snapshot=snapshot)
                        pred_graph = pred_graph[-1]
                        pred_kernel, true_kernel, distance = self.evaluator.evaluateOrca(pred_graph, self.training_graphs[snapshot])
                        graphlet_loss = graphlet_loss_fn(to_tensor(pred_kernel, device=device).unsqueeze(0), to_tensor(true_kernel, device=device).unsqueeze(0))
                        
                        loss = 0.5*loss_fn(preds, y) + 0.5*graphlet_loss
                        loss.backward()
                        optimizer.step()
                        train_loss[flag].append(loss.item())
                        
                        # Add to our labels for evaluation
                        train_preds.extend(preds.detach().cpu().numpy())
                        train_labels.extend(y.detach().cpu().numpy())

                    # Assign embeddings for all the training_nodes
                    curr_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=training_graphs, encoder_model=self.encoder_model, device=device)
                    constructing_graph = get_node_features(constructing_graph.copy(), self.training_graphs[:snapshot], self.thresholds, self.graph_descriptions[snapshot], node_types["old_nodes"], node_types["new_nodes"])
                    sampled_edges = predict_edges(constructing_graph, edge_type=flag, node_types=node_types, edgebank=self.all_edgebanks[snapshot], link_prediction_decoder=self.link_prediction_decoder, 
                                old_node_embeddings=curr_embeddings, top_k=self.current_target_count[flag], graph_num=snapshot, device=device)
                    constructing_graph.add_edges_from(list(sampled_edges))
                    update_degrees(constructing_graph)
                    
                    # Update the training_graphs to involve with the constructing graph
                    if flag == 'o-o-nobank':
                        training_graphs.append(constructing_graph)
                    else:
                        training_graphs[-1] = constructing_graph 
                    
                    if len(np.unique(train_labels)) < 2:
                        train_auc.append(0)
                    else:
                        train_auc[flag].append(roc_auc_score(train_labels, train_preds))  # Calculate scores
                        
            # Validation
            self.run_validation(validation_samples=validation_samples, batch_size=batch_size, epoch=epoch)
            
            # Record the Training Loss, AUC 
            for flag in self.all_edge_types:
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    epochMessage = f"Epoch {epoch+1:02d} | Edge Type: {flag} | Train Loss: {np.mean(train_loss[flag]):.4f} | Train AUCROC {np.mean(train_auc[flag]):.4f}"
                    print(epochMessage)
                    with open(rf"{self.file_visualization_path}\{args.dataset}\{args.embeddingType}\multiheadMLP_performance.txt", "a") as f:
                        f.write(epochMessage + "\n")
            

        return self.link_prediction_decoder, self.encoder_model

    def train_models(self):
        """
        Create and train the models used for graph construction, these will be used for later graph construction
        
        Args:
            lr (float): The learning rate to use for the model
            seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
            
        Returns:
            link_prediction_decoder (MLP NN): The trained MLP, either single or multiheaded
        """
        MAX_SAMPLES = 1000000  # 1 Million

        old_training_nodes = set().union(*[g.nodes() for g in self.training_graphs]) 
        
        # Prepare training data
        training_sorted_samples, training_new_edges_count = generate_training_data_cached(training_graphs=self.training_graphs,
                                                all_edgebanks=self.all_edgebanks, MAX_SAMPLES=MAX_SAMPLES, 
                                                dataset=args.dataset, seed=global_seed, 
                                                saved_data_file_path=self.saved_input)

        # Prepare validation data
        # We pass all_edgebanks of the training snapshots edgebanks
        validation_sorted_samples, training_new_edges_count = generate_validation_data_cached(training_graphs=self.validation_graphs, old_training_nodes=old_training_nodes, 
                                                all_edgebanks=self.all_edgebanks[self.train_end], MAX_SAMPLES=MAX_SAMPLES, 
                                                dataset=args.dataset, seed=global_seed, 
                                                type_data="validation", saved_data_file_path=self.saved_input)
        # Prepare test data
        # We pass all_edgebanks of the training snapshots edgebanks
        test_sorted_samples, training_new_edges_count = generate_validation_data_cached(training_graphs=self.test_graphs, old_training_nodes=old_training_nodes, 
                                                all_edgebanks=self.all_edgebanks[self.val_end], MAX_SAMPLES=MAX_SAMPLES, 
                                                dataset=args.dataset, seed=global_seed, 
                                                type_data="test", saved_data_file_path=self.saved_input)
        
        print('Training') 
    
        self.link_prediction_decoder = self.train_multi_head(training_samples=training_sorted_samples, validation_samples=validation_sorted_samples, 
                                                                 epochs=500, batch_size=64, training_new_edges_count=training_new_edges_count)
        
        return self.link_prediction_decoder, self.encoder_model
            
    # ======================= BUILD GRAPH =======================
    def build_accumulating_filtration_sequence_with_edgebank(self, current_target_snapshot):
        """
        Our main driver function to build graphs, takes in various arguments to guide the graph construction
        Specifically, this version uses an MLP to assign edges to two nodes based on the probability of them forming an edge
        But, this version also creates a new MLP before each new graph construction. A process called "continual learning"
        
        Args:
            current_target_graph_description (list): The TopER current_target_graph_description to guide construction of the graph, stores the number of nodes and edges to add to the graph
            thresholds (list): The thresholds for node degrees 'maxDegree' as dicted by TopER
            seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
            
        Returns:
            filtration_graphs (list(nx.DiGraph)): A list of nx Graphs that we built up from our TopER current_target_graph_description
            node_types (dict): A dictionary that stores 'old_nodes' and 'new_nodes' organized into lists
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Get the edgebank up to the current target snapshot
        edgebank = self.all_edgebanks[current_target_snapshot]
        current_target_graph_description = self.graph_descriptions[current_target_snapshot]
        prev_graphs = [graph[-1] for graph in self.target_graphs[max(0, current_target_snapshot - 5):current_target_snapshot]]
        
        V_total = int(current_target_graph_description[-1][0])
        E_total = int(current_target_graph_description[-1][1])
        W_total = current_target_graph_description[-1][2] 

        # Sample old nodes
        probs = compute_reappearance_probabilities(graphs=prev_graphs,
                                                   t_curr=current_target_snapshot)
        node_ids = list(probs.keys())
        weights = list(probs.values())

        old_nodes = list(np.random.choice(node_ids, size=self.current_target_count_old_nodes, replace=False, p=np.array(weights)/np.sum(weights)))  # Makes sure that we select only unique nodes each time

        # Create new node IDs
        current_target_old_nodes = set().union(*[g[-1].nodes() for g in self.target_graphs[:current_target_snapshot]])
        if current_target_old_nodes:
            max_id = max(current_target_old_nodes)
        else:
            max_id = 0

        new_nodes = list(range(max_id + 1, max_id + 1 + self.current_target_count_new_nodes))
        all_nodes = old_nodes + new_nodes

        constructing_graph = nx.DiGraph()  # A graph for computing node embeddings easily
        
        node_types = {
            "old_nodes": old_nodes,
            "new_nodes": new_nodes
        } 
        
        # Assign maximum degrees
        constructing_graph = get_node_features(constructing_graph, prev_graphs, self.thresholds, current_target_graph_description, old_nodes, new_nodes)  

        # Assign embeddings for all the training_nodes
        curr_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=prev_graphs, encoder_model=self.encoder_model, device=device)
        
        # Assign zero vector for new nodes
        for new_node in new_nodes:
            curr_embeddings[new_node] = np.zeros(len(curr_embeddings[old_nodes[0]]))
            
        # SAMPLE EDGES STEP
        # Get edges of each type
        edge_pool = []
        
        # Sample edges 4 phases
        for flag in self.all_edge_types:
            sampled_edges = predict_edges(constructing_graph, edge_type=flag, node_types=node_types, edgebank=edgebank, link_prediction_decoder=self.link_prediction_decoder, 
                                old_node_embeddings=curr_embeddings, top_k=self.current_target_count[flag], graph_num=current_target_snapshot, device=device)
        
            constructing_graph.add_edges_from(sampled_edges)
            update_degrees(constructing_graph)
            new_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=prev_graphs + [constructing_graph], encoder_model=self.encoder_model, device=device)
            curr_embeddings.update(new_embeddings)  # Recompute old node embeddings
        
            edge_pool = edge_pool + sampled_edges
            
        weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
        edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

        # ======== START GRAPH CONSTRUCTION ========
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

        return filtration_graphs, node_types

    # ======================= Evaluate =======================
    def evaluate(self, pred_graph, true_graph, node_types):
        # Evaluate the graph of o-n 
        pred_on_graph = create_on_graph(node_types["new_nodes"], self.current_target_old_nodes, pred_graph.copy())
        true_on_graph = create_on_graph(node_types["new_nodes"], self.current_target_old_nodes, true_graph.copy())
        
        on_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_on_graph, true_on_graph, mode="total")

        with open(rf"{self.file_visualization_path}\{args.dataset}\{args.embeddingType}\kl_results_on.txt", "a") as f:
            f.write(f"{self.current_target_snapshot + 1}, {on_kl_divergence_results:.6f}\n")
            
        # Evaluate the graph of n-n 
        pred_nn_graph = create_nn_graph(node_types["new_nodes"], pred_graph.copy())
        true_nn_graph = create_nn_graph(node_types["new_nodes"], true_graph.copy())

        nn_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_nn_graph, true_nn_graph, mode="total")

        with open(rf"{self.file_visualization_path}\{args.dataset}\{args.embeddingType}\kl_results_nn.txt", "a") as f:
            f.write(f"{self.current_target_snapshot + 1}, {nn_kl_divergence_results:.6f}\n")
            
        # Evaluate the graph of old nodes
        oldG = pred_graph.subgraph(self.current_target_old_nodes).copy()
        target_oldG = true_graph.subgraph(self.current_target_old_nodes).copy()

        # results_edges = self.evaluator.evaluateEdges(pred_graph, true_graph, curr_edgebank_pred, all_edgebanks[i], graph_num=i)
        results_true_structure = self.evaluator.evaluateSingleStructure(target_oldG, graph_num=self.current_target_snapshot)
        results_pred_structure = self.evaluator.evaluateSingleStructure(oldG, graph_num=self.current_target_snapshot)
        pred_kernel, true_kernel, distance = self.evaluator.evaluateOrca(oldG, target_oldG)
        
        # Store all results
        pd.DataFrame([results_true_structure]).to_csv(f"{self.structure_dir}/structure_true.csv", mode='a', header=False, index=False)
        pd.DataFrame([results_pred_structure]).to_csv(f"{self.structure_dir}/structure_pred.csv", mode='a', header=False, index=False)
        pd.DataFrame([pred_kernel]).to_csv(f"{self.kernel_dir}/kernel_pred.csv", mode='a', header=False, index=False)
        pd.DataFrame([true_kernel]).to_csv(f"{self.kernel_dir}/kernel_true.csv", mode='a', header=False, index=False)
        
    def run(self):             
        print("INFO: Dataset: {}".format(args.dataset))
        encoder_model_path = os.path.join(self.saved_input, rf"saved_models/encoder_{args.embeddingType}_{self.seed}")
        decoder_model_path = os.path.join(self.saved_input, rf"saved_data/decoder_MLP_{self.seed}")

        if os.path.exists(encoder_model_path) and os.path.exists(decoder_model_path):
            self.link_prediction_decoder.load_state_dict(torch.load(decoder_model_path, map_location=device))
            self.encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=device))
            
            self.link_prediction_decoder.to(device)
            self.encoder_model.to(device)
            
            self.link_prediction_decoder.eval()
            self.encoder_model.eval()
            print(f"✅ Link Prediction Decoder loaded from: {decoder_model_path}")
            print(f"✅ Ecoder loaded from: {encoder_model_path}")
        else:
            # Train the Decoder and Encoder model
            print('Training the Link Prediction Decoder and Encoder')
            self.link_prediction_decoder, self.encoder_model = self.train_models()
            print('Finished training the Link Prediction Decoder and Encoder; Start Graph Construction')
            
            # saving the trained model
            print("INFO: Saving the model...")
            torch.save(self.link_prediction_decoder.state_dict(), self.model_path)
            torch.save(self.encoder_model.state_dict(), self.model_path)
            print("INFO: The model is saved. Done.")
       
        # Old graphs that we know up to now
        self.old_graphs = [self.target_graphs[0], self.target_graphs[1]]
        
        # To predict snapshot i, we use snapshot 0,...,i-1 to train
        for i in range(2, len(self.probabilities)): 
            print("INFO: >>> Temporal Graph Construction <<<")
            print("INFO: Predict snapshot: ", i)
            print("======================================")

            self.current_target_snapshot = i
            
            # Get all old nodes
            self.current_target_old_nodes = set().union(*[g[-1].nodes() for g in self.old_graphs])
            
            # Get the true count of 4 edges type and number of new, old nodes of the target snapshot
            self.current_target_count_old_nodes = self.probabilities[i][0]
            self.current_target_count_new_nodes = self.probabilities[i][1]
            self.current_target_count = {
                    edge_type: self.probabilities[i][j + 2]
                    for j, edge_type in enumerate(self.all_edge_types)
                }
            
            # Build the filtration sequence using the current parameters
            filtration_sequence, node_types = self.build_accumulating_filtration_sequence_with_edgebank(current_target_snapshot=i)
            
            # Evaluate generated graph
            self.evaluate(pred_graph=filtration_sequence[-1], true_graph=self.target_graphs[i], node_types=node_types)
            
            # Add to the old graphs
            self.old_graphs.append(self.target_graphs[i])
            
if __name__ == '__main__':
    runner = Runner()
    runner.run()

# To run the script
# python GraphGeneration/scripts/topoGED_end_to_end.py --embeddingType=LSTM --dataset=CollegeMsg --nfeat=64