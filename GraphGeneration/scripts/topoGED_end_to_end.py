import numpy as np 
import networkx as nx
import random
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from GraphGeneration.utils.Evaluator import Evaluator
from GraphGeneration.models.temporal_gnn.script.config import args
from load_data import load_data, generate_training_data_cached, generate_validation_data_cached
from GraphGeneration.utils.sampling_edges_utils import sample_edges
from GraphGeneration.utils.graph_construction_utils import compute_reappearance_probabilities, get_node_features, update_degrees
# Models in use
from GraphGeneration.models.model import setupMLP, load_encoder_model

# Import all node embedding methods
from compute_embedding import compute_embedding, node2vec_batch_words
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
        
        # Set up Evaluator
        self.evaluator = Evaluator()
        
        # Some default file path
        self.file_visualization_path = "GraphGeneration/scripts/Visualize"
        self.saved_input = os.path.abspath(f'data/input/cached/{args.dataset}/saved_data')
        
        # Current target snapshot we want to predict
        self.current_target_snapshot = 2
        
        # Load the global encoder & decoder model
        self.encoder_model, input_dim = load_encoder_model(args, device=device, node2vec_dimensions=args.nfeat + node2vec_batch_words, hidden_dim=64)
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
        val_end = int(0.9 * self.num_snapshots)

        # Assign snapshots
        self.training_graphs = [self.target_graphs[i][-1] for i in range(self.train_end)]
        self.validation_graphs = [self.target_graphs[i][-1] for i in range(self.train_end, val_end)]
        self.test_graphs = [self.target_graphs[i][-1] for i in range(val_end, self.num_snapshots)]

    # ======================= TRAIN LINK PREDICTION MODEL =======================
    def train_multi_head(self, edge_type, X_train, y_train, X_val=None, y_val=None, lr=1e-3, epochs=250, batch_size=64, top_k=0):
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
            for snapshot in range(1, len(self.training_graphs)):
                curr_X = [x.cpu().detach().numpy() if torch.is_tensor(x) else x for x in X_train[snapshot]]
                curr_X = np.array(curr_X)
                curr_y = np.array(y_train[snapshot])

                X_train_curr, y_train_curr = shuffle(curr_X, curr_y, random_state=self.seed)
                print('y_train_curr:', y_train_curr)
                temp_X_train = torch.tensor(X_train_curr, dtype=torch.float32).to(device)
                temp_y_train = torch.tensor(y_train_curr, dtype=torch.float32).to(device)
                train_loader = DataLoader(TensorDataset(temp_X_train, temp_y_train), batch_size=batch_size, shuffle=False)
                for (x, y) in train_loader:
                    optimizer.zero_grad()
                    node_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=self.training_graphs[:snapshot], encoder_model=self.encoder_model)
                    
                    # Get current embeddings
                    src_nodes = [int(n) for n in x[:, 0].tolist()]                
                    dst_nodes = [int(n) for n in x[:, 1].tolist()]
                    src_embed = torch.stack([node_embeddings[n] for n in src_nodes])
                    dst_embed = torch.stack([node_embeddings[n] for n in dst_nodes])

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
                    # if (epoch + 1) % 10 == 0:  # Run graphlet loss every 10 epochs
                    #     pred_graph = self.build_graph_edge_type_with_edgebank(target_snapshot_graph_description, edge_type=edge_type, graph_num=self.current_target_snapshot, 
                    #                                                           self.current_target_count_old_nodes=self.current_target_count_old_nodes, top_k=top_k, thresholds=self.thresholds, embeddings=node_embeddings, 
                    #                                                           edgebank=self.all_edgebanks[i], mlp=self.link_prediction_decoder)
                    #     pred_graph = pred_graph[-1]
                    #     old_nodes = set().union(*[g.nodes() for g in self.training_graphs[:-1]])
                        
                    #     # Get all previous edges
                    #     previous_edges = set()
                    #     for g in self.training_graphs[:-1]:
                    #         previous_edges.update(g.edges())
                            
                    #     if edge_type == "o-o-bank":
                    #         # Get edges of the current graph
                    #         current_graph = self.training_graphs[-1]
                    #         current_edges = set(current_graph.edges())

                    #         # Get only new edges (those not in previous self.training_graphs)
                    #         oo_bank = current_edges & previous_edges
                    #         target_graph = nx.DiGraph()
                    #         target_graph.add_edges_from(oo_bank)
                    #     elif edge_type == "o-o-nobank":
                    #         current_graph = self.training_graphs[-1]
                    #         current_edges = set(current_graph.edges())

                    #         # Get only new edges (those not in previous self.training_graphs)
                    #         new_edges = current_edges - previous_edges
                    #         oo_nobank = set()
                    #         for u, v in new_edges:
                    #             if u in old_nodes and v in old_nodes:
                    #                 oo_nobank.add((u, v))
                    #         target_graph = nx.DiGraph()
                    #         target_graph.add_edges_from(oo_nobank)
                
                    #     pred_kernel, true_kernel, distance = self.evaluator.evaluateOrca(pred_graph, target_graph)
                    #     graphlet_loss = graphlet_loss_fn(to_tensor(pred_kernel, device).unsqueeze(0), to_tensor(true_kernel, device).unsqueeze(0))
                    # else:
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
                    node_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=self.training_graphs[:snapshot], encoder_model=self.encoder_model)
                    
                    # Get current embeddings
                    src_nodes = [int(n) for n in x[:, 0].tolist()]                
                    dst_nodes = [int(n) for n in x[:, 1].tolist()]
                    src_embed = torch.stack([node_embeddings[n] for n in src_nodes])
                    dst_embed = torch.stack([node_embeddings[n] for n in dst_nodes])

                    if src_embed.dim() == 1:
                        src_embed = src_embed.unsqueeze(1)  
                    if dst_embed.dim() == 1:
                        dst_embed = dst_embed.unsqueeze(1) 
                        
                    pred_val = self.link_prediction_decoder(src_embed=src_embed, dst_embed=dst_embed, edge_type=edge_type)
                    
                    if preds_val.dim() == 0:
                        preds_val = preds_val.unsqueeze(0)
                    if y.dim() == 0:  # scalar value like torch.tensor(0.5)
                        y = y.unsqueeze(0)  # make it [1]
                    elif y.dim() == 2 and y.size(1) == 1:  # shape [batch_size, 1]
                        y = y.view(-1)

                    # Calculate the loss and accuracy
                    val_loss = loss_fn(preds_val, y_val).item()
                    if len(np.unique(y_val)) < 2:
                        val_aucroc = float('inf')
                    else:
                        val_aucroc = roc_auc_score(y_val.cpu().numpy(), preds_val.cpu().numpy())  # Calculate scores
                    
                self.link_prediction_decoder.train()
                
                if (epoch + 1) % 100 == 0 or epoch == 0:
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

    def train_models(self, lr=0.001):
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
        old_training_nodes = set().union(*[g.nodes() for g in self.training_graphs]) 
        
        # Prepare training data
        training_sorted_samples, training_new_edges_count = generate_training_data_cached(training_graphs=self.training_graphs, old_nodes=old_nodes, 
                                                all_edgebanks=self.all_edgebanks, MAX_SAMPLES=MAX_SAMPLES, dataset=args.dataset, seed=global_seed, saved_data_file_path=self.saved_input)

        # Prepare validation data
        # We pass all_edgebanks of the training snapshots edgebanks
        validation_sorted_samples, training_new_edges_count = generate_validation_data_cached(training_graphs=self.validation_graphs, old_training_nodes=old_training_nodes, 
                                                all_edgebanks=self.all_edgebanks[self.train_end], MAX_SAMPLES=MAX_SAMPLES, dataset=args.dataset, seed=global_seed, type_data="validation", saved_data_file_path=self.saved_input)
        # Prepare test data
        # We pass all_edgebanks of the training snapshots edgebanks
        test_sorted_samples, training_new_edges_count = generate_validation_data_cached(training_graphs=self.test_graphs, old_training_nodes=old_training_nodes, 
                                                all_edgebanks=self.all_edgebanks[self.train_end], MAX_SAMPLES=MAX_SAMPLES, dataset=args.dataset, seed=global_seed, type_data="test", saved_data_file_path=self.saved_input)
        
        
        # Set up the training, validation, test data       
        print('Data setup')
        
        # Set up the training and validation data
        training_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
        }
        
        validation_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
        }
        
        test_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
        }
           
        # Sort all necessary data
        for flag in ['o-o-bank', 'o-o-nobank']:
            curr_training_X = training_sorted_samples[flag]['X']
            curr_training_y = training_sorted_samples[flag]['y']
            curr_validation_X = validation_sorted_samples[flag]['X']
            curr_validation_y = validation_sorted_samples[flag]['y']
            curr_test_X = test_sorted_samples[flag]['X']
            curr_test_y = test_sorted_samples[flag]['y']

            if len(curr_training_X) == 0 or len(curr_training_y) == 0:
                continue           
            
            # Cast validation to Numpy for sklearn
            curr_validation_X = [x.cpu().detach().numpy() if torch.is_tensor(x) else x for x in curr_validation_X]
            curr_validation_X = np.array(curr_validation_X)
            curr_validation_y = np.array(curr_validation_y)
            
            # Cast test to Numpy for sklearn
            curr_test_X = [x.cpu().detach().numpy() if torch.is_tensor(x) else x for x in curr_test_X]
            curr_test_X = np.array(curr_test_X)
            curr_test_y = np.array(curr_test_y)

            # Shuffle training, validation, test               
            training_samples[flag]['X'] = curr_training_X
            training_samples[flag]['y'] = curr_training_y
            validation_samples[flag]['X'] = curr_validation_X
            validation_samples[flag]['y'] = curr_validation_y
            test_samples[flag]['X'] = curr_test_X
            test_samples[flag]['y'] = curr_test_y
        
        print('Training') 
        for flag in ['o-o-bank', 'o-o-nobank']:
            X_train = training_samples[flag]['X']
            y_train = training_samples[flag]['y']
            X_val = validation_samples[flag]['X']
            y_val = validation_samples[flag]['y']
            X_test = test_samples[flag]['X']
            y_test = test_samples[flag]['y']
    
            if len(X_train) == 0 or len(y_train) == 0:
                print(len(X_train), len(y_train))
                print(f'No samples for edge type: {flag}')
                continue
    
            self.link_prediction_decoder = self.train_multi_head(edge_type=flag, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, 
                                                                 lr=lr, epochs=500, batch_size=64, top_k=training_new_edges_count[flag])
        
        return self.link_prediction_decoder
            
    # ======================= BUILD GRAPH =======================
    def build_accumulating_filtration_sequence_with_edgebank(self):
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
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Get the edgebank up to the current target snapshot
        edgebank = self.all_edgebanks[self.current_target_snapshot]
        current_target_graph_description = self.graph_descriptions[self.current_target_snapshot]
        
        V_total = int(current_target_graph_description[-1][0])
        E_total = int(current_target_graph_description[-1][1])
        W_total = current_target_graph_description[-1][2] 

        # Sample old nodes
        probs = compute_reappearance_probabilities(self.current_target_old_nodes, self.current_target_snapshot)
        node_ids = list(probs.keys())
        weights = list(probs.values())

        old_nodes = list(np.random.choice(node_ids, size=self.current_target_count_old_nodes, replace=False, p=np.array(weights)/np.sum(weights)))  # Makes sure that we select only unique nodes each time

        # Create new node IDs
        if self.current_target_old_nodes:
            max_id = max(self.current_target_old_nodes)
        else:
            max_id = 0

        new_nodes = list(range(max_id + 1, max_id + 1 + self.current_target_count_new_nodes))
        
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
            
        # Assign maximum degrees
        get_node_features(tmp_graph, self.thresholds, current_target_graph_description, old_nodes, new_nodes)  

        # Assign embeddings for all the training_nodes
        curr_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=self.training_graphs, encoder_model=self.sencoder_model)
        
        # Assign zero vector for new nodes
        for new_node in new_nodes:
            curr_embeddings[new_node] = np.zeros(len(curr_embeddings[old_nodes[0]]))
            
        # SAMPLE EDGES STEP
        # Get edges of each type
        oo_bank_edges = []
        oo_nobank_edges = []
        
        # Phase 1: o-o-bank
        oo_bank_edges = sample_edges(src_list=old_nodes, dst_list=old_nodes, count=self.current_target_count_oo, edgebank=edgebank, edges=edges, 
                 tmp_graph=tmp_graph, node_types=node_types, link_prediction_decoder=self.link_prediction_decoder, 
                 curr_embeddings=curr_embeddings, graph_num=self.current_target_snapshot, device=device, edge_type="o-o-bank")
        tmp_graph.add_edges_from(oo_bank_edges)
        update_degrees(tmp_graph)
        new_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=self.training_graphs + [tmp_graph], encoder_model=self.sencoder_model)
        curr_embeddings.update(new_embeddings)  # Recompute old node embeddings

        # Phase 2: o-o-nobank
        oo_nobank_edges = sample_edges(src_list=old_nodes, dst_list=old_nodes, count=self.current_target_count_oon, edges=edges, 
                 tmp_graph=tmp_graph, node_types=node_types, link_prediction_decoder=self.link_prediction_decoder, 
                 curr_embeddings=curr_embeddings, graph_num=self.current_target_snapshot, device=device, edge_type="o-o-nobank")
        tmp_graph.add_edges_from(oo_nobank_edges)
        update_degrees(tmp_graph)
        new_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=self.training_graphs + [tmp_graph], encoder_model=self.encoder_model)
        curr_embeddings.update(new_embeddings)  # Recompute old node embeddings

        # Phase 3: o-n
        on_edges = sample_edges(src_list=old_nodes, dst_list=new_nodes, count=self.current_target_count_on, edges=edges, 
                 tmp_graph=tmp_graph, node_types=node_types, link_prediction_decoder=self.link_prediction_decoder, 
                 curr_embeddings=curr_embeddings, graph_num=self.current_target_snapshot, device=device, edge_type="o-n")
        tmp_graph.add_edges_from(on_edges)
        update_degrees(tmp_graph)
        new_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=self.training_graphs + [tmp_graph], encoder_model=self.encoder_model)
        curr_embeddings.update(new_embeddings)  # Recompute old and new node embeddings

        # Phase 4: n-n
        nn_edges = sample_edges(src_list=new_nodes, dst_list=new_nodes, count=self.current_target_count_nn, edges=edges, 
                 tmp_graph=tmp_graph, node_types=node_types, link_prediction_decoder=self.link_prediction_decoder, 
                 curr_embeddings=curr_embeddings, graph_num=self.current_target_snapshot, device=device, edge_type="n-n")
        tmp_graph.add_edges_from(nn_edges)
        update_degrees(tmp_graph)
        new_embeddings = compute_embedding(embeddingType=args.embeddingType, graphs=self.training_graphs + [tmp_graph], encoder_model=self.encoder_model)
        curr_embeddings.update(new_embeddings)  # Final update  
        
        edge_pool = (oo_bank_edges + oo_nobank_edges + on_edges + nn_edges)
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

        return filtration_graphs

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
            self.link_prediction_decoder = self.train_models(lr=args.lr)
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
            self.current_target_count_oo = self.probabilities[i][2]
            self.current_target_count_nn = self.probabilities[i][3]
            self.current_target_count_on = self.probabilities[i][4]
            self.current_target_count_oon = self.probabilities[i][5]
            
            # Build the filtration sequence using the current parameters
            filtration_sequence = self.build_accumulating_filtration_sequence_with_edgebank()
            
            # Add to the old graphs
            self.old_graphs.append(self.target_graphs[i])
            
if __name__ == '__main__':
    print("INFO: Dataset: {}".format(args.dataset))
    runner = Runner()
    runner.run()

# To run the script
# python GraphGeneration/scripts/topoGED_end_to_end.py --embeddingType=LSTM --dataset=CollegeMsg --nfeat=126