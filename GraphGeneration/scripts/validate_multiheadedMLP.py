# New strat
# Keep the embeddings of old node:
    # One with a dict of {id: embedding}
    # Other with a dict of {degree: [available nodes]}
# Take the old node embeddings we expect to see in this new graph
# Cluster nodes together based on expected degree
# Take the average of their embeddings and use that to compute the expected embedding for new nodes
# Then plug into MLP
# Embed nodes at the end of every graph
# Train a new MLP before starting each new graph
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
from load_data import load_data
from utils.visualizers import Visualizer
# Models in use
from GraphGeneration.models.model import setupMLP, load_encoder_model
from itertools import product

# Import all embedding methods

from compute_embedding import compute_linear_gnn_embeddings, compute_node2vec_embeddings, compute_node_embeddings_GCLSTM, compute_node_embeddings_HTGN, compute_node_embeddings_LSTM, node2vec_dimensions
from process_data import modifyGraphIds, build_edgebanks_from_start, process_starter_graph
from create_sub_graphs import create_nn_graph, create_on_graph, create_onn_with_hops_graph
from torch.utils.data import DataLoader

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

# Dummy encoder_model variable
# encoder_model = {'o-o-bank': load_encoder_model(args, device=device, node2vec_dimensions=node2vec_dimensions),
#                  'o-o-nobank': load_encoder_model(args, device=device, node2vec_dimensions=node2vec_dimensions),
#                  'o-n': load_encoder_model(args, device=device, node2vec_dimensions=node2vec_dimensions),
#                  'n-n': load_encoder_model(args, device=device, node2vec_dimensions=node2vec_dimensions)}
encoder_model = load_encoder_model(args, device=device, node2vec_dimensions=node2vec_dimensions)

# Each of these are (2 * node_embedding_size)
if args.embeddingType in ['Node2Vec', 'LSTM', 'GCLSTM', 'HTGN']:
    embedding_dim = 128
elif args.embeddingType == 'Linear':
    embedding_dim = 8  # Must be same as the number of features

# Create file for visualization
file_visualization_path = "GraphGeneration/scripts/Visualize"
if not os.path.exists(f"{file_visualization_path}/{args.dataset}/{args.embeddingType}"):
    os.makedirs(rf"{file_visualization_path}/{args.dataset}/{args.embeddingType}")
with open(f"{file_visualization_path}/{args.dataset}/{args.embeddingType}/multiheadMLP_performance_{args.seed}.txt", "w") as f:
    f.write("")
with open(f"{file_visualization_path}/{args.dataset}/{args.embeddingType}/kl_results_on_{args.seed}.txt", "w") as f:
    f.write("snapshot, kl-divergence\n")
with open(rf"{file_visualization_path}/{args.dataset}/{args.embeddingType}/kl_results_nn_{args.seed}.txt", "w") as f:
    f.write("snapshot, kl-divergence\n")
with open(rf"{file_visualization_path}/{args.dataset}/{args.embeddingType}/kl_results_oonn_{args.seed}.txt", "w") as f:
    f.write("snapshot, kl-divergence\n")
with open(rf"{file_visualization_path}/{args.dataset}/{args.embeddingType}/kernel_results_pred_oonn_{args.seed}.txt", "w") as f:
    categories = [f"Graphlet{i}" for i in range(1, 22)]
    f.write("snapshot," + ",".join(categories) + "\n")
with open(rf"{file_visualization_path}/{args.dataset}/{args.embeddingType}/kernel_results_true_oonn_{args.seed}.txt", "w") as f:
    categories = [f"Graphlet{i}" for i in range(1, 22)]
    f.write("snapshot," + ",".join(categories) + "\n")
with open(rf"{file_visualization_path}/{args.dataset}/{args.embeddingType}/picking_nodes_on_{args.seed}.txt", "w") as f:
    f.write(f"snapshot, precison_on, recall_on, f1_on\n")
    
def save_result_time(model_name,dataset,time,seed):
    partial_path = "data/output/results/time"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    result_path = f"{partial_path}/time_saving.csv"
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(
            columns=["model", "dataset","time","seed"])
    else:
        result_df = pd.read_csv(result_path)

    result_df = result_df.append({'model': model_name,
                                  'dataset': dataset,
                                  "time": time,
                                  "seed" : seed
                                  }, ignore_index=True)
    result_df.to_csv(result_path, index=False)

# Utility function for CUDA
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)

def to_tensor(x):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32, device=device)
    return x.to(device=device, dtype=torch.float32) if device else x


def generate_negative_edges(G, num_samples, edge_type, edgebank=None):
    """
    For training the MLP, we need some negative edges that did not occur in the graph to predict
    
    Args:
        G (nx.DiGraph): The graph we are trying to generate samples on, we use its structure to check what edges dont exist
        num_samples (int): How many negative samples we want to create (we aim for equal amounts of positive and negative)
        edge_type (string): The type of edge we are attempting to generate negative samples for
        edgebank (dict):  A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        
    Returns:
        list(negatives) (list): A list of negative edges for training the MLP
    """
    all_nodes = list(G.nodes())
    negatives = set()
    
    # Remove if unnecessary
    max_attempts = 250000
    attempts = 0
    print(f'For edge type {edge_type}')
    while len(negatives) < num_samples and attempts < max_attempts:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        
        # Skip if u == v (self-loops not allowed)
        if u == v:
            continue
        
        # Filter edges based on edge_type
        if edge_type == 'o-o-bank':
            # This may stall, so there is a precaution to stop this
            if G.nodes[u]['feat']['type'] == 0 and G.nodes[v]['feat']['type'] == 0 and v in edgebank.get(u, []):
                if not G.has_edge(u, v) and (u, v) not in negatives:
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'o-o-nobank':
            if G.nodes[u]['feat']['type'] == 0 and G.nodes[v]['feat']['type'] == 0 and v not in edgebank.get(u, []):
                if not G.has_edge(u, v) and (u, v) not in negatives:
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'n-n':
            if G.nodes[u]['feat']['type'] == 1 and G.nodes[v]['feat']['type'] == 1:
                if not G.has_edge(u, v) and (u, v) not in negatives:
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'o-n':
            if (G.nodes[u]['feat']['type'], G.nodes[v]['feat']['type']) in [(0, 1), (1, 0)]:
                if not G.has_edge(u, v) and (u, v) not in negatives:
                    negatives.add((u, v))
            else:
                attempts += 1
                    
    negatives = list(negatives)
    print(f"{edge_type}: {len(negatives)}")
    if len(negatives) < num_samples:
        print(f"Only {len(negatives)} unique negative edges found for type {edge_type}, requested {num_samples}")

    return negatives

# ======================= TRAIN MODEL =======================
def train_multi_head(model, edge_type, X_train, y_train, X_val=None, y_val=None, lr=1e-3, epochs=250, batch_size=64):
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
        model (Multiheaded MLP): The trained MLP
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(model.parameters()), lr=lr)
    loss_fn = nn.BCELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # Can choose to use validation split, typically I don't
    if X_val is not None and y_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Train
    for epoch in range(epochs):
        train_loss = 0
        # For computing AUC Scores
        train_preds = []
        train_labels = []
        
        for x, y in train_loader:
            optimizer.zero_grad()
            
            # Get current embeddings
            half = x.shape[1] // 2  
            src_embed = x[:, :half]  
            dst_embed = x[:, half:] 

            if src_embed.dim() == 1:
                src_embed = src_embed.unsqueeze(1)  
            if dst_embed.dim() == 1:
                dst_embed = dst_embed.unsqueeze(1) 
                
            preds = model(src_embed=src_embed, dst_embed=dst_embed, edge_type=edge_type)
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if y.dim() == 0:  # scalar value like torch.tensor(0.5)
                y = y.unsqueeze(0)  # make it [1]

            elif y.dim() == 2 and y.size(1) == 1:  # shape [batch_size, 1]
                y = y.view(-1)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Add to our labels for evaluation
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(y.detach().cpu().numpy())

        if len(np.unique(train_labels)) < 2:
            train_aucroc = float('inf')
        else:
            train_aucroc = roc_auc_score(train_labels, train_preds)  # Calculate scores

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                half = X_val.shape[2] // 2 
                
                src_embed = X_val[:, :, :half] 
                dst_embed = X_val[:, :, half:]  

                if src_embed.dim() == 2:  
                    src_embed = src_embed.unsqueeze(1)
                if dst_embed.dim() == 2:
                    dst_embed = dst_embed.unsqueeze(1)

                preds_val = model(src_embed, dst_embed)
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
                
            model.train()
            if (epoch + 1) % 100 == 0:
                epochMessage = f"Epoch {epoch+1:02d} | Edge Type: {edge_type} | Train Loss: {train_loss:.4f} | Train AUCROC {train_aucroc:.4f} | Val Loss: {val_loss:.4f} | Val AUCROC: {val_aucroc:.4f}"
                print(epochMessage)
                with open(rf"{file_visualization_path}/{args.dataset}/{args.embeddingType}/multiheadMLP_performance.txt", "a") as f:
                    f.write(epochMessage + "\n")
        else:
            if (epoch + 1) % 100 == 0:
                epochMessage = f"Epoch {epoch+1:02d} | Edge Type: {edge_type} | Train Loss: {train_loss:.4f} | Train AUCROC {train_aucroc:.4f}"
                print(epochMessage)
                with open(rf"{file_visualization_path}/{args.dataset}/{args.embeddingType}/multiheadMLP_performance.txt", "a") as f:
                    f.write(epochMessage + "\n")
    
    return model

def train_models(prev_graphs, edgebanks, lr=0.001, seed=42):
    """
    Create and train the models used for graph construction, these will be used for later graph construction
    
    Args:
        prev_graphs (list): A list of networkx graphs that we want to use to train
        edgebanks (list): A list of edgebanks used to verify edge types
        prev_embeddings (dict): The embeddings of previously seen nodes in the previous graphs (currently unused; will be implemented to make code more efficient)
        lr (float): The learning rate to use for the model
        seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
        
    Returns:
        mlp (MLP NN): The trained MLP, either single or multiheaded
    """
    mlp = setupMLP(embedding_dim, strategy=args.strategy, embedding=args.embedding, mlpEncoding=args.mlpEncoding, embedOld=args.embedOld)
    
    MAX_SAMPLES = 1000000  # 1 Million
    embeddings = {}
    degree_clusters = {}
    
    old_nodes = set().union(*[g.nodes() for g in prev_graphs])  # A set of old nodes used to differentiate node types

    sorted_samples = {
        'o-o-bank': {'X': [], 'y': []},
        'o-o-nobank': {'X': [], 'y': []},
        'o-n': {'X': [], 'y': []},
        'n-n': {'X': [], 'y': []},
        }  # A dict to sort embeddings for multiheaded MLP training
    
    # Generate embedding inputs and labels
    for i, graph in enumerate(prev_graphs[1:]):  # Since we go one graph back for predictions
        prev_graph = prev_graphs[i]
        
        # Embeddings depend on our strategy
        if args.embeddingType == 'Node2Vec':
            final_embeddings = compute_node2vec_embeddings(prev_graph)
        elif args.embeddingType == 'Linear':
            final_embeddings = compute_linear_gnn_embeddings(prev_graph)
        elif args.embeddingType == 'LSTM':
            # graph_snapshots = [G_0, G_1, ..., G_T]  # each G must have node['feat']
            final_embeddings = compute_node_embeddings_LSTM(prev_graphs[:i+1], encoder_model)
        elif args.embeddingType == 'GCLSTM':
            final_embeddings = compute_node_embeddings_GCLSTM(prev_graphs[:i+1], encoder_model)
        elif args.embeddingType == 'HTGN':
            final_embeddings = compute_node_embeddings_HTGN(prev_graphs[:i+1], encoder_model)
        else: final_embeddings = {}
        embeddings.update(final_embeddings)  # Update our embeddings to reflect the new node ids
        
        # Update the degree clusters
        for node in prev_graph.nodes():
            degree = prev_graph.nodes[node]['feat']['maxDegree']

            curr_embedding = embeddings[node]
            old_embedding = degree_clusters.get(degree, [])
            
            # Average the embeddings if both exist
            if old_embedding is not None and len(old_embedding) > 0:
                new_embedding = (to_numpy(curr_embedding) + to_numpy(old_embedding)) / 2
            else:
                new_embedding = to_tensor(curr_embedding)
                
            degree_clusters[degree] = new_embedding  # Add the embedding
            
            
        curr_embeddings = {}  # The current embeddings we are working with
           
        # Mimicks how we assign new node ids later
        for node, data in graph.nodes(data=True):   
            if node in embeddings:
                base_embedding = embeddings[node]
            else:
                base_embedding = degree_clusters.get(data['feat']['maxDegree'], [])
                
                # Protects from crashes
                if base_embedding is None or len(base_embedding) == 0:
                    # base_embedding = np.zeros(64) if args.embeddingType == 'Node2Vec' else np.zeros(4)
                    base_embedding = np.zeros(len(embeddings[0]))
                
            # Convert to tensor for concatenation
            base_embedding = to_tensor(base_embedding)

            additional_features = []  # If needed according to arguments

            if 'NodeType' in args.embedding:
                node_type_feat = torch.tensor([data['feat']['type']], dtype=torch.float32).to(device)  # Ensure 1D
                additional_features.append(node_type_feat)

            if 'Position' in args.embedding:
                pos_feat = torch.tensor([math.cos(i + 1)], dtype=torch.float32).to(device)  # Ensure 1D
                additional_features.append(pos_feat)

            if additional_features:
                base_embedding = torch.cat([base_embedding] + additional_features, dim=0).to(device)

            curr_embeddings[node] = base_embedding

        
        num_new_edges_oo = 0
        num_new_edges_oon = 0
        num_new_edges_on = 0
        num_new_edges_nn = 0
        
        # Generate positive labels
        for u, v in graph.edges(data=False):
            try:
                edge_type = 'any'  # Default/fallback type

                # Determine edge type based on node categories and edgebank
                if u in old_nodes and v in old_nodes:
                    if v in edgebanks[i].get(u, []):
                        edge_type = 'o-o-bank'
                        num_new_edges_oo += 1
                    else:
                        edge_type = 'o-o-nobank'
                        num_new_edges_oon += 1
                elif (u in old_nodes and v not in old_nodes) or (u not in old_nodes and v in old_nodes):
                    edge_type = 'o-n'
                    num_new_edges_on += 1
                elif u not in old_nodes and v not in old_nodes:
                    edge_type = 'n-n'
                    num_new_edges_nn += 1

                # Now fetch the embeddings
                try:
                    emb_u = curr_embeddings[u]
                    emb_v = curr_embeddings[v]
                except KeyError as ke:
                    print(f"[KeyError] Missing embedding for node {ke.args[0]} (edge: {u}, {v})")
                    continue
                except Exception as e:
                    print(f"[ERROR] Unexpected error accessing embeddings for edge ({u}, {v}): {e}")
                    continue

                # Store the sample
                sorted_samples[edge_type]['X'].append(torch.cat([emb_u, emb_v], dim=0))
                sorted_samples[edge_type]['y'].append(1)

            except Exception as e:
                print(f"[FATAL] Unexpected failure at outer loop for edge ({u}, {v}): {type(e).__name__} - {e}")
            
        print('Generating negative samples')
            
        # Generate an equal amount of negative labels for each type of edge
        negative_edges_oo = generate_negative_edges(graph, num_new_edges_oo, edge_type='o-o-bank', edgebank=edgebanks[i + 1])
        negative_edges_oon = generate_negative_edges(graph, num_new_edges_oon, edge_type='o-o-nobank', edgebank=edgebanks[i + 1])
        negative_edges_on = generate_negative_edges(graph, num_new_edges_on, edge_type='o-n', edgebank=edgebanks[i + 1])
        negative_edges_nn = generate_negative_edges(graph, num_new_edges_nn, edge_type='n-n', edgebank=edgebanks[i + 1])
        print("Negative edges nn, on:")
        print(len(negative_edges_nn), len(negative_edges_on))
        tmp_samples_oo = [torch.cat([curr_embeddings[u], curr_embeddings[v]], dim=0) for u, v in negative_edges_oo]
        tmp_samples_oon = [torch.cat([curr_embeddings[u], curr_embeddings[v]], dim=0) for u, v in negative_edges_oon]
        tmp_samples_on = [torch.cat([curr_embeddings[u], curr_embeddings[v]], dim=0) for u, v in negative_edges_on]
        tmp_samples_nn = [torch.cat([curr_embeddings[u], curr_embeddings[v]], dim=0) for u, v in negative_edges_nn]
        
        # Add to our samples
        sorted_samples['o-o-bank']['X'].extend(tmp_samples_oo)
        sorted_samples['o-o-bank']['y'].extend([0 for _ in range(len(negative_edges_oo))])
        sorted_samples['o-o-nobank']['X'].extend(tmp_samples_oon)
        sorted_samples['o-o-nobank']['y'].extend([0 for _ in range(len(negative_edges_oon))])
        sorted_samples['o-n']['X'].extend(tmp_samples_on)
        sorted_samples['o-n']['y'].extend([0 for _ in range(len(negative_edges_on))])
        sorted_samples['n-n']['X'].extend(tmp_samples_nn)
        sorted_samples['n-n']['y'].extend([0 for _ in range(len(negative_edges_nn))])
        
        old_nodes.update(graph.nodes())  # Add the old nodes
        
    # If we need to remove some samples to prevent OOM crashes
    total_samples = sum(len(sorted_samples[key]['X']) for key in sorted_samples)
    if total_samples > MAX_SAMPLES:
        print(f"Total samples exceed {MAX_SAMPLES}. Truncating samples.")
        # Randomly sample to reduce to MAX_SAMPLES
        for edge_type in sorted_samples:
            num_samples_to_remove = total_samples - MAX_SAMPLES
            if num_samples_to_remove > 0:
                indices_to_remove = random.sample(range(len(sorted_samples[edge_type]['X'])), num_samples_to_remove)
                sorted_samples[edge_type]['X'] = [x for i, x in enumerate(sorted_samples[edge_type]['X']) if i not in indices_to_remove]
                sorted_samples[edge_type]['y'] = [y for i, y in enumerate(sorted_samples[edge_type]['y']) if i not in indices_to_remove]
                total_samples = sum(len(sorted_samples[key]['X']) for key in sorted_samples)

        
    # Set up the training data (optional validation split)
    VALID_PERCENT = 0.0  # Constant
    flags = []  # Only used in Multiheaded MLP
    
    print('Data setup')
    
    if args.strategy == 'MultiheadedMLP':
        # Set up the training and validation data
        training_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
            'o-n': {'X': [], 'y': []},
            'n-n': {'X': [], 'y': []},
        }
        valid_samples = {
            'o-o-bank': {'X': [], 'y': []},
            'o-o-nobank': {'X': [], 'y': []},
            'o-n': {'X': [], 'y': []},
            'n-n': {'X': [], 'y': []},
        }
        
        if args.embedOld == 'True':
            flags = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
            
        else:
            flags = ['o-o-nobank', 'o-n', 'n-n']
            
        # Sort all necessary data
        for flag in flags:
            
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
    # We train the heads separately
    if args.strategy == 'MultiheadedMLP':  
        for flag in flags:
            X_train = training_samples[flag]['X']
            y_train = training_samples[flag]['y']
            X_val = valid_samples[flag]['X']
            y_val = valid_samples[flag]['y']
    
            if len(X_train) == 0 or len(y_train) == 0:
                print(len(X_train), len(y_train))
                print(f'No samples for edge type: {flag}')
                continue
    
            mlp = train_multi_head(mlp, flag, X_train, y_train, X_val=X_val, y_val=y_val, lr=lr, epochs=500, batch_size=64)
    
    return mlp  # TODO Let this reduce repeating embedding graphs
        
    
def generate_candidates(graph:nx.DiGraph, nodes_1, flag, nodes_2=None, edgebank=None):
    """
    Generate all possible edges that we could add (directed)
    
    Args:
        graph (nx.DiGraph): The current graph that we are constructing
        nodes_1 (list): The set of source node ids
        flag (string): The edge type that we are making candidates for
        nodes_2 (list): The set of destination node ids
        edgebank (dict):  A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
    
    Returns:
        candidates (list): A list of tuples for all possible edges that can be added given the nodes
    """
    candidates = []  # The edges that we could add

    # Different processing depending on available nodes for the edge
    if flag == 'o-n':
        candidates = [(node_1, node_2) for node_1, node_2 in product(nodes_1, nodes_2) if node_1 != node_2 and (node_1, node_2) not in graph.edges() and ((node_1 in edgebank.keys() and node_2 not in edgebank.keys()) or (node_1 not in edgebank.keys() and node_2 in edgebank.keys()))and (graph.degree(node_1) < graph.nodes[node_1]['feat']['maxDegree']) and (graph.degree(node_2) < graph.nodes[node_2]['feat']['maxDegree'])]
    elif flag == 'o-o-nobank':
        candidates = [(node_1, node_2) for node_1, node_2 in product(nodes_1, nodes_1) if node_1 != node_2 and (node_1, node_2) not in graph.edges() and node_2 not in edgebank.get(node_1, []) and (graph.degree(node_1) < graph.nodes[node_1]['feat']['maxDegree']) and (graph.degree(node_2) < graph.nodes[node_2]['feat']['maxDegree'])]
    elif flag == 'o-o-bank':
        candidates = [(node_1, node_2) for node_1, node_2 in product(nodes_1, nodes_1) if node_1 != node_2 and (node_1, node_2) not in graph.edges() and node_2 in edgebank.get(node_1, []) and (graph.degree(node_1) < graph.nodes[node_1]['feat']['maxDegree']) and (graph.degree(node_2) < graph.nodes[node_2]['feat']['maxDegree'])]
    elif flag == 'n-n':
        for u in nodes_1:
            for v in nodes_1:
                if u != v and not graph.has_edge(u, v) and (graph.degree(u) < graph.nodes[u]['feat']['maxDegree']) and (graph.degree(v) < graph.nodes[v]['feat']['maxDegree']):  # Skip edges that already exist in the graph
                    candidates.append((u, v))
    
    return candidates

def predict_edges(graph, edge_type, node_types, edgebank, mlp, embeddings, top_k, graph_num):
    """
    Predict what edges we will see in the graph, this is done by passing the node embeddings into the MLP and selecting the top_k most likely edges
    
    Args:
        graph (nx.DiGraph): The graph that we are currently constructing
        edge_type (string): The current edge type we are predicting edges for
        node_types (dict): A dictionary storing the old nodes and new nodes in ['old_nodes'] and ['new_nodes'] respectively
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        mlp (MLP NN): An MLP that predicts the probability of an edge occurring
        embeddings (dict): The embeddings of all old nodes we have seen up to this point
        top_k (int): How many edges we are going to select
        graph_num (int): Used for assigning a positional encoding onto the node embedding
    
    Returns:
        top_edges (list): The top_k edges that we have decided to add here
    """
    if edge_type == 'o-o-bank' or edge_type == 'o-o-nobank':
        available_nodes = node_types['old_nodes']
        candidate_edges = generate_candidates(graph, nodes_1=available_nodes, nodes_2=None, flag=edge_type, edgebank=edgebank)

    elif edge_type == 'n-n':
        available_nodes = node_types['new_nodes']
        candidate_edges = generate_candidates(graph, nodes_1=available_nodes, nodes_2=None, flag=edge_type, edgebank=edgebank)

    elif edge_type == 'o-n':
        nodes = node_types['old_nodes'] + node_types['new_nodes']  # Since all nodes are valid candidates
        candidate_edges = generate_candidates(graph, nodes_1=nodes, nodes_2=nodes, flag=edge_type, edgebank=edgebank) #TODO kha: check this
    
    # Predict edge probabilities using the MLP
    edge_probs = []
    for u, v in candidate_edges:
        src_embed = embeddings[u]
        dst_embed = embeddings[v]

        # Convert to torch.Tensor if necessary
        if isinstance(src_embed, np.ndarray):
            src_embed = torch.tensor(src_embed, dtype=torch.float32).to(device)
        if isinstance(dst_embed, np.ndarray):
            dst_embed = torch.tensor(dst_embed, dtype=torch.float32).to(device)

        # Add batch dimension if needed
        if src_embed.dim() == 1:
            src_embed = src_embed.unsqueeze(0)
        if dst_embed.dim() == 1:
            dst_embed = dst_embed.unsqueeze(0)

        # Append onto the end
        if 'NodeType' in args.embedding:
            src_type = torch.tensor([[1.0]] if u in node_types['new_nodes'] else [[0.0]], device=device)
            dst_type = torch.tensor([[1.0]] if v in node_types['new_nodes'] else [[0.0]], device=device)
            src_embed = torch.cat([src_embed, src_type], dim=1)
            dst_embed = torch.cat([dst_embed, dst_type], dim=1)

        if 'Position' in args.embedding:
            cos_val = torch.tensor([[math.cos(graph_num)]], dtype=torch.float32, device=device)
            src_embed = torch.cat([src_embed, cos_val], dim=1)
            dst_embed = torch.cat([dst_embed, cos_val], dim=1)
            
        # Predict edge probability

        prob = mlp(src_embed, dst_embed, edge_type)
        
        edge_probs.append((u, v, prob.item()))

    # Sort and select top_k
    edge_probs.sort(key=lambda x: x[2], reverse=True)
    top_edges = [(u, v) for u, v, _ in edge_probs[:top_k]]

    # Likely to be an issue in the early graphs
    if top_k != len(top_edges):
        print(f'[WARNING] There was an incorrect amount of predicted edges for Graph #{graph_num} and edgetype: {edge_type}')
        print(f'[WARNING] There were {len(top_edges)} edges when there was supposed to be {top_k} edges with {len(candidate_edges)} options')

    return top_edges

def compute_reappearance_probabilities(nodes, t_curr, decay_factor=3.0, alpha=1.0, epsilon=1e-8):
    """
    Compute the probability for each node to reappear given how long ago it was seen and its latest degree
    Nodes of higher degree, and nodes seen more recently are preferred
    
    Args:
        nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing probabilities
        t_curr (int): The current graph number we are on, used to compute probabilities
        decay_factor (float): How quickly the recency of a node decays. Higher means that the nodes seen long ago decay slower
        alpha (float): Our decay constant, controls how influential degree is (alpha > 1 means that it prefers degree, alpha < 1 means that it matters less)
        epsilon (float): Prevents having 0 probabilities for a node, and thus prevents numpy errors later on
    
    Returns:
        probs (dict):  A dictionary of {node_id: percent probability} probabilities for each node in nodes
    """
    if not nodes:
        return {}

    max_degree = max(degree for _, (_, degree) in nodes.items())

    probs = {}
    for node_id, (last_seen, degree) in nodes.items():
        recency_score = np.exp(-max(0, t_curr - last_seen) / decay_factor)
        degree_score = (degree / max_degree) ** alpha if max_degree > 0 else epsilon
        raw_score = recency_score * degree_score
        probs[node_id] = max(raw_score, epsilon)  # Apply epsilon floor to avoid exact 0

    # Normalize to make a valid probability distribution
    total = sum(probs.values())
    for node in probs:
        probs[node] /= total

    return probs

        
def get_node_features(graph, thresholds, embedding, old_nodes, new_nodes):
    """
    Assign the maximum degree of a node, either using its last seen degree (if args.oldDegree == True) or randomly giving it one
    
    Args:
        graph (nx.DiGraph): The Graph we are attempting to construct, we assign node features here
        thresholds (list): The thresholds according to TopER, assigns the maximum degree of nodes
        embedding (list): The current TopER graph embedding vector used to figure out how many nodes have a degree
        old_nodes (list): The list of old nodes that are in the graph
        new_nodes (list): The list of new nodes that are in the graph
        
    Returns:
        None
    """
    degree_counts = [embedding[i][0] for i in range(0, len(embedding))]
    
    degree_dict = {thresholds[i]: degree_counts[i] for i in range(len(thresholds))}
    
    degree_assignment = []  # This will store the assigned degrees
    
    for degree, count in degree_dict.items():
        degree_assignment.extend([degree] * count)
        
    random.shuffle(degree_assignment)
    
    if args.oldDegree == 'True':
        for node in old_nodes:
            old_degree = existing_nodes[node][1]

            # Find the smallest degree in degree_assignment ≥ old_degree
            suitable_degrees = [d for d in degree_assignment if d >= old_degree]
            if suitable_degrees:
                assigned_degree = min(suitable_degrees)
            else:
                assigned_degree = degree_assignment.pop()

            if not degree_assignment:
                pass
            else:
                degree_assignment.remove(assigned_degree)
            
            graph.nodes[node]['feat']['currDegree'] = 0
            graph.nodes[node]['feat']['maxDegree'] = assigned_degree
        
        # Give the node a random new degree    
        for node in new_nodes:
            assigned_degree = degree_assignment.pop()

            graph.nodes[node]['feat']['currDegree'] = 0
            graph.nodes[node]['feat']['maxDegree'] = assigned_degree
            
    else:
        for i, node in enumerate(graph.nodes):        
            # Assign features to the node as an attribute
            graph.nodes[node]['feat']['currDegree'] = 0  # Starts at 0
            graph.nodes[node]['feat']['maxDegree'] = degree_assignment[i]
 

def update_degrees(graph: nx.DiGraph):
    """
    After updating the graph, between edge types, update the nodes current degree feature
    
    Args:
        graph (nx.DiGraph): The current graph in construction
        
    Returns:
        None
    """
    for node in graph.nodes(data=False):
        graph.nodes[node]['feat']['currDegree'] = graph.degree(node)
        
     
def update_edgebank(graph, edgebank):
    """
    Update the edgebank based on the current graph
    
    Args:
        graph (nx.Graph): The current graph to update based on
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        
    Returns:
        edgebank (dict): The updated edgebank; updated in place
    """
    for u, v in graph.edges():
        edgebank.setdefault(u, []).append(v)
        
    return edgebank


# ======================= BUILD GRAPH =======================
def build_accumulating_filtration_sequence_with_edgebank(embedding, prev_graphs, graph_num, 
    p_old_nodes, p_new_nodes, E_oo, E_nn, E_on, E_oon, thresholds, embeddings=None, 
    degree_clusters=None, edgebank=None, existing_nodes=None, mlp=None, seed=42):
    """
    Our main driver function to build graphs, takes in various arguments to guide the graph construction
    Specifically, this version uses an MLP to assign edges to two nodes based on the probability of them forming an edge
    But, this version also creates a new MLP before each new graph construction. A process called "continual learning"
    
    Args:
        graphs (list): list of graph from 0 to current
        embedding (list): The TopER embedding to guide construction of the graph, stores the number of nodes and edges to add to the graph
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
        mlp (MLP NN): An MLP that predicts the probability of an edge occurring
        seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
        
    Returns:
        filtration_graphs (list(nx.DiGraph)): A list of nx Graphs that we built up from our TopER embedding
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

    V_total = int(embedding[-1][0])
    E_total = int(embedding[-1][1])
    W_total = embedding[-1][2] 

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
    edge_type_map = {}  # For calculating AUC scores later 
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
    
    get_node_features(tmp_graph, thresholds, embedding, old_nodes, new_nodes)  # Assign maximum degrees

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


    def sample_edges(src_list, dst_list, count, edge_type=None):
        sampled = set()
        attempts = 0

        if args.embedOld == False and edge_type == "o-o-bank" and edgebank is not None:
            for u in src_list:
                if u in edgebank:
                    for v in edgebank[u]:
                        if v in dst_list and u != v and v in edgebank.get(u, []) and (u, v) not in edges:
                            sampled.add((u, v))
                            edge_type_map.setdefault(edge_type, []).append((u, v))
                            edges.add((u, v))
                            if len(sampled) >= count:
                                return list(sampled)

        else:
            if count > 0:
                sampled = predict_edges(tmp_graph, edge_type, node_types, edgebank, mlp, curr_embeddings, top_k=count, graph_num=graph_num)
                
        return list(sampled)

    # Get edges of each type
    oo_bank_edges = []
    oo_nobank_edges = []
    

    oo_bank_edges = sample_edges(old_nodes, old_nodes, count=E_oo, edge_type="o-o-bank")
    tmp_graph.add_edges_from(oo_bank_edges)
    update_degrees(tmp_graph)
    
    oo_nobank_edges = sample_edges(old_nodes, old_nodes, count=E_oon, edge_type="o-o-nobank")
    tmp_graph.add_edges_from(oo_nobank_edges)
    update_degrees(tmp_graph)
    
    on_edges = sample_edges(old_nodes, new_nodes, count=E_on, edge_type="o-n")
    tmp_graph.add_edges_from(on_edges)
    update_degrees(tmp_graph)
    
    nn_edges = sample_edges(new_nodes, new_nodes, count=E_nn, edge_type="n-n")
    tmp_graph.add_edges_from(nn_edges)
    update_degrees(tmp_graph)
    
    
    edge_pool = (oo_bank_edges + oo_nobank_edges + on_edges + nn_edges)
    weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
    edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

    G = nx.DiGraph()
    used_edges = set()
    filtration_graphs = []

    for i, (v_target, e_target, w_target) in enumerate(embedding):
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
    if args.embeddingType == 'Node2Vec':
        final_embeddings = compute_node2vec_embeddings(tmp_graph)
    elif args.embeddingType == 'Linear':
        final_embeddings = compute_linear_gnn_embeddings(tmp_graph)
    elif args.embeddingType == 'LSTM':       
        # graph_snapshots = [G_0, G_1, ..., G_T]  # each G must have node['feat']
        final_embeddings = compute_node_embeddings_LSTM(graphs, encoder_model)
    elif args.embeddingType == 'GCLSTM':
        final_embeddings = compute_node_embeddings_GCLSTM(graphs, encoder_model)
    elif args.embeddingType == 'HTGN':
        final_embeddings = compute_node_embeddings_HTGN(graphs, encoder_model)
    
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
 
def build_oo_graph_with_edgebank(embedding, prev_graphs, graph_num, p_old_nodes, E_oo, E_oon, thresholds, embeddings=None, degree_clusters=None, edgebank=None, existing_nodes=None, mlp=None, seed=global_seed):
    """
    Our main driver function to build graphs, takes in various arguments to guide the graph construction
    Specifically, this version uses an MLP to assign edges to two nodes based on the probability of them forming an edge
    But, this version also creates a new MLP before each new graph construction. A process called "continual learning"
    
    Args:
        embedding (list): The TopER embedding to guide construction of the graph, stores the number of nodes and edges to add to the graph
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
        mlp (MLP NN): An MLP that predicts the probability of an edge occurring
        seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
        
    Returns:
        old_graphs (list(nx.DiGraph)): A list of nx Graphs that we built up from our TopER embedding
    """
    random.seed(seed)
    np.random.seed(seed)


    if existing_nodes is None:
        existing_nodes = {}

    W_total = embedding[-1][2] 

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
    edge_type_map = {}  # For calculating AUC scores later 
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
    
    get_node_features(tmp_graph, thresholds, embedding, old_nodes, [])  # Assign maximum degrees

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


    def sample_edges(src_list, dst_list, count, edge_type=None):
        sampled = set()
        attempts = 0

        if args.embedOld == False and edge_type == "o-o-bank" and edgebank is not None:
            for u in src_list:
                if u in edgebank:
                    for v in edgebank[u]:
                        if v in dst_list and u != v and v in edgebank.get(u, []) and (u, v) not in edges:
                            sampled.add((u, v))
                            edge_type_map.setdefault(edge_type, []).append((u, v))
                            edges.add((u, v))
                            if len(sampled) >= count:
                                return list(sampled)

        else:
            if count > 0:
                sampled = predict_edges(tmp_graph, edge_type, node_types, edgebank, mlp, curr_embeddings, top_k=count, graph_num=graph_num)
                
        return list(sampled)

    # Get edges of each type
    oo_bank_edges = sample_edges(old_nodes, old_nodes, count=E_oo, edge_type="o-o-bank")
    tmp_graph.add_edges_from(oo_bank_edges)
    update_degrees(tmp_graph)
    
    oo_nobank_edges = sample_edges(old_nodes, old_nodes, count=E_oon, edge_type="o-o-nobank")
    tmp_graph.add_edges_from(oo_nobank_edges)
    update_degrees(tmp_graph)
    
    
    edge_pool = (oo_bank_edges + oo_nobank_edges)
    weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
    edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

    G = nx.DiGraph()
    used_edges = set()
    old_graphs = []

    for i, (v_target, e_target, w_target) in enumerate(embedding):
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

        old_graphs.append(G.copy())  
                
    return old_graphs

# # Data Loading and Prep

dataset = args.dataset
my_loader = Loader()
my_evaluator = Evaluator()
my_visualizer = Visualizer(dataset=dataset, task='regression')

# # Construct csv
# run_number = 1
# structure_pred_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}_embeddingType{args.embeddingType}/structure_pred.csv'
# structure_true_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}_embeddingType{args.embeddingType}/structure_true.csv'
# structure_diff_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}_embeddingType{args.embeddingType}/structure_diff.csv'
# kernel_pred_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}_embeddingType{args.embeddingType}/kernel_pred.csv'
# kernel_true_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}_embeddingType{args.embeddingType}/kernel_true.csv'
# edge_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}_embeddingType{args.embeddingType}/edge_analysis.csv'
# topER_file_path = f'GraphGeneration/output/results/topER/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}_embeddingType{args.embeddingType}/toper_diff.csv'
# animation_path = f'GraphGeneration/output/results/animations/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}_embeddingType{args.embeddingType}/pred_vs_true.mp4'

probabilities, features, thresholds, target_graphs = load_data(args.dataset, args.strategy, args.embedding, args.mlpEncoding, args.embedOld, args.trainingStyle, args.embeddingType)
# Initialize list for predicted graphs
pred_graphs = []


start = timeit.default_timer()

# Build the edgebanks for construction
tmp_target_graphs, _ = modifyGraphIds(target_graphs, thresholds)
all_edgebanks = build_edgebanks_from_start(tmp_target_graphs)

print('Starting training')
num_trainers = 2  # The number of graphs used to initialize
trainer_graphs = [tmp_target_graphs[i][-1] for i in range(num_trainers)]
on_graph_history = [] # The history of o-n graph
old_nodes = set().union(*[g.nodes() for g in trainer_graphs]) #Get all old nodes up to current snapshot
encoder_model = load_encoder_model(args, device=device, node2vec_dimensions=node2vec_dimensions, 
                                       HTGN_nodelist=old_nodes)
embeddings, degree_clusters, existing_nodes, curr_edgebank_pred = process_starter_graph(trainer_graphs, thresholds, encoder_model=encoder_model)  # We need a graph to get things going

curr_edgebank_pred = all_edgebanks[num_trainers]  # We start with an edgebank

# Exclusive to retraining the models, helps with args.trainingStyle
TEST_GRAPH_PERCENT = 0.3
split_idx = int((1.0 - TEST_GRAPH_PERCENT) * len(tmp_target_graphs))
mlp_training_graphs = [tmp_target_graphs[i][-1] for i in range(num_trainers)]  # The graphs we will use to train the MLPs, must start with our starter

# Graph Creation

# Iterate through each graph in the dataset
for i in range(num_trainers, len(probabilities)):  # We don't use first two graphs because we need old edges to train on for the MLP, and we need a primer graph
    print('Constructing graph number: ', i + 1)
    
    print(f'Preparing Encoder: {args.embeddingType}')
    old_nodes = set().union(*[g[-1].nodes() for g in tmp_target_graphs[:i+1]]) #Get all old nodes up to current snapshot
    encoder_model = load_encoder_model(args, device=device, node2vec_dimensions=node2vec_dimensions, 
                                       HTGN_nodelist=old_nodes)

    
    # Get the number of resources available for this graph
    count_old = probabilities[i][0]
    count_new = probabilities[i][1]
    p0 = probabilities[i][2]
    p1 = probabilities[i][3]
    p2 = probabilities[i][4]
    p3 = probabilities[i][5]

    # Get the embedding and reshape it
    embedding = features[i]
    embedding = list(zip(embedding[0::3], embedding[1::3], embedding[2::3]))

    print('Training the MLP')
    mlp = train_models(mlp_training_graphs, all_edgebanks, lr=0.001, seed=global_seed)
    print('Finished training the MLP; Beginning Construction')
    print("New nodes: ", count_new)
    # Build the filtration sequence using the current parameters
    filtration_sequence, node_types, existing_nodes, edge_type_map, curr_edgebank_pred, embeddings, degree_clusters = build_accumulating_filtration_sequence_with_edgebank(
        embedding=embedding, prev_graphs=mlp_training_graphs, graph_num=i, p_old_nodes=count_old, p_new_nodes=count_new, 
        E_oo=p0, E_nn=p1, E_on=p2, E_oon=p3, thresholds=thresholds, embeddings=embeddings, 
        degree_clusters=degree_clusters, edgebank=curr_edgebank_pred, existing_nodes=existing_nodes, mlp=mlp,
        seed= global_seed
    )
    
    # Evaluate the graph of o-n 
    pred_on_graph = create_on_graph(node_types["new_nodes"], old_nodes, filtration_sequence[-1].copy())
    # true_on_graph = create_on_graph(node_types["new_nodes"], old_nodes, tmp_target_graphs[i][-1].copy())
    # on_graph_history.append(true_on_graph.copy())
    
    # precison_on, recall_on, f1_on = my_evaluator.calculate_precision_picking_nodes(pred_on_graph, true_on_graph, old_nodes=old_nodes)
    # on_kl_divergence_results = my_evaluator.kl_divergence_graphs(pred_on_graph, true_on_graph, mode="total")

    # with open(rf"{file_visualization_path}\{args.dataset}\{args.embeddingType}\kl_results_on.txt", "a") as f:
    #     f.write(f"{i + 1}, {on_kl_divergence_results:.6f}\n")
    
    # with open(rf"{file_visualization_path}\{args.dataset}\{args.embeddingType}\picking_nodes_on.txt", "a") as f:
    #     f.write(f"{i + 1}, {precison_on:.6f}, {recall_on:.6f}, {f1_on:.6f}\n")
        
    # Evaluate the graph of oo-nn
    pred_oonn_graph = create_onn_with_hops_graph(node_types["new_nodes"], filtration_sequence[-1].copy())
    # true_oonn_graph = create_onn_with_hops_graph(node_types["new_nodes"], tmp_target_graphs[i][-1].copy())
    # try:
    #     pred_kernel, true_kernel, distance = my_evaluator.evaluateOrca(pred_oonn_graph, true_oonn_graph)
    #     on_kl_divergence_results = my_evaluator.kl_divergence_graphs(pred_oonn_graph, true_oonn_graph, mode="total")

    #     with open(rf"{file_visualization_path}\{args.dataset}\{args.embeddingType}\kl_results_oonn.txt", "a") as f:
    #         f.write(f"{i + 1}, {on_kl_divergence_results:.6f}\n")

    #     with open(rf"{file_visualization_path}\{args.dataset}\{args.embeddingType}\kernel_results_pred_oonn.txt", "a") as f:
    #         csv_string = ",".join(f"{x:.3f}" for x in pred_kernel) + ","
    #         f.write(f"{i + 1}," + csv_string + "\n")

    #     with open(rf"{file_visualization_path}\{args.dataset}\{args.embeddingType}\kernel_results_true_oonn.txt", "a") as f:
    #         csv_string = ",".join(f"{x:.3f}" for x in true_kernel) + ","
    #         f.write(f"{i + 1}," + csv_string + "\n")

    # except Exception as e:
    #     print(f"[Error at step {i + 1}] ORCA evaluation failed: {e}")

        
    # Evaluate the graph of n-n 
    pred_nn_graph = create_nn_graph(node_types["new_nodes"], filtration_sequence[-1].copy())
    # true_nn_graph = create_nn_graph(node_types["new_nodes"], tmp_target_graphs[i][-1].copy())

    # nn_kl_divergence_results = my_evaluator.kl_divergence_graphs(pred_nn_graph, true_nn_graph, mode="total")

    # with open(rf"{file_visualization_path}\{args.dataset}\{args.embeddingType}\kl_results_nn.txt", "a") as f:
    #     f.write(f"{i + 1}, {nn_kl_divergence_results:.6f}\n")
        
    # Evaluate the graph of old nodes
    oldG = build_oo_graph_with_edgebank(embedding, prev_graphs=mlp_training_graphs, graph_num=i, p_old_nodes=count_old, E_oo=p0, E_oon=p3, 
                                        thresholds=thresholds, embeddings=embeddings, degree_clusters=degree_clusters, edgebank=curr_edgebank_pred, 
                                        existing_nodes=existing_nodes, mlp=mlp)
    # oldG = oldG[-1]
    # target_oldG = tmp_target_graphs[i][-1].subgraph(tmp_target_graphs[i-1][-1].nodes()).copy()
    # results_diff_structure = my_evaluator.evaluateTwoStructure(oldG, target_oldG, graph_num=i)
    
    # results_edges = my_evaluator.evaluateEdges(filtration_sequence[-1], tmp_target_graphs[i][-1], curr_edgebank_pred, all_edgebanks[i], graph_num=i)
    # results_true_structure = my_evaluator.evaluateSingleStructure(target_oldG, graph_num=i)
    # results_pred_structure = my_evaluator.evaluateSingleStructure(oldG, graph_num=i)
    # pred_kernel, true_kernel, distance = my_evaluator.evaluateOrca(oldG, target_oldG)
    
    # results_diff_structure['Kernel Distance'] = distance  # The kernel distance will be part of our structure evaluation

    
    # Store all results
    # pd.DataFrame([results_diff_structure]).to_csv(structure_diff_file_path, mode='a', header=False, index=False)
    # pd.DataFrame([results_edges]).to_csv(edge_file_path, mode='a', header=False, index=False)
    # pd.DataFrame([results_true_structure]).to_csv(structure_true_file_path, mode='a', header=False, index=False)
    # pd.DataFrame([results_pred_structure]).to_csv(structure_pred_file_path, mode='a', header=False, index=False)
    # pd.DataFrame([pred_kernel]).to_csv(kernel_pred_file_path, mode='a', header=False, index=False)
    # pd.DataFrame([true_kernel]).to_csv(kernel_true_file_path, mode='a', header=False, index=False)

        
    # Visualize predGraph vs trueGraph
    # print(len(oldG.nodes()))
    # print(len(target_oldG))
    # print(f"snapshot {i}")
    # my_visualizer.display_pred_graph_vs_true_graph(oldG[-1], target_oldG)
    
    # Append the last graph from the filtration (assumed to be the "predicted" one)
    pred_graphs.append(filtration_sequence)  # The kernel distance will be part of our structure evaluation
    
    # Add to our training graphs depending on args.trainingStyle
    if args.trainingStyle == 'TrueGraphs':
        mlp_training_graphs.append(tmp_target_graphs[i][-1])
    elif args.trainingStyle == 'PredGraphs':
        mlp_training_graphs.append(pred_graphs[i - 2][-1])
    elif args.trainingStyle == 'MixedGraphs':
        if i < split_idx:
            mlp_training_graphs.append(tmp_target_graphs[i][-1])
        elif i >= split_idx:
            mlp_training_graphs.append(pred_graphs[i - 2][-1])

end = timeit.default_timer()
duration = end - start
save_result_time(args.embeddingType,args.dataset,duration,global_seed)

