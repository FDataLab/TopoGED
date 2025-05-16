# New strat
# Keep the embeddings of old node:
    # One with a dict of {id: embedding}
    # Other with a dict of {degree: [available nodes]}
# Take the old node embeddings we expect to see in this new graph
# Cluster nodes together based on expected degree
# Take the average of their embeddings and use that to compute the expected embedding for new nodes
# Then plug into MLP
# Embed GCN at the end of every graph

from collections import defaultdict
import numpy as np 
import networkx as nx
import pandas as pd 
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import copy
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from GraphGeneration.utils.Evaluator import Evaluator


# Models in use
from GraphGeneration.models.MultiHeadedEdgePredictor import MultiHeadedEdgePredictor
from GraphGeneration.models.EdgePredictor import EdgePredictorMLP
from GraphGeneration.models.GCNEmbedder import GCNEmbedder

from torch_geometric.utils import from_networkx
from itertools import product

# Import all embedding methods
from utils.embedding_methods.degree import EmbedDegree


# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkcindicator', 'networkiconomi', 'networkdgd'])
parser.add_argument("--strategy", type=str, required=True, choices=['MultiheadedMLP', 'SingleMLP'], help="The type of MLP NN to use")
parser.add_argument("--embedding", type=str, required=True, choices=['Position', 'NodeType', 'Position+NodeType', 'None'], help="Allows appending positional encodings or an integer node type onto the end of the embeddings")
parser.add_argument("--mlpEncoding", type=str, required=True, choices=['Concat', 'Product', 'Addition', 'Subtraction'], help="How you want to input node embeddings to the MLP")  # Product and addition lead to potential noise as we use directed graphs
parser.add_argument("--embedOld", type=str, required=True, choices=['True', 'False'], help="If you want to let the MLP predict edge type \'o-o-bank\', otherwise these edges are randomly added")
parser.add_argument("--oldDegree", type=str, required=True, choices=['True', 'False'], help="If you want reappearing nodes to reuse their most recent degree")
#parser.add_argument("--oldLate", type=str, required=True, choices=['True', 'False'])  # Currently unused
parser.add_argument("--trainingStyle", type=str, required=True, choices=['TrueGraphs', 'PredGraphs', 'MixedGraphs'], help="When training the MLP, decides if you use real graphs, predicted graphs (with first real as starter), or real then pred for MLP training")
args = parser.parse_args()


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
                if not G.has_edge(u, v):
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'o-o-nobank':
            if G.nodes[u]['feat']['type'] == 0 and G.nodes[v]['feat']['type'] == 0 and v not in edgebank.get(u, []):
                if not G.has_edge(u, v):
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'n-n':
            if G.nodes[u]['feat']['type'] == 1 and G.nodes[v]['feat']['type'] == 1:
                if not G.has_edge(u, v):
                    negatives.add((u, v))
            else:
                attempts += 1
        elif edge_type == 'o-n':
            if (G.nodes[u]['feat']['type'], G.nodes[v]['feat']['type']) in [(0, 1), (1, 0)]:
                if not G.has_edge(u, v):
                    negatives.add((u, v))
            else:
                attempts += 1
                    
    if attempts >= max_attempts:
        print(f'Max attempts for type {edge_type} reached')
                    
    if len(negatives) < num_samples and len(negatives) > 0:
        deficit = num_samples - len(negatives)
        negatives = list(negatives)
        negatives += random.choices(negatives, k=deficit)                    

    return list(negatives)


def setupMLP():   
    """
    Set up the MLP based on the arguments provided in the command line starter
    
    Args:
        None
    
    Returns:
        None
    """ 
    input_dim = 64  # Starting input dimension (two 32-dim node embeddings)
    
    # Input size changes if we are doing different methods, this keeps it consistent
    if 'Position' in args.embedding:
        input_dim += 2
    if 'NodeType' in args.embedding:
        input_dim += 2
        
    # Set up the MLPs according to arguments
    if args.strategy == 'SingleMLP':
        mlp = EdgePredictorMLP(in_channels=input_dim, hidden_channels=32, input_type=args.mlpEncoding)
        
    elif args.strategy == 'MultiheadedMLP':
        if args.embedOld == 'True':
            flags = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
        else:
            flags = ['o-o-nobank', 'o-n', 'n-n']
        mlp = MultiHeadedEdgePredictor(in_channels=input_dim, hidden_channels=32, edge_types=flags, input_type=args.mlpEncoding)
        
    return mlp
    
    
def train_single_head(model, X_train, y_train, X_val=None, y_val=None, lr=1e-3, epochs=250, batch_size=64):
    """
    Train a single headed MLP neural network
    
    Args:
        model (EdgePredictor): The MLP that we want to train
        X_train (np.array): The training features. A tuple of two node embeddings
        y_train (np.array): The training labels (aiming for a mix of positive and negative labels)
        X_val (np.array): The validation features for training verification
        y_val (np.array): The validation labels for training verification
        lr (float): The learning rate to use for the model
        epochs (int): The number of epochs to train for
        batch_size (int): The batch size to use for the training data
        
    Returns:
        mlp (EdgePredictor): The trained MLP
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # Can choose to use validation split, typically I don't
    if X_val is not None and y_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(epochs):
        train_loss = 0
        # For computing AUC Scores
        train_preds = []
        train_labels = []
        
        for x, y in train_loader:
            optimizer.zero_grad()
            
            # Get current embeddings
            src_embed = x[:, 0]
            dst_embed = x[:, 1]
            
            preds = model(src_embed, dst_embed)
            
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            try:
                # Shape fix: handle common target shape issues
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                elif y.dim() == 2 and y.size(1) == 1:
                    y = y.view(-1)

                loss = loss_fn(preds, y)

            except ValueError as e:
                print(f"[Shape Mismatch Error] preds shape: {preds.shape}, target shape: {y.shape}")
                continue
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Add to our labels for evaluation
            train_preds.extend(preds.detach().numpy())
            train_labels.extend(y.detach().numpy())

        train_aucroc = roc_auc_score(train_labels, train_preds)  # Calculate scores

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                src_embed, dst_embed = X_val[:, 0], X_val[:, 1]
                preds_val = model(src_embed, dst_embed)
                if preds_val.dim() == 0:
                    preds_val = preds_val.unsqueeze(0)
                if y_val.dim() == 0:  # scalar value like torch.tensor(0.5)
                    y_val = y_val.unsqueeze(0)  # make it [1]

                elif y_val.dim() == 2 and y_val.size(1) == 1:  # shape [batch_size, 1]
                    y_val = y_val.view(-1)

                # Calculate the loss and accuracy
                val_loss = loss_fn(preds_val, y_val).item()
                val_aucroc = roc_auc_score(y_val.numpy(), preds_val.numpy())
                
            model.train()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train AUCROC {train_aucroc:.4f} | Val Loss: {val_loss:.4f} | Val AUCROC: {val_aucroc:.4f}")
        else:
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train AUCROC {train_aucroc:.4f}")

    return model


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
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # Can choose to use validation split, typically I don't
    if X_val is not None and y_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

    # Train
    for epoch in range(epochs):
        train_loss = 0
        # For computing AUC Scores
        train_preds = []
        train_labels = []
        
        for x, y in train_loader:
            optimizer.zero_grad()
            
            # Get current embeddings
            src_embed = x[:, 0]
            dst_embed = x[:, 1]
            
            preds = model(src_embed, dst_embed, edge_type)
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
            train_preds.extend(preds.detach().numpy())
            train_labels.extend(y.detach().numpy())

        if len(np.unique(train_labels)) < 2:
            train_aucroc = float('inf')
        else:
            train_aucroc = roc_auc_score(train_labels, train_preds)  # Calculate scores

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                src_embed, dst_embed = X_val[:, 0], X_val[:, 1]
                preds_val = model(src_embed, dst_embed, edge_type)
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
                    val_aucroc = roc_auc_score(y_val.numpy(), preds_val.numpy())  # Calculate scores
                
            model.train()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:02d} | Edge Type: {edge_type} | Train Loss: {train_loss:.4f} | Train AUCROC {train_aucroc:.4f} | Val Loss: {val_loss:.4f} | Val AUCROC: {val_aucroc:.4f}")
        else:
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:02d} | Edge Type: {edge_type} | Train Loss: {train_loss:.4f} | Train AUCROC {train_aucroc:.4f}")

    return model


# TODO Update to use prev_embeddings
def train_models(prev_graphs, edgebanks, prev_embeddings=None, lr=0.001, seed=42):
    """
    Create and train the models used for graph construction, these will be used for later graph construction
    
    Args:
        prev_graphs (list): A list of networkx graphs that we want to use to train
        edgebanks (list): A list of edgebanks used to verify edge types
        prev_embeddings (dict): The embeddings of previously seen nodes in the previous graphs (currently unused; will be implemented to make code more efficient)
        lr (float): The learning rate to use for the model
        seed (int): The seed for reproducibility purposes, controls our randomness in this strategy
        
    Returns:
        gcn (GCN Model): The trained GCN model (verify training strategy)
        mlp (MLP NN): The trained MLP, either single or multiheaded
    """
    gcn = GCNEmbedder(in_channels=4, hidden_channels=16, out_channels=32)
    mlp = setupMLP()
    
    MAX_SAMPLES = 1000000  # 1 Million
    embeddings = {}
    degree_clusters = {}
    
    old_nodes = set(prev_graphs[0].nodes())  # A set of old nodes used to differentiate node types
    sorted_samples = {
        'o-o-bank': {'X': [], 'y': []},
        'o-o-nobank': {'X': [], 'y': []},
        'o-n': {'X': [], 'y': []},
        'n-n': {'X': [], 'y': []},
        }  # A dict to sort embeddings for multiheaded MLP training
    
    # Generate embedding inputs and labels
    for i, graph in enumerate(prev_graphs[1:]):  # Since we go one graph back for predictions
        prev_graph = prev_graphs[i]
        
        # Embed previous graph
        original_nodes = list(prev_graph.nodes())
        id_to_idx = {node_id: idx for idx, node_id in enumerate(original_nodes)}
        idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}

        # Step 2: Convert to PyG data with consistent feature ordering
        pyg_data = from_networkx(prev_graph)

        # Ensure features are ordered according to original_nodes
        pyg_data.x = torch.stack([
            torch.tensor(list(prev_graph.nodes[n]['feat'].values()), dtype=torch.float32)
            for n in original_nodes
        ])

        # Step 3: Pass through GCN
        new_embeddings = gcn(pyg_data.x, pyg_data.edge_index)

        # Step 4: Map embeddings back to original node IDs
        final_embeddings = {
            idx_to_id[i]: new_embeddings[i].detach()
            for i in range(new_embeddings.shape[0])
        }
        
        embeddings.update(final_embeddings)  # Update our embeddings to reflect the new node ids
        
        # Update the degree clusters
        for node in prev_graph.nodes():
            degree = prev_graph.nodes[node]['feat']['maxDegree']

            curr_embedding = embeddings[node]
            old_embedding = degree_clusters.get(degree, [])
            
            # Average the embeddings if both exist
            if old_embedding is not None and len(old_embedding) > 0:
                new_embedding = (np.array(curr_embedding) + np.array(old_embedding)) / 2
            else:
                new_embedding = curr_embedding
                
            degree_clusters[degree] = new_embedding  # Add the embedding
            
            
        curr_embeddings = {}  # The current embeddings we are working with
           
        # Mimicks how we assign new node ids later
        for node, data in graph.nodes(data=True):   
            if node in embeddings:
                base_embedding = embeddings[node]
            else:
                base_embedding = degree_clusters.get(data['feat']['maxDegree'], [])
                
                # Protects from crashes
                if base_embedding is None and len(base_embedding) == 0:
                    base_embedding = [np.zeros(32)]
                
            # Convert to tensor for concatenation
            base_embedding = torch.tensor(base_embedding, dtype=torch.float32)

            additional_features = []  # If needed according to arguments

            if 'NodeType' in args.embedding:
                node_type_feat = torch.tensor([data['feat']['type']], dtype=torch.float32)  # Ensure 1D
                additional_features.append(node_type_feat)

            if 'Position' in args.embedding:
                pos_feat = torch.tensor([math.cos(i + 1)], dtype=torch.float32)  # Ensure 1D
                additional_features.append(pos_feat)

            if additional_features:
                base_embedding = torch.tensor(base_embedding, dtype=torch.float32)
                base_embedding = torch.cat([base_embedding] + additional_features, dim=0)

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
                sorted_samples[edge_type]['X'].append((emb_u, emb_v))
                sorted_samples[edge_type]['y'].append(1)

            except Exception as e:
                print(f"[FATAL] Unexpected failure at outer loop for edge ({u}, {v}): {type(e).__name__} - {e}")
            
        print('Generating negative samples')
            
        # Generate an equal amount of negative labels for each type of edge
        negative_edges_oo = generate_negative_edges(graph, num_new_edges_oo, edge_type='o-o-bank', edgebank=edgebanks[i + 1])
        negative_edges_oon = generate_negative_edges(graph, num_new_edges_oon, edge_type='o-o-nobank', edgebank=edgebanks[i + 1])
        negative_edges_on = generate_negative_edges(graph, num_new_edges_on, edge_type='o-n', edgebank=edgebanks[i + 1])
        negative_edges_nn = generate_negative_edges(graph, num_new_edges_nn, edge_type='n-n', edgebank=edgebanks[i + 1])
        
        tmp_samples_oo = [(curr_embeddings[u], curr_embeddings[v]) for u, v in negative_edges_oo]
        tmp_samples_oon = [(curr_embeddings[u], curr_embeddings[v]) for u, v in negative_edges_oon]
        tmp_samples_on = [(curr_embeddings[u], curr_embeddings[v]) for u, v in negative_edges_on]
        tmp_samples_nn = [(curr_embeddings[u], curr_embeddings[v]) for u, v in negative_edges_nn]
        
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
    
    else:
        # All of our data
        X_all = []
        y_all = []
        
        # Determines if we want to allow predicting old edges and include this in MLP training
        if args.embedOld == 'True':
            X_all.extend(sorted_samples['o-o-bank']['X'])
            y_all.extend(sorted_samples['o-o-bank']['y'])
            
        # Nothing extra to do
        else:
            pass 

        # Add the data
        X_all.extend(sorted_samples['o-o-nobank']['X'])
        y_all.extend(sorted_samples['o-o-nobank']['y'])
        X_all.extend(sorted_samples['o-n']['X'])
        y_all.extend(sorted_samples['o-n']['y'])
        X_all.extend(sorted_samples['n-n']['X'])
        y_all.extend(sorted_samples['n-n']['y'])
            
        # Numpy for sklearn
        X_all = np.array(X_all)
        y_all = np.array(y_all)

        if VALID_PERCENT > 0.0:
            X_train_single, X_val_single, y_train_single, y_val_single = train_test_split(X_all, y_all, test_size=VALID_PERCENT, random_state=seed, shuffle=True)
        else:
            X_train_single, y_train_single = shuffle(X_all, y_all, random_state=seed)
            X_val_single = None
            y_val_single = None

    
    print('Training')
    # We train the heads separately
    if args.strategy == 'MultiheadedMLP':  
        for flag in flags:
            X_train = training_samples[flag]['X']
            y_train = training_samples[flag]['y']
            X_val = valid_samples[flag]['X']
            y_val = valid_samples[flag]['y']
    
            if len(X_train) == 0 or len(y_train) == 0:
                print(f'No samples for edge type: {flag}')
                continue
    
            mlp = train_multi_head(mlp, flag, X_train, y_train, X_val=X_val, y_val=y_val, lr=lr, epochs=500, batch_size=64)

    # We train all together
    else:
        mlp = train_single_head(mlp, X_train_single, y_train_single, X_val=X_val_single, y_val=y_val_single, lr=lr, epochs=1000, batch_size=64)
                        
    return gcn, mlp  # TODO Let this reduce repeating embedding graphs
        
    
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
        candidate_edges = generate_candidates(graph, nodes_1=nodes, nodes_2=nodes, flag=edge_type, edgebank=edgebank)
    
    # Predict edge probabilities using the MLP
    edge_probs = []
    for u, v in candidate_edges:
        src_embed = embeddings[u]
        dst_embed = embeddings[v]

        # Convert to torch.Tensor if necessary
        if isinstance(src_embed, np.ndarray):
            src_embed = torch.tensor(src_embed, dtype=torch.float32)
        if isinstance(dst_embed, np.ndarray):
            dst_embed = torch.tensor(dst_embed, dtype=torch.float32)

        # Add batch dimension if needed
        if src_embed.dim() == 1:
            src_embed = src_embed.unsqueeze(0)
        if dst_embed.dim() == 1:
            dst_embed = dst_embed.unsqueeze(0)

        # Append onto the end
        if 'NodeType' in args.embedding:
            src_type = torch.tensor([[1.0]] if u in node_types['new_nodes'] else [[0.0]])
            dst_type = torch.tensor([[1.0]] if v in node_types['new_nodes'] else [[0.0]])
            src_embed = torch.cat([src_embed, src_type], dim=1)
            dst_embed = torch.cat([dst_embed, dst_type], dim=1)

        if 'Position' in args.embedding:
            cos_val = torch.tensor([[math.cos(graph_num)]], dtype=torch.float32)
            src_embed = torch.cat([src_embed, cos_val], dim=1)
            dst_embed = torch.cat([dst_embed, cos_val], dim=1)
            
        # Predict edge probability
        if args.strategy == 'SingleMLP':
            prob = mlp(src_embed, dst_embed)
        elif args.strategy == 'MultiheadedMLP':
            prob = mlp(src_embed, dst_embed, edge_type)
        
        edge_probs.append((u, v, prob.item()))

    # Sort and select top_k
    edge_probs.sort(key=lambda x: x[2], reverse=True)
    top_edges = [(u, v) for u, v, _ in edge_probs[:top_k]]

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
            if not suitable_degrees:
                assigned_degree = degree_assignment.pop()

            assigned_degree = min(suitable_degrees)
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


def build_accumulating_filtration_sequence_with_edgebank(embedding, graph_num, p_old_nodes, p_new_nodes, E_oo, E_nn, E_on, E_oon, thresholds, embeddings=None, degree_clusters=None, edgebank=None, existing_nodes=None, gcn=None, mlp=None, seed=42):
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
        degree_clusters (dict): A dictionary of {'degree': [nodes with degree]} that we use to compute the embeddings for new nodes
        edgebank (dict): A dict of {node_id: [neighbors]} built up over time to store the previously seen edges
        existing_nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing reappearance probabilities
        gcn (GCN Model): A GCN Model used to embed graphs for node embeddings
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
            if new_embedding is None and len(new_embedding) == 0:
                new_embedding = [np.zeros(32)]

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
                if count != len(sampled):
                    print(f'[WARNING] There was an incorrect amount of predicted edges for Graph #{graph_num}')
                    print(f'There were {len(sampled)} edges when there was supposed to be {count} edges')
                    print(f'The edges: \n {sampled}')
                
        return list(sampled)

    # Get edges of each type
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

    original_nodes = list(tmp_graph.nodes())
    id_to_idx = {node_id: idx for idx, node_id in enumerate(original_nodes)}
    idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}

    # Step 2: Convert to PyG data with consistent feature ordering
    pyg_data = from_networkx(tmp_graph)

    # Ensure features are ordered according to original_nodes
    pyg_data.x = torch.stack([
        torch.tensor(list(tmp_graph.nodes[n]['feat'].values()), dtype=torch.float32)
        for n in original_nodes
    ])

    # Step 3: Pass through GCN
    new_embeddings = gcn(pyg_data.x, pyg_data.edge_index)

    # Step 4: Map embeddings back to original node IDs
    final_embeddings = {
        idx_to_id[i]: new_embeddings[i].detach()
        for i in range(new_embeddings.shape[0])
    }
    
    embeddings.update(final_embeddings)  # Blindly overwrites the previously existing embeddings
    
    for node in tmp_graph.nodes():
        degree = tmp_graph.nodes[node]['feat']['maxDegree']
        
        curr_embedding = embeddings[node]
        old_embedding = degree_clusters.get(degree, [])
        
        # Average the embeddings if both exist
        if old_embedding is not None and len(old_embedding) > 0:
            new_embedding = (np.array(curr_embedding) + np.array(old_embedding)) / 2
        else:
            new_embedding = curr_embedding
            
        degree_clusters[degree] = new_embedding  # Add the embedding

    return filtration_graphs, node_types, existing_nodes, edge_type_map, edgebank, embeddings, degree_clusters


def modifyGraphIds(graphs, thresholds):
    '''
    For the target graphs, modify their ids to start at 0 for an instance of a node, then increment throughout the graphs
    
    Args:
        graphs (list(nx.Graph)): A list of graphs to modify
        
    Returns:
        graphs (list(nx.Graph)): The modified graphs (operations performed in-place)       
    '''
    # This dictionary will store the mapping of original node IDs to new node IDs
    node_mapping = {}
    new_id = 0
    updated_graphs = []

    # Iterate over all graphs in the list of lists (where each graph is a subgraph in the list)
    # First pass: assign a new ID to every unique node
    for graph_list in graphs:
        updated_sublist = []
        for graph in graph_list:
            curr_mapping = {}  # Mapping applies to this specific graph

            for node in graph.nodes:
                # Ensure that 'feat' exists and is properly initialized
                if 'feat' not in graph.nodes[node]:
                    graph.nodes[node]['feat'] = {}

                # Mark the node as new or old
                if node not in node_mapping:
                    node_mapping[node] = new_id
                    new_id += 1
                    graph.nodes[node]['feat']['type'] = 1  # Node is new
                else:
                    graph.nodes[node]['feat']['type'] = 0  # Node is old

                # Map the node and update the ID in the feature dictionary
                curr_mapping[node] = node_mapping[node]
                graph.nodes[node]['feat']['id'] = node_mapping[node]

                node_degree = graph.degree(node)  # Current node's degree

                # If thresholds are available, calculate the max degree based on thresholds
                if np.any(thresholds):
                    graph.nodes[node]['feat']['currDegree'] = node_degree
                    graph.nodes[node]['feat']['maxDegree'] = next((t for t in thresholds if node_degree <= t), thresholds[-1])
                else:
                    # If no thresholds, use degree as maxDegree
                    graph.nodes[node]['feat']['currDegree'] = node_degree
                    graph.nodes[node]['feat']['maxDegree'] = node_degree

            # Relabel the graph nodes according to the new IDs
            relabeled_graph = nx.relabel_nodes(graph, curr_mapping, copy=True)

            # Preserve features for relabeled nodes
            for old_node, new_node in curr_mapping.items():
                relabeled_graph.nodes[new_node]['feat'] = graph.nodes[old_node]['feat'].copy()
                # print(f"Old Node: {old_node}, Features: {graph.nodes[old_node]['feat']}")
                # print(f"New Node: {new_node}, Features: {relabeled_graph.nodes[new_node]['feat']}")

            updated_sublist.append(relabeled_graph)
        updated_graphs.append(updated_sublist)

    return updated_graphs, len(node_mapping)
 

def build_edgebanks_from_start(graphs):
    """
    Build the edgebanks for each graph in graphs, stores all edges from graph i-1 in each index i
    
    Args:
        graphs (list(nx.Graph)): A list of nx Graphs that we will build our edgebanks from
        
    Returns:
        edgebanks (list(dict)): A list of dictionary edgebanks that store all edges from the previous graphs in each index
    """
    edgebanks = [{}]  # Initialize an empty list for edgebanks

    # Loop over all graphs (starting from the second graph)
    for i in range(1, len(graphs)):
        curr_edgebank = {}

        # Add edges from all previous graphs (not the current graph)
        for j in range(i):  # Loop through all previous graphs (graphs 0 to i-1)
            for u, v in graphs[j][-1].edges():  # Accessing the graph directly
                u_key = u
                v_key = v
                curr_edgebank.setdefault(u_key, []).append(v_key)  # Add edge from u to v

        edgebanks.append(curr_edgebank)  # Append the current edgebank to the list

    return edgebanks


def process_starter_graph(graph: nx.DiGraph, gcn, thresholds):
    """
    Process our very first graph, this is our 'primer' used to construct the later graphs
    We do this since we need some node embeddings and features to start with
    
    Args:
        graph (nx.DiGraph): The first graph in the dataset, which we are embedding the nodes for
        gcn (GCN Model): The GCN network that we will use to embed our graph
        thresholds (list): A list of integers, from TopER, used to assign the max degree of a node
    """
    # Assign base features
    for node in graph.nodes():
        graph.nodes[node]['feat'] = {}  # Set up the dictionary
        graph.nodes[node]['feat']['id'] = node
        graph.nodes[node]['feat']['type'] = 1
        node_degree = graph.degree(node)  # The current nodes degree
        
        if np.any(thresholds):
            graph.nodes[node]['feat']['currDegree'] = node_degree
            graph.nodes[node]['feat']['maxDegree'] = next((t for t in thresholds if node_degree <= t), thresholds[-1])
        else:
            graph.nodes[node]['feat']['currDegree'] = node_degree
            graph.nodes[node]['feat']['maxDegree'] = graph.degree(node)
        
        
    original_nodes = list(graph.nodes())
    id_to_idx = {node_id: idx for idx, node_id in enumerate(original_nodes)}
    idx_to_id = {idx: node_id for node_id, idx in id_to_idx.items()}

    # Convert to PyG format
    pyg_data = from_networkx(graph)

    # Consistent feature ordering (define keys explicitly if needed)
    feature_keys = list(next(iter(graph.nodes(data=True)))[1]['feat'].keys())

    # Build the feature matrix
    pyg_data.x = torch.stack([
        torch.tensor([graph.nodes[n]['feat'][k] for k in feature_keys], dtype=torch.float32)
        for n in original_nodes
    ])

    # GCN forward pass
    new_embeddings = gcn(pyg_data.x, pyg_data.edge_index)

    # Map back to node IDs
    final_embeddings = {
        idx_to_id[i]: new_embeddings[i].detach()
        for i in range(new_embeddings.shape[0])
    }
    
    existing_nodes = {}
    # process the nodes for old node evaluation
    for node in graph.nodes(data=False):
        existing_nodes[node] = (0, graph.degree(node))
            
    # Build the edgebank
    edgebank = {}
    for u, v in graph.edges():  # Accessing the graph directly
        edgebank.setdefault(u, []).append(v)
    
    # Process the degree clusters for generating the embeddings for new nodes
    degree_clusters = defaultdict(list)
    for node in graph.nodes():        
        degree = graph.nodes[node]['feat']['maxDegree']
        
        curr_embedding = final_embeddings[node]
        old_embedding = degree_clusters.get(degree, [])
        
        # Average the embeddings if both exist
        if old_embedding is not None and len(old_embedding) > 0:
            new_embedding = (np.array(curr_embedding) + np.array(old_embedding)) / 2
        else:
            new_embedding = curr_embedding
            
        degree_clusters[degree] = new_embedding  # Add the embedding
    
    return final_embeddings, degree_clusters, existing_nodes, edgebank


# Data Loading and Prep

dataset = args.dataset
my_loader = Loader()
my_evaluator = Evaluator()

# Construct csv
run_number = 1
structure_pred_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}/structure_pred.csv'
structure_true_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}/structure_true.csv'
structure_diff_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}/structure_diff.csv'
kernel_pred_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}/kernel_pred.csv'
kernel_true_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}/kernel_true.csv'
edge_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}/edge_analysis.csv'
topER_file_path = f'GraphGeneration/output/results/topER/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}/toper_diff.csv'
animation_path = f'GraphGeneration/output/results/animations/{dataset}/model_gen_retrain_{args.strategy}_embedding{args.embedding}_mlpEncoding{args.mlpEncoding}_embedOld{args.embedOld}_trainingStyle{args.trainingStyle}/pred_vs_true.mp4'

# Create file paths if needed
for path in [structure_pred_file_path, structure_true_file_path, structure_diff_file_path, kernel_pred_file_path, 
             kernel_true_file_path, edge_file_path, topER_file_path, animation_path]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

columns_structure = ['Graph Number', 'Average Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortivity Coefficient',
            'Clustering Coefficient', 'Density', 'Number of Weakly Connected Components',
            'Number of Strongly Connected Components', 'Number of Nodes', 'Number of Edges',
            'Eigenvalue_1', 'Eigenvalue_2', 'Eigenvalue_3', 'Eigenvalue_4', 'Eigenvalue_5', ]
removed = ['Betweenness Centrality', 'Closeness Centrality', 'Number of Cliques', 'Diameter', 'Number of 3-Motifs',  'Number of Cycles', ]

# Write the header and empty content
pd.DataFrame(columns=columns_structure).to_csv(structure_pred_file_path, index=False)
pd.DataFrame(columns=columns_structure).to_csv(structure_true_file_path, index=False)
pd.DataFrame(columns=columns_structure + ['Kernel Distance']).to_csv(structure_diff_file_path, index=False)

columns_edges = ['Graph Number', 'precision overall', 'recall overall', 'tp_overall', 'fp_overall','tn_overall','fn_overall', 'precision oo', 'recall oo', 'tp_oo', 'fp_oo','tn_oo','fn_oo', 'precision oon', 'recall oon', 'tp_oon', 'fp_oon','tn_oon','fn_oon',  'precision on', 'recall on', 'tp_on', 'fp_on','tn_on','fn_on', 'precision nn', 'recall nn', 'tp_nn', 'fp_nn','tn_nn','fn_nn', 
                     'Correct Node IDs', 'Correct Old Node IDs', 'Precision Old IDs', 'Recall Old IDs',  'Correct New Node IDs', 'Precision New IDs', 
                     'Recall New IDs', 'Correct Overall IDs', 'Precision Overall IDs', 'Recall Overall IDs']

# Write the header and empty content
pd.DataFrame(columns=columns_edges).to_csv(edge_file_path, index=False)

columns_kernel = ['Subgraph 1', 'Subgraph 2', 'Subgraph 3', 'Subgraph 4']

# Write the header and empty content
pd.DataFrame(columns=columns_kernel).to_csv(kernel_pred_file_path, index=False)
pd.DataFrame(columns=columns_kernel).to_csv(kernel_true_file_path, index=False)

# Load probabilities
probabilities_df = pd.read_csv(f'ReinforcementLearning/output/probabilities/{dataset}_1back.csv').iloc[:, 1:]
probabilities = probabilities_df.values.tolist()

# Load all features, thresholds, and target subgraphs
features, _ = my_loader.load_data(dataset, activation='Degree', type='features', include_weights=True)
thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=True)
target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)

# Initialize list for predicted graphs
pred_graphs = []

# Build the edgebanks for construction
tmp_target_graphs, _ = modifyGraphIds(target_graphs, thresholds)
all_edgebanks = build_edgebanks_from_start(tmp_target_graphs)

print('Starting training')
gcn = GCNEmbedder(in_channels=4, hidden_channels=16, out_channels=32)
embeddings, degree_clusters, existing_nodes, curr_edgebank_pred = process_starter_graph(target_graphs[0][-1], gcn, thresholds)  # We need a graph to get things going

old_nodes_true = set()
curr_edgebank_pred = {}

# Exclusive to retraining the models, helps with args.trainingStyle
TEST_GRAPH_PERCENT = 0.3
split_idx = int((1.0 - TEST_GRAPH_PERCENT) * len(tmp_target_graphs))
mlp_training_graphs = [tmp_target_graphs[0][-1], tmp_target_graphs[1][-1]]  # The graphs we will use to train the MLPs, must start with our starter


# Graph Creation

# Iterate through each graph in the dataset
for i in range(2, len(probabilities)):  # We don't use first two graphs because we need old edges to train on for the MLP, and we need a primer graph
    print('Constructing graph number: ', i + 1)
    
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
    gcn, mlp = train_models(mlp_training_graphs, all_edgebanks, lr=0.001, seed=42)
    print('Finished training the MLP; Beginning Construction')

    # Build the filtration sequence using the current parameters
    filtration_sequence, node_types, existing_nodes, edge_type_map, curr_edgebank_pred, embeddings, degree_clusters = build_accumulating_filtration_sequence_with_edgebank(
        embedding, graph_num=i, p_old_nodes=count_old, p_new_nodes=count_new, E_oo=p0, E_nn=p1, E_on=p2, E_oon=p3, thresholds=thresholds, embeddings=embeddings, degree_clusters=degree_clusters, edgebank=curr_edgebank_pred, existing_nodes=existing_nodes, gcn=gcn, mlp=mlp
    )
    
    
    # Evaluate the graphs
    results_diff_structure = my_evaluator.evaluateTwoStructure(filtration_sequence[-1], target_graphs[i][-1], graph_num=i)
    results_edges = my_evaluator.evaluateEdges(filtration_sequence[-1], target_graphs[i][-1], curr_edgebank_pred, all_edgebanks[i], graph_num=i)
    results_true_structure = my_evaluator.evaluateSingleStructure(target_graphs[i][-1], graph_num=i)
    results_pred_structure = my_evaluator.evaluateSingleStructure(filtration_sequence[-1], graph_num=i)
    pred_kernel, true_kernel, distance = my_evaluator.evaluateOrca(filtration_sequence[-1], target_graphs[i][-1])

    results_diff_structure['Kernel Distance'] = distance  # The kernel distance will be part of our structure evaluation

    # Store all results
    pd.DataFrame([results_diff_structure]).to_csv(structure_diff_file_path, mode='a', header=False, index=False)
    pd.DataFrame([results_edges]).to_csv(edge_file_path, mode='a', header=False, index=False)
    pd.DataFrame([results_true_structure]).to_csv(structure_true_file_path, mode='a', header=False, index=False)
    pd.DataFrame([results_pred_structure]).to_csv(structure_pred_file_path, mode='a', header=False, index=False)
    pd.DataFrame([pred_kernel]).to_csv(kernel_pred_file_path, mode='a', header=False, index=False)
    pd.DataFrame([true_kernel]).to_csv(kernel_true_file_path, mode='a', header=False, index=False)
    
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


# TopER Comparison and G/S Eval

# Delete first two graphs from targets
del target_graphs[0]
del target_graphs[0]

# Flatten the graphs to embed them, only take the last graphs so that we don't mess anything up
embedding_graphs = [inner_list[-1] for inner_list in pred_graphs]
embedder = EmbedDegree(include_weights=False)

# Load the TopER embeddings and labels for the G/S task and TopER comparisons
all_embeddings, _, _ = embedder.process_graphs_for_embeddings(embedding_graphs)
true_embeddings, labels = my_loader.load_data(dataset, 'Degree', include_weights=False)
labels = np.array(labels)

# Compute the Growth/Shrink labels for the predicted graphs
pred_gs_labels = [1]  # TODO Double check this works properly

# Generate predictions for G/S; based on the number of edges in the graph
# 1 if the current graph has more edges than the previous graph; 0 otherswise
for i in range(1, len(embedding_graphs)):
    prev_edges = embedding_graphs[i - 1].number_of_edges()
    curr_edges = embedding_graphs[i].number_of_edges()
    pred_gs_labels.append(1 if curr_edges > prev_edges else 0)

predictions = np.array(pred_gs_labels)

tmp_labels = labels[1:]  # Since we don't do the first graph

# Compute metrics for Growth/Shrink task
aucroc = roc_auc_score(tmp_labels, predictions)
aucpr = average_precision_score(tmp_labels, predictions)

# Display results
print(f'G/S AUCROC: {aucroc}')
print(f'G/S AUCPR: {aucpr}')

# Used to store topER evaluation
columns = ['graph_num', 'l2_norm', 'cosine_similarity', 'g/s_pred_label', 'g/s_true_label']
for i in range(10):
    columns.append(f'node_diff_{i+1}')
    columns.append(f'edge_diff_{i+1}')

# Write the header and empty content
pd.DataFrame(columns=columns).to_csv(topER_file_path, index=False)


true_embeddings = list(true_embeddings)[2:]  # We ignore the first

# Loop through all embeddings for evaluating topER comparisons
for idx, (embedding, true_embedding) in enumerate(zip(all_embeddings, true_embeddings)):
    # Set graph_num based on index (1-based)
    graph_num = idx + 1
    pred_label = predictions[idx]
    true_label = labels[idx] 

    # Compare embeddings and get the result
    result = my_evaluator.evaluateTopER(embedding, true_embedding, pred_label=pred_label, true_label=true_label, graph_num=graph_num)

    # Append the result to the CSV
    pd.DataFrame([result]).to_csv(topER_file_path, mode='a', header=False, index=False)


# Animation Creation

from itertools import chain

# Flatten the graphs for the animation purposes
predicted_flat = list(chain(*pred_graphs))
target_flat = list(chain(*target_graphs))

# Used to create the animation; it takes a while so it is commented out for now
#my_evaluator.create_animation(predicted_flat, target_flat, output_file=animation_path)