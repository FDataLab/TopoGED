from json import encoder
from math import e
import random
import numpy as np
import argparse
import os
import sys
from sklearn.metrics import recall_score
from sqlalchemy import all_
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from sympy import use
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from GraphGeneration.utils.Evaluator import Evaluator
from nn.custom_model import Decoder
from GraphGeneration.utils.graph_construction_utils import compute_reappearance_probabilities, generate_tgcn_node_features, get_node_features, update_degrees, generate_gnn_node_embeddings
from GraphGeneration.models.model import setupMLP, load_encoder_model
from process_data import modifyGraphIds, build_edgebanks_from_start
from GraphGeneration.scripts.compute_embedding import compute_node2vec_embeddings

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
    


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
    

def compute_reappearance_probabilities_new(graphs, current_target_count_old_nodes, decay_factor, alpha, beta, epsilon=1e-10):
    nodes = dict()
    frequency = defaultdict(int)
    # Create the nodes dict degree history
    # nodes (dict): A dict of {node_id: (last_seen_timestamp, last_seen_degree)} used for computing probabilities
    
    for t, G in enumerate(graphs):
        for node in G[-1].nodes():
            nodes[node] = (t, G[-1].degree(node))
            frequency[node] += 1
    
    max_degree = max(degree for _, (_, degree) in nodes.items())
    
    probs = {}
    t_curr = len(graphs)  # Makes the formula work best
    
    for node_id, (last_seen, degree) in nodes.items():
        recency_score = np.exp(-(t_curr - last_seen) / decay_factor) if decay_factor > 0 else 1.0
        degree_score = (degree / max_degree) ** alpha if max_degree > 0 else 1.0
        frequency_score = (frequency[node_id] ** beta) if beta > 0 else 1.0
        
        raw_score = recency_score * degree_score * frequency_score
        
        probs[node_id] = max(raw_score, epsilon)  # Apply epsilon floor to avoid exact 0

    # Normalize to make a valid probability distribution
    total = sum(probs.values())
    for node in probs:
        probs[node] /= total

    node_ids = list(probs.keys())
    weights = list(probs.values())

    sampled_old_nodes = list(np.random.choice(node_ids, size=current_target_count_old_nodes, replace=False, p=np.array(weights)/np.sum(weights)))  # Makes sure that we select only unique nodes each time
    
    return sampled_old_nodes
    
    

def evaluate_metrics(model, data_loader, criterion, device):
    """Evaluates the model on a given data loader and calculates metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(-1) # outputs are logits
            loss = criterion(outputs, targets.float())
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate binary metrics
    try:
        aucroc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # AUCROC requires at least one positive and one negative sample
        aucroc = 0.0 
        
    # Convert probabilities to binary predictions for Precision, Recall, F1 (using 0.5 threshold)
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    precision = precision_score(all_targets, binary_preds, zero_division=0)
    recall = recall_score(all_targets, binary_preds, zero_division=0)
    f1 = f1_score(all_targets, binary_preds, zero_division=0)

    return avg_loss, aucroc, precision, recall, f1

# ====================================================================
# MAIN TRAINING FUNCTION (Filled out)
# ====================================================================

def train_mlp_for_node_classification(encoder_model, target_number_nodes, train_data, val_data, test_data, lr=0.001, input_dim=64, output_dim=1, hidden_2=32, num_layer=2, combo=['MLP'], dropout=0.1, epochs=250, batch_size=32, model_key='GCN'):
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_aucroc": [],
        "val_aucroc": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
    }  # For plotting
    path = 'GraphGeneration/output/results/old_node_optimization/networkadex/training_plots2/'
    os.makedirs(path, exist_ok=True)
    
    all_graph_predictions = []  # For storing nodes
    
    # Make the MLP model
    #model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    
    model = SimpleMLP(input_dim=64, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layer, dropout=dropout)
    model.to(device)
    
    # Loss function (BCEWithLogitsLoss is numerically stable for binary classification)
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer (Adam is a robust choice)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Data Loading ---
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(train_data['X'], dtype=torch.float32)
    y_train_tensor = torch.tensor(train_data['y'], dtype=torch.float32)
    X_val_tensor = torch.tensor(val_data['X'], dtype=torch.float32)
    y_val_tensor = torch.tensor(val_data['y'], dtype=torch.float32)
    X_test_tensor = torch.tensor(test_data['X'], dtype=torch.float32)
    y_test_tensor = torch.tensor(test_data['y'], dtype=torch.float32)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)
    
    best_val_aucroc = -1
    best_epoch = 0
    
    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs).squeeze(-1) # Output is logits
            loss = criterion(outputs, targets.float())
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Average training loss
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Step ---
        #val_loss, val_aucroc, _, _, _ = evaluate_metrics(model, val_loader, criterion, device)
        
        train_loss, train_aucroc, train_precision, train_recall, train_f1 = evaluate_metrics(
            model, train_loader, criterion, device
        )
        val_loss, val_aucroc, val_precision, val_recall, val_f1 = evaluate_metrics(
            model, val_loader, criterion, device
        )
        
        # Simple Early Stopping Check
        if val_aucroc > best_val_aucroc:
            best_val_aucroc = val_aucroc
            best_epoch = epoch + 1
            # Save best model weights
            torch.save(model.state_dict(), 'GraphGeneration/output/results/old_node_optimization/best_Node2Vecmlp_model.pt')
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val AUCROC: {val_aucroc:.4f}")

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_aucroc"].append(float(train_aucroc))
        history["val_aucroc"].append(float(val_aucroc))
        history["train_precision"].append(float(train_precision))
        history["val_precision"].append(float(val_precision))
        history["train_recall"].append(float(train_recall))
        history["val_recall"].append(float(val_recall))
        history["train_f1"].append(float(train_f1))
        history["val_f1"].append(float(val_f1))

    # --- Final Evaluation (on best model) ---
    model.load_state_dict(torch.load('GraphGeneration/output/results/old_node_optimization/best_Node2Vecmlp_model.pt', map_location=device))
    
    # Evaluate on all three sets using the final best model
    train_loss, train_aucroc, train_precision, train_recall, train_f1 = evaluate_metrics(model, train_loader, criterion, device)
    valid_loss, valid_aucroc, valid_precision, valid_recall, valid_f1 = evaluate_metrics(model, val_loader, criterion, device)
    test_loss, test_aucroc, test_precision, test_recall, test_f1 = evaluate_metrics(model, test_loader, criterion, device)

    # --- Output results ---
    res = {
        'model_type': model_key,
        'lr': lr, # Added learning rate to results
        'feature_dim': input_dim, # Changed to input_dim as this is the input feature dimension
        'hidden_dim': hidden_2,
        'num_layers': num_layer,
        'dropout': dropout,
        'batch_size': batch_size,
        'trained_epochs': best_epoch,
        'predictor_type': 'MLP', # Corrected model_type key to mlp_type to avoid confusion
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'test_loss': test_loss,
        'train_aucroc': train_aucroc,
        'valid_aucroc': valid_aucroc,
        'test_aucroc': test_aucroc,
        'train_precision': train_precision,
        'valid_precision': valid_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'valid_recall': valid_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'valid_f1': valid_f1,
        'test_f1': test_f1,
    }
    
    # Clean up saved model file
    # os.remove('best_mlp_model.pt')
    
    # plt.figure()
    # plt.plot(history["train_aucroc"])
    # plt.plot(history["val_aucroc"])
    # plt.xlabel("Epoch")
    # plt.ylabel("AUCROC")
    # plt.title("Train vs Validation AUCROC")
    # plt.legend(["Train", "Validation"])
    # plt.savefig(path + 'aucroc.png')
    # plt.close()
    # # ----- Precision -----
    # plt.figure()
    # plt.plot(history["train_precision"])
    # plt.plot(history["val_precision"])
    # plt.xlabel("Epoch")
    # plt.ylabel("Precision")
    # plt.title("Train vs Validation Precision")
    # plt.legend(["Train", "Validation"])
    # plt.savefig(path + 'precision.png')
    # plt.close()
    # # ----- Recall -----
    # plt.figure()
    # plt.plot(history["train_recall"])
    # plt.plot(history["val_recall"])
    # plt.xlabel("Epoch")
    # plt.ylabel("Recall")
    # plt.title("Train vs Validation Recall")
    # plt.legend(["Train", "Validation"])
    # plt.savefig(path + 'recall.png')
    # plt.close()
    # # ----- F1 -----
    # plt.figure()
    # plt.plot(history["train_f1"])
    # plt.plot(history["val_f1"])
    # plt.xlabel("Epoch")
    # plt.ylabel("F1 Score")
    # plt.title("Train vs Validation F1")
    # plt.legend(["Train", "Validation"])
    # plt.savefig(path + 'f1.png')
    # plt.close()
    return res



def prepare_data_for_mlp(encoder_model, target_graphs, node_features, starting_graph, num_back, model_type, device, embedding_dim=64, method='gnn'):
    train_part = 0.80
    val_part = 0.10
    
    n_graphs = len(target_graphs)
    
    train_end_idx = int(train_part * n_graphs)
    val_end_idx = train_end_idx + int(val_part * n_graphs)
    
    training_samples = {'X': [], 'y': []}
    val_samples = {'X': [], 'y': []}
    test_samples = {'X': [], 'y': []}
    
    for i in range(starting_graph, n_graphs):
        curr_nodes = target_graphs[i][-1].nodes()
        all_options = set().union(*[set(g[-1].nodes()) for g in target_graphs[max(0, i - num_back):i]])
        curr_nodes = curr_nodes & all_options  # Only consider old nodes
        other_options = all_options - curr_nodes  # Nodes that could have appeared, but didn't
        
        if method == 'gnn':
            embeddings = generate_gnn_node_embeddings(encoder_model, model_type, node_features, target_graphs[:i], num_back, embedding_dim=embedding_dim, curr_nodes=target_graphs[i][-1].nodes(), device=device)
        elif method == 'node2vec':
            embeddings = {id: torch.zeros(32) for id in curr_nodes} 
            history = [target_graphs[j][-1] for j in range(max(0, i - num_back), i)]
            for j, graph in enumerate(history):
                embeddings.update(compute_node2vec_embeddings(graph, device='cpu'))
        #curr_embeddings_existing = torch.stack([embeddings[node] for node in curr_nodes])
        #curr_embeddings_noexisting = torch.stack([embeddings[node] for node in other_options])
        
        # Filter
        for node in curr_nodes:
            if i < train_end_idx:
                training_samples['X'].append(embeddings[node].detach().cpu().numpy())
                training_samples['y'].append(1)  # Existing node
            elif i < val_end_idx:
                val_samples['X'].append(embeddings[node].detach().cpu().numpy())
                val_samples['y'].append(1)  # Existing node
            else:
                test_samples['X'].append(embeddings[node].detach().cpu().numpy())
                test_samples['y'].append(1)  # Existing node

        # Get the proper number of negative samples 
        other_options_ids = list(other_options)
        
        other_options = random.sample(other_options_ids, min(len(curr_nodes), len(other_options_ids)))
        for node in other_options:
            if i < train_end_idx:
                training_samples['X'].append(embeddings[node].detach().cpu().numpy())
                training_samples['y'].append(0)  # Existing node
            elif i < val_end_idx:
                val_samples['X'].append(embeddings[node].detach().cpu().numpy())
                val_samples['y'].append(0)  # Existing node
            else:
                test_samples['X'].append(embeddings[node].detach().cpu().numpy())
                test_samples['y'].append(0)  # Existing node
                    
    return training_samples, val_samples, test_samples
    
    
def numpy_mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    index = np.argmax(counts)
    return values[index]


if __name__ == "__main__":
    my_loader = Loader()
    num_back = 5
    use_predicted = False 
    starting_graph = 2
    
    
    
    mlp_res_path = ""
    
    # Load nodes and compute reappearance probabilities for different parameters
    for dataset in ['networkadex', 'networkaion', 'networkbancor', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon', 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd']:
        probabilities_df = my_loader.load_data(type='probabilities', dataset=dataset, activation='', normalized=True, use_predicted=use_predicted, num_back=num_back)
        probabilities = probabilities_df.values.tolist()
        thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=False)
        graph_descriptions, _ = my_loader.load_data(dataset, activation='Degree', type='features', use_predicted=use_predicted, include_weights=False)
        graph_descriptions = [[(lst[i], lst[i+1]) for i in range(0, len(lst), 2)] for lst in graph_descriptions]
        target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)
        target_graphs, _ = modifyGraphIds(target_graphs, thresholds, num_back)
        
        target_number_nodes = [int(probabilities[i][0] * graph_descriptions[i][-1][0]) for i in range(starting_graph, len(target_graphs))]
        
        equation_res_path = f"GraphGeneration/output/results/old_node_optimization/{dataset}_equation_results.csv"
        os.makedirs(os.path.dirname(equation_res_path), exist_ok=True)
        mlp_res_path = f"GraphGeneration/output/results/old_node_optimization/{dataset}_Node2Vecmlp_results_new.csv"
        os.makedirs(os.path.dirname(mlp_res_path), exist_ok=True)
        
        # for decay_factor in [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0]:
        #     for alpha in [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0]:
        #         for beta in [0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0]:
        #             if decay_factor == 0 and alpha == 0 and beta == 0:
        #                 continue  # Skip trivial case
                                        
        #             # Predicted nodes, and true nodes
        #             preds = []
        #             trues = []
                    
        #             for i in range(starting_graph, len(probabilities)): 
        #                 current_target_snapshot = i
                                                                        
        #                 V_total = int(graph_descriptions[current_target_snapshot][-1][0])  # Used to convert probabilities
                        
        #                 # Get the true count of 4 edges type and number of new, old nodes of the target snapshot (probabilities are fed in as percents)
        #                 current_target_count_old_nodes = int(round(probabilities[current_target_snapshot][0] * V_total))
                    
        #                 sampled_old_nodes = compute_reappearance_probabilities_new(target_graphs[max(i - num_back, 0):i], current_target_count_old_nodes, decay_factor, alpha, beta)
                                            
        #                 # For later evaluation
        #                 preds.append(sampled_old_nodes)
        #                 trues.append(list(target_graphs[current_target_snapshot][-1].nodes()))
                    
        #             # For results storage
        #             precisions = []
        #             recalls = []
        #             f1s = [] 
                    
        #             for pred_nodes, true_nodes in zip(preds, trues):
        #                 pred_set = set(pred_nodes)
        #                 true_set = set(true_nodes)

        #                 tp = len(pred_set & true_set)
        #                 fp = len(pred_set - true_set)
        #                 fn = len(true_set - pred_set)

        #                 precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        #                 recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

        #                 if precision + recall > 0:
        #                     f1 = 2 * precision * recall / (precision + recall)
        #                 else:
        #                     f1 = 0

        #                 precisions.append(precision)
        #                 recalls.append(recall)
        #                 f1s.append(f1)
                    
        #             precisions = np.array(precisions)
        #             recalls = np.array(recalls)
        #             f1s = np.array(f1s)
                    
        #             # Return mean, std, mode, min, max of precision, recall, accuracy, f1 and input variables
        #             res = {
        #                 'dataset': dataset,
        #                 'decay_factor': decay_factor,
        #                 'alpha': alpha,
        #                 'beta': beta,
        #                 'mean_precision': np.mean(precisions),
        #                 'std_precision': np.std(precisions),
        #                 'mode_precision': numpy_mode(precisions),
        #                 'min_precision': np.min(precisions),
        #                 'max_precision': np.max(precisions),
        #                 'mean_recall': np.mean(recalls),
        #                 'std_recall': np.std(recalls),
        #                 'mode_recall': numpy_mode(recalls),
        #                 'min_recall': np.min(recalls),
        #                 'max_recall': np.max(recalls),
        #                 'mean_f1': np.mean(f1s),
        #                 'std_f1': np.std(f1s),
        #                 'mode_f1': numpy_mode(f1s),
        #                 'min_f1': np.min(f1s),
        #                 'max_f1': np.max(f1s)
        #             }
                    
        #             # Output res to the csv for analysis
        #             write_header = not os.path.exists(equation_res_path)
        #             pd.DataFrame([res]).to_csv(equation_res_path, mode='a', header=write_header, index=False) 
                
        # # Try the MLP training for node classification
        for model_name, encoder_config in [
            ("GCN_binary_lr0.001", {"feature_dim": 32, "feature_type": 'binary', "embedding_dim": 64, "hidden_dim": 128, "num_layers": 1}),
            ("GAT_binary_lr0.001", {"feature_dim": 32, "feature_type": 'binary', "embedding_dim": 64, "hidden_dim": 128, "num_layers": 1})
            ]:
            if dataset != 'networkadex':
                continue 
            if model_name == "GCN_binary_lr0.001":
                continue 
            encoder_config["encoder_model"] = {"other_models": {}}
            encoder_config["encoder_model"]["nodeEmbeddingType"] = model_name.split('_')[0]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feature_dim = encoder_config['feature_dim']
            feature_type = encoder_config['feature_type']
            total_edges = sum(len(g[-1].edges()) for g in target_graphs)
            edge_features = np.zeros((total_edges, 0), dtype=np.float32)  # Since TGN needs features, provide dummy features
            node_features = generate_tgcn_node_features(target_graphs, feature_dim, feature_type=feature_type, device=device)
            embedding_dim = encoder_config['embedding_dim']
            hidden_dim = encoder_config['hidden_dim']
            encoder_config["encoder_model"]["other_models"]["feature_dim"] = feature_dim
            encoder_config["encoder_model"]["other_models"]["embedding_dim"] = embedding_dim
            num_layers = encoder_config['num_layers']
            encoder_config["encoder_model"]["other_models"]["num_layers"] = num_layers
            
            encoder_model, input_dim = load_encoder_model(encoder_config, device=device, node2vec_dimensions=embedding_dim, 
                                                                hidden_dim=hidden_dim, num_layers=num_layers,
                                                                node_features=node_features, edge_features=edge_features)
            
            # Load the existing model
            encoder_model_path = f"data/input/cached/{dataset}/saved_data_gnn_{model_name}/saved_models/embedder_1024"  # This should work but doesnt, i think its the '.' character
            if not os.path.exists(encoder_model_path):
                # This code runs if the file IS NOT found.
                print(f"WARNING: Encoder model not found at: {encoder_model_path}")
            encoder_model.load_state_dict(torch.load(encoder_model_path, map_location=device))            
            encoder_model.to(device)
            encoder_model.eval()
            
            train_data, val_data, test_data = prepare_data_for_mlp(encoder_model, target_graphs, node_features, starting_graph, num_back, model_type=model_name.split('_')[0], device=device, embedding_dim=embedding_dim, method='node2vec')
            
            
            # A grid search for the MLP 
            skip = True 
            for lr in [0.01, 0.001, 0.0001]:
                for hidden_dim in [16, 32, 64, 128]:
                    for dropout in [0, 0.1, 0.2, 0.3]:
                        for num_layer in [1, 2, 3]:      
                            
                            
                            # Continue from here when have time for GCN
                            # if lr == 0.01 and hidden_dim == 64 and num_layer == 2 and dropout == 0.2:
                            #     skip = True
                            #     continue
                            
                            # if skip:
                            #     continue
                            if not (lr == 0.001 and hidden_dim ==128 and num_layer == 3 and dropout == 0.2):
                                continue              
                            res = train_mlp_for_node_classification(encoder_model, target_number_nodes, train_data, val_data, test_data, lr=lr, input_dim=input_dim, output_dim=1, hidden_2=hidden_dim, num_layer=num_layer, combo=['MLP'], dropout=dropout, epochs=250, batch_size=32, model_key=model_name)
                            
                            # Output res to the csv for analysis
                            write_header = not os.path.exists(mlp_res_path)
                            pd.DataFrame([res]).to_csv(mlp_res_path, mode='a', header=write_header, index=False) 