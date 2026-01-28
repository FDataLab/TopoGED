import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import pickle
import ast

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader
from torch.utils.data import DataLoader
from nn.custom_model import Decoder
from torch.utils.data import DataLoader, Dataset
from utils.utils import Utils
from utils.dataset import DeltaEmbeddingDataset
from utils.visualizer import Visualizer


# Define the new results path
RESULTS_CSV = 'data/output/ProbabilityTesting/data/bayesian_training_results.csv'


# Constants
seed = 42
FLOAT_MAX = np.finfo(np.float32).max

# Device Selection logic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Activation name map
activation_map = {
    "Degree": ['Degree'],
    "Betweenness": ['Betweenness'],
    "Forman": ['Forman'],
    "Closeness": ['Closeness'],
    "Weight": ['Weight'],
    "Degree_Betweenness_Closeness": ['Degree', 'Betweenness', 'Closeness'],
    "Degree_Forman_Weight": ['Degree', 'Forman', 'Weight'],
    "Degree_Forman_Closeness": ['Degree', 'Forman', 'Closeness'],
    "Degree_Weight_Closeness": ['Degree', 'Weight', 'Closeness'],
    "Degree_Forman": ['Degree', 'Forman'],
    "Degree_Weight": ['Degree', 'Weight'],
    "Degree_Betweenness": ['Degree', 'Betweenness'],
    "Degree_Closeness": ['Degree', 'Closeness'],
}

combo_map = {
    "['LSTM', 'ReLU']": ['LSTM', 'ReLU'], 
    "['GRU', 'ReLU']": ['GRU', 'ReLU'], 
    "['LSTM', 'GRU', 'ReLU']": ['LSTM', 'GRU', 'ReLU'], 
    "['RNN']": ['RNN'],
    "['GRU']": ['GRU'],
    "['LSTM']": ['LSTM'],
    "['RNN', 'FC']": ['RNN', 'FC'],
    "['LSTM', 'FC']": ['LSTM', 'FC'],
    "['GRU', 'FC']": ['GRU', 'FC'],
    "['LSTM', 'GRU', 'FC']": ['LSTM', 'GRU', 'FC'],
    "['LSTM', 'FC', 'FC']": ['LSTM', 'FC', 'FC'],
    "['GRU', 'FC', 'FC']": ['GRU', 'FC', 'FC'],
    "['RNN', 'MLP']": ['RNN', 'MLP'],
    "['LSTM', 'MLP']": ['LSTM', 'MLP'], 
    "['GRU', 'MLP']": ['GRU', 'MLP'], 
    "['LSTM', 'GRU', 'MLP']": ['LSTM', 'GRU', 'MLP']
}

class ProbabilityWrapper(nn.Module):
    def __init__(self, base_model):
        super(ProbabilityWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        logits = self.base_model(x)
        if logits.dim() == 3:
            logits = logits[:, -1, :] 

        # Group 1: Indices 0, 1 | Group 2: Indices 2, 3, 4, 5
        group1 = torch.softmax(logits[:, :2], dim=1)
        group2 = torch.softmax(logits[:, 2:], dim=1)

        return torch.cat([group1, group2], dim=1)

def train_and_eval_probs(dataset, window_size, norm, num_layer, dropout, hidden_1, hidden_2, lr_val, l2_val, batch_size, combo, counter, seed):
    # 1. Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 6
    input_dim = 6
    patience = 25
    num_epochs = 500
    
    # Load raw probabilities
    probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities', num_back='all')  
    if norm:
        # Assuming your custom normalization handles the two probability groups correctly
        embeddings = probabilities_df.apply(
            lambda row: my_utils.normalize_vector_by_groups(row.values),
            axis=1
        )
        embeddings = np.vstack(embeddings.values).astype(np.float32) 
    else:
        embeddings = probabilities_df.values.astype(np.float32)
        
    run_name = f"{dataset}_probs_{counter}"
        
    # 2. Data Splitting (70/15/15)
    n = len(embeddings)
    train_end = int(0.70 * n)
    val_start = train_end - window_size
    val_end = int(0.85 * n)
    test_start = val_end - window_size

    X_train = embeddings[:train_end]
    X_val = embeddings[val_start:val_end]
    X_test = embeddings[test_start:]

    # Dataset provides (x, y_delta, x_last). We reconstruct P(t+1) = x_last + y_delta
    train_dataset = DeltaEmbeddingDataset(X_train, k=window_size)
    valid_dataset = DeltaEmbeddingDataset(X_val, k=window_size)
    test_dataset = DeltaEmbeddingDataset(X_test, k=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                                    
            
    # 4. Model Setup (Base Decoder + Probability Constraint Wrapper)
    base_decoder = Decoder(
        in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], 
        hids_size_other=[hidden_2], num_layers=[num_layer], layers=combo, 
        bias=[True], dropout=[dropout]
    )
    model = ProbabilityWrapper(base_decoder).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    criterion = nn.MSELoss().to(device)
    
    curr_batch_best_loss = float('inf')    
    best_vectors = None 
    no_improvement_counter = 0
      
    # 5. Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (x, y_delta, x_last) in enumerate(train_loader):
            x, y_delta, x_last = x.to(device), y_delta.to(device), x_last.to(device)
            optimizer.zero_grad()
            
            # Ground Truth Target: Actual P(t+1) state
            y_target = x_last + y_delta 
            predicted_probs = model(x) 
            
            loss = criterion(predicted_probs, y_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation Pass
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for i, (x, y_delta, x_last) in enumerate(valid_loader):
                x, y_delta, x_last = x.to(device), y_delta.to(device), x_last.to(device)
                y_target = (x_last + y_delta).float()
                predicted_probs = model(x)
                valid_loss += criterion(predicted_probs, y_target).item()
        
        valid_loss /= len(valid_loader)

        # 6. Best Moment Tracking & Full Trajectory Generation
        if valid_loss <= curr_batch_best_loss:
            curr_batch_best_loss = valid_loss
            no_improvement_counter = 0

            # Autoregressive sequence reconstruction for PKL
            temp_vectors = []
            # Context Window: Prepend the "missed" ground truth vectors
            for k in range(window_size):
                temp_vectors.append(embeddings[k])
            
            # Inference on full data: Use window size 1 context to predict next step
            # We use embeddings (input data) to generate the full predicted TopER sequence
            full_inference_dataset = DeltaEmbeddingDataset(embeddings, k=window_size)
            inf_loader = DataLoader(full_inference_dataset, batch_size=1, shuffle=False)
            
            model.eval()
            with torch.no_grad():
                for x_inf, _, x_last in inf_loader:
                    x_inf = x_inf.to(device)
                    x_last = x_last.to(device) # The state at time t
                    
                    # The model outputs the absolute probabilities for t+1
                    # because of the ProbabilityWrapper softmax groups
                    p_pred_abs = model(x_inf) 
                    
                    # To treat it as a delta: 
                    # Delta = Predicted_Absolute - Last_Known_Absolute
                    delta = p_pred_abs - x_last
                    
                    # Re-adding them (Result = x_last + delta)
                    # This effectively uses the model's prediction while 
                    # maintaining the mathematical flow you requested.
                    final_vector = (x_last + delta).cpu().numpy()[0]
                    
                    # 3. Final grouping check (Optional safety)
                    # Even though ProbabilityWrapper does this, doing it here
                    # ensures the saved PKL is 100% clean
                    final_vector = my_utils.normalize_vector_by_groups(final_vector)
                    
                    temp_vectors.append(final_vector)
            
            best_vectors = np.array(temp_vectors)
        else:
            if epoch >= 100:
                no_improvement_counter += 1
            
        if no_improvement_counter >= patience:
            break
        
    pred_df = pd.DataFrame(best_vectors)
    real_df = pd.DataFrame(embeddings)
    figures_output_path = f"data/output/ProbabilityTesting/data/delta_plots/"
    os.makedirs(figures_output_path, exist_ok=True)
    for i in range(0, 6):
        # Plot Nodes/Edges (-2 and -1 are the last threshold pair)
        Visualizer.plot_scatter(pred_df.iloc[:, i], real_df.iloc[:, i], 
                            f"{figures_output_path}{dataset}_idx{i}_delta.png", mode="nodes" if i < 3 else "edges")
        
    # 7. Final Export
    pickle_dir = f'data/input/cached/{dataset}/predValues/'
    os.makedirs(pickle_dir, exist_ok=True)
    pickle_path = os.path.join(pickle_dir, f"{dataset}_probs_all_back.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(best_vectors, f)
    
    

def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} not found.")
        return

    # 1. Read the CSV and find the best trial for each dataset
    df = pd.read_csv(RESULTS_CSV)
    
    # Sort by valid_loss (ascending) and take the first (best) for each dataset
    best_trials = df.sort_values('valid_loss').groupby('dataset').first().reset_index()
    
    print(f"Found best trials for {len(best_trials)} datasets.")

    # 2. Iterate through the best trials and re-run training/generation
    for _, row in best_trials.iterrows():
        dataset = row['dataset']
        print(f"\n>>> Re-running best trial for: {dataset} (Best Val Loss: {row['valid_loss']:.6f})")

        # Map string combo back to list if necessary
        # Depending on how it was saved, it might be a string like "['LSTM', 'FC']"
        combo_val = row['combo']
        if isinstance(combo_val, str):
            import ast
            combo_list = ast.literal_eval(combo_val)
        else:
            combo_list = combo_val

        # 3. Call your training function with the exact best parameters
        # This will trigger the epoch loop, find the best moment, and save the .pkl
        train_and_eval_probs(
            dataset=dataset,
            window_size=int(row['window_size']),
            norm=row['normalization'],
            num_layer=int(row['num_layers']),
            dropout=row['dropout'],
            hidden_1=int(row['hidden_1']),
            hidden_2=int(row['hidden_2']),
            lr_val=row['learning_rate'],
            l2_val=row['l2_regularization'],
            batch_size=int(row['batch_size']),
            combo=combo_list,
            counter="FINAL_GEN", # Marker for run_name
            seed=42
        )
        
        print(f"Completed {dataset}")

if __name__ == "__main__":
    main()