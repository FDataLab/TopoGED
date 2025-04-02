from stable_baselines3 import PPO, SAC
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# Update path for imports
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ReinforcementLearning.reinforcement_utils.environment_adj_noremoval import GraphReconstructionEnvAdjMatNoRemoval
from ReinforcementLearning.reinforcement_utils.environment_adj import GraphReconstructionEnvAdjMat
from ReinforcementLearning.reinforcement_utils.environment_adj_p2 import GraphReconstructionEnvAdjMatGrouped
from ReinforcementLearning.reinforcement_utils.environment_resources_only import GraphReconstructionEnvResources
from ReinforcementLearning.reinforcement_utils.imitation.policy_mlp import ImitationPolicyMLP

from ReinforcementLearning.reinforcement_utils.visualizer import GraphVisualizer
from utils.loader import Loader

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, required=True, choices=['base', 'grouped', 'no_removal', 'no_removal_grouped', 'no_matrix'])
parser.add_argument("--imitation", type=str, required=False, default='False')  # If we should use imitation learning
args = parser.parse_args()

dataset = 'CollegeMsg'
my_loader = Loader()
probabilities_df = pd.read_csv(f'ReinforcementLearning/output/probabilities/{dataset}_1back.csv').iloc[:, 1:]  # Need to make a loader for this

# Load the features and their subgraphs
features, _ = my_loader.load_data(dataset, activation='Degree', type='features', include_weights=False)
thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=False) 
probabilities = probabilities_df.values.tolist()
target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)

print('Data Loaded Successfully')

# Split the features (85% train/15% test)ss
split_idx = int(len(features) * 0.85)
features_train = features[:split_idx]
probabilities_train = probabilities[:split_idx]
target_graphs_train = target_graphs[:split_idx]
# Since thresholds are the same across all graphs, we ignore


if args.strategy == 'base':
    tmp_env = GraphReconstructionEnvAdjMat(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)  # If doing imitation learning, this is needed
    train_env = GraphReconstructionEnvAdjMat(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)
    test_env = GraphReconstructionEnvAdjMat(features, thresholds, probabilities, target_graphs)
    
elif args.strategy == 'grouped':
    tmp_env = GraphReconstructionEnvAdjMatGrouped(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)  # If doing imitation learning, this is needed
    train_env = GraphReconstructionEnvAdjMatGrouped(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)
    test_env = GraphReconstructionEnvAdjMatGrouped(features, thresholds, probabilities, target_graphs)
    
elif args.strategy == 'no_removal':
    tmp_env = GraphReconstructionEnvAdjMatNoRemoval(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)  # If doing imitation learning, this is needed
    train_env = GraphReconstructionEnvAdjMatNoRemoval(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)
    test_env = GraphReconstructionEnvAdjMatNoRemoval(features, thresholds, probabilities, target_graphs)
    
elif args.strategy == 'no_matrix':
    tmp_env = GraphReconstructionEnvResources(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)  # If doing imitation learning, this is needed
    train_env = GraphReconstructionEnvResources(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)
    test_env = GraphReconstructionEnvResources(features, thresholds, probabilities, target_graphs)
    
if args.imitation == 'True':
    base_decisions = tmp_env.gen_expert_decisions(target_graphs[:3])
    
    print('Generating states for imiation')
    expert_data = []
    state, _ = tmp_env.reset()
    print('starting actions')
    for action in base_decisions:
        next_state, _, _, _, _ = tmp_env.step(action)
        expert_data.append((state, action))
        state = next_state

    state_dim = len(expert_data[0][0])
    action_dim = len(expert_data[0][-1])

    policy = ImitationPolicyMLP(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 500
    batch_size = 16

    print('Starting training imitation')

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, len(expert_data), batch_size):
            batch = expert_data[i:i+batch_size]

            # Convert batch to tensors while ensuring states and actions remain aligned
            filtered_batch = [(s, a) for s, a in batch if len(s) > 0]
            if len(filtered_batch) == 0:
                continue  # Skip this batch if no valid states

            states, actions = zip(*filtered_batch)
            
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)  # Shape: (batch, seq_len, state_dim)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32)  # Shape: (batch, action_dim)

            optimizer.zero_grad()
            predicted_actions = policy(states_tensor)
            loss = loss_fn(predicted_actions, actions_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(expert_data)}")

    print('Imitation complete!')


#train_env.reset()
num_graphs = len(features)  # The number of graphs we will train on
max_steps_per_graph = 20000
max_steps_overall = max_steps_per_graph * num_graphs 
max_steps_train = max_steps_per_graph * len(features_train)

print('Training starting')

model_path = f"ReinforcementLearning/models/latest_model_{args.strategy}.zip"

if args.imitation == 'True':
    model_path = f"ReinforcementLearning/models/latest_model_{args.strategy}_imitation.zip"

# Train PPO with the new environment
# Check if a saved model exists and load it
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model = PPO.load(model_path, env=train_env)
else:
    print("No existing model found. Initializing a new one.")
    model = PPO("MlpPolicy", train_env, verbose=1, gamma=0.99, n_steps=4096)  # Try reducing gamma or increasing entropy_coefficient or increasing gae_lambda or increasing n_steps
    
    # Start imiation if using
    model.policy.load_state_dict(policy.state_dict())


try:
    # Start training
    model.learn(total_timesteps=max_steps_overall)
except KeyboardInterrupt:
    print("\nTraining interrupted! Saving model...")
    model.save(model_path)
    print(f"Model saved at {model_path}")
    exit()  # Exit gracefully

print('Training finished')
model.save(model_path)
print(f"Model saved at {model_path}")

# Test trained model on all graphs

obs, _ = test_env.reset()

for _ in range(max_steps_overall):  
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = test_env.step(action)

    # Completed all graphs
    if done == True:
        break

print('Testing finished')

# Visualize the results
animation_path = 'initial_anim.mp4'
recon_adj_matrices, target_adj_matrices = model.get_all_states()
my_visualizer = GraphVisualizer(recon_adj_matrices, target_adj_matrices)
my_visualizer.animate(file_name=animation_path)