from stable_baselines3 import PPO, SAC
import pandas as pd

# Update path for imports
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ReinforcementLearning.reinforcement_utils.environment_adj_noremoval import GraphReconstructionEnvAdjMatNoRemoval
from ReinforcementLearning.reinforcement_utils.environment_adj import GraphReconstructionEnvAdjMat
from ReinforcementLearning.reinforcement_utils.environment_adj_p2 import GraphReconstructionEnvAdjMatGrouped


from ReinforcementLearning.reinforcement_utils.visualizer import GraphVisualizer
from utils.loader import Loader

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, required=True, choices=['base', 'grouped', 'no_removal', 'no_removal_grouped'])
args = parser.parse_args()

dataset = 'CollegeMsg'
my_loader = Loader()
probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities')  # Activation doesn't matter here

# Load the features and their subgraphs
features, _ = my_loader.load_data(dataset, activation='Degree', type='features', include_weights=False)
thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=False) 
probabilities = probabilities_df.values.tolist()
target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)

print('Data Loaded Successfully')

# Split the features (10% demo/80% train/10% test)
split_idx_demo = int(len(features) * 0.10)
split_idx_train = int(len(features) * 0.9)

features_demo = features[:split_idx_demo]
probabilities_demo = probabilities[:split_idx_demo]
target_graphs_demo = target_graphs[:split_idx_demo]

features_train = features[split_idx_demo :split_idx_train]
probabilities_train = probabilities[split_idx_demo : split_idx_train]
target_graphs_train = target_graphs[split_idx_demo : split_idx_train]
# Since thresholds are the same across all graphs, we ignore

if args.strategy == 'base':
    train_env = GraphReconstructionEnvAdjMat(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)
    test_env = GraphReconstructionEnvAdjMat(features, thresholds, probabilities, target_graphs)
    
elif args.strategy == 'grouped':
    train_env = GraphReconstructionEnvAdjMatGrouped(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)
    test_env = GraphReconstructionEnvAdjMatGrouped(features, thresholds, probabilities, target_graphs)
    
elif args.strategy == 'no_removal':
    train_env = GraphReconstructionEnvAdjMatNoRemoval(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)
    test_env = GraphReconstructionEnvAdjMatNoRemoval(features, thresholds, probabilities, target_graphs)
    
#train_env.reset()
num_graphs = len(features)  # The number of graphs we will train on
max_steps_per_graph = 20000
max_steps_overall = max_steps_per_graph * num_graphs 
max_steps_train = max_steps_per_graph * len(features_train)

print('Training starting')

model_path = f"ReinforcementLearning/models/latest_model_{args.strategy}.zip"

# Train PPO with the new environment
# Check if a saved model exists and load it
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    model = PPO.load(model_path, env=train_env)
else:
    print("No existing model found. Initializing a new one.")
    model = PPO("MlpPolicy", train_env, verbose=1, gamma=0.99, n_steps=4096)  # Try reducing gamma or increasing entropy_coefficient or increasing gae_lambda or increasing n_steps
    # NOTE
    # Also try SAC, DDPG, or TD3 (better in continuous action spaces)
    

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