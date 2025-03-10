from stable_baselines3 import PPO
import pandas as pd

# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ReinforcementLearning.reinforcement_utils.environment_adj import GraphReconstructionEnvAdjMat
from ReinforcementLearning.reinforcement_utils.visualizer import GraphVisualizer
from utils.loader import Loader


dataset = 'CollegeMsg'
my_loader = Loader()
probabilities_df = pd.read_csv(f'ReinforcementLearning/output/probabilities/{dataset}_1back.csv')  # Need to make a loader for this

# Load the features and their subgraphs
features, _ = my_loader.load_data(dataset, activation='Degree', type='features', include_weights=False)
thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=False) 
probabilities = probabilities_df.values.tolist()
target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)

print('Data Loaded Successfully')

# Split the features (85% train/15% test)
split_idx = int(len(features) * 0.85)
features_train = features[:split_idx]
probabilities_train = probabilities[:split_idx]
target_graphs_train = target_graphs[:split_idx]
# Since thresholds are the same across all graphs, we ignore


train_env = GraphReconstructionEnvAdjMat(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train)

num_graphs = len(features)  # The number of graphs we will train on
max_steps_per_graph = 1250
max_steps_overall = max_steps_per_graph * num_graphs 
max_steps_train = max_steps_per_graph * len(features_train)

print('Training starting')

train_env.reset()
print(train_env.step(train_env.action_space.sample()))  # Check return values
# Train PPO with the new environment
model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=max_steps_overall)

print('Training finished')

test_env = GraphReconstructionEnvAdjMat(features, thresholds, probabilities, target_graphs)

# Test trained model on all graphs

obs = test_env.reset()

for _ in range(max_steps_overall):  
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = test_env.step(action)

    # Completed all graphs
    if done[0] == True:
        break

print('Testing finished')

# Visualize the results
animation_path = 'initial_anim.mp4'
recon_adj_matrices, target_adj_matrices = model.get_all_states()
my_visualizer = GraphVisualizer(recon_adj_matrices, target_adj_matrices)
my_visualizer.animate(file_name=animation_path)