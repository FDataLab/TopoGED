from stable_baselines3 import PPO, SAC
import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from imitation.algorithms import bc

# Update path for imports
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ReinforcementLearning.reinforcement_utils.nx_envs.environment_contids_noremoval import GraphReconstructionNxContidsNoRemoval
from ReinforcementLearning.reinforcement_utils.nx_envs.environment_contids_noremoval_structured import GraphReconstructionNxContidsNoRemovalStructured
from ReinforcementLearning.reinforcement_utils.nx_envs.environment_contids_noremoval_grouped import GraphReconstructionNxContidsNoRemovalGrouped
from ReinforcementLearning.reinforcement_utils.nx_envs.environment_node2vec import GraphReconstructionNxNode2Vec
from ReinforcementLearning.reinforcement_utils.imitation.policy_mlp import ImitationPolicyMLP
from ReinforcementLearning.reinforcement_utils.models.EpsilonGreedyPPO import EpsilonGreedyPPO

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from ReinforcementLearning.reinforcement_utils.visualizer import GraphVisualizer
from utils.loader import Loader

from stable_baselines3.common.callbacks import BaseCallback


# Move to a file if this works
class StopOnDoneCallback(BaseCallback):
    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            if isinstance(dones, np.ndarray) and np.any(dones):
                print("Stopping training: at least one environment is done")
                return False
        return True

def mask_fn(env: gym.Env):
    return env.get_action_mask()


parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, required=True, choices=['base', 'grouped', 'no_removal', 'no_removal_grouped', 'no_matrix', 'no_removal_structured', 'node2vec'])
parser.add_argument("--imitation", type=str, required=False, default='False')  # If we should use imitation learning
parser.add_argument("--model", type=str, required=False, default='PPO', choices=['PPO', 'EpsilonGreedyPPO', 'MaskablePPO'])  # If we should use imitation learning
parser.add_argument("--cloning", type=str, required=False, default='False')
args = parser.parse_args()

dataset = 'CollegeMsg'
my_loader = Loader()
probabilities_df = pd.read_csv(f'ReinforcementLearning/output/probabilities/all_back/{dataset}_from_start.csv').iloc[:, 1:]  # Need to make a loader for this

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

max_steps_per_graph = 25000

    
if args.strategy == 'no_removal':
    # TODO make embedding not constant
    tmp_env = GraphReconstructionNxContidsNoRemoval(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True)  # If doing imitation learning, this is needed
    train_env = GraphReconstructionNxContidsNoRemoval(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True)
    test_env = GraphReconstructionNxContidsNoRemoval(features, thresholds, probabilities, target_graphs, max_steps_per_graph=max_steps_per_graph,  embed_graph=True)
    
    if args.model == 'MaskablePPO':
        tmp_env = ActionMasker(GraphReconstructionNxContidsNoRemoval(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True, all_graphs=target_graphs), mask_fn)  # If doing imitation learning, this is needed
        train_env = ActionMasker(GraphReconstructionNxContidsNoRemoval(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True, all_graphs=target_graphs), mask_fn)
        test_env = ActionMasker(GraphReconstructionNxContidsNoRemoval(features, thresholds, probabilities, target_graphs, max_steps_per_graph=max_steps_per_graph,  embed_graph=True), mask_fn)
        
if args.strategy == 'no_removal_structured':
    tmp_env = GraphReconstructionNxContidsNoRemovalStructured(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True)  # If doing imitation learning, this is needed
    train_env = GraphReconstructionNxContidsNoRemovalStructured(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True)
    test_env = GraphReconstructionNxContidsNoRemovalStructured(features, thresholds, probabilities, target_graphs, max_steps_per_graph=max_steps_per_graph,  embed_graph=True)
    
    if args.model == 'MaskablePPO':
        tmp_env = ActionMasker(GraphReconstructionNxContidsNoRemovalStructured(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True, all_graphs=target_graphs), mask_fn)  # If doing imitation learning, this is needed
        train_env = ActionMasker(GraphReconstructionNxContidsNoRemovalStructured(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True, all_graphs=target_graphs), mask_fn)
        test_env = ActionMasker(GraphReconstructionNxContidsNoRemovalStructured(features, thresholds, probabilities, target_graphs, max_steps_per_graph=max_steps_per_graph,  embed_graph=True), mask_fn)
        
if args.strategy == 'node2vec':
    tmp_env = GraphReconstructionNxNode2Vec(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True)  # If doing imitation learning, this is needed
    train_env = GraphReconstructionNxNode2Vec(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True)
    test_env = GraphReconstructionNxNode2Vec(features, thresholds, probabilities, target_graphs, max_steps_per_graph=max_steps_per_graph,  embed_graph=True)
    
    if args.model == 'MaskablePPO':
        tmp_env = ActionMasker(GraphReconstructionNxNode2Vec(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True, all_graphs=target_graphs), mask_fn)  # If doing imitation learning, this is needed
        train_env = ActionMasker(GraphReconstructionNxNode2Vec(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True, all_graphs=target_graphs), mask_fn)
        test_env = ActionMasker(GraphReconstructionNxNode2Vec(features, thresholds, probabilities, target_graphs, max_steps_per_graph=max_steps_per_graph,  embed_graph=True), mask_fn)
        


elif args.strategy == 'no_removal_grouped':
    # TODO make embedding not constant
    tmp_env = GraphReconstructionNxContidsNoRemovalGrouped(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True)  # If doing imitation learning, this is needed
    train_env = GraphReconstructionNxContidsNoRemovalGrouped(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True)
    test_env = GraphReconstructionNxContidsNoRemovalGrouped(features, thresholds, probabilities, target_graphs, max_steps_per_graph=max_steps_per_graph,  embed_graph=True)
    
    if args.model == 'MaskablePPO':
        tmp_env = ActionMasker(GraphReconstructionNxContidsNoRemovalGrouped(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True, all_graphs=target_graphs), mask_fn)  # If doing imitation learning, this is needed
        train_env = ActionMasker(GraphReconstructionNxContidsNoRemovalGrouped(feature_vectors=features_train, filtration_thresholds=thresholds, probabilities=probabilities_train, target_graphs=target_graphs_train, max_steps_per_graph=max_steps_per_graph, embed_graph=True, all_graphs=target_graphs), mask_fn)
        test_env = ActionMasker(GraphReconstructionNxContidsNoRemovalGrouped(features, thresholds, probabilities, target_graphs, max_steps_per_graph=max_steps_per_graph,  embed_graph=True), mask_fn)
        

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
    print(state_dim, action_dim)

    policy = ImitationPolicyMLP(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 2000
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
max_steps_overall = max_steps_per_graph * num_graphs 
max_steps_train = max_steps_per_graph * len(features_train)

print('Training starting')

model_path = f"ReinforcementLearning/models/nx/latest_model_{args.strategy}_{args.model}.zip"

if args.imitation == 'True':
    model_path = f"ReinforcementLearning/models/nx/latest_model_{args.strategy}_imitation.zip"

# Train PPO with the new environment
# Check if a saved model exists and load it
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}")
    if args.model == 'PPO':
        model = PPO.load(model_path, env=train_env)
    elif args.model == 'EpsilonGreedyPPO':
        model = EpsilonGreedyPPO.load(model_path, env=train_env)
    elif args.model == 'MaskablePPO':
        model = MaskablePPO.load(model_path, env=train_env)
else:
    print("No existing model found. Initializing a new one.")
    
    if args.model == 'PPO':
        model = PPO("MlpPolicy", train_env, verbose=1, gamma=0.95, n_steps=4096, ent_coef=0.9)  # Try reducing gamma or increasing entropy_coefficient or increasing gae_lambda or increasing n_steps
    elif args.model == 'EpsilonGreedyPPO':
        print('EpsilonGreedy Policy')
        model = EpsilonGreedyPPO("MlpPolicy", train_env, verbose=1, gamma=0.99, n_steps=4096, ent_coef=0.1, learning_rate=0.0001, epsilon=1.0)  # Try reducing gamma or increasing entropy_coefficient or increasing gae_lambda or increasing n_steps
    elif args.model == 'MaskablePPO':
        print('MaskablePPO Policy')
        model = MaskablePPO(MaskableActorCriticPolicy, train_env, verbose=1, gamma=0.99, n_steps=4096, ent_coef=0.01, learning_rate=0.0001)  # Try reducing gamma or increasing entropy_coefficient or increasing gae_lambda or increasing n_steps
        
    if args.imitation == 'True':
        with torch.no_grad():
            model.policy.mlp_extractor.policy_net[0].weight.copy_(policy.fc1.weight)
            model.policy.mlp_extractor.policy_net[0].bias.copy_(policy.fc1.bias)
            model.policy.mlp_extractor.policy_net[2].weight.copy_(policy.fc2.weight)
            model.policy.mlp_extractor.policy_net[2].bias.copy_(policy.fc2.bias)
            print("Custom MLP fc3 weight shape:", policy.fc3.weight.shape)
            print("SB3 action_net weight shape:", model.policy.action_net.weight.shape)
            print("Custom MLP fc3 bias shape:", policy.fc3.bias.shape)
            print("SB3 action_net bias shape:", model.policy.action_net.bias.shape)
            model.policy.action_net.weight.copy_(policy.fc3.weight)
            model.policy.action_net.bias.copy_(policy.fc3.bias)

if args.cloning == 'False':
    try:
        # Start training
        print('Starting training')
        callback = StopOnDoneCallback()
        model.learn(total_timesteps=max_steps_overall, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        model.save(model_path)
        print(f"Model saved at {model_path}")
        exit()  # Exit gracefully
        
elif args.cloning == 'True':
    try:
        # Start training
        print('Starting training Behavior Cloning')
        base_decisions = tmp_env.gen_expert_decisions(target_graphs[:10])
        
        # Get the states
        expert_data = []
        states = []
        state, _ = tmp_env.reset()
        print('starting actions')
        for action in base_decisions:
            next_state, _, _, _, _ = tmp_env.step(action)
            expert_data.append((state, action))
            states.append(state)
            state = next_state
        
        formatted_expert_data = []
        for state, action in expert_data:
            # Reshape to ensure that 'obs' and 'acts' are properly formatted
            formatted_expert_data.append({
                'obs': np.expand_dims(state, axis=0),  # Add a batch dimension for the state
                'acts': np.expand_dims(action, axis=0)  # Add a batch dimension for the action
            })
        # formatted_expert_data = {
        #     'obs': states,
        #     'acts': base_decisions
        # }
        
        bc_trainer = bc.BC(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            demonstrations=formatted_expert_data,  # This should contain the state-action pairs from the expert
            rng=np.random.RandomState(), 
            batch_size=1
        )

        # Train the model
        n_epochs = 10  # Example: train for 10 epochs
        n_batches = None  # You can specify the number of batches or leave as None for default behavior
        
        bc_trainer.train(n_epochs=n_epochs, n_batches=n_batches)
        
        done = False
        total_reward = 0
        obs, _ = test_env.reset()

        for _ in range(max_steps_overall):  
            action = bc_trainer.policy.predict(obs)[0]
            obs, reward, done, _, _ = test_env.step(action)
            
            total_reward += reward

            # Completed all graphs
            if done == True:
                break
        print(f'BC accumulated {total_reward} reward points!')
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        model.save(model_path)
        print(f"Model saved at {model_path}")
        exit()  # Exit gracefully
        
print('Training finished')
model.save(model_path)
print(f"Model saved at {model_path}")

# Test trained model on all graphs

obs, info = test_env.reset()

for _ in range(max_steps_overall): 
    if args.model == 'MaskablePPO':
        action, _ = model.predict(obs, action_masks=info["action_mask"])
    else:     
        action, _ = model.predict(obs)
    obs, reward, done, _, info = test_env.step(action)

    # Completed all graphs
    if done == True:
        break

print('Testing finished')

# Visualize the results
animation_path = f'initial_anim_{args.strategy}_{args.model}_nx.mp4'
'''recon_adj_matrices, target_adj_matrices = model.get_final_graphs()
my_visualizer = GraphVisualizer(recon_adj_matrices, target_adj_matrices)
my_visualizer.animate(file_name=animation_path)'''
# TODO Update GraphVisualizer when more awake

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

def create_animation(predicted_graphs, target_graphs, output_file="graph_animation.gif"):
    # Check that both lists have the same number of graphs
    if len(predicted_graphs) != len(target_graphs):
        raise ValueError("Both lists of graphs must have the same length.")
    
    # Set up the figure for side-by-side plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    ax_left, ax_right = axes

    # Layout the nodes for better visualization
    pos_left = {}
    pos_right = {}

    def update_frame(i):
        ax_left.clear()
        ax_right.clear()

        # Draw the 'predicted' graph on the left
        G_pred = predicted_graphs[i]
        pos_left = nx.spring_layout(G_pred, seed=42)  # Use spring layout for the predicted graph
        nx.draw(G_pred, pos=pos_left, ax=ax_left, with_labels=True, node_size=500, node_color="skyblue", edge_color="gray")
        ax_left.set_title("Predicted Graph")
        
        # Draw the 'target' graph on the right
        G_target = target_graphs[i]
        pos_right = nx.spring_layout(G_target, seed=42)  # Use spring layout for the target graph
        nx.draw(G_target, pos=pos_right, ax=ax_right, with_labels=True, node_size=500, node_color="lightgreen", edge_color="gray")
        ax_right.set_title("Target Graph")
        
        ax_left.set_axis_off()
        ax_right.set_axis_off()

    # Create the animation
    ani = FuncAnimation(fig, update_frame, frames=len(predicted_graphs), interval=1000, repeat=False)

    # Save the animation
    ani.save(output_file, writer='imagemagick', fps=1)  # Saves as a .gif file (or change to .mp4)

    plt.show()

# Example usage:
# Assuming you have two lists of DiGraph objects: `predicted_graphs` and `target_graphs`
predicted_graphs, target_graphs = test_env.get_final_graphs()  
with open(f"ReinforcementLearning/output/graphs/{args.strategy}_{args.model}.pkl", "wb") as f:
    pickle.dump((predicted_graphs, target_graphs), f)

print("Graphs saved to 'graphs.pkl'.")

# # === LOAD ===
# with open("graphs.pkl", "rb") as f:
#     loaded_predicted, loaded_target = pickle.load(f)
# if(predicted_graphs is None or target_graphs is None):
#     print('Graphs are none')
# create_animation(predicted_graphs, target_graphs, output_file=animation_path)