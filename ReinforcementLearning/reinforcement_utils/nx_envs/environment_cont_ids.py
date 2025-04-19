import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.GCNEmbedder import GCNEmbedder

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
from gymnasium import spaces
from stable_baselines3 import PPO
from sympy import Q
from sklearn.metrics.pairwise import cosine_similarity


class GraphReconstructionNxBase(gym.Env):
    
    def __init__(self, feature_vectors, filtration_thresholds, probabilities, target_graphs, max_steps_per_graph = 1250, expert_trajectories=None, embed_graph=False):
        super(GraphReconstructionNxBase, self).__init__()
        
        # For guiding construction and comparing to a target
        self.curr_graph = -1  # For tracking which graph we are working on (updated in reset()) (for some reason we need to start with -2)
        self.feature_vectors = feature_vectors  
        self.filtration_thresholds = filtration_thresholds  # Static across all graphs
        self.probabilities = probabilities
        self.target_graphs = self.custom_graphs_to_adj(target_graphs)
        
        
        # Other variables to guide construction
        self.resources = np.zeros(6)  # A vector storing [num old nodes, num new nodes, num edges oo, num edges nn, num edges on, num edges oon]
        self.filtration_vector = np.zeros(20)  # The filtration vector guiding construction
        self.curr_edges = set()  # A list of tuples of current edges (becomes the edge bank after construction)
        self.curr_stage = 0  # The current stage of graph filtration (1-10)
        self.nodes_to_place = 0  # Current node budget
        self.edges_to_place = 0  # Current edge budget
        self.steps = 0  # Track the number of steps taken on this graph
        self.max_steps_per_graph = max_steps_per_graph
        self.done = False  # If we have not gone through all graphs, this is False
        self.last_n_actions = []  # Prevents cyclic actions
        
        # Track old nodes carried over from previous graphs
        self.old_node_map = {}  # Maps the current nodes idx in the matrix to the previous idx it had (used for referencing the edge bank)
        self.old_nodes = set()  # For simplicity, it is the keys of self.old_node_map
        self.edge_bank = {
            # 1: [2, 3, 4, ...],
            # 2: [1, 3, 4, ...],
            # ...
        }
        self.total_nodes = 0

        # For imitation learning
        self.use_expert = expert_trajectories is not None  # Updated once completed
        self.expert_trajectories = expert_trajectories
        self.current_expert_index = 0
        
        self.graph = nx.DiGraph()  # An networkx DiGraph  (node features are {Degree: int, currDegree: int, Type: 'Old'/'New'})
        self.max_num_nodes = max(row[18] for row in self.feature_vectors)  # Figure out the maximum number of nodes for the observation space if we are not embedding
        
        # The following will be dynamically assigned based on each graph on each reset
        self.action_space = self.action_space = spaces.MultiDiscrete((
            8,  # Action type (0-7)
            1,  # Node 1 to interact with (expands as it adds nodes)
            1   # Node 2 to interact with (expands as it adds nodes)
        ))  # A three-tuple of (action, node1, node2)
        
        self.in_channels = 3  # The number of features to embed
        self.embedding_dim = 128
        if embed_graph:
            self.feature_dim = len(filtration_thresholds) + len(probabilities[0]) + len(feature_vectors[0])  # Should be 36
            obs_size = self.embedding_dim + self.feature_dim
            self.encoder = GCNEmbedder(in_channels=self.in_channels, hidden_dim=256, out_dim=self.embedding_dim)
            
        else:
            self.feature_dim = len(filtration_thresholds) + len(probabilities[0]) + len(feature_vectors[0])  # Should be 36
            obs_size = self.max_num_nodes * self.max_num_nodes + self.feature_dim
            self.observation_space = spaces.Box(low=0, high=2**62, shape=(obs_size,), dtype=np.float32)  # Observation space: Flattened adjacency matrix + thresholds + filtration vector + resources (Updated on each reset)

        # For visualization afterward
        self.pred_graph_storage = []
        self.target_graph_storage = []
        
        
    #######################################################################
    # Below functions are for embedding the graph with the GCN
    def graphToData(self):
        g = nx.convert_node_labels_to_integers(self.graph)  # Reindex nodes from 0 to N-1

        num_nodes = g.number_of_nodes()
        
        # Stack real node features from graph attributes
        features = [g.nodes[n]['feat'] for n in range(num_nodes)]
        x = torch.tensor(features, dtype=torch.float)

        # Get directed edge_index
        edges = list(g.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        batch = torch.zeros(num_nodes, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, batch=batch)
        
        
    # Embed the graph for the GCN
    def embedGraph(self):
        graph_data = self.graphToData()
        graph_embedding = self.encoder(graph_data.x, graph_data.edge_index, graph_data.batch)
        
        return graph_embedding


    # Take an action
    def step(self, action):
        """
        Take an action
        
        Available Actions:
            - Use a previously seen edge
            - Make an edge between two new nodes
            - Make an edge between one new and one old node
            - Make an edge between two old nodes that did not previously have an edge
            - Remove an edge
            - Activate old node
            - Activate new node
            - Remove node (in reverse activation order)
        """
        
        # For ease of customization
        node_reward = 1
        old_edge_reward = 100
        other_edge_reward = 75
        removal_reward = 0
        repeat_removal_reward = -10
        
        # If we are using imitation learning
        if self.use_expert:
            action = self.expert_trajectories[self.current_expert_index]
            self.current_expert_index += 1
            
        action_type, node1, node2 = action  # Unpack our action tuple
        print(action_type)
        # Ensure node1 != node2 to avoid self-loops (only checked if we are placing a node)
        if action_type < 4 and node1 == node2:
            #print('self loop attempted')
            return self._get_observation(), -2, False, False, {}

        reward = -1  # Default penalty for invalid moves

        # Edge type 0: Use a previously seen edge
        if action_type == 0:  
            #print('attempting to add a previously seen edge')
            # We have seen this edge before and it is not already in this graph
            if (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and node1 in self.old_nodes and node2 in self.old_nodes and (tuple(sorted([self.old_node_map[node1], self.old_node_map[node2]]))) in self.edge_bank and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[2] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[2] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1  # Undirected graph
                    self.curr_edges.add(tuple(sorted([node1, node2])))

                    reward = old_edge_reward
                    #print('add a previously seen edge success')


        # Edge type 1: New -> New
        elif action_type == 1:  
            #print(f'attempting to add an edge between two new nodes with node {node1} and {node2}')
            if (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and node1 not in self.old_nodes and node2 not in self.old_nodes and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[3] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[3] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1
                    self.curr_edges.add(tuple(sorted([node1, node2])))
                    
                    reward = other_edge_reward  
                    #print('add an edge between two new nodes success')


        # Edge type 2: Old -> New
        elif action_type == 2:  
            #print('attempting to add an edge between one new and one old node')
            if (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and (node1 in self.old_nodes and node2 not in self.old_nodes) and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[4] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[4] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1
                    self.curr_edges.add(tuple(sorted([node1, node2])))

                    reward = other_edge_reward
                    #print('add an edge between one new and one old node success')

            elif (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and (node2 in self.old_nodes and node1 not in self.old_nodes) and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[4] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[4] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1
                    self.curr_edges.add(tuple(sorted([node1, node2])))

                    reward = other_edge_reward
                    #print('add an edge between one new and one old node success')


        # Edge type 3: Old -> Old (new edge)
        elif action_type == 3:  
            #print('attempting to add an edge between two old nodes that havent had an edge')
            if (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and node1 in self.old_nodes and node2 in self.old_nodes and ((self.old_node_map.get(node1, -1), self.old_node_map.get(node2, -1)) not in self.edge_bank) and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[5] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[5] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1
                    self.curr_edges.add(tuple(sorted([node1, node2])))

                    reward = other_edge_reward
                    #print('add an edge between two old nodes that havent had an edge success')
                

        # Remove an edge
        elif action_type == 4:
            #print('attempting to remove an edge')
            if(self.graph.has_edge(i, node_to_remove) or self.graph.has_edge(node_to_remove, i)):
                # Remove the node
                removal_status = False
                try:
                    self.graph.remove_edge(i, node_to_remove)
                    removal_status = True
                except:
                    pass 
                if not removal_status:
                    try:
                        self.graph.remove_edge(node_to_remove, i)
                    except:
                        print('Couldn\'t remove specified edge in action node removal')
                self.curr_edges.remove(tuple(sorted([node1, node2])))

                # Give them back their degree budget
                self.capacities[node1] += 1
                self.capacities[node2] += 1
                self.edges_to_place += 1  # and the edge budget
                
                self.last_n_actions.append(action_type)  # Add to last successful actions

                # Figure out where to add back to our budget
                # Two old nodes, previously seen edge
                if (node1 < self.new_nodes_start_idx and node2 < self.new_nodes_start_idx) and tuple(sorted([self.old_node_map[node1], self.old_node_map[node2]])) in self.edge_bank:
                    self.resources[2] += 1  
                # Two new nodes
                elif (node1 >= self.new_nodes_start_idx and node2 >= self.new_nodes_start_idx):
                    self.resources[3] += 1
                # One new, one old
                elif (node1 < self.new_nodes_start_idx and node2 >= self.new_nodes_start_idx) or (node1 >= self.new_nodes_start_idx and node2 < self.new_nodes_start_idx):
                    self.resources[4] += 1
                # Both old, no previous edge
                elif node1 < self.new_nodes_start_idx and node2 < self.new_nodes_start_idx:
                    self.resources[5] += 1
                reward = removal_reward  # Not necessarily good or bad to remove an edge, if done strategically
                #print('remove an edge success')


        # Activate a node we have previously seen
        elif action_type == 5:
            #print('attempting to activate an old node')
             # Check if node1 is an old node, then map the id of the activated node to it for the edge bank
            if self.nodes_to_place > 0 and self.resources[0] > 0 and node1 in self.old_nodes and (node1 not in self.old_node_map.values()) and self.count_old_nodes + 1 < self.new_nodes_start_idx:  # Make sure we have space and havent already mapped this node
                self.count_old_nodes += 1
                self.nodes_to_place -= 1
                self.resources[0] -= 1
                self.capacities[self.count_old_nodes] = self.filtration_thresholds[self.curr_stage - 1]
                
                self.graph.add_node(node1, feat={
                    'currDegree': 0,
                    'maxDegree': self.filtration_thresholds[self.curr_stage - 1],
                    'Type': 1
                })
                
                self.last_n_actions.append(action_type)  # Add to last successful actions

                self.old_node_map[node1] = node1  # Do I need this
                #print(f'mapped slot {self.count_old_nodes} to {node1}')

                reward = node_reward
                #print('activate an old node success')
                
            elif self.nodes_to_place > 0 and self.resources[0] > 0 and self.count_old_nodes + 1 >= self.new_nodes_start_idx:
                print('TRYING TO ACTIVATE TOO MANY OLD NODES')
                print(f'need {self.nodes_to_place} nodes and resources are {self.resources}')
            
        
        # Activate a brand new node
        elif action_type == 6:
            #print('attempting to add a brand new node')
            if self.nodes_to_place > 0 and self.resources[1] > 0 and self.new_nodes_start_idx + self.count_new_nodes + 1 < self.num_nodes and node1 not in self.graph.nodes(data=False):
                #print(f'Attempting to place a node when new nodes start at index {self.new_nodes_start_idx}')
                self.nodes_to_place -= 1
                self.resources[1] -= 1
                self.graph.add_node(node1, feat={
                    'currDegree': 0,
                    'maxDegree': self.filtration_thresholds[self.curr_stage - 1],
                    'Type': 1
                })
                
                self.last_n_actions.append(action_type)  # Add to last successful actions
                
                reward = node_reward
                #print('add a brand new node success')
                
            elif self.nodes_to_place > 0 and self.resources[1] > 0 and self.new_nodes_start_idx + self.count_new_nodes + 1 >= self.num_nodes:
                print('TRYING TO ACTIVATE TOO MANY NEW NODES')
                print(f'need {self.nodes_to_place} nodes and resources are {self.resources}')


        # Remove the most recently activated node
        # NOTE If using a networkx graph, it can remove any node with ease. However for my design of the adjacency matrix, it can only remove the most recently activated
        elif action_type == 7:
            #print('attempting to remove node')
            # Check if any nodes have been activated
            if len(self.graph.number_of_nodes()) > 0:
                node_type = ''  # For graph cleanup
                node_to_remove = node1 
                
                reward = removal_reward  # If done strategically, this is a good move
                self.last_n_actions.append(action_type)  # Add to last successful actions
                if (5 in self.last_n_actions or 6 in self.last_n_actions):
                    reward += repeat_removal_reward                
                

                # Check node type to add back to our budget
                if(node_to_remove < self.new_nodes_start_idx):
                    self.resources[0] += 1
                    self.count_old_nodes -= 1
                    node_type = 'old'

                elif(node_to_remove >= self.new_nodes_start_idx):
                    self.resources[1] += 1
                    self.count_new_nodes -= 1
                    node_type = 'new'
                    
                self.nodes_to_place += 1
                self.capacities[node_to_remove] = -1

                # Clean up the edges in the graph
                for i in range(self.num_nodes):
                    if(self.graph.has_edge(i, node_to_remove) or self.graph.has_edge(node_to_remove, i)):
                        # Remove the node
                        removal_status = False
                        try:
                            self.graph.remove_edge(i, node_to_remove)
                            removal_status = True
                        except:
                            pass 
                        if not removal_status:
                            try:
                                self.graph.remove_edge(node_to_remove, i)
                            except:
                                print('Couldn\'t remove specified edge in action node removal')
                        self.curr_edges.remove(tuple(sorted([i, node_to_remove])))
                        self.edges_to_place += 1

                        # We know the node is old
                        if(node_type == 'old'):
                            if(i < self.new_nodes_start_idx):
                                # Old edge, old nodes
                                if(tuple(sorted([self.old_node_map[node_to_remove], self.old_node_map[i]])) in self.edge_bank):
                                    self.resources[2] += 1  # Both old nodes
                                else:
                                    self.resources[5] += 1
                            # One old, one new
                            elif(i >= self.new_nodes_start_idx):
                                self.resources[4] += 1  # One old, one new

                        # We know the node is new
                        elif(node_type == 'new'):
                            if(i < self.new_nodes_start_idx):
                                self.resources[4] += 1  # One old, one new

                            elif(i >= self.new_nodes_start_idx):
                                self.resources[3] += 1  # Both new

                        else:
                            print('LOGIC ERROR ACTION 7')

                #print('remove node success')


        # Should we move on to the next filtration threshold
        if self.curr_stage < 10 and self.is_filtration_complete():
            self.curr_stage += 1
            print(f'Stage is now: {self.curr_stage}')
            # Update our budgets
            self.nodes_to_place += self.filtration_vector[self.curr_stage * 2 - 2] - self.filtration_vector[self.curr_stage * 2 - 4]  # Calculate then number of nodes needed now
            self.edges_to_place += self.filtration_vector[self.curr_stage * 2 - 1] - self.filtration_vector[self.curr_stage * 2 - 3]  # Calculate then number of edges needed now
            
            reward += 500  # Encourage the agent to complete subgraphs

            reward += self.compute_reward()  # Make it happen every single time

        # Check if we have complete the entire graph
        elif self.curr_stage >= 10:
            completion_status, tmp_reward = self.is_graph_complete()
            if completion_status:
                reward += tmp_reward  # Encourage the agent to complete graphs (reward based on number of edges and nodes placed)
                reward += self.compute_reward()

                obs = self.reset()

                # No rewards given for expert moves
                if self.use_expert:
                    reward = 0

                return self._get_observation(), reward, self.done, False, {}

        # Add to storage for later animation
        self.pred_graph_storage.append(self.graph)
        self.target_graph_storage.append(self.target_graphs[self.curr_graph])
        
        if(len(self.last_n_actions) > 3):
            self.last_n_actions.pop(0)
        
        # No rewards given for expert moves TODO I KINDA WANT TO TRY GIVING REWARD FOR EXPERTS
        if self.use_expert:
            reward = 0
        
        if reward >= 0:
            print(f'On graph number {self.curr_graph} at stage number {self.curr_stage} with resources: {self.resources} and {self.nodes_to_place} nodes and {self.edges_to_place} edges to place')
        
        return self._get_observation(), reward, self.done, False, {}
    

    def compute_reward(self):
        reward = 0

        # Compute a reward for each edge type, dependent on the filtration stage
        reward += self.compute_reward_edge_oo()
        reward += self.compute_reward_edge_nn()
        reward += self.compute_reward_edge_on()
        reward += self.compute_reward_edge_oon()
        reward += self.compute_eigen_reward()

        return reward
    
    
    def compute_eigen_reward(self):
        target_mat = np.array(self.target_graphs[self.curr_graph][self.curr_stage - 1])
        curr_mat = np.array(self.graph)
        
        # Sort and get the top 10 eigenvalues (use absolute values in case of complex eigenvalues)
        eigenvalues_target = np.linalg.eigvals(target_mat)
        eigenvalues_curr = np.linalg.eigvals(curr_mat)
        
        # Sort the eigenvalues in descending order and take the top k
        top_k_eigenvalues_target = np.sort(np.abs(eigenvalues_target))[-10:]  # Top 10 eigenvalues (use absolute values)
        top_k_eigenvalues_curr = np.sort(np.abs(eigenvalues_curr))[-10:]  # Top 10 eigenvalues (use absolute values)

        # Define the reward based on similarity (Cosine Similarity or Euclidean Distance)

        # Cosine similarity
        similarity = cosine_similarity(top_k_eigenvalues_target.reshape(1, -1), top_k_eigenvalues_curr.reshape(1, -1))[0][0]

        reward = similarity * 25  # Higher similarity yields higher rewards (25 is our scalar since this is important)

        return reward
    

    def compute_reward_edge_oo(self):
        reward = 0



        return reward


    def compute_reward_edge_nn(self):
        reward = 0


        
        return reward
    

    def compute_reward_edge_on(self):
        reward = 0


        
        return reward
    

    def compute_reward_edge_oon(self):
        reward = 0


        
        return reward
    
    
    def kiarash_reward(self):
        pass 


    # Might put a separate step limit on this as well
    def is_filtration_complete(self):
        if (self.edges_to_place == 0 and self.nodes_to_place == 0) or (self.steps > self.max_steps_per_graph):
            return True
        
        return False


    def is_graph_complete(self):
        reward = 1000  # Base reward amount for completing a graph

        # Check if all resources used
        if (self.edges_to_place == 0 and self.nodes_to_place == 0):
            return True, reward 
        
        # If we have timed out
        elif (self.steps > self.max_steps_per_graph):
            # Remove reward for each missing node or edge
            reward -= (self.num_nodes)  # First the nodes
            reward -= (self.filtration_vector[19] - self.len(self.curr_edges))  # Subtract the necessary edges from the amount of existing edges

            return True, reward

        return False, reward  # Reward not used in this scenario


    def _get_observation(self):
        """Return the flattened adjacency matrix concatenated with the feature vector."""
        embedded_graph = self.embedding_dim()
        return np.concatenate([embedded_graph, self.filtration_thresholds, self.filtration_vector, self.resources])


    def _update_action_space(self):
        """Update the action space when the number of nodes changes."""
        # Prevents destroying the action space from too many removals
        if self.total_nodes == 0:
            self.action_space = spaces.MultiDiscrete((
            8,  # Action type (0-7)
            1,  # First node ID (sometimes unused)
            1   # Second node ID (sometimes unused)
        )) 
            
        else:
            self.action_space = spaces.MultiDiscrete((
                8,  # Action type (0-7)
                self.total_nodes,  # First node ID (sometimes unused)
                self.total_nodes   # Second node ID (sometimes unused)
            )) 


    def reset(self, seed=None, options=None):
        """Reset the environment and optionally set a new feature vector."""
        super().reset(seed=seed)

        # Reset if out of range
        if(self.curr_graph >= len(self.feature_vectors)):
            self.done = True
            self.curr_graph = -1  # Safeguards going out of range in case done doesnt work

        self.curr_graph += 1  # Moving on to the next graph 
        print(f'Working on graph number: {self.curr_graph}')
        # Extract our three vectors that guide graph construction
        self.filtration_vector = self.feature_vectors[self.curr_graph]  # Get the current TopER filtration vector
        self.resources = self.probabilities[self.curr_graph]
        self.feature_dim = len(self.filtration_vector) + len(self.filtration_thresholds) + len(self.resources)  # Should be 36

        print(f'Our resources for graph {self.curr_graph}: {self.resources}')

        # Set up other variables and safegaurds
        self.old_nodes = set(range(self.num_nodes))
        self.num_nodes = self.filtration_vector[-2]  # This position stores the number of nodes in the entire graph
        self.graph = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int8)  # Make an empty binary matrix
        self.edge_bank = set(self.curr_edges)  # Store the old edges
        self.curr_edges = set()  # Clear the set of edges
        self.curr_stage = 1  # Start at the first stage on each reset
        self.last_n_actions = []  # Prevents cyclic actions

        # Get budget for first threshold
        self.nodes_to_place = self.filtration_vector[0]
        self.edges_to_place = self.filtration_vector[1]

        # Update storage with a new slot
        self.pred_graph_storage.append([])
        self.target_graph_storage.append([])

        self._update_action_space()  # Update the action space based on current graph size
        #print('the space: ', self.action_space)

        return self._get_observation(), {}
    

    def render(self, mode="human"):
        print("Adjacency Matrix:\n", self.graph)
        print("Feature Vector:", self.feature_vector)
        print("Resources: ", self.resources)


    # This should somewhat fix the error of different adjacency matrices representing the same graph
    def custom_graphs_to_adj(self, graphs):
        """
        Let old nodes take up the first part, then new nodes, all nodes are ordered by degree in their respective part
        We compare our current subgraph to the previous complete graph
        
        Input:
            graphs (list(list(nx.Graph))): A list of lists of subgraphs for each timestamp
        """
        adj_matrix_collection = [[]]
        curr_graph_idx = 0
        
        # First graph has different processing in that every node is new
        for graph in graphs[0]:
            curr_graph = graph.to_undirected()
            complete_graph = graphs[0][-1].to_undirected()
            nodes = list(complete_graph.nodes())
            nodes_ordered = sorted(nodes, key=lambda x: complete_graph.degree(x), reverse=True)  # Sort all nodes by their degree
            
            adj_matrix = np.zeros((len(nodes_ordered), len(nodes_ordered)), dtype=int)  # Create an empty matrix
            
            # Fill our adjacency matrix
            for i, node1 in enumerate(nodes_ordered):
                for j, node2 in enumerate(nodes_ordered):
                    if curr_graph.has_edge(node1, node2):  # Check if there's an edge in the new graph
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1
                        
            adj_matrix_collection[curr_graph_idx].append(adj_matrix)  # Add to our matrices
                        
        for i in range(1, len(graphs)):
            # For processing in order later
            adj_matrix_collection.append([])
            curr_graph_idx += 1
            
            # Get the previous complete graph to get old nodes from
            prev_graph = graphs[i - 1][-1]  
            prev_graph = prev_graph.to_undirected()

            # To get the nodes
            complete_graph = graphs[i][-1].to_undirected()
            all_nodes = set(list(complete_graph.nodes()))
            
            old_nodes = set(list(prev_graph.nodes()))

            
            # Get the nodes by type
            curr_old_nodes = old_nodes.intersection(all_nodes)  # Get the old nodes that are in this current graph
            curr_new_nodes = all_nodes - curr_old_nodes  # Get the new nodes in this graph
            

            old_nodes_ordered = sorted(curr_old_nodes, key=lambda x: complete_graph.degree(x), reverse=True)
            new_nodes_ordered = sorted(curr_new_nodes, key=lambda x: complete_graph.degree(x), reverse=True)
            
            nodes_ordered = old_nodes_ordered + new_nodes_ordered
            
            # Loop over the subgraphs from the current graph filtration
            for subgraph in graphs[i]:
                subgraph = subgraph.to_undirected()
                
                adj_matrix = np.zeros((len(nodes_ordered), len(nodes_ordered)), dtype=int)  # Create an empty matrix
            
                # Fill our adjacency matrix
                for i, node1 in enumerate(nodes_ordered):
                    for j, node2 in enumerate(nodes_ordered):
                        if subgraph.has_edge(node1, node2):  # Check if there's an edge in the new graph
                            adj_matrix[i, j] = 1
                            adj_matrix[j, i] = 1
                            
                adj_matrix_collection[curr_graph_idx].append(adj_matrix)
                
        
        return adj_matrix_collection
                
    
    def gen_expert_decisions(self, target_graphs):
        # - Make an edge between two new nodes
        # - Make an edge between one new and one old node
        # - Make an edge between two old nodes that did not previously have an edge
        # - Remove an edge
        # - Activate old node
        # - Activate new node
        # - Remove node (in reverse activation order)
        # I honestly don't know how I'd make this right now
        
        decision_map = {
            'edge_oo': 0,
            'edge_on': 1,
            'edge_nn': 2,
            'edge_oon': 3,
            'edge_remove': 4,  # Not needed
            'add_old_node': 5,
            'add_new_node': 6,
            'remove_node': 7  # Not needed
        }
                
        
        decisions = []  # We will append our decisions to here
        curr_graph = 0
        
        # Handle the first graph
        for subgraph in target_graphs[0]:
            # All new nodes and new edges
            # Activate the nodes first:
            for node in subgraph.nodes():
                decisions.append((decision_map['add_new_node'], 0, 0))  # The second two numbers (node1, node2) don't matter
            
            for edge in subgraph.edges():
                first, second = tuple(sorted([edge[0], edge[1]]))
                decisions.append((decision_map['edge_nn'], first, second))
        
        # Loop over all graphs
        for graph_idx in range(1, len(target_graphs)):
            curr_graph += 1  # Move to the next graph
            old_graph = target_graphs[graph_idx - 1][-1]
            old_graph_nodes = set(old_graph.nodes())
            old_graph_edges = set(old_graph.edges())
            old_graph_node_map = {}
            old_node_idx = 0
            # Expecting error here
            new_node_idx = len(set(target_graphs[graph_idx][-1].nodes()).intersection(old_graph_nodes))   # Calculate where the new nodes start
            
            existing_nodes = set()
            existing_edges = set()
            
            for subgraph in target_graphs[graph_idx]:
                # We want to add nodes, then edges
                
                # Take the old final graph, make the node ids into a set
                #
                
                # We need to figure out how many of each are needed, we can apply probabilities for this
            
                for node in subgraph.nodes():
                    if(node in existing_nodes):
                        continue
                    else:
                        if(node in old_graph_nodes):
                            decisions.append((decision_map['add_old_node'], 0, 0))
                            old_graph_node_map[node] = old_node_idx  # assign the mapping
                            old_node_idx += 1
                        else:
                            decisions.append((decision_map['add_new_node'], 0, 0))
                            old_graph_node_map[node] = new_node_idx  # assign the mapping
                            new_node_idx += 1
                        
                        # Map the node to an id
                        existing_nodes.add(node)
                            
                            
                for edge in subgraph.edges():
                    if edge in existing_edges:
                        continue
                    else:
                        node1, node2 = edge  # Unpack the edge
                        # Gotta figure out what X, X will be
                        if(edge in old_graph_edges):
                            first, second = tuple(sorted([old_graph_node_map[node1], old_graph_node_map[node2]]))  # since i have it sorted in impelmentation
                            decisions.append((decision_map['edge_oo'], first, second))
                        elif node1 in old_graph_nodes and node2 in old_graph_nodes and edge not in old_graph_edges:
                            first, second = tuple(sorted([old_graph_node_map[node1], old_graph_node_map[node2]]))  # since i have it sorted in impelmentation
                            decisions.append((decision_map['edge_oon'], first, second))
                        elif node1 not in old_graph_nodes and node2 not in old_graph_nodes:
                            first, second = tuple(sorted([old_graph_node_map[node1], old_graph_node_map[node2]]))  # since i have it sorted in impelmentation
                            decisions.append((decision_map['edge_nn'], first, second))
                        else:
                            first, second = tuple(sorted([old_graph_node_map[node1], old_graph_node_map[node2]]))  # since i have it sorted in impelmentation
                            decisions.append((decision_map['edge_on'], first, second))
                        
                        existing_edges.add(edge)  # Error potentially here
                
                
                # I dont expect this to work perfectly, but it should work alright

        return decisions

    # Get the states to animate
    def get_all_states(self):
        return self.pred_graph_storage, self.target_graph_storage
                
'''
TODO
Rewards for each edge type
Probably add a check when adding edges that the nodes are activated
'''