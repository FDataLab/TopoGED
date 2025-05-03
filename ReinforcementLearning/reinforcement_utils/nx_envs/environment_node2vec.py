# This environment will use Node2Vec to embed nodes, then we will select a random node with a given embedding to work on


import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import networkx as nx
import numpy as np
from sympy import Q
import torch
from torch_geometric.data import Data
from gymnasium import spaces
from collections import defaultdict
from node2vec import Node2Vec
from gensim.models import Word2Vec
import random
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score
from ReinforcementLearning.reinforcement_utils.rewards.dos_reward import get_dos, cosine_similarity
from ReinforcementLearning.reinforcement_utils.models.GCNEmbedder import GCNEmbedder

class GraphReconstructionNxNode2Vec(gym.Env):

    def __init__(self, feature_vectors, filtration_thresholds, probabilities, target_graphs, max_steps_per_graph = 1250, expert_trajectories=None, embed_graph=False, all_graphs=None):
        super(GraphReconstructionNxNode2Vec, self).__init__()
        
        # For guiding construction and comparing to a target
        self.curr_graph = -1  # For tracking which graph we are working on (updated in reset()) (for some reason we need to start with -2)
        self.feature_vectors = feature_vectors  
        self.filtration_thresholds = filtration_thresholds  # Static across all graphs
        self.probabilities = probabilities
        self.target_graphs, self.num_unique_ids = self.modifyGraphIds(target_graphs)  # A list of lists of networkx graphs
        
        # For the action space to work properly with mask
        if all_graphs is not None:
            _, self.num_unique_ids = self.modifyGraphIds(all_graphs)
        
        # Other variables to guide construction
        self.resources = np.zeros(6)  # A vector storing [num old nodes, num new nodes, num edges oo, num edges nn, num edges on, num edges oon]
        self.filtration_vector = np.zeros(20)  # The filtration vector guiding construction
        self.curr_stage = 0  # The current stage of graph filtration (1-10)
        self.nodes_to_place = 0  # Current node budget
        self.edges_to_place = 0  # Current edge budget
        self.steps = 0  # Track the number of steps taken on this graph
        self.max_steps_per_graph = max_steps_per_graph
        self.done = False  # If we have not gone through all graphs, this is False
        self.last_n_actions = []  # Prevents cyclic actions
        
        # Track old nodes carried over from previous graphs
        self.old_nodes_end = 0
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
        
        
        self.in_channels = 4  # The number of features to embed
        self.embedding_dim = 128
        if embed_graph:
            self.feature_dim = len(filtration_thresholds) + len(probabilities[0]) + len(feature_vectors[0])  # Should be 36
            obs_size = self.embedding_dim + self.feature_dim
            self.encoder = GCNEmbedder(in_channels=self.in_channels, hidden_dim=256, out_dim=self.embedding_dim)
            self.observation_space = spaces.Box(low=0, high=2**62, shape=(obs_size,), dtype=np.float32)
            
        # TODO fix
        else:
            self.feature_dim = len(filtration_thresholds) + len(probabilities[0]) + len(feature_vectors[0])  # Should be 36
            obs_size = self.max_num_nodes * self.max_num_nodes + self.feature_dim
            self.observation_space = spaces.Box(low=0, high=2**62, shape=(obs_size,), dtype=np.float32)  # Observation space: Flattened adjacency matrix + thresholds + filtration vector + resources (Updated on each reset)

        # For visualization afterward
        self.pred_graph_storage = []
        self.target_graph_storage = []

        # For the Node2Vec requirements
        self.max_available_embeddings = 15000
        self.node2vec_dim = 64
        self.successful_edges = 0
        self.node2vec_mapping = defaultdict(list)  # Takes a node2vec embedding id and maps it to available nodes
        self.node2vec_id_mapping = []  # Stores (embedding, key) pairs (could also just hash with a tuple in the above)
        
        # The following will be dynamically assigned based on each graph on each reset
        self.action_space = spaces.MultiDiscrete((
            self.max_available_embeddings,  # Action type (0-5)
            self.max_available_embeddings,  # First node ID (sometimes unused)
            self.max_available_embeddings   # Second node ID (sometimes unused)
        ))
        
        
    # Transform the graph into torch_geometric Data for embedding with GCN
    def graphToData(self):
        g = nx.convert_node_labels_to_integers(self.graph)  # Reindex nodes from 0 to N-1

        num_nodes = g.number_of_nodes()
        
        # Stack real node features from graph attributes
        features = []
        for n in range(num_nodes):
            node_data = g.nodes[n]['feat']
            # Extract numerical features from the dictionary (you can choose which ones to use)
            feature_vector = [
                node_data['currDegree'],
                node_data['maxDegree'],
                node_data['Type'],
                node_data['id']
            ]
            features.append(feature_vector)
            
        x = torch.tensor(features, dtype=torch.float)

        # Get directed edge_index
        edges = list(g.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        batch = torch.zeros(num_nodes, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, batch=batch)
        
        
    # Embed the graph for the GCN
    def embedGraph(self):
        # print('Embedding graph')
        graph_data = self.graphToData()
        # print("graph_data.x:", graph_data.x)
        # print("graph_data.x.shape:", graph_data.x.shape)
        # print("graph_data.edge_index.shape:", graph_data.edge_index.shape)
        # print("graph_data.batch.shape:", graph_data.batch.shape)
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
            - Activate old node
            - Activate new node
        """
        
        # For ease of customization
        #node_reward = 5  # Encourage node creation, but it's not as important as edges
        old_edge_reward = 75  # Lower reward for old edges (since they're more frequent)
        other_edge_reward = 150  # High reward for new edges to encourage exploration
        old_node_reward = 3  # Low reward for adding old nodes (since they're more frequent)
        new_node_reward = 15  # Higher reward for new nodes to incentivize exploration
        
        # TODO prob update it to remove nodes from the available
            
        action_type, vec_id_1, vec_id_2 = action  # Unpack our action tuple
        node1, node2 = self.get_nodes_from_embedding(vec_id_1, vec_id_2)  # Get the nodes to work on

        self.steps += 1  # Increment our step counter
        # print(action_type, node1, node2)
        print(action_type)
        # Debugging
        # if(node1 != 0 or node2 != 0):
        #     print('WE DID IT')
            
        # print('Checking: ', self.action_space.sample())
        
        # Ensure node1 != node2 to avoid self-loops (only checked if we are placing an edge)
        '''if action_type < 4 and node1 == node2:
            #print('self loop attempted')
            return self._get_observation(), -2, False, False, {"action_mask": self.get_action_mask()}'''

        reward = -1  # Default penalty for invalid moves

        # Edge type 0: Use a previously seen edge
        if action_type == 0:  
            #print('attempting to add a previously seen edge')
            # We have seen this edge before and it is not already in this graph
            if (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and node1 < self.old_nodes_end and node2 < self.old_nodes_end and node2 in self.edge_bank.get(node1, []) and not self.graph.has_edge(node1, node2):
                if self.graph.nodes[node1]['feat']['currDegree'] < self.graph.nodes[node1]['feat']['maxDegree'] and self.graph.nodes[node2]['feat']['currDegree'] < self.graph.nodes[node2]['feat']['maxDegree'] and self.resources[2] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.graph.nodes[node1]['feat']['currDegree'] += 1
                    self.graph.nodes[node2]['feat']['currDegree'] += 1
                    self.resources[2] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph.add_edge(node1, node2)  # Directed

                    self.successful_edges += 1  # Another successful edge

                    reward = old_edge_reward + self.compute_reward_single_edge(node1, node2)
                    print('add a previously seen edge edge success')


        # Edge type 1: New -> New
        elif action_type == 1:  
            #print(f'attempting to add an edge between two new nodes with node {node1} and {node2}')
            if (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and node1 >= self.old_nodes_end and node2 >= self.old_nodes_end and not self.graph.has_edge(node1, node2) and not self.graph.has_edge(node2, node1):
                if self.graph.nodes[node1]['feat']['currDegree'] < self.graph.nodes[node1]['feat']['maxDegree'] and self.graph.nodes[node2]['feat']['currDegree'] < self.graph.nodes[node2]['feat']['maxDegree'] and self.resources[3] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.graph.nodes[node1]['feat']['currDegree'] += 1
                    self.graph.nodes[node2]['feat']['currDegree'] += 1
                    self.resources[3] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph.add_edge(node1, node2)  # Directed

                    self.successful_edges += 1  # Another successful edge
                    
                    reward = other_edge_reward + self.compute_reward_single_edge(node1, node2)
                    print('add an edge between two new nodes edge success')


        # Edge type 2: Old -> New
        elif action_type == 2:  
            #print('attempting to add an edge between one new and one old node')
            if (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and (node1 < self.old_nodes_end and node2 >= self.old_nodes_end) and not self.graph.has_edge(node1, node2):
                if self.graph.nodes[node1]['feat']['currDegree'] < self.graph.nodes[node1]['feat']['maxDegree'] and self.graph.nodes[node2]['feat']['currDegree'] < self.graph.nodes[node2]['feat']['maxDegree'] and self.resources[4] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.graph.nodes[node1]['feat']['currDegree'] += 1
                    self.graph.nodes[node2]['feat']['currDegree'] += 1
                    self.resources[4] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph.add_edge(node1, node2)  # Directed

                    self.successful_edges += 1  # Another successful edge

                    reward = other_edge_reward + self.compute_reward_single_edge(node1, node2)
                    print('add an edge between one new and one old node edge success')

            elif (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and (node2 < self.old_nodes_end and node1 >= self.old_nodes_end) and not self.graph.has_edge(node2, node1):
                if self.graph.nodes[node1]['feat']['currDegree'] < self.graph.nodes[node1]['feat']['maxDegree'] and self.graph.nodes[node2]['feat']['currDegree'] < self.graph.nodes[node2]['feat']['maxDegree'] and self.resources[4] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.graph.nodes[node1]['feat']['currDegree'] += 1
                    self.graph.nodes[node2]['feat']['currDegree'] += 1
                    self.resources[4] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph.add_edge(node2, node1)  # Directed

                    self.successful_edges += 1  # Another successful edge

                    reward = other_edge_reward + self.compute_reward_single_edge(node2, node1)
                    print('add an edge between one new and one old node edge success')


        # Edge type 3: Old -> Old (new edge)
        elif action_type == 3:  
            #print('attempting to add an edge between two old nodes that havent had an edge')
            if (node1 in self.graph.nodes(data=False) and node2 in self.graph.nodes(data=False)) and node1 < self.old_nodes_end and node2 < self.old_nodes_end and node2 not in self.edge_bank.get(node1, []) and not self.graph.has_edge(node1, node2) and not self.graph.has_edge(node2, node1):
                if self.graph.nodes[node1]['feat']['currDegree'] < self.graph.nodes[node1]['feat']['maxDegree'] and self.graph.nodes[node2]['feat']['currDegree'] < self.graph.nodes[node2]['feat']['maxDegree'] and self.resources[5] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.graph.nodes[node1]['feat']['currDegree'] += 1
                    self.graph.nodes[node2]['feat']['currDegree'] += 1
                    self.resources[5] -= 1
                    self.edges_to_place -= 1
                    
                    self.last_n_actions.append(action_type)  # Add to last successful actions

                    # Add to the graph
                    self.graph.add_edge(node1, node2)  # Directed

                    self.successful_edges += 1  # Another successful edge

                    reward = other_edge_reward + self.compute_reward_single_edge(node1, node2)
                    print('add an edge between two old nodes that havent had an edge success')
            

        # Activate a node we have previously seen
        elif action_type == 4:
            #print('attempting to activate an old node')
             # Check if node1 is an old node, then map the id of the activated node to it for the edge bank
            if self.nodes_to_place > 0 and self.resources[0] > 0 and node1 < self.old_nodes_end and node1 not in self.graph.nodes(data=False) and self.graph.number_of_nodes() < self.target_num_nodes:  # Make sure we have space and havent already mapped this node
                self.nodes_to_place -= 1
                self.total_nodes += 1
                self.resources[0] -= 1
                
                self.graph.add_node(node1, feat={
                    'currDegree': 0,
                    'maxDegree': self.filtration_thresholds[self.curr_stage - 1],
                    'Type': 0,  # This means old
                    'id': node1
                })
                
                self.last_n_actions.append(action_type)  # Add to last successful actions
                
                self.node2vec_mapping[tuple(np.zeros(self.node2vec_dim))].append(node1)  # Add as a new option
                self.generate_node_embeddings(self.graph)
                self._update_action_space()  # Update the action space based on current graph size
                
                
                reward = old_node_reward + self.compute_reward_single_node(node1)
                print(f'activate an old node success with node_id {node1}')
                
        
        # Activate a brand new node
        elif action_type == 5:
            #print('attempting to add a brand new node')
            if self.nodes_to_place > 0 and self.resources[1] > 0 and self.graph.number_of_nodes() < self.target_num_nodes:
                #print(f'Attempting to place a node when new nodes start at index {self.new_nodes_start_idx}')
                self.nodes_to_place -= 1
                self.resources[1] -= 1
                node_id = self.total_nodes  # Set here because its the next available id
                self.total_nodes += 1  # We have added a new node
                self.graph.add_node(node_id, feat={
                    'currDegree': 0,
                    'maxDegree': self.filtration_thresholds[self.curr_stage - 1],
                    'Type': 1,
                    'id': node_id
                })
                
                self.last_n_actions.append(action_type)  # Add to last successful actions
                
                self.node2vec_mapping[tuple(np.zeros(self.node2vec_dim))].append(node_id)  # Add as a new option
                self.generate_node_embeddings(self.graph)
                self._update_action_space()  # Update the action space based on current graph size
                
                
                reward = new_node_reward + self.compute_reward_single_node(node_id)
                print('add a brand new node success')
                
            # TODO what?
            '''    
            elif self.nodes_to_place > 0 and self.resources[1] > 0 and self.new_nodes_start_idx + self.count_new_nodes + 1 >= self.target_num_nodes:
                print('TRYING TO ACTIVATE TOO MANY NEW NODES')
                print(f'need {self.nodes_to_place} nodes and resources are {self.resources}')'''

        # Can make this happen on X step intervals
        if reward >= 0:
            reward += self.compute_reward()  # Make it happen every single time

        # Should we move on to the next filtration threshold
        if self.curr_stage < 10 and self.is_filtration_complete():
            self.curr_stage += 1
            print(f'Stage is now: {self.curr_stage}')
            # Update our budgets
            self.nodes_to_place += self.filtration_vector[self.curr_stage * 2 - 2] - self.filtration_vector[self.curr_stage * 2 - 4]  # Calculate then number of nodes needed now
            self.edges_to_place += self.filtration_vector[self.curr_stage * 2 - 1] - self.filtration_vector[self.curr_stage * 2 - 3]  # Calculate then number of edges needed now
            
            self.pred_graph_storage.append(self.graph)
            self.target_graph_storage.append(self.target_graphs[self.curr_graph][self.curr_stage - 1])
            
            self.generate_node_embeddings(self.graph)

            reward += 500  # Encourage the agent to complete subgraphs

        # Check if we have complete the entire graph
        elif self.curr_stage >= 10:
            completion_status, tmp_reward = self.is_graph_complete()
            if completion_status:
                reward += tmp_reward  # Encourage the agent to complete graphs (reward based on number of edges and nodes placed)

                # Add to storage for later animation
                self.pred_graph_storage.append(self.graph)
                self.target_graph_storage.append(self.target_graphs[self.curr_graph][-1])
                
                obs = self.reset()

                # No rewards given for expert moves
                if self.use_expert:
                    reward = 0

                return self._get_observation(), reward, self.done, False, {"action_mask": self.get_action_mask()}

        
        
        if(self.successful_edges >= 5):
            self.generate_node_embeddings(self.graph)
            self.successful_edges = 0
        
        # No rewards given for expert moves TODO I KINDA WANT TO TRY GIVING REWARD FOR EXPERTS
        if self.use_expert:
            reward = 0
        
        if reward >= 0:
            print(f'On graph number {self.curr_graph} at stage number {self.curr_stage} with resources: {self.resources} and {self.nodes_to_place} nodes and {self.edges_to_place} edges to place')
        
        
        return self._get_observation(), reward, self.done, False, {"action_mask": self.get_action_mask()}
    

    def compute_reward(self):
        reward = 0

        # Compute a reward for each edge type, dependent on the filtration stage
        reward += self.compute_reward_edges()
        reward += self.kiarash_reward()

        return reward
    

    def compute_reward_edges(self):
        '''
        Use set operations to compute a reward by taking the intersection of current edges and the target edges
        '''
        multiplier = 10  # The multiplier for correct edges # * self.steps % 100 (we mod steps to ensure linear penalty)
        
        curr_edges = set(self.graph.edges(data=False))
        target_edges = set(self.target_graphs[self.curr_graph][self.curr_stage - 1].edges(data=False))  # Get the last graph's edges
        
        matches = target_edges.intersection(curr_edges)
        
        reward = len(matches) * multiplier  # This could be better but its a start
    
        return reward
    
    
    def compute_reward_single_edge(self, u, v):
        '''
        A Reward to get an edge exactly correct
        '''
        multiplier = 50  # The reward for correct edges
        
        if ((u, v) in self.target_graphs[self.curr_graph][self.curr_stage - 1].edges(data=False)):
            return multiplier
        else:
            return multiplier * -1
    
    
    def compute_reward_single_node(self, u):
        '''
        A Reward to get a node exactly correct
        '''
        multiplier = 2  # The reward for correct edges
        
        if (u in self.target_graphs[self.curr_graph][self.curr_stage - 1].nodes(data=False)):
            return multiplier
        else:
            return multiplier * -1
        
    
    def kiarash_reward(self):
        multiplier = 25
        
        curr_graph = self.graph 
        target_graph = self.target_graphs[self.curr_graph][self.curr_stage - 1]
        
        L_curr = nx.to_numpy_array(curr_graph, dtype=np.float64)
        L_target = nx.to_numpy_array(target_graph, dtype=np.float64)

        # Compute DOS vectors
        dos1 = get_dos(L_curr, adj=True)
        dos2 = get_dos(L_target, adj=True)

        # Make sure both DOS vectors are the same length (they should be by design)
        if len(dos1) != len(dos2):
            raise ValueError("DOS vectors must be the same length for cosine similarity.")

        # Compute cosine similarity
        return cosine_similarity(dos1, dos2) * multiplier


    # Might put a separate step limit on this as well
    def is_filtration_complete(self):
        if (self.edges_to_place == 0 and self.nodes_to_place == 0) or (self.steps > self.max_steps_per_graph):
            if(self.steps > self.max_steps_per_graph):
                print('TIMEOUT ERROR: FORCING THE AGENT TO MOVE TO THE NEXT FILTRATION')
            return True
        
        return False


    def is_graph_complete(self):
        reward = 1000  # Base reward amount for completing a graph
        multiplier = 10
        
        # Check if all resources used
        if (self.edges_to_place == 0 and self.nodes_to_place == 0):
            return True, reward 
        
        # If we have timed out
        elif (self.steps > self.max_steps_per_graph):
            # Remove reward for each missing node or edge
            reward -= (self.target_num_nodes) * multiplier  # First the nodes
            reward -= (self.filtration_vector[19] - self.graph.number_of_edges()) * multiplier  # Subtract the necessary edges from the amount of existing edges
            
            print('TIMEOUT ERROR: FORCING THE AGENT TO MOVE TO THE NEXT GRAPH')
            
            return True, reward

        return False, 0  # Reward not used in this scenario


    def _get_observation(self):
        """Return the flattened adjacency matrix concatenated with the feature vector."""
        embedded_graph = self.embedGraph()
        
        if embedded_graph.size(0) == 0:
            embedded_graph = torch.zeros(1, embedded_graph.size(1))  # Shape: [1, 128]
        # print('Shape of embedded_graph: ', embedded_graph.shape)
        # print('Shape of flattened graph: ', embedded_graph.flatten().shape)
        if isinstance(embedded_graph, torch.Tensor):
            # If it's a tensor, detach it and then convert it to NumPy
            embedded_graph = embedded_graph.detach().numpy()
    
        return np.concatenate([embedded_graph.flatten(), self.filtration_thresholds, self.filtration_vector, self.resources])


    def _update_action_space(self):
        """Update the action space when the number of nodes changes."""
        # Prevents destroying the action space from too many removals
        '''if self.total_nodes == 0:
            print('Updating action space 0 nodes')
            self.action_space = spaces.MultiDiscrete((
                6,  # Action type (0-5)
                1,  # First node ID (sometimes unused)
                1   # Second node ID (sometimes unused)
            )) 
            
        else:
            print(f'Updating action space else, there are {self.total_nodes} total nodes')
            self.action_space = spaces.MultiDiscrete((
                6,  # Action type (0-5)
                self.total_nodes,  # First node ID (sometimes unused)
                self.total_nodes   # Second node ID (sometimes unused)
            )) 
        print(self.action_space[0], self.action_space[1], self.action_space[2])'''

        # Using this for now before customizing
        self.action_space = spaces.MultiDiscrete((
            self.max_available_embeddings,  # Action type (0-5)
            self.max_available_embeddings,  # First node2vec vector (sometimes unused)
            self.max_available_embeddings   # Second node2vec vector (sometimes unused)
        ))


    def _embedding_rounding(self, embedding):
        # Also makes embedding a tuple for dict hashing
        decimals = 3  # The number of decimals to round to
        embedding = tuple(np.round(embedding, decimals=decimals))
        return embedding


    def generate_node_embeddings(self, graph: nx.DiGraph):
        # Parameters for Node2Vec
        self.node2vec_dim = 64
        walk_length = 20
        num_walks = 30
        p = 1
        q = 0.5
        workers = 1

        tmp_graph = graph.to_undirected()

        node2vec = Node2Vec(tmp_graph, dimensions=self.node2vec_dim, walk_length=walk_length,
                            num_walks=num_walks, workers=workers, p=p, q=q, quiet=True)
        walks = node2vec.walks

        model = Word2Vec(sentences=walks,window=5,min_count=1,sg=1, workers=workers, vector_size=self.node2vec_dim  )

        self.node2vec_mapping = defaultdict(list)

        for node in tmp_graph.nodes():
            # Skip complete nodes
            if graph.nodes[node]['feat']['currDegree'] == graph.nodes[node]['feat']['maxDegree']:
                continue

            try:
                embedding = model.wv[str(node)]
                embedding_key = self._embedding_rounding(embedding)  # You handle rounding
            except KeyError:
                # Disconnected node → assign zero vector
                embedding_key = tuple(np.zeros(self.node2vec_dim))

            # Add for action indexing
            if embedding_key not in self.node2vec_id_mapping:
                self.node2vec_id_mapping.append(embedding_key)

            self.node2vec_mapping[embedding_key].append(node)

    
    def get_nodes_from_embedding(self, embedding_idx_1, embedding_idx_2):
        node_1 = 0
        node_2 = 0

        # Just in case the embedding doesnt work properly
        try:
            node_1 = random.choice(self.node2vec_mapping[self.node2vec_id_mapping[embedding_idx_1]])
            node_2 = random.choice(self.node2vec_mapping[self.node2vec_id_mapping[embedding_idx_2]])

        except:
            print('Missing the list to choose nodes from')
            # print(len(self.node2vec_mapping.keys()))
            # print(embedding_idx_1)
            # print(embedding_idx_2, '\n')

        return node_1, node_2
        
        
    def get_action_mask(self):
        mask = []
        
        first_mask = np.zeros(self.max_available_embeddings, dtype=np.int8)
        first_mask[:6] = 1
        mask.append(first_mask)
        
        if len(self.node2vec_id_mapping) > 0:
            # Fill node idxs
            for _ in range(2):
                submask = np.zeros(self.max_available_embeddings, dtype=np.int8)
                submask[:self.total_nodes] = 1
                mask.append(submask)
        else:
            for _ in range(2):
                submask = np.zeros(self.max_available_embeddings, dtype=np.int8)
                submask[0] = 1
                mask.append(submask)
                
        # Don't bother with embeddings that dont have nodes available        
        for i in range(len(self.node2vec_id_mapping)):
            embedding = self.node2vec_id_mapping[i]
            if(len(self.node2vec_mapping[embedding]) == 0):
                mask[1][i] = 0
                mask[2][i] = 0        
                
        return mask        
    

    def reset(self, seed=None, options=None):
        """Reset the environment and optionally set a new feature vector."""
        super().reset(seed=seed)
        
        self.curr_graph += 1  # Moving on to the next graph 
        # Reset if out of range
        if(self.curr_graph >= len(self.feature_vectors)):
            self.done = True
            self.curr_graph = -1  # Safeguards going out of range in case done doesnt work

        
        print(f'Working on graph number: {self.curr_graph}')
        # Extract our three vectors that guide graph construction
        self.filtration_vector = self.feature_vectors[self.curr_graph]  # Get the current TopER filtration vector
        self.resources = self.probabilities[self.curr_graph]
        self.feature_dim = len(self.filtration_vector) + len(self.filtration_thresholds) + len(self.resources)  # Should be 36

        print(f'Our resources for graph {self.curr_graph}: {self.resources}')

        # Set up other variables and safegaurds
        self.target_num_nodes = self.filtration_vector[-2]  # This position stores the number of nodes in the entire graph
        
        # Update the edge bank
        for u, v in self.graph.edges(data=False):
            # Ensure that u is in the edge bank
            if u not in self.edge_bank:
                self.edge_bank[u] = []
            
            # Add v if not in there already
            if v not in self.edge_bank[u]:
                self.edge_bank[u].append(v)  # Store the old edges
            
        self.graph = nx.DiGraph()  # Make an empty nx graph
        self.curr_stage = 1  # Start at the first stage on each reset
        self.last_n_actions = []  # Prevents cyclic actions
        self.old_nodes_end = self.total_nodes
        self.steps = 0  # Track the number of steps taken on this graph

        # Get budget for first threshold
        self.nodes_to_place = self.filtration_vector[0]
        self.edges_to_place = self.filtration_vector[1]

        self.successful_edges = 0  # Reset

        # Update storage with a new slot
        # self.pred_graph_storage.append([])
        # self.target_graph_storage.append([])

        #self._update_action_space()  # Update the action space based on current graph size
        #print('the space: ', self.action_space)

        return self._get_observation(), {"action_mask": self.get_action_mask()}
    

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
    
    
    def modifyGraphIds(self, graphs):
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

        # Iterate over all graphs in the list of lists (where each graph is a subgraph in the list)
        for graph_list in graphs:
            # Each graph_list contains multiple subgraphs, iterate over the subgraphs
            for graph in graph_list:
                # Create a new dictionary to store the relabeled nodes for the current subgraph
                mapping_for_current_graph = {}
                
                # Iterate over all nodes in the current graph
                for node in graph.nodes:
                    # If the node is already in the node_mapping, use the existing ID
                    if node not in node_mapping:
                        # If not, assign it a new ID
                        node_mapping[node] = new_id
                        new_id += 1
                    
                    # Store the relabeled node ID in the current graph mapping
                    mapping_for_current_graph[node] = node_mapping[node]
                
                # Relabel the nodes in the graph using the mapping
                nx.relabel_nodes(graph, mapping_for_current_graph, copy=True)
        
        return graphs, len(node_mapping)
        
        
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
            'add_old_node': 4,
            'add_new_node': 5,
        }
                
        
        decisions = []  # We will append our decisions to here
        curr_graph = 0
        
        existing_nodes = set()
        existing_edges = set()
        
        # Handle the first graph
        for subgraph in target_graphs[0]:
            # All new nodes and new edges
            # Activate the nodes first:
            for node in subgraph.nodes(data=False):
                if node not in existing_nodes:
                    decisions.append((decision_map['add_new_node'], 0, 0))  # The second two numbers (node1, node2) don't matter
                    existing_nodes.add(node)
            
            for edge in subgraph.edges(data=False):
                if edge not in existing_edges:
                    first, second = edge
                    decisions.append((decision_map['edge_nn'], first, second))
                    existing_nodes.add((first, second))

        old_nodes = existing_nodes
        old_edges = existing_edges
        
        # Loop over all graphs
        for graph_idx in range(1, len(target_graphs)):
            curr_graph += 1  # Move to the next graph
            old_graph = target_graphs[graph_idx - 1][-1]
            
            new_nodes = set()
            new_edges = set()
            
            
            existing_nodes = set()
            existing_edges = set()
            
            for subgraph in target_graphs[graph_idx]:
                for node in subgraph.nodes(data=False):
                    if(node in existing_nodes):
                        continue
                    else:
                        if(node in old_nodes):
                            decisions.append((decision_map['add_old_node'], node, 0))
                        else:
                            decisions.append((decision_map['add_new_node'], 0, 0))
                            new_nodes.add(node)
                        
                        # Map the node to an id
                        existing_nodes.add(node)
                            
                            
                for edge in subgraph.edges(data=False):
                    if edge in existing_edges:
                        continue
                    else:
                        node1, node2 = edge  # Unpack the edge

                        if(edge in old_edges):
                            decisions.append((decision_map['edge_oo'], node1, node2))
                        elif node1 in old_nodes and node2 in old_nodes and edge not in old_edges:
                            decisions.append((decision_map['edge_oon'], node1, node2))
                            new_edges.add((node1, node2))
                        elif node1 in new_nodes and node2 in new_nodes:
                            decisions.append((decision_map['edge_nn'], node1, node2))
                            new_edges.add((node1, node2))
                        elif node in old_nodes and node2 in new_nodes:
                            decisions.append((decision_map['edge_on'], node1, node2))
                            new_edges.add((node1, node2))
                        elif node in new_nodes and node2 in old_nodes:
                            decisions.append((decision_map['edge_on'], node2, node1))
                            new_edges.add((node2, node1))
                        
                        existing_edges.add(edge)  # Error potentially here
                
                
                # I dont expect this to work perfectly, but it should work alright

            old_nodes = old_nodes.intersection(new_nodes)
            old_edges = old_edges.intersection(new_edges)

        return decisions


    def get_final_graphs(self):
        return self.pred_graph_storage, self.target_graph_storage
