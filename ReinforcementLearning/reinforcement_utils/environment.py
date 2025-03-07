import gymnasium as gym
import networkx as nx
import numpy as np
import random
from gymnasium import spaces

class ReconstructionEnv(gym.Env):
    def __init__(self, feature_vector, probabilities, threshold_vals, target_graphs):
        super(ReconstructionEnv, self).__init__()
        
        
        self.action_space = spaces.Discrete(8)
        self.step_counter = 0
        self.node_bank = {}  # It'll be difficult to store these nodes and compare them, as the patterns are important in later graphs
        self.state = nx.Graph()  # Use a regular graph for now, we can move to DiGraphs if need be
        
        
        # Observation space: two feature vectors per graph (30-dimensional for nodes + 5-dimensional for graph stats)
        self.observation_space = spaces.Box(low=0, high=2**16, shape=(35,), dtype=np.int32)  # Feature vector without weights, probabilities, threshold_values
        
        # Initialize variables
        self.graph = {}
        self.node_id_counter = 0
        self.total_edges = 0
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.stage = 0  # The stage of filtration we are working on
        
        # The target graphs, for reference
        self.target_graphs = target_graphs
        self.current_graph_idx = 0  # Index to track the current graph

        # The feature vectors to guide graph reconstruction
        self.feature_vector = feature_vector
        self.probabilities = probabilities
        self.threshold_vals = threshold_vals
        
    
    def step(self, action):
        '''
        Take an action
        
        Action List:
            - Add a new node
            - Add an old node
            - Add an old edge
            - Add an edge between one new and one old node
            - Add an edge between two new nodes
            - Add an edge between two old nodes that didn't have an edge
            - Remove a node
            - Remove an edge
        '''
        # Add a new node with a new id
        if action == 0:
            # We work our way up from the start of the group when making new ids
        
        # Add a node we have seen before (one that has a high degree)
        elif action == 1:
            
            
        # Add an edge from p_oo
        elif action == 1:
            
            
        # Add an edge from p_no
        elif action == 1:
            
            
        # Add an edge from p_nn
        elif action == 1:
            
            
        # Add an edge from p_oon
        elif action == 1:
            
        
        # Remove node
        elif action == 1:
            
            
        # Remove edge
        elif action == 1:
        
        
    
    def calculateContinuousReward(self):
        '''
        Calculate a reward for having more satisfactory nodes/edges
        Also reward for moving closer to a proper graph
        '''
    
    
    
    def calculateIntermittentReward(self):
        '''
        Calculate a reward, using graph edit distance at either each filtration end or the final graph
        '''
        
        

# Notes
# Might need to provide initial node ids

        
# IT DOESNT WORK NEARLY HOW I WANT BUT ITS A GOOD START        

import random
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from networkx.algorithms.isomorphism import GraphMatcher

class GraphEnvironment(gym.Env):
    def __init__(self, target_graphs, target_features, node_bank_capacity=100, max_nodes=10, max_edges=20):
        super(GraphEnvironment, self).__init__()

        self.action_space = spaces.Discrete(4)  # [0] Add node, [1] Add edge, [2] Remove node, [3] Remove edge
        
        # Observation space: two feature vectors per graph (30-dimensional for nodes + 5-dimensional for graph stats)
        self.observation_space = spaces.Box(low=0, high=1, shape=(max_nodes * 2, 30 + 5), dtype=np.float32)
        
        # Initialize variables
        self.node_bank = {}  # Node bank persists across graphs
        self.graph = {}
        self.node_id_counter = 0
        self.total_edges = 0
        self.node_bank_capacity = node_bank_capacity
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.stage = 0
        
        # The target graphs and their feature vectors
        self.target_graphs = target_graphs
        self.target_features = target_features
        self.current_graph_idx = 0  # Index to track the current graph

        # The 30-dimensional feature vector for the graph construction
        self.feature_vector = np.random.rand(30)
        self.stage_features = self.feature_vector.reshape(10, 3)  # 10 stages, 3 values per stage

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state, but keep node_bank
        self.graph = {}
        self.node_id_counter = 0
        self.total_edges = 0
        self.stage = 0
        
        # Get the current graph and its corresponding feature vector
        target_graph = self.target_graphs[self.current_graph_idx]
        target_feature_vector = self.target_features[self.current_graph_idx]

        # Break the feature vector into 10 stages, each with 3 values
        self.stage_features = self.feature_vector.reshape(10, 3)
        
        # Reset observation: empty graph and node features
        node_features = np.zeros((self.max_nodes, 30), dtype=np.float32)
        graph_features = np.zeros((self.max_nodes, 5), dtype=np.float32)  # Placeholder for graph-level stats
        
        # Add the target feature vector to the observation
        target_features = np.expand_dims(target_feature_vector, axis=0)  # Shape (1, 30)
        
        # Combine node features, graph-level features, and target features into the observation
        observation = np.concatenate([node_features, graph_features, target_features], axis=1)
        return observation, {}

    def step(self, action):
        reward = 0
        done = False

        # Retrieve the current stage's target values (node count and edge count)
        target_node_count, target_edge_count, _ = self.stage_features[self.stage]

        # Perform action: Add/remove nodes and edges based on the current stage
        if action == 0:  # Add node
            if len(self.node_bank) < self.node_bank_capacity and self.node_id_counter < self.max_nodes:
                node_id = self.node_id_counter
                self.node_bank[node_id] = np.random.rand(30)  # Random 30-dimensional features for the node
                self.graph[node_id] = []
                self.node_id_counter += 1
                reward = 1  # Reward for successfully adding a node

        elif action == 1:  # Add edge
            if self.total_edges < self.max_edges:
                nodes = list(self.graph.keys())
                if len(nodes) >= 2:
                    node1, node2 = random.sample(nodes, 2)
                    self.graph[node1].append(node2)
                    self.graph[node2].append(node1)
                    self.total_edges += 1
                    reward = 1  # Reward for successfully adding an edge

        elif action == 2:  # Remove node
            if self.node_id_counter > 0:
                node_to_remove = random.choice(list(self.graph.keys()))
                del self.node_bank[node_to_remove]
                del self.graph[node_to_remove]
                reward = -1  # Negative reward for removing a node

        elif action == 3:  # Remove edge
            if self.total_edges > 0:
                node = random.choice(list(self.graph.keys()))
                if self.graph[node]:
                    edge_to_remove = random.choice(self.graph[node])
                    self.graph[node].remove(edge_to_remove)
                    self.graph[edge_to_remove].remove(node)
                    self.total_edges -= 1
                    reward = -1  # Negative reward for removing an edge

        # After each stage, check if the graph construction is finished
        if self.is_stage_complete(target_node_count, target_edge_count):
            self.stage += 1  # Move to the next stage after the current one is completed

        # Create the observation (node features and graph-level features)
        node_features = np.zeros((self.max_nodes, 30), dtype=np.float32)
        for idx, node_id in enumerate(self.graph.keys()):
            if node_id < self.max_nodes:
                node_features[node_id] = self.node_bank[node_id]
        
        # Example graph-level features could include things like the number of edges or average degree
        graph_features = np.zeros((self.max_nodes, 5), dtype=np.float32)  # Placeholder for graph-level stats

        # Add the target feature vector for the current graph to the observation
        target_feature_vector = self.target_features[self.current_graph_idx]
        target_features = np.expand_dims(target_feature_vector, axis=0)  # Shape (1, 30)

        # Combine node features, graph-level features, and target features into one observation
        observation = np.concatenate([node_features, graph_features, target_features], axis=1)

        # After each filtration (stage), check if the current graph is isomorphic to the target graph
        if self.is_isomorphic(self.graph, self.target_graphs[self.current_graph_idx]):
            reward = 10  # Large reward if the current graph is isomorphic to the target graph
            done = True

        # End condition after 10 stages
        if self.stage >= 10:
            done = True
            # Move to the next graph after finishing the current one
            self.current_graph_idx += 1
            if self.current_graph_idx >= len(self.target_graphs):  # If we've finished all graphs, we're done
                done = True

        return observation, reward, done, False, {}


    def is_isomorphic(self, graph1, graph2):
        """Check if two graphs are isomorphic"""
        G1 = nx.Graph(graph1)
        G2 = nx.Graph(graph2)
        matcher = GraphMatcher(G1, G2)
        return matcher.is_isomorphic()


    def is_stage_complete(self, target_node_count, target_edge_count, step_count):
        """Check if the current stage has been completed based on the target node and edge count or taken too many steps to be feasible"""
        current_node_count = len(self.graph)
        current_edge_count = self.total_edges
        
        # Check if the number of nodes and edges match the target for this stage
        if current_node_count >= target_node_count and current_edge_count >= target_edge_count or step_count > self.max_steps_per_stage:
            return True
        return False


    def render(self):
        print(f"Graph with {len(self.graph)} nodes and {self.total_edges} edges.")
        for node, edges in self.graph.items():
            print(f"Node {node}: {edges}")








# MAKING SOME FUNCTIONS TO INCLUDE LATER

# Dont make this a function, this is just templating
def networkx_to_matrix(graph):
    graph = graph.to_undirected()  # Make an undirected graph without edge weights
    adj_matrix = nx.to_numpy_array(graph, dtype=int)


# Use &
def compute_similarity(mat_recon, mat_target):
    res_matrix = mat_recon & mat_target
    similarity_score = res_matrix.sum()
    
    return similarity_score


# Use ^ (this will probably be better to penalize differences, we can apply negative diff_score)
def compute_diff(mat_recon, mat_target):
    res_matrix = mat_recon ^ mat_target
    diff_score = res_matrix.sum()
    
    return diff_score


def construct_empty_matrix(num_old_nodes, num_new_nodes):
    n = num_old_nodes + num_new_nodes  # Compute size of matrix
    matrix = np.zeros((n, n), dtype=int)
    self.old_limit = num_old_nodes
    return matrix


def add_edge_p_oo():
    pass 


def add_edge_p_oo():
    pass 


def add_edge_p_oo():
    pass 


def add_edge_p_oo():
    pass 


def add_node_p_oo():
    pass 


def add_node_p_oo():
    pass 


def remove_node():
    pass 


def remove_edge():
    pass 

