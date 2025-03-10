import gymnasium as gym
import networkx as nx
import numpy as np
from collections import defaultdict
from gymnasium import spaces
from stable_baselines3 import PPO

class GraphReconstructionEnvAdjMat(gym.Env):
    
    def __init__(self, feature_vectors, filtration_thresholds, probabilities, target_graphs, max_steps_per_graph = 1250):
        super(GraphReconstructionEnvAdjMat, self).__init__()

        # For guiding construction and comparing to a target
        self.curr_graph = -1  # For tracking which graph we are working on (updated in reset())
        self.feature_vectors = feature_vectors  
        self.filtration_thresholds = filtration_thresholds  # Static across all graphs
        self.probabilities = probabilities
        self.target_graphs = self.custom_graphs_to_adj(target_graphs)
        
        
        # Other variables to guide construction
        self.new_nodes_start_idx = 0  # Where the new nodes end and the old nodes start
        self.count_old_nodes = -1  # Used for indexing the activated nodes and checking if a node is activated
        self.count_new_nodes = -1  # Used for indexing the activated nodes and checking if a node is activated
        self.activation_order = []  # Used for checking which node to remove on action 7
        self.num_nodes = 0  # The number of nodes in the complete graph
        self.resources = np.zeros(6)  # A vector storing [num old nodes, num new nodes, num edges oo, num edges nn, num edges on, num edges oon]
        self.filtration_vector = np.zeros(20)  # The filtration vector guiding construction
        self.capacities = defaultdict(int)  # Stores a budget for each node based on a filtration vector
        self.curr_edges = set()  # A list of tuples of current edges (becomes the edge bank after construction)
        self.curr_stage = 0  # The current stage of graph filtration (1-10)
        self.nodes_to_place = 0  # Current node budget
        self.edges_to_place = 0  # Current edge budget
        self.steps = 0  # Track the number of steps taken on this graph
        self.max_steps_per_graph = max_steps_per_graph
        self.done = False  # If we have not gone through all graphs, this is False

        # Track old nodes carried over from previous graphs
        self.old_node_map = {}  # Maps the current nodes idx in the matrix to the previous idx it had (used for referencing the edge bank)
        self.old_nodes = set()  # For simplicity, it is the keys of self.old_node_map
        self.edge_bank = set()  # Will store tuples of edges for reference
        self.activated_nodes = set()

        # The following will be dynamically assigned based on each graph on each reset
        self.action_space = None  # A three-tuple of (action, node1, node2)
        self.graph = np.zeros((1, 1), dtype=np.int8)  # An empty binary matrix
        self.max_num_nodes = max(row[18] for row in self.feature_vectors)  # Figure out the maximum number of nodes for the observation space
        
        self.feature_dim = len(filtration_thresholds) + len(probabilities[0]) + len(feature_vectors[0])  # Should be 36
        obs_size = self.max_num_nodes * self.max_num_nodes + self.feature_dim
        self.observation_space = spaces.Box(low=0, high=2**62, shape=(obs_size,), dtype=np.float32)  # Observation space: Flattened adjacency matrix + thresholds + filtration vector + resources (Updated on each reset)

        # For visualization afterward
        self.pred_graph_storage = []
        self.target_graph_storage = []
        
        self.reset()


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
        action_type, node1, node2 = action  # Unpack our action tuple

        # Ensure node1 != node2 to avoid self-loops (only checked if we are placing a node)
        if action_type < 4 and node1 == node2:
            return self._get_observation(), -1, False, {}

        reward = -1  # Default penalty for invalid moves

        # Edge type 0: Use a previously seen edge
        if action_type == 0:  
            # We have seen this edge before and it is not already in this graph
            if (node1 in self.activated_nodes and node2 in self.activated_nodes) and node1 in self.old_nodes and node2 in self.old_nodes and (tuple(sorted([self.old_node_map[node1], self.old_node_map[node2]]))) in self.edge_bank and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[2] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[2] -= 1
                    self.edges_to_place -= 1

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1  # Undirected graph
                    self.curr_edges.add(tuple(sorted([node1, node2])))

                    reward = 1


        # Edge type 1: New -> New
        elif action_type == 1:  
            if (node1 in self.activated_nodes and node2 in self.activated_nodes) and node1 not in self.old_nodes and node2 not in self.old_nodes and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[3] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[3] -= 1
                    self.edges_to_place -= 1

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1
                    self.curr_edges.add(tuple(sorted([node1, node2])))
                    
                    reward = 2


        # Edge type 2: Old -> New
        elif action_type == 2:  
            if (node1 in self.activated_nodes and node2 in self.activated_nodes) and (node1 in self.old_nodes and node2 not in self.old_nodes) and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[4] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[4] -= 1
                    self.edges_to_place -= 1

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1
                    self.curr_edges.add(tuple(sorted([node1, node2])))

                    reward = 2

            elif (node1 in self.activated_nodes and node2 in self.activated_nodes) and (node2 in self.old_nodes and node1 not in self.old_nodes) and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[4] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[4] -= 1
                    self.edges_to_place -= 1

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1
                    self.curr_edges.add(tuple(sorted([node1, node2])))

                    reward = 2


        # Edge type 3: Old -> Old (new edge)
        elif action_type == 3:  
            if (node1 in self.activated_nodes and node2 in self.activated_nodes) and node1 in self.old_nodes and node2 in self.old_nodes and ((self.old_node_map.get(node1, -1), self.old_node_map.get(node2, -1)) not in self.edge_bank) and self.graph[node1][node2] == 0:
                if self.capacities[node1] > 0 and self.capacities[node2] > 0 and self.resources[5] > 0 and self.edges_to_place > 0:
                    # Take away from the budgets
                    self.capacities[node1] -= 1
                    self.capacities[node2] -= 1
                    self.resources[5] -= 1
                    self.edges_to_place -= 1

                    # Add to the graph
                    self.graph[node1][node2] = 1
                    self.graph[node2][node1] = 1
                    self.curr_edges.add(tuple(sorted([node1, node2])))

                    reward = 2
                

        # Remove an edge
        elif action_type == 4:
            if self.graph[node1][node2] == 1:
                self.graph[node1][node2] = 0
                self.graph[node2][node1] = 0
                self.curr_edges.remove(tuple(sorted([node1, node2])))

                # Give them back their degree budget
                self.capacities[node1] += 1
                self.capacities[node2] += 1
                self.edges_to_place += 1  # and the edge budget

                # Figure out where to add back to our budget
                if (node1 < self.new_nodes_start_idx and node2 < self.new_nodes_start_idx) and (self.old_node_map[node1], self.old_node_map[node2]) in self.edge_bank:
                    self.resources[2] += 1  # TODO CHECK THESE INDICES
                elif node1 < self.new_nodes_start_idx and node2 < self.new_nodes_start_idx:
                    self.resources[3] += 1
                elif (node1 < self.new_nodes_start_idx and node2 >= self.new_nodes_start_idx) or (node1 >= self.new_nodes_start_idx and node2 < self.new_nodes_start_idx):
                    self.resources[4] += 1
                elif (node1 >= self.new_nodes_start_idx and node2 >= self.new_nodes_start_idx):
                    self.resources[5] += 1
                
                reward = 0  # Not necessarily good or bad to remove an edge, if done strategically


        # Activate a node we have previously seen
        elif action_type == 5:
             # Check if node1 is an old node, then map the id of the activated node to it for the edge bank
            if self.nodes_to_place > 0 and node1 in self.old_nodes:
                self.count_old_nodes += 1
                self.nodes_to_place -= 1
                self.capacities[self.count_old_nodes] = self.filtration_thresholds[self.curr_stage - 1]
                self.activation_order.append(self.count_old_nodes)
                self.activated_nodes.add(self.count_old_nodes)

                self.old_node_map[self.count_old_nodes] = node1 

                reward = 2
            
        
        # Activate a brand new node
        elif action_type == 6:
            if self.nodes_to_place > 0:
                self.count_new_nodes += 1
                self.nodes_to_place -= 1
                node_id = self.new_nodes_start_idx + self.count_new_nodes
                self.capacities[node_id] = self.filtration_thresholds[self.curr_stage - 1]
                self.activation_order.append(node_id)
                self.activated_nodes.add(node_id)
                reward = 2


        # Remove the most recently activated node
        # NOTE If using a networkx graph, it can remove any node with ease. However for my design of the adjacency matrix, it can only remove the most recently activated
        elif action_type == 7:
            # Check if any nodes have been activated
            if len(self.activation_order) > 0:
                node_type = ''  # For graph cleanup
                node_to_remove = self.activation_order.pop()  # Get the most recently activated node
                self.activated_nodes.remove(node_to_remove)

                # Check node type to add back to our budget
                if(node_to_remove < self.new_nodes_start_idx):
                    self.resources[0] += 1
                    self.count_old_nodes -= 1
                    node_type = 'old'

                elif(node_to_remove >= self.new_nodes_start_idx):
                    self.resources[1] += 1
                    self.count_new_nodes -= 1
                    node_type = 'new'
                
                self.capacities[node_to_remove] = -1

                # Clean up the edges in the graph
                for i in range(self.num_nodes):
                    if(self.graph[i][node_to_remove] == 1):
                        # Remove the node
                        self.graph[i][node_to_remove] = 0
                        self.graph[node_to_remove][i] = 0

                        # We know the node is old
                        if(node_type == 'old'):
                            if(i < self.new_nodes_start_idx):
                                # Old edge, old nodes
                                if((self.old_node_map[node_to_remove], self.old_node_map[i]) in self.edge_bank):
                                    self.resources[2] += 1  # Both old nodes
                                else:
                                    self.resources[5] += 1

                            elif(i >= self.new_nodes_start_idx):
                                self.resources[3] += 1  # One old, one new

                        # We know the node is new
                        elif(node_type == 'new'):
                            if(i < self.new_nodes_start_idx):
                                self.resources[3] += 1  # One old, one new

                            elif(i >= self.new_nodes_start_idx):
                                self.resources[4] += 1  # Both new

                        else:
                            print('LOGIC ERROR ACTION 7')


                reward = 0  # If done strategically, this is a good move


        # Should we move on to the next filtration threshold
        if self.curr_stage < 10 and self.is_filtration_complete():
            self.curr_stage += 1
            # Update our budgets
            self.nodes_to_place += self.filtration_vector[self.curr_stage * 2 - 2] - self.filtration_vector[self.curr_stage * 2 - 4]  # Calculate then number of nodes needed now
            self.edges_to_place += self.filtration_vector[self.curr_stage * 2 - 1] - self.filtration_vector[self.curr_stage * 2 - 3]  # Calculate then number of edges needed now
            
            reward += 5  # Encourage the agent to complete subgraphs

            reward += self.compute_reward()

        # Check if we have complete the entire graph
        elif self.curr_stage >= 10:
            completion_status, tmp_reward = self.is_graph_complete()
            if completion_status:
                reward += tmp_reward  # Encourage the agent to complete graphs (reward based on number of edges and nodes placed)
                reward += self.compute_reward()

                obs = self.reset()

                terminated = self.done
                return np.array([self._get_observation()]), np.array([reward]), np.array([self.done]), np.array([False]), [{}]

        # Add to storage for later animation
        self.pred_graph_storage.append(self.graph)
        self.target_graph_storage.append(self.target_graphs[self.curr_graph])
        
        terminated = self.done
        return np.array([self._get_observation()]), np.array([reward]), np.array([self.done]), np.array([False]), [{}]
    

    def compute_reward(self):
        reward = 0

        # Compute a reward for each edge type, dependent on the filtration stage
        reward += self.compute_reward_edge_oo()
        reward += self.compute_reward_edge_nn()
        reward += self.compute_reward_edge_on()
        reward += self.compute_reward_edge_oon()

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


    # Might put a separate step limit on this as well
    def is_filtration_complete(self):
        if (self.edges_to_place == 0 and self.nodes_to_place == 0) or (self.steps > self.max_steps_per_graph):
            return True
        
        return False


    def is_graph_complete(self):
        reward = 10  # Base reward amount for completing a graph

        # Check if all resources used
        if (self.edges_to_place == 0 and self.nodes_to_place == 0):
            return True, reward 
        
        # If we have timed out
        elif (self.steps > self.max_steps_per_graph):
            # Remove reward for each missing node or edge
            reward -= (self.num_nodes - len(self.activation_order))  # First the nodes
            reward -= (self.filtration_vector[19] - self.len(self.curr_edges))  # Subtract the necessary edges from the amount of existing edges

            return True, reward

        return False, reward  # Reward not used in this scenario


    def _get_observation(self):
        """Return the flattened adjacency matrix concatenated with the feature vector."""
        amt_to_pad = self.max_num_nodes - self.num_nodes
        tmp_graph = np.pad(np.array(self.graph), ((0, amt_to_pad), (0, amt_to_pad)), mode='constant', constant_values=0)  # Since it needs to be a specific size
        return np.concatenate([tmp_graph.flatten(), self.filtration_thresholds, self.filtration_vector, self.resources])


    def _update_action_space(self):
        """Update the action space when the number of nodes changes."""
        self.action_space = spaces.MultiDiscrete((
            5,  # Action type (0-4)
            self.num_nodes,  # First node ID (sometimes unused)
            self.num_nodes   # Second node ID (sometimes unused)
        )) 


    def reset(self, seed=None, options=None):
        """Reset the environment and optionally set a new feature vector."""
        super().reset(seed=seed)

        # Reset if out of range
        if(self.curr_graph >= len(self.feature_vectors)):
            self.done = True
            self.curr_graph = -1  # Safeguards going out of range in case done doesnt work

        self.curr_graph += 1  # Moving on to the next graph 

        # Extract our three vectors that guide graph construction
        self.filtration_vector = self.feature_vectors[self.curr_graph]  # Get the current TopER filtration vector
        self.resources = self.probabilities[self.curr_graph]
        self.feature_dim = len(self.filtration_vector) + len(self.filtration_thresholds) + len(self.resources)  # Should be 36

        # Set up other variables and safegaurds
        self.old_nodes = set(range(self.num_nodes))
        self.num_nodes = self.filtration_vector[-2]  # This position stores the number of nodes in the entire graph
        self.graph = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int8)  # Make an empty binary matrix
        self.edge_bank = set(self.curr_edges)  # Store the old edges
        self.curr_edges = set()  # Clear the set of edges
        self.capacities = {i: -1 for i in range(self.num_nodes)}  # Stores a budget for each node based on a filtration vector
        self.activation_order = []  # No nodes are activated anymore
        self.activated_nodes = set()  # Track the activated nodes (O(1) lookup)
        self.new_nodes_start_idx = self.resources[0]  # This is the end of the old nodes (aka the amount of existing old nodes)
        self.curr_stage = 1  # Start at the first stage on each reset

        # Get budget for first threshold
        self.nodes_to_place = self.filtration_vector[0]
        self.edges_to_place = self.filtration_vector[1]

        # Update storage with a new slot
        self.pred_graph_storage.append([])
        self.target_graph_storage.append([])

        self._update_action_space()  # Update the action space based on current graph size

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
                

    # Get the states to animate
    def get_all_states(self):
        return self.pred_graph_storage, self.target_graph_storage
                
'''
TODO
Rewards for each edge type
Probably add a check when adding edges that the nodes are activated
'''