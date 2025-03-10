import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

class GraphConstructionEnv(gym.Env):
    def __init__(self, num_nodes=10):
        super(GraphConstructionEnv, self).__init__()
        
        self.num_nodes = num_nodes
        self.graph = nx.Graph()
        self.old_nodes = set()
        self.seen_edges = set()

        # Define action space
        self.action_space = spaces.Tuple((
            spaces.Discrete(4),  # Edge type (0-3)
            spaces.Discrete(num_nodes),  # First node
            spaces.Discrete(num_nodes)   # Second node
        ))

        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_nodes, num_nodes), dtype=np.int32
        )

    def _get_observation(self):
        """Returns the adjacency matrix as the observation."""
        adj_matrix = nx.to_numpy_array(self.graph, dtype=np.int32)
        return adj_matrix

    def step(self, action):
        modification_type, node1, node2 = action

        if node1 == node2:  # Prevent self-loops
            return self._get_observation(), -1, False, {}

        reward = 0

        if modification_type == 0:  # Reuse a seen edge
            if (node1, node2) in self.seen_edges:
                self.graph.add_edge(node1, node2)
                reward = 1
            else:
                reward = -1  # Invalid move
        
        elif modification_type == 1:  # New -> New
            if node1 not in self.old_nodes and node2 not in self.old_nodes:
                self.graph.add_edge(node1, node2)
                self.old_nodes.update([node1, node2])
                reward = 2
            else:
                reward = -1
        
        elif modification_type == 2:  # Old -> New
            if node1 in self.old_nodes and node2 not in self.old_nodes:
                self.graph.add_edge(node1, node2)
                self.old_nodes.add(node2)
                reward = 2
            elif node2 in self.old_nodes and node1 not in self.old_nodes:
                self.graph.add_edge(node1, node2)
                self.old_nodes.add(node1)
                reward = 2
            else:
                reward = -1

        elif modification_type == 3:  # Old -> Old (new edge)
            if node1 in self.old_nodes and node2 in self.old_nodes and not self.graph.has_edge(node1, node2):
                self.graph.add_edge(node1, node2)
                reward = 2
            else:
                reward = -1

        self.seen_edges.add((node1, node2))  # Track seen edges
        done = len(self.graph.edges) >= self.num_nodes * 2  # Stop after a set number of edges

        return self._get_observation(), reward, done, {}

    def reset(self, seed=None, options=None):
        self.graph.clear()
        self.old_nodes.clear()
        self.seen_edges.clear()
        return self._get_observation(), {}