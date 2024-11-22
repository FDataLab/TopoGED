import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# Function to generate graphs with a cosine cycle pattern for nodes with random edges
def generate_cosine_pattern_graphs(num_graphs, max_nodes, avg_edges_per_node, period_days, start_size):
    graphs = []
    
    # Cosine pattern with a cycle (adjusted period for nodes)
    node_pattern = (max_nodes / 2) * (1 + np.cos(2 * np.pi * np.arange(num_graphs) / period_days))

    # Create graphs based on the node pattern
    for i in range(num_graphs):
        # Add more randomness to the number of nodes around the cosine pattern to increase variation
        num_nodes = start_size + max(2, min(int(node_pattern[i] + random.uniform(-0.3 * max_nodes, 0.3 * max_nodes)), max_nodes))

        # Create an empty graph with num_nodes
        G = nx.empty_graph(num_nodes)

        # Ensure each node has at least one edge
        for node in G.nodes():
            possible_edges = [edge for edge in nx.non_edges(G) if node in edge]
            if possible_edges:
                selected_edge = random.choice(possible_edges)
                G.add_edge(*selected_edge)
        
        # Randomly add additional edges based on the average number of edges per node
        num_edges = avg_edges_per_node * num_nodes
        possible_edges = list(nx.non_edges(G))
        if num_edges > len(possible_edges):
            num_edges = len(possible_edges)
        selected_edges = np.random.choice(len(possible_edges), int(num_edges), replace=False)
        G.add_edges_from([possible_edges[j] for j in selected_edges])

        # Print the graph details for verification
        print(f"Graph {i}: Nodes = {G.number_of_nodes()}, Edges = {G.number_of_edges()}")
        graphs.append(G)

    return graphs

if __name__ == "__main__":
    # Parameters for graph generation
    num_graphs = 100 # Number of graphs for training
    max_nodes = 100  # Maximum number of nodes
    avg_edges_per_node = 10  # Average number of edges per node
    period_days = 30  # Set the period of the cosine cycle
    start_size = 10  # Starting size for nodes

    # Generate graphs with cosine patterns for nodes and random edges
    synthetic_graphs = generate_cosine_pattern_graphs(num_graphs, max_nodes, avg_edges_per_node, period_days, start_size)

    # Optional: Plot the number of nodes over time
    node_pattern = (max_nodes / 2) * (1 + np.cos(2 * np.pi * np.arange(num_graphs) / period_days))
    num_nodes_with_randomness = [
        start_size + max(2, min(int(n + np.random.uniform(-0.3 * max_nodes, 0.3 * max_nodes)), max_nodes))
        for n in node_pattern
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(num_nodes_with_randomness, label="Number of Nodes", color="blue")
    plt.title(f"Number of Nodes Over Time with {period_days}-Day Cosine Pattern")
    plt.xlabel("Time (Graph Index)")
    plt.ylabel("Number of Nodes")
    plt.grid(True)
    plt.legend()
    plt.show()
