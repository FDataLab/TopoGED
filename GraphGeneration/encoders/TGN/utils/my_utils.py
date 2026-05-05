import numpy as np

class GraphData:  # Dummy class for just assigning attributes
    pass

def build_graph_data(graph, timestamp_start=0):
    edges = list(graph.edges())
    sources = np.array([u for u, v in edges])
    destinations = np.array([v for u, v in edges])
    timestamps = np.arange(len(edges)) + timestamp_start  # unique timestamps
    edge_idxs = np.arange(len(edges))  # unique edge IDs

    data = GraphData()
    data.sources = sources
    data.destinations = destinations
    data.timestamps = timestamps
    data.edge_idxs = edge_idxs
    return data