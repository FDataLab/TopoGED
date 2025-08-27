import os
from collections import defaultdict

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader

import time
from GraphGeneration.utils.OrcaEvaluator import get_five_node_graphlet_vector
import statistics 

from math import comb
def get_num_samples(G, percentage=1.0):
    """
    Compute number of 5-node subgraphs to sample based on a percentage.
    
    Args:
        G: networkx Graph
        percentage: float between 0 and 1, fraction of all possible subgraphs
    Returns:
        num_samples: int, capped at the maximum possible
    """
    n = G.number_of_nodes()
    max_subgraphs = comb(n, 5) if n >= 5 else 0
    num_samples = int(max_subgraphs * percentage)
    return max(1, num_samples)  # at least 1 sample

my_loader = Loader()
graph_dataset_tmp = my_loader.load_data(dataset='CollegeMsg', activation='Degree', type='subgraphs')
graph_dataset = [tmp[-1] for tmp in graph_dataset_tmp]

# Paths to binaries (replace with actual paths)
orca_path = "GraphGeneration/utils/orca/orca"
osn_path = "GraphGeneration/utils/GraphletCountOSN/build/subgraphCounts"
gtrie_path = "GraphGeneration/utils/gtrieScanner/gtrieScanner"

num_samples = 20000  # for GraphletCountOSN

methods = [
    ("ORCA", "orca", orca_path),
    ("Gtrie", "gtrie", gtrie_path),
    # ("pyfglt", "pyfglt", None)
]

timings = defaultdict(list)

for i, G in enumerate(graph_dataset):
    print(f"\nGraph {i+1}/{len(graph_dataset)}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    G_undirected = G.to_undirected()

    for name, method, path in methods:
        
        if method == "osn":            
            num_samples = get_num_samples(G_undirected, percentage=0.8)
            num_samples = min(num_samples, 100)
        
        start_time = time.time()
        counts = get_five_node_graphlet_vector(
            G_undirected,
            method=method,
            binary_path=path,
            num_samples=num_samples
        )
        elapsed = time.time() - start_time
        timings[name].append(elapsed)   # store all timings
        print(f"{name} completed in {elapsed:.3f}s")

# Compute averages
print("\n=== Average runtimes across dataset ===")
avg_timings = {name: statistics.mean(times) for name, times in timings.items()}
for name, avg in avg_timings.items():
    print(f"{name}: {avg:.3f}s")

# Find fastest method on average
fastest = min(avg_timings, key=avg_timings.get)
print(f"\nFastest method on average: {fastest} ({avg_timings[fastest]:.3f}s)")