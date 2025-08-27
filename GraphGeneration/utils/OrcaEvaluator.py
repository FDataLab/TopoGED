import networkx as nx
import numpy as np
import subprocess
import tempfile
import os
import platform
import pyfglt
from collections import Counter


# ORCA's orbit-to-graphlet mapping for 5-node graphlets (21 graphlets)
# ORCA paper Appendix B: https://arxiv.org/pdf/1704.06664.pdf
# Each index in this list contains the orbit indices (starting from 0) that belong to a single 5-node graphlet
five_node_orbit_to_graphlet = {
    0:  [0, 1, 2],     # G1
    1:  [3, 4, 5],
    2:  [6, 7, 8],
    3:  [9, 10, 11],
    4:  [12, 13],
    5:  [14, 15, 16],
    6:  [17, 18, 19],
    7:  [20, 21, 22, 23],
    8:  [24, 25],
    9:  [26, 27],
    10: [28, 29, 30],
    11: [31, 32, 33],
    12: [34, 35],
    13: [36, 37, 38],
    14: [39, 40],
    15: [41, 42, 43, 44],
    16: [45, 46, 47],
    17: [48, 49],
    18: [50, 51, 52, 53],
    19: [54, 55, 56, 57],
    20: [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]
}

def write_graph_for_gtrie(G, file_path):
    G_copy = nx.convert_node_labels_to_integers(G, first_label=1)  # nodes start at 1
    G_copy.remove_edges_from(nx.selfloop_edges(G_copy))
    
    with open(file_path, 'w') as f:
        for u, v in G_copy.edges():
            f.write(f"{u} {v}\n")


def write_graph_to_orca_input(G, file_path):
    G_copy = G.copy()  # 🛡️ Make a copy so original G is unchanged
    G_copy = nx.convert_node_labels_to_integers(G_copy)
    G_copy.remove_edges_from(nx.selfloop_edges(G_copy))  # Remove self-loops in the copy

    with open(file_path, 'w') as f:
        f.write(f"{G_copy.number_of_nodes()} {G_copy.number_of_edges()}\n")
        for u, v in G_copy.edges():
            f.write(f"{u} {v}\n")
            

def write_graph_for_osn(G, file_path):
    # Relabel nodes to integers 0..n-1
    G_copy = nx.convert_node_labels_to_integers(G)
    
    n = G_copy.number_of_nodes()
    e = G_copy.number_of_edges()
    
    with open(file_path, 'w') as f:
        # First line: number of nodes and edges
        f.write(f"{n} {e}\n")
        
        # Then the edges
        for u, v in G_copy.edges():
            f.write(f"{u} {v}\n")


def read_orca_output(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return np.array([list(map(int, line.strip().split())) for line in lines])


def count_graphlets_from_orbits(orbit_matrix):
    num_orbits = orbit_matrix.shape[1]
    #print(f"[DEBUG] ORCA orbit matrix shape: {orbit_matrix.shape}")

    orbit_totals = orbit_matrix.sum(axis=0)
    graphlet_counts = []

    for i in range(len(five_node_orbit_to_graphlet)):
        indices = five_node_orbit_to_graphlet[i]
        valid_indices = [idx for idx in indices if 0 <= idx < num_orbits]

        if len(valid_indices) != len(indices):
            for idx in indices:
                if idx not in valid_indices:
                    print(f"[WARNING] Orbit index {idx} is out of bounds (max {num_orbits - 1}) — skipping.")

        count = sum(orbit_totals[idx] for idx in valid_indices)
        #print(f"[DEBUG] Graphlet {i}, orbit indices: {indices}, total orbits: {count}")
        graphlet_counts.append(count)

    return np.array(graphlet_counts)


def get_five_node_graphlet_vector(G, method='orca', binary_path=None, num_samples=20000):
    """
    Unified wrapper for ORCA (exact), GraphletCountOSN (estimated), and pyfglt (exact).
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph
    method : str
        'orca', 'osn', 'gtrie', or 'pyfglt'
    binary_path : str
        Path to compiled binary (ORCA or GraphletCountOSN)
    num_samples : int
        Number of samples (only used for OSN)
    """
    
    
    if method == 'pyfglt':
        # pyfglt expects an adjacency dict
        adj = {n: list(G.neighbors(n)) for n in G.nodes()}
        counts = pyfglt.count_graphlets(adj, k=5)  # 5-node graphlets
        return np.array(counts)

    if binary_path is None or not os.path.exists(binary_path):
        raise FileNotFoundError(f"Binary not found at {binary_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        if method == 'orca':
            in_path = os.path.join(tmpdir, "input.txt")
            out_path = os.path.join(tmpdir, "output.txt")
            write_graph_to_orca_input(G, in_path)
            subprocess.run([binary_path, "5", in_path, out_path], check=True)
            orbit_matrix = read_orca_output(out_path)
            res = count_graphlets_from_orbits(orbit_matrix)
            print(res)
            return res

        elif method == 'osn':
            graph_file = os.path.join(tmpdir, "graph.txt")
            write_graph_for_osn(G, graph_file)
            result = subprocess.run([binary_path, graph_file, str(num_samples)],
                                    capture_output=True, text=True, check=True)
            counts = [int(x) for x in result.stdout.strip().split()]
            return np.array(counts)
        
        elif method == 'gtrie':
            graph_path = f"{tmpdir}/graph.txt"
            output_path = f"{tmpdir}/occurrences.txt"
            write_graph_for_gtrie(G, graph_path)
            
            cmd = [
                binary_path,
                "-s", "5",
                "-g", graph_path,
                "-m", "subgraphs",
                "GraphGeneration/utils/gtrieScanner/lists/undir5.str",
                "-oc", output_path
            ]
                        
            subprocess.run(cmd, check=True)
            
            counts_dict = Counter()
            with open(output_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    motif_code, nodes = line.split(':', 1)
                    motif_code = motif_code.strip()
                    counts_dict[motif_code] += 1

            # If you have a predefined list of motifs in order (from undir5.str), map them to a vector
            predefined_motifs = []
            with open("GraphGeneration/utils/gtrieScanner/lists/undir5.str", 'r') as f:
                for motif in f:
                    predefined_motifs.append(motif.strip())

            # Build the vector
            counts = [counts_dict.get(motif, 0) for motif in predefined_motifs]
        
            print(counts)
            return np.array(counts)
        
        else:
            raise ValueError("method must be 'orca', 'osn', or 'pyfglt'")