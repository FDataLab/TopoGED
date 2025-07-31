import networkx as nx
import numpy as np
import subprocess
import tempfile
import os
import platform

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


def write_graph_to_orca_input(G, file_path):
    G_copy = G.copy()  # 🛡️ Make a copy so original G is unchanged
    G_copy = nx.convert_node_labels_to_integers(G_copy)
    G_copy.remove_edges_from(nx.selfloop_edges(G_copy))  # 🚫 Remove self-loops in the copy

    with open(file_path, 'w') as f:
        f.write(f"{G_copy.number_of_nodes()} {G_copy.number_of_edges()}\n")
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


def get_five_node_graphlet_vector(G, orca_path=None):
    # Set up the binary path
    if orca_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if platform.system() == "Windows":
            orca_path = os.path.join(base_dir, "orca", "orca.exe")  # Windows uses .exe
        else:
            orca_path = os.path.join(base_dir, "orca", "orca")  # Linux does not use .exe

    #print(f"[DEBUG] Using orca_binary_path: {orca_path}")  # DEBUG LINE

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.txt")
        out_path = os.path.join(tmpdir, "output.txt")

        write_graph_to_orca_input(G, in_path)
        # print("Running ORCA with:", [orca_path, "5", in_path, out_path])
        subprocess.run([orca_path, "5", in_path, out_path], check=True)

        # subprocess.run([orca_path, "5", in_path, out_path], check=True)

        orbit_matrix = read_orca_output(out_path)
        graphlet_vector = count_graphlets_from_orbits(orbit_matrix)

    return graphlet_vector