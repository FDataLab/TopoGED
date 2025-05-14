import numpy as np 
import networkx as nx
import pandas as pd 
import matplotlib.pyplot as plt 
import random
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from GraphGeneration.utils.Evaluator import Evaluator

# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 'networkcentra', 'networkcindicator', 'networkcoindash', 'networkdgd', 'networkiconomi', 'Reddit_B'])
args = parser.parse_args()


def build_accumulating_filtration_sequence_with_edgebank(embedding, p_old_nodes, E_oo, E_nn, E_on, E_oon, edgebank=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    V_total = int(embedding[-1][0])
    E_total = int(embedding[-1][1])
    W_total = embedding[-1][2] 

    # Sample old/new node labels
    is_old = np.random.binomial(1, p_old_nodes / V_total, V_total)
    old_nodes = [i for i in range(V_total) if is_old[i] == 1]
    new_nodes = [i for i in range(V_total) if is_old[i] == 0]
    all_nodes = old_nodes + new_nodes

    edges = set()
    edge_type_map = {}  # For calculating AUC scores later 

    def sample_edges(src_list, dst_list, count, edge_type=None):
        sampled = set()
        attempts = 0

        if edge_type == "o-o-bank" and edgebank is not None:
            for u in src_list:
                if u in edgebank:
                    for v in edgebank[u]:
                        if v in dst_list and u != v and (u, v) not in edges:
                            sampled.add((u, v))
                            edge_type_map.setdefault(edge_type, []).append((u, v))
                            edges.add((u, v))
                            if len(sampled) >= count:
                                return list(sampled)

        if edge_type == "o-o-nobank" and edgebank is not None:
            for u in src_list:
                if u in edgebank:
                    for v in edgebank[u]:
                        if v in dst_list and u != v and (u, v) not in edges:
                            sampled.add((u, v))
                            edge_type_map.setdefault(edge_type, []).append((u, v))
                            edges.add((u, v))
                            if len(sampled) >= count:
                                return list(sampled)

        # Random fallback
        while len(sampled) < count and attempts < count * 10:
            if not src_list or not dst_list:
                break
            u = random.choice(src_list)
            v = random.choice(dst_list)
            if u != v and (u, v) not in edges:
                sampled.add((u, v))
                edge_type_map.setdefault(edge_type, []).append((u, v))
                edges.add((u, v))
            attempts += 1
        return list(sampled)

    # Use directly passed-in counts
    edge_pool = (
        sample_edges(old_nodes, old_nodes, E_oo, edge_type="o-o-bank")
        + sample_edges(old_nodes, new_nodes, E_on, edge_type="o-n")
        + sample_edges(new_nodes, new_nodes, E_nn, edge_type="n-n")
        + sample_edges(old_nodes, old_nodes, E_oon, edge_type="o-o-nobank")
    )

    weights = np.random.dirichlet(np.ones(len(edge_pool))) * W_total
    edge_weight_map = {edge: w for edge, w in zip(edge_pool, weights)}

    G = nx.DiGraph()
    used_edges = set()
    filtration_graphs = []

    for i, (v_target, e_target, w_target) in enumerate(embedding):
        v_target = int(v_target)
        e_target = int(e_target)

        current_nodes = set(all_nodes[:v_target])
        G.add_nodes_from(current_nodes)

        available_edges = [
            (u, v) for (u, v) in edge_pool
            if u in current_nodes and v in current_nodes and (u, v) not in used_edges
        ]

        needed = e_target - G.number_of_edges()
        selected_edges = available_edges[:needed]

        for (u, v) in selected_edges:
            G.add_edge(u, v, weight=edge_weight_map[(u, v)])
            used_edges.add((u, v))

        filtration_graphs.append(G.copy())

    node_types = {
        "old_nodes": old_nodes,
        "new_nodes": new_nodes
    }

    return filtration_graphs, node_types, edge_type_map


def build_edgebank(graph):
    # Relabel nodes to be 0..n-1
    mapping = {old_label: new_label for new_label, old_label in enumerate(graph.nodes())}
    G = nx.relabel_nodes(graph, mapping)

    edgebank = {}
    for u, v in G.edges():
        u_key = u
        v_key = v
        edgebank.setdefault(u_key, []).append(v_key)
        # Add reverse edge for undirected graphs
        if not G.is_directed():
            edgebank.setdefault(v_key, []).append(u_key)

    return edgebank



dataset = args.dataset
my_loader = Loader()
my_evaluator = Evaluator()

# Construct csv
run_number = 1
structure_pred_file_path = f'GraphGeneration/output/results/topology/{dataset}/random_gen_contids_degree_pred.csv'
structure_true_file_path = f'GraphGeneration/output/results/topology/{dataset}/random_gen_contids_degree_true.csv'
structure_diff_file_path = f'GraphGeneration/output/results/topology/{dataset}/random_gen_contids_degree_diff.csv'
edge_file_path = f'GraphGeneration/output/results/structure/{dataset}/random_gen_contids_degree.csv'
topER_file_path = f'GraphGeneration/output/results/topER/{dataset}/random_gen_contids_degree.csv'
animation_path = f'GraphGeneration/output/results/animations/{dataset}/random_gen_contids_degree.mp4'

# Create file paths if needed
for path in [structure_pred_file_path, structure_true_file_path, structure_diff_file_path, edge_file_path, topER_file_path, animation_path]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

columns = ["graph_num", "filtration_num", 
           "edge_oo_aucroc", "edge_oon_aucroc", "edge_on_aucroc", "edge_nn_aucroc", 
           "edge_oo_aucpr", "edge_oon_aucpr", "edge_on_aucpr", "edge_nn_aucpr", "edge_overall_aucroc", "edge_overall_aucpr", "edge_overall_true_positive", "edge_overall_true_negative", "edge_overall_false_positive", "edge_overall_false_negative",
           "num_oo_edges_pred", "num_oo_edges_true", "num_oon_edges_pred", "num_oon_edges_true", "num_on_edges_pred", "num_on_edges_true", "num_nn_edges_pred", "num_nn_edges_true", 
           "node_id_precision", "node_id_recall", "num_correct_ids"]

# Write the header and empty content
pd.DataFrame(columns=columns).to_csv(auc_file_path, index=False)


dataset = 'CollegeMsg'
my_loader = Loader()

# Load probabilities
probabilities_df = pd.read_csv(f'ReinforcementLearning/output/probabilities/{dataset}_1back.csv').iloc[:, 1:]
probabilities = probabilities_df.values.tolist()

# Load all features, thresholds, and target subgraphs
features, _ = my_loader.load_data(dataset, activation='Degree', type='features', include_weights=True)
thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=True)
target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)

# Initialize list for predicted graphs
pred_graphs = []

# Iterate through each graph in the dataset
for i in range(len(probabilities)):
    print('Constructing graph number: ', i + 1)
    count_old = probabilities[i][0]
    count_new = probabilities[i][1]
    p0 = probabilities[i][2]
    p1 = probabilities[i][3]
    p2 = probabilities[i][4]
    p3 = probabilities[i][5]

    # Build the edgebank from the graph one step back (if i == 0, skip or use empty edgebank)
    if i == 0:
        edgebank = {}  # or use some empty placeholder
    else:
        edgebank = build_edgebank(target_graphs[i - 1][-1])
    
    # Get the embedding and reshape it
    embedding = features[i]
    embedding = list(zip(embedding[0::3], embedding[1::3], embedding[2::3]))

    # Build the filtration sequence using the current parameters
    filtration_sequence, node_types, edge_type_map = build_accumulating_filtration_sequence_with_edgebank(
        embedding, p_old_nodes=count_old, E_oo=p0, E_nn=p1, E_on=p2, E_oon=p3, edgebank=edgebank
    )
    
    results  = compute_auc_scores(filtration_sequence, target_graphs[i], target_graphs[i - 1][-1] if i > 0 else nx.DiGraph(), node_types, edgebank, edge_type_map, graph_num=i)

    pd.DataFrame(results).to_csv(auc_file_path, mode='a', header=False, index=False)

    # Append the last graph from the filtration (assumed to be the "predicted" one)
    pred_graphs.append(filtration_sequence)


def create_animation(predicted_graphs, target_graphs, output_file="graph_animation.gif"):
    # Check that both lists have the same number of graphs
    # Check that both lists have the same number of graphs
    if len(predicted_graphs) != len(target_graphs):
        raise ValueError("Both lists of graphs must have the same length.")

    num_graphs = len(predicted_graphs)
    filtration_per_graph = 10

    # Precompute shared positions using the union of node sets
    pos_list = []
    for G_pred, G_target in zip(predicted_graphs, target_graphs):
        combined_nodes = set(G_pred.nodes()).union(set(G_target.nodes()))
        combined_graph = nx.Graph()
        combined_graph.add_nodes_from(combined_nodes)

        # Use consistent layout for both by computing layout on the combined graph
        pos_combined = nx.spring_layout(combined_graph, seed=42)
        
        # Subset positions for each graph (some nodes may be missing)
        pos_pred = {node: pos_combined[node] for node in G_pred.nodes()}
        pos_target = {node: pos_combined[node] for node in G_target.nodes()}
        
        pos_list.append((pos_pred, pos_target))

    # Setup the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_left, ax_right = axes

    def update_frame(i):
        ax_left.clear()
        ax_right.clear()

        G_pred = predicted_graphs[i]
        G_target = target_graphs[i]

        graph_index = (i // filtration_per_graph) + 1
        filtration_number = (i % filtration_per_graph) + 1

        pos_pred, pos_target = pos_list[i]

        nx.draw(G_pred, pos=pos_pred, ax=ax_left, with_labels=True,
                node_size=150, node_color="skyblue", edge_color="gray", width=1.25, alpha=0.8)
        ax_left.set_title(f"Graph {graph_index}, Filtration {filtration_number}")

        nx.draw(G_target, pos=pos_target, ax=ax_right, with_labels=True,
                node_size=150, node_color="lightgreen", edge_color="gray", width=1.25, alpha=0.8)
        ax_right.set_title(f"Graph {graph_index}, Filtration {filtration_number}")

        ax_left.set_axis_off()
        ax_right.set_axis_off()

    ani = FuncAnimation(fig, update_frame, frames=num_graphs, interval=1000, repeat=False)

    print('Animation created')

    try:
        writer = FFMpegWriter(fps=1)
        ani.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
    except Exception as e:
        print(f"FFmpeg failed with error: {e}")
        fallback_file = os.path.splitext(output_file)[0] + ".gif"
        ani.save(fallback_file, writer=PillowWriter(fps=1))
        print(f"Fallback: animation saved as GIF to {fallback_file}")

    plt.show()


predicted, target = pred_graphs, target_graphs
embedding_graphs = [inner_list[-1] for inner_list in predicted]
my_loader = Loader()
embedder = EmbedDegree(include_weights=False)

all_embeddings, _, _ = embedder.process_graphs_for_embeddings(embedding_graphs)
true_embeddings, labels = my_loader.load_data('CollegeMsg', 'Degree', include_weights=False)
labels = np.array(labels)

import pickle
with open('data/input/cached/CollegeMsg/CollegeMsg.pkl', 'rb') as f:
    data = pickle.load(f)

    
pred_gs_labels = [1]

for i in range(1, len(embedding_graphs)):
    prev_edges = embedding_graphs[i - 1].number_of_edges()
    curr_edges = embedding_graphs[i].number_of_edges()
    # print(f'In the predicted graphs there are {curr_edges} vs {prev_edges}')
    # print(f'In the true graphs there are {true_embeddings[i][-1]} vs {true_embeddings[i - 1][-1]}')
    # print(f'For checking purposes: {data[i][0].number_of_edges()} vs {data[i - 1][0].number_of_edges()}')
    # print(f'For checking purposes: {data[i][1]}')
    # print(f'Meaning we have a label of {labels[i]}\n')
    pred_gs_labels.append(1 if curr_edges > prev_edges else 0)

predictions = np.array(pred_gs_labels)


# Compute metrics
aucroc = roc_auc_score(labels, predictions)
aucpr = average_precision_score(labels, predictions)

print(f'AUCROC: {aucroc}')
print(f'AUCPR: {aucpr}')

csv_file_path = f'GraphGeneration/output/topERComparisons/CollegeMsg_cuneyt_gen.csv'

columns = ['graph_num', 'l2_norm', 'cosine_similarity', 'g/s_pred_label', 'g/s_true_label']
for i in range(10):
    columns.append(f'node_diff_{i+1}')
    columns.append(f'edge_diff_{i+1}')

# Write the header and empty content
pd.DataFrame(columns=columns).to_csv(csv_file_path, index=False)


# Function to compare vectors with named node and edge differences
def compare_vectors_with_named_columns(A, B, pred_label, true_label, graph_num=0):
    A = np.array(A)
    B = np.array(B)

    if len(A) != 20 or len(B) != 20:
        raise ValueError("Each vector must contain exactly 20 elements.")

    # Core metrics
    l2_norm = np.linalg.norm(A - B)
    cosine_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    # Node-edge diff labels (pointwise absolute differences)
    diff = A - B
    node_edge_diffs = {}
    for i in range(10):
        node_edge_diffs[f'node_diff_{i+1}'] = diff[2*i]  # node_diff for even indexes (0, 2, 4, ...)
        node_edge_diffs[f'edge_diff_{i+1}'] = diff[2*i + 1]  # edge_diff for odd indexes (1, 3, 5, ...)

    # Final combined dictionary to be added to the DataFrame
    result = {
        'graph_num': graph_num,
        'l2_norm': l2_norm,
        'cosine_similarity': cosine_sim,
        'g/s_pred_label': pred_label,
        'g/s_true_label': true_label,
        **node_edge_diffs
    }

    return result

graph_num = 1
filtration_num = 0

# Loop through all embeddings
for idx, (embedding, true_embedding) in enumerate(zip(all_embeddings, true_embeddings)):
    # Set graph_num based on index (1-based)
    graph_num = idx + 1
    pred_label = predictions[idx]
    true_label = labels[idx] 

    # Compare embeddings and get the result
    result = compare_vectors_with_named_columns(embedding, true_embedding, pred_label=pred_label, true_label=true_label,graph_num=graph_num)

    # Append the result to the CSV
    pd.DataFrame([result]).to_csv(csv_file_path, mode='a', header=False, index=False)
    
    
animation_path = f'GraphGeneration/output/animations/initial_anim_cuneyt_gen.mp4'
# print('Predicted')
# print(loaded_predicted)
# print('Target')
# print(loaded_target)


from itertools import chain

# Flatten a list of lists into a single list of NetworkX graphs
predicted_flat = list(chain(*predicted))
target_flat = list(chain(*target))

# Call the create_animation function with the flattened lists
create_animation(predicted_flat, target_flat, output_file=animation_path)


# TODO THIS WHOLE FILE