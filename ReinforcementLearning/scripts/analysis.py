import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import os
import sys
import argparse
import pickle
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader

# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, required=True, choices=['base', 'grouped', 'no_removal', 'no_removal_grouped', 'no_matrix'])
parser.add_argument("--model", type=str, required=False, default='PPO', choices=['PPO', 'EpsilonGreedyPPO', 'MaskablePPO'])  # If we should use imitation learning
args = parser.parse_args()



def create_animation(predicted_graphs, target_graphs, output_file="graph_animation.gif"):
    # Check that both lists have the same number of graphs
    if len(predicted_graphs) != len(target_graphs):
        raise ValueError("Both lists of graphs must have the same length.")
    
    num_graphs = len(predicted_graphs)
    filtration_per_graph = 10

    # Precompute the layout for all graphs
    pos_pred_list = [nx.spring_layout(G_pred, seed=42) for G_pred in predicted_graphs]
    pos_target_list = [nx.spring_layout(G_target, seed=42) for G_target in target_graphs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_left, ax_right = axes

    def update_frame(i):
        # No need to clear axes
        ax_left.clear()
        ax_right.clear()

        G_pred = predicted_graphs[i]
        G_target = target_graphs[i]

        # Determine graph and filtration indices
        graph_index = i // filtration_per_graph
        filtration_index = i % filtration_per_graph

        # Use precomputed positions for the graphs
        pos_pred = pos_pred_list[i]
        pos_target = pos_target_list[i]

        # Draw the graphs
        nx.draw(G_pred, pos=pos_pred, ax=ax_left, with_labels=True,
                node_size=500, node_color="skyblue", edge_color="gray")
        ax_left.set_title(f"Predicted Graph {graph_index + 1}, Filtration {filtration_index + 1}")

        nx.draw(G_target, pos=pos_target, ax=ax_right, with_labels=True,
                node_size=500, node_color="lightgreen", edge_color="gray")
        ax_right.set_title(f"Target Graph {graph_index + 1}, Filtration {filtration_index + 1}")

        ax_left.set_axis_off()
        ax_right.set_axis_off()

    # Create the animation
    ani = FuncAnimation(fig, update_frame, frames=num_graphs, interval=1000, repeat=False)
    
    print('Animation created')
    
    # Try to save as .mp4, fallback to .gif
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

# Example usage:
# Assuming you have two lists of DiGraph objects: `predicted_graphs` and `target_graphs`
with open(f"ReinforcementLearning/output/graphs/{args.strategy}_{args.model}.pkl", "rb") as f:
    loaded_predicted, loaded_target = pickle.load(f)
    
my_loader = Loader()
embedder = EmbedDegree(include_weights=False)

print('Start')
print(loaded_predicted[0].number_of_nodes())

all_embeddings, _, _ = embedder.process_graphs_for_embeddings(loaded_predicted)
true_embeddings, labels = my_loader.load_data('CollegeMsg', 'Degree', include_weights=False)

graph_num = 1
filtration_num = 1

csv_file_path = f'ReinforcementLearning/output/topERComparisons/CollegeMsg_{args.strategy}_{args.model}.csv'

if not os.path.isfile(csv_file_path):
    # Create columns: 'node_diff_1', 'edge_diff_1', ..., 'node_diff_10', 'edge_diff_10' (alternating)
    columns = ['graph_num', 'filtration_num', 'l2_norm', 'cosine_similarity']
    for i in range(10):
        columns.append(f'node_diff_{i+1}')
        columns.append(f'edge_diff_{i+1}')
    pd.DataFrame(columns=columns).to_csv(csv_file_path, index=False)


# Function to compare vectors with named node and edge differences
def compare_vectors_with_named_columns(A, B, graph_num=0, filtration_num=0):
    A = np.array(A)
    B = np.array(B)

    if len(A) != 20 or len(B) != 20:
        raise ValueError("Each vector must contain exactly 20 elements.")

    # Core metrics
    l2_norm = np.linalg.norm(A - B)
    cosine_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    # Node-edge diff labels (pointwise absolute differences)
    diff = np.abs(A - B)
    node_edge_diffs = {}
    for i in range(10):
        node_edge_diffs[f'node_diff_{i+1}'] = diff[2*i]  # node_diff for even indexes (0, 2, 4, ...)
        node_edge_diffs[f'edge_diff_{i+1}'] = diff[2*i + 1]  # edge_diff for odd indexes (1, 3, 5, ...)

    # Final combined dictionary to be added to the DataFrame
    result = {
        'graph_num': graph_num,
        'filtration_num': filtration_num,
        'l2_norm': l2_norm,
        'cosine_similarity': cosine_sim,
        **node_edge_diffs
    }

    return result

graph_num = 1
filtration_num = 1

# Loop through all embeddings
for idx, (embedding, true_embedding) in enumerate(zip(all_embeddings, true_embeddings)):
    print('Pred: ', embedding)
    print('True: ', true_embedding)
    # Increment graph_num every 10 embeddings and reset filtration_num to 1
    if idx % 10 == 0 and idx != 0:
        graph_num += 1
        filtration_num = 1
    else:
        filtration_num += 1

    # Compare embeddings and get the result
    result = compare_vectors_with_named_columns(embedding, true_embedding, graph_num=graph_num, filtration_num=filtration_num)

    # Append the result to the CSV
    pd.DataFrame([result]).to_csv(csv_file_path, mode='a', header=False, index=False)    
    
animation_path = f'ReinforcementLearning/output/animations/initial_anim_{args.strategy}_{args.model}_nx.mp4'
# print('Predicted')
# print(loaded_predicted)
# print('Target')
# print(loaded_target)


create_animation(loaded_predicted, loaded_target, output_file=animation_path)
