import numpy as np 
import networkx as nx
import pandas as pd 
import matplotlib.pyplot as plt 
import random
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import Loader

# Import all embedding methods
from utils.embedding_methods.betweenness import EmbedBetweenness
from utils.embedding_methods.closeness import EmbedCloseness
from utils.embedding_methods.degree import EmbedDegree
from utils.embedding_methods.forman_ricci import EmbedForman
from utils.embedding_methods.weight import EmbedWeight

def build_accumulating_filtration_sequence_with_edgebank(embedding, p_old_nodes, p_new_nodes, E_oo, E_nn, E_on, E_oon, edgebank=None, existing_nodes=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    if existing_nodes is None:
        existing_nodes = []

    V_total = int(embedding[-1][0])
    E_total = int(embedding[-1][1])
    W_total = embedding[-1][2] 

    # Sample old nodes
    old_nodes = random.sample(existing_nodes, p_old_nodes)
    # Create new node IDs
    new_nodes = [f"v{i}" for i in range(len(existing_nodes), len(existing_nodes) + p_new_nodes)]

    all_nodes = old_nodes + new_nodes
    existing_nodes += new_nodes

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

        elif edge_type == "o-o-nobank" and edgebank is not None:
            for u in src_list:
                if u in edgebank:
                    for v in edgebank[u]:
                        if u != v and (u, v) not in edgebank.get(u, []) and (u, v) not in edges:
                            sampled.add((u, v))
                            edge_type_map.setdefault(edge_type, []).append((u, v))
                            edges.add((u, v))
                            if len(sampled) >= count:
                                return list(sampled)

        # Random fallback
        else:
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

    return filtration_graphs, node_types, existing_nodes, edge_type_map


def modifyGraphIds(graphs):
    '''
    For the target graphs, modify their ids to start at 0 for an instance of a node, then increment throughout the graphs
    
    Args:
        graphs (list(nx.Graph)): A list of graphs to modify
        
    Returns:
        graphs (list(nx.Graph)): The modified graphs (operations performed in-place)       
    '''
    # This dictionary will store the mapping of original node IDs to new node IDs
    node_mapping = {}
    new_id = 0

    # Iterate over all graphs in the list of lists (where each graph is a subgraph in the list)
    for graph_list in graphs:
        # Each graph_list contains multiple subgraphs, iterate over the subgraphs
        for graph in graph_list:
            # Create a new dictionary to store the relabeled nodes for the current subgraph
            mapping_for_current_graph = {}
            
            # Iterate over all nodes in the current graph
            for node in graph.nodes:
                # If the node is already in the node_mapping, use the existing ID
                if node not in node_mapping:
                    # If not, assign it a new ID
                    node_mapping[node] = new_id
                    new_id += 1
                
                # Store the relabeled node ID in the current graph mapping
                mapping_for_current_graph[node] = node_mapping[node]
            
            # Relabel the nodes in the graph using the mapping
            nx.relabel_nodes(graph, mapping_for_current_graph, copy=True)
    
    return graphs, len(node_mapping)


def build_edgebanks_from_start(graphs):
    edgebanks = [{}]  # First graph has an empty edgebank

    curr_edgebank = {}

    # Loop over all graphs and add their edges
    for i in range(1, len(graphs)):
        for u, v in graphs[i - 1].edges():  # We look one graph back to add from
            u_key = f"v{u}"
            v_key = f"v{v}"
            curr_edgebank.setdefault(u_key, []).append(v_key)  # Directed graphs

        edgebanks.append(curr_edgebank)

    return edgebanks
    

# Maps node ids by computing a degree cost
def generate_node_mapping_by_id(pred_graph, true_graph, old_nodes_true, old_nodes_pred, curr_mapping):
    from scipy.optimize import linear_sum_assignment

    unmapped_pred = [n for n in old_nodes_pred if n not in curr_mapping]

    # Filter out already mapped true nodes
    already_mapped_true = set(curr_mapping.values())
    unmapped_true = [n for n in old_nodes_true if n not in already_mapped_true]

    if not unmapped_pred or not unmapped_true:
        return curr_mapping

    # Cost matrix based on degree difference
    cost_matrix = np.zeros((len(unmapped_pred), len(unmapped_true)))

    for i, pred_id in enumerate(unmapped_pred):
        deg_pred = pred_graph.degree[pred_id]
        for j, true_id in enumerate(unmapped_true):
            deg_true = true_graph.degree[true_id]
            cost_matrix[i, j] = abs(deg_pred - deg_true)

    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Add new mappings
    for i, j in zip(row_ind, col_ind):
        curr_mapping[unmapped_pred[i]] = unmapped_true[j]

    return curr_mapping


# For this, I will assume no self loops
def create_possible_edges(node_ids, directed=True):
    edges = []

    if directed:
        for u in node_ids:
            for v in node_ids:
                if u == v:
                    continue
                
                edges.append((u, v))
    
    elif not directed:
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                # Just a precaution
                if i == j:
                    continue
                
                edges.append((node_ids[i], node_ids[j]))
                edges.append((node_ids[j], node_ids[i]))

    return edges


def compute_edge_auc(true_edges, pred_edges, possible_edges):
    y_pred = []
    y_true = []

    true_edges = set(true_edges)
    pred_edges = set(pred_edges)

    for edge in possible_edges:
        y_true.append(1 if edge in true_edges else 0)
        y_pred.append(1 if edge in pred_edges else 0)

    # AUC is undefined if all values are same (a precaution)
    if len(set(y_true)) < 2:
        return float('inf')

    return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)

 
def compute_auc_scores(filtration_graphs, true_graphs, prev_old_nodes_true, node_types, edgebank: dict, edge_type_map, graph_num):
    results = []
    mapping = {}  # For mapping nodes

    old_nodes_pred = node_types["old_nodes"]
    new_nodes_pred = node_types["new_nodes"]

    if set(new_nodes_pred) != set(new_nodes_true):
        print('The new nodes arent the same!')

    
    # If graph number is 0, there are no old nodes, therefore we don't compute auc
    if graph_num > 0:
        #for i, (pred_graph, true_graph) in enumerate(zip(filtration_graphs, true_graphs)):
            # Since just using final graph for now
            pred_graph = filtration_graphs[-1]
            true_graph = true_graphs[-1] 

            # Get the nodes and edges in use
            old_nodes_true = set.intersection(true_graph.nodes(), prev_old_nodes_true)
            new_nodes_true = set(true_graph.nodes()) - prev_old_nodes_true
            all_nodes_true = old_nodes_true.union(new_nodes_true)

            oo_true_edges =  [edge for edge in true_graph.edges(data=False) if edge[1] in edgebank.get(edge[0])]
            oon_true_edges = [edge for edge in true_graph.edges(data=False) if (edge[1] not in edgebank.get(edge[0]) and edge[0] in old_nodes_true and edge[1] in old_nodes_true)]
            on_true_edges = [edge for edge in true_graph.edges(data=False) if ((edge[0] in old_nodes_true and edge[1] in new_nodes_true) or (edge[0] in new_nodes_true and edge[1] in old_nodes_true))]
            nn_true_edges = [edge for edge in true_graph.edges(data=False) if (edge[0] in new_nodes_true and edge[1] in new_nodes_true)]


            # Make a node mapping to compute auc scores
            mapping = generate_node_mapping_by_id(pred_graph, true_graph, old_nodes_true, old_nodes_pred, mapping)
            tmp_pred_graph = nx.relabel_nodes(pred_graph, mapping, copy=True)


            # Compute scores for edge type oo
            oo_pred_edges = [(mapping[u], mapping[v]) for (u, v) in edge_type_map["o-o-bank"]
                if u in mapping and v in mapping and (mapping[u], mapping[v]) in tmp_pred_graph.edges()
            ]  # Need to map node ids for the alignment
            oo_possible_edges = create_possible_edges(old_nodes_true, directed=True)
            oo_aucroc, oo_aucpr = compute_edge_auc(oo_true_edges, oo_pred_edges, oo_possible_edges)
            
            # Compute scores for edge type oon
            oon_pred_edges = [(mapping[u], mapping[v]) for (u, v) in edge_type_map["o-o-nobank"]
                if u in mapping and v in mapping and (mapping[u], mapping[v]) in tmp_pred_graph.edges()
            ]  # Need to map node ids for the alignment
            oon_possible_edges = create_possible_edges(old_nodes_true, directed=True)
            oon_aucroc, oon_aucpr = compute_edge_auc(oon_true_edges, oon_pred_edges, oon_possible_edges)
            
            # Compute scores for edge type on
            on_pred_edges = [(mapping[u], mapping[v]) for (u, v) in edge_type_map["o-n"]
                if (u in mapping and v in mapping and (mapping[u], mapping[v]) in tmp_pred_graph.edges()) and
                    (u in old_nodes_pred and v in new_nodes_pred) or (v in old_nodes_pred and u in new_nodes_pred)
            ]  # Need to map node ids for the alignment
            on_possible_edges = create_possible_edges(all_nodes_true, directed=True)
            on_aucroc, on_aucpr = compute_edge_auc(on_true_edges, on_pred_edges, on_possible_edges)

            # Compute scores for edge type nn
            nn_pred_edges = [(mapping[u], mapping[v]) for (u, v) in edge_type_map["n-n"]
                if u in mapping and v in mapping and (mapping[u], mapping[v]) in tmp_pred_graph.edges()
            ]  # Need to map node ids for the alignment
            nn_possible_edges = create_possible_edges(new_nodes_true, directed=True)  
            nn_aucroc, nn_aucpr = compute_edge_auc(nn_true_edges, nn_pred_edges, nn_possible_edges)
            
            # Compute overall aucroc
            overall_true_edges = true_graph.edges()
            overall_pred_edges = tmp_pred_graph.edges()  # Need to map node ids for the alignment
            overall_possible_edges = create_possible_edges(new_nodes_pred, directed=True)  
            overall_aucroc, overall_aucpr = compute_edge_auc(overall_true_edges, overall_pred_edges, overall_possible_edges)
           

            # Compute score for correct node ids
            predicted_node_ids = pred_graph.nodes(data=False)
            true_node_ids = true_graph.nodes(data=False)

            correct_ids = list(predicted_node_ids & true_node_ids)
            num_correct_ids = len(correct_ids)

            id_precision = len(correct_ids) / len(predicted_node_ids) if predicted_node_ids else 0
            id_recall = len(correct_ids) / len(true_node_ids) if true_node_ids else 0
            

            curr_results = {
                "graph_num": graph_num, 
                "filtration_num": i,

                "edge_oo_aucroc": oo_aucroc,
                "edge_oon_aucroc": oon_aucroc,
                "edge_on_aucroc": on_aucroc,
                "edge_nn_aucroc": nn_aucroc,
                "edge_oo_aucpr": oo_aucpr,
                "edge_oon_aucpr": oon_aucpr,
                "edge_on_aucpr": on_aucpr,
                "edge_nn_aucpr": nn_aucpr,
                "edge_overall_aucroc": overall_aucroc,
                "edge_overall_aucpr": overall_aucpr,

                "num_oo_edges_pred": len(oo_pred_edges),
                "num_oo_edges_true": len(oo_true_edges),
                "num_oon_edges_pred": len(oon_pred_edges),
                "num_oon_edges_true": len(oon_true_edges),
                "num_on_edges_pred": len(on_pred_edges),
                "num_on_edges_true": len(on_true_edges),
                "num_nn_edges_pred": len(nn_pred_edges),
                "num_nn_edges_true": len(nn_true_edges),
                "num_edges_pred": len(overall_pred_edges),
                "num_edges_true": len(overall_true_edges),

                "node_id_precision": id_precision,
                "node_id_recall": id_recall,
                "num_correct_ids": num_correct_ids
            }

            results.append(curr_results)


    else:
        # No relabeling graph needed
        #for i, (pred_graph, true_graph) in enumerate(zip(filtration_graphs, true_graphs)):
            # Since just using final graph for now
            pred_graph = filtration_graphs[-1]
            true_graph = true_graphs[-1] 

            # Compute scores for edge type nn
            nn_true_edges = [edge for edge in true_graph.edges(data=False) if (edge[0] in new_nodes_true and edge[1] in new_nodes_true)]

            nn_pred_edges = [edge for edge in edge_type_map["n-n"] if edge in pred_graph]  # It is just all edges right now
            nn_possible_edges = create_possible_edges(new_nodes_true, directed=True)  
            nn_aucroc, nn_aucpr = compute_edge_auc(nn_true_edges, nn_pred_edges, nn_possible_edges)
            
            # Compute overall aucroc
            overall_true_edges = true_graph.edges()
            overall_pred_edges = pred_graph.edges()  # Need to map node ids for the alignment
            overall_possible_edges = create_possible_edges(new_nodes_pred, directed=True)  
            overall_aucroc, overall_aucpr = compute_edge_auc(overall_true_edges, overall_pred_edges, overall_possible_edges)
           

            # Compute score for correct node ids
            predicted_node_ids = pred_graph.nodes(data=False)
            true_node_ids = true_graph.nodes(data=False)

            correct_ids = list(predicted_node_ids & true_node_ids)
            num_correct_ids = len(correct_ids)

            id_precision = len(correct_ids) / len(predicted_node_ids) if predicted_node_ids else 0
            id_recall = len(correct_ids) / len(true_node_ids) if true_node_ids else 0
            

            curr_results = {
                "graph_num": graph_num, 
                #"filtration_num": i,
                "filtration_num": 10,

                "edge_oo_aucroc": float('inf'),  # Cant compute
                "edge_oon_aucroc": float('inf'),  # Cant compute
                "edge_on_aucroc": float('inf'),  # Cant compute
                "edge_nn_aucroc": nn_aucroc, 
                "edge_oo_aucpr": float('inf'),  # Cant compute
                "edge_oon_aucpr": float('inf'),  # Cant compute
                "edge_on_aucpr": float('inf'),  # Cant compute
                "edge_nn_aucpr": nn_aucpr, 
                "edge_overall_aucroc": overall_aucroc,  # This should be the same as nn_aucroc
                "edge_overall_aucpr": overall_aucpr,  # This should be the same as nn_aucpr
 
                "num_oo_edges_pred": float('inf'),  # Cant compute
                "num_oo_edges_true": float('inf'),  # Cant compute
                "num_oon_edges_pred": float('inf'),  # Cant compute
                "num_oon_edges_true": float('inf'),  # Cant compute
                "num_on_edges_pred": float('inf'),  # Cant compute
                "num_on_edges_true": float('inf'),  # Cant compute
                "num_nn_edges_pred": pred_graph.number_of_edges(),  # All edges are nn
                "num_nn_edges_true": true_graph.number_of_edges(),  # All edges are nn
                
                "node_id_precision": id_precision,
                "node_id_recall": id_recall,
                "num_correct_ids": num_correct_ids
            }

            results.append(curr_results)


    return results


# Construct csv
auc_file_path = f'GraphGeneration/output/aucResults/CollegeMsg_cuneyt_gen_contids.csv'

columns = ["graph_num", "filtration_num", 
           "edge_oo_aucroc", "edge_oon_aucroc", "edge_on_aucroc", "edge_nn_aucroc", 
           "edge_oo_aucpr", "edge_oon_aucpr", "edge_on_aucpr", "edge_nn_aucpr", "edge_overall_aucroc", "edge_overall_aucpr", 
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

# Build the edgebanks for construction
tmp_target_graphs, _ = modifyGraphIds(target_graphs)
all_edgebanks = build_edgebanks_from_start(tmp_target_graphs)
existing_nodes = []  # The current nodes we have seen for continuous id implementation

# Iterate through each graph in the dataset
for i in range(len(probabilities)):
    print('Constructing graph number: ', i + 1)
    count_old = probabilities[i][0]
    count_new = probabilities[i][1]
    p0 = probabilities[i][2]
    p1 = probabilities[i][3]
    p2 = probabilities[i][4]
    p3 = probabilities[i][5]

    edgebank = all_edgebanks[i]

    # Get the embedding and reshape it
    embedding = features[i]
    embedding = list(zip(embedding[0::3], embedding[1::3], embedding[2::3]))

    # Build the filtration sequence using the current parameters
    filtration_sequence, node_types, existing_nodes, edge_type_map = build_accumulating_filtration_sequence_with_edgebank(
        embedding, p_old_nodes=count_old, E_oo=p0, E_nn=p1, E_on=p2, E_oon=p3, edgebank=edgebank, existing_nodes=existing_nodes
    )

    results  = compute_auc_scores(filtration_sequence, target_graphs[i], target_graphs[i - 1][-1] if i > 0 else nx.Graph(), node_types, edgebank, edge_type_map, graph_num=i)

    pd.DataFrame(results).to_csv(auc_file_path, mode='a', header=False, index=False)
    
    # Append the last graph from the filtration (assumed to be the "predicted" one)
    pred_graphs.append(filtration_sequence)


# Analysis


def create_animation(predicted_graphs, target_graphs, output_file="graph_animation.gif"):
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
print(data)
    
pred_gs_labels = [1]

for i in range(1, len(embedding_graphs)):
    prev_edges = embedding_graphs[i - 1].number_of_edges()
    curr_edges = embedding_graphs[i].number_of_edges()
    print(f'In the predicted graphs there are {curr_edges} vs {prev_edges}')
    print(f'In the true graphs there are {true_embeddings[i][-1]} vs {true_embeddings[i - 1][-1]}')
    print(f'For checking purposes: {data[i][0].number_of_edges()} vs {data[i - 1][0].number_of_edges()}')
    print(f'For checking purposes: {data[i][1]}')
    print(f'Meaning we have a label of {labels[i]}\n')
    pred_gs_labels.append(1 if curr_edges > prev_edges else 0)

predictions = np.array(pred_gs_labels)


# Compute metrics
aucroc = roc_auc_score(labels, predictions)
aucpr = average_precision_score(labels, predictions)

print(f'G/S AUCROC: {aucroc}')
print(f'G/S AUCPR: {aucpr}')

csv_file_path = f'GraphGeneration/output/topERComparisons/CollegeMsg_cuneyt_gen_contids.csv'

columns = ['graph_num', 'l2_norm', 'cosine_similarity']
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
    pred_label = predictions[i]
    true_label = labels[i] 

    # Compare embeddings and get the result
    result = compare_vectors_with_named_columns(embedding, true_embedding, pred_label=pred_label, true_label=true_label,graph_num=graph_num)

    # Append the result to the CSV
    pd.DataFrame([result]).to_csv(csv_file_path, mode='a', header=False, index=False)
    
    
animation_path = f'GraphGeneration/output/animations/initial_anim_cuneyt_gen_contids.mp4'
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


