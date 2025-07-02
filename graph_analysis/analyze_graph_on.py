import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.loader import Loader
import argparse
from GraphGeneration.scripts.load_data import load_data
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
# Process arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=False, default='CollegeMsg', choices=['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkcindicator', 'networkiconomi', 'networkdgd'])
parser.add_argument("--strategy", type=str, required=False, default='MultiheadedMLP', choices=['MultiheadedMLP', 'SingleMLP', 'Multiheaded_LSTM_oo'], help="The type of MLP NN to use")
parser.add_argument("--embedding", type=str, required=False, default='Position', choices=['Position', 'NodeType', 'Position+NodeType', 'None'], help="Allows appending positional encodings or an integer node type onto the end of the embeddings")
parser.add_argument("--mlpEncoding", type=str, required=False, default='Concat', choices=['Concat', 'Product', 'Addition', 'Subtraction'], help="How you want to input node embeddings to the MLP")  # Product and addition lead to potential noise as we use directed graphs
parser.add_argument("--embedOld", type=str, required=False, default='True', choices=['True', 'False'], help="If you want to let the MLP predict edge type \'o-o-bank\', otherwise these edges are randomly added")
parser.add_argument("--oldDegree", type=str, required=False, default='False' ,choices=['True', 'False'], help="If you want reappearing nodes to reuse their most recent degree")
parser.add_argument("--trainingStyle", type=str, required=False, default='TrueGraphs', choices=['TrueGraphs', 'PredGraphs', 'MixedGraphs'], help="When training the MLP, decides if you use real graphs, predicted graphs (with first real as starter), or real then pred for MLP training")
parser.add_argument("--embeddingType", type=str, required=False, default='Node2Vec', choices=['Linear', 'Node2Vec', 'LSTM'], help="How nodes should be embedded. Either with Node2Vec or with a Linear mutliplication of adjacency matrix by node feature matrix")
parser.add_argument("--snapshot", type=int, required=False, default=2)
args = parser.parse_args()

my_loader = Loader()
# probabilities, features, thresholds, target_graphs = load_data(args.dataset, args.strategy, args.embedding, args.mlpEncoding, args.embedOld, args.trainingStyle, args.embeddingType)
# target_graphs, _ = modifyGraphIds(target_graphs, thresholds)

def create_layered_graph(new_nodes, old_nodes, graph):
    """
        Constructs a layered subgraph layout from a directed graph based on connections 
        between new and old nodes.

        Parameters:
            new_nodes (list or set): Set of newly added nodes.
            old_nodes (list or set): Set of previously existing nodes.
            graph (networkx.DiGraph): The full graph.

        Returns:
            pos (dict): A dictionary mapping each node to its (x, y) position for layout.
            layers (dict): A dictionary mapping layer index to list of nodes in that layer.
            B (networkx.DiGraph): A subgraph of `graph` containing relevant edges and nodes.
    """
    new_nodes = set(new_nodes)
    old_nodes = set(old_nodes)

    B = nx.DiGraph()
    connected_nodes = set()
    
    # Add edges between new and old nodes (both directions)
    for u in new_nodes:
        for v in old_nodes:
            if graph.has_edge(u, v):
                B.add_edge(u, v)
                connected_nodes.update([u, v])
            if graph.has_edge(v, u):
                B.add_edge(v, u)
                connected_nodes.update([v, u])

    # Add edges between new nodes (both directions)
    for u in new_nodes:
        for v in new_nodes:
            if u != v and graph.has_edge(u, v):
                B.add_edge(u, v)
                connected_nodes.update([u, v])
            if graph.has_edge(v, u):
                B.add_edge(v, u)
                connected_nodes.update([v, u])

    if len(B.edges) == 0:
        print("No connections to show.")
        return 

    new_nodes &= connected_nodes
    old_nodes &= connected_nodes

    # Step 1: BFS layer assignment (undirected)
    layer_map = {}
    visited = set()
    queue = deque()
    B_undirected = B.to_undirected()
    for node in old_nodes:
        layer_map[node] = 0
        visited.add(node)
        queue.append((node, 0))

    
    
    while queue:
        current, curr_layer = queue.popleft()
        for neighbor in B_undirected.neighbors(current):  # undirected traversal
            if neighbor not in visited:
                visited.add(neighbor)
                layer_map[neighbor] = curr_layer + 1
                queue.append((neighbor, curr_layer + 1))

        

    # Group nodes by layer
    layers = defaultdict(list)
    for node, layer in layer_map.items():
        layers[layer].append(node)

    # Step 2: Assign positions based on layers
    pos = {}
    for layer, nodes in layers.items():
        for i, node in enumerate(nodes):
            pos[node] = (i * 1.5, -layer)  # Space horizontally, stack vertically

    for i, node in enumerate(B.nodes):
        if node not in pos:
            pos[node] = (i, -0.5)  # push off-screen or stack at default
            layers[0.5].append(node)
    
    return pos, layers, B

def visualize_layered_bipartite_graph(pos, layers, graph, new_nodes, dataset_name="Dataset"):
        
    # Draw graph
    plt.figure(figsize=(12, 7))
    # # Ensure all nodes in B have positions (fallback to y = -999)
    visualized_nodes = pos

    for layer, nodes in layers.items():
        color = (
            'skyblue' if layer == 0 else
            'orange' if layer == 1 else
            'lightgreen' if layer == 2 else
            'red' if layer == 0.5 else
            'violet'
        )
        nx.draw_networkx_nodes(graph, visualized_nodes, nodelist=nodes, node_color=color, label=f'Layer {layer}')
    nx.draw_networkx_edges(graph, visualized_nodes, arrows=True)
    nx.draw_networkx_labels(graph, visualized_nodes)

    # Legend and title
    plt.title(f"Layered Graph: New vs Old Nodes ({dataset_name}) At Snapshot {args.snapshot}")
    plt.legend()
    plt.axis('off')

    # Table
    table_data = [
        ["# Old Nodes", len(layers[0])],
        ["# New Nodes", len(new_nodes)],
        ["# Layers", max(layers.keys())],
        ["# Edges", graph.number_of_edges()]
    ]
    table = plt.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="lower right",
        cellLoc='center',
        colWidths=[0.25, 0.15]
    )
    table.scale(1, 1.2)
    for _, cell in table.get_celld().items():
        cell.set_fontsize(12)

    plt.tight_layout()
    plt.show()

def visualize_pure_nn_graph(pos, graph, dataset_name="Dataset"):
    layer_half_nodes = [node for node in graph.nodes if pos.get(node, (0, 1))[1] == -0.5]

    if not layer_half_nodes:
        print("No nodes in layer 0.5 to plot.")
        return

    # Extract subgraph
    subgraph = graph.subgraph(layer_half_nodes).copy()
    sub_pos = nx.spring_layout(subgraph, seed=42)

    # Draw
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(subgraph, sub_pos, node_color='red', label='Layer 0.5', node_size=700)
    nx.draw_networkx_edges(subgraph, sub_pos, arrows=True)
    nx.draw_networkx_labels(subgraph, sub_pos, font_size=8, font_color="white")

    plt.title(f"Pure nn Visualization ({dataset_name}) At Snapshot {args.snapshot}")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def record_layer_dataset():
    folder_path = rf".\data\input\cached\{args.dataset}\graph_on_layered"
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "graph_on_layer.txt"), "w") as f:
        f.write('Snapshot,L_0,L_1,L_gt_2,L_inf\n')
    
    prev_graphs = [graph[-1] for graph in target_graphs[:2]]
    prev_nodes = set().union(*[graph.nodes() for graph in prev_graphs])
    
    for snapshot in range(2, len(target_graphs)):
        
        new_nodes=set(target_graphs[snapshot][-1].nodes()) - prev_nodes
        
        try:
            pos, layers, graph = create_layered_graph(graph=target_graphs[snapshot][-1], old_nodes=prev_nodes,
                                                            new_nodes=new_nodes)
        except Exception as e:
            pass
  
        
        with open(os.path.join(folder_path, "graph_on_layer.txt"), "a") as f:
            f.write(f'{snapshot},{len(layers[0])},{len(layers[1])},{sum([len(layers[i]) for i in layers.keys() if i >= 2])},{len(layers[0.5])}\n')
        prev_nodes = prev_nodes.union(target_graphs[snapshot][-1].nodes())
        

# snapshot = args.snapshot
# prev_graphs = [graph[-1] for graph in target_graphs[:snapshot]]
# prev_nodes = set().union(*[graph.nodes() for graph in prev_graphs])
# new_nodes=set(target_graphs[snapshot][-1].nodes()) - prev_nodes
# pos, layers, graph = create_layered_graph(graph=target_graphs[snapshot][-1], old_nodes=prev_nodes,
#     new_nodes=new_nodes)
# visualize_layered_bipartite_graph(dataset_name=args.dataset, graph=graph, 
#                                   pos=pos, layers=layers , new_nodes=new_nodes)
# visualize_pure_nn_graph(pos=pos, graph=graph,dataset_name=args.dataset)
for dt in ['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkcindicator', 'networkiconomi', 'networkdgd']:
    args.dataset = dt
    probabilities, features, thresholds, target_graphs = load_data(args.dataset, args.strategy, args.embedding, args.mlpEncoding, args.embedOld, args.trainingStyle, args.embeddingType)
    record_layer_dataset()