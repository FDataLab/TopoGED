import networkx as nx
import numpy as np
import os
from grakel import GraphKernel
from grakel import Graph
from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from .OrcaEvaluator import get_five_node_graphlet_vector
from collections import Counter
from scipy.special import rel_entr

class Evaluator():
    def __init__(self):
        pass 
    
    def evaluateSingleStructure(self, graph, graph_num=1):
        # print('Generating statistics for one graph')
        
        # Compute the eigenvalues first
        eigenvals = self.__calculateEigenvalues(graph, num_values=5)
        eigen_dict = {f'Eigenvalue_{i + 1}': eigenvals[i] for i in range(len(eigenvals))}
        
        res = {
            'Graph Number': graph_num,
            'Average Node Degree': self.__calculateAvgDegree(graph),
            'Unique Degree Count': self.__calculateUniqueDegrees(graph),
            #'Betweenness Centrality': self.__calculateBetweenness(graph),
            #'Closeness Centrality': self.__calculateCloseness(graph),
            'Degree Centrality': self.__calculateDegreeCentrality(graph),
            'Assortivity Coefficient': self.__calculateAssortivity(graph),
            'Clustering Coefficient': self.__calculateClustering(graph),
            'Density': self.__calculateDensity(graph),
            #'Diameter': self.__calculateDiameter(graph),
            #'Number of 3-Motifs': self.__countMotifs(graph),
            #'Number of Cliques': self.__countCliques(graph),
            #'Number of Cycles': self.__countCycles(graph), 
            'Number of Weakly Connected Components': self.__countWeakComponents(graph),
            'Number of Strongly Connected Components': self.__countStrongComponents(graph),
            'Number of Nodes': graph.number_of_nodes(),
            'Number of Edges': graph.number_of_edges(),
        }  
        
        res.update(eigen_dict)  # Add the dicts together
        
        return res
        
    
    def evaluateTwoStructure(self, pred_graph: nx.Graph, true_graph: nx.Graph, graph_num=1):
        # print('Comparing two graphs')
        # print('Note: Plus means that the predicted graph had too many/too high of the associated value')
    
        # Evaluate both graphs independently
        pred_res = self.evaluateSingleStructure(pred_graph)
        true_res = self.evaluateSingleStructure(true_graph)
        
        res = {k: true_res[k] - pred_res[k] for k in true_res}  # Since they share keys we can do this
        
        return res
    
    
    # Add this as a class, and let it keep track of old node ids
    def evaluateEdges(self, pred_graph: nx.Graph, true_graph: nx.Graph, edgebank_pred: dict, edgebank_true: dict,graph_num=1):
        # Get shared nodes and get subgraphs for analysis
        common_nodes = set(pred_graph.nodes()).intersection(true_graph.nodes())
        pred_sub = pred_graph.subgraph(common_nodes)
        true_sub = true_graph.subgraph(common_nodes)

        # Get edges
        pred_edges = set(pred_sub.edges())
        true_edges = set(true_sub.edges())
        possible_edges = set((u, v) for u in common_nodes for v in common_nodes if u != v)

        # Compute for overall
        tp = len(pred_edges & true_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)
        tn = len(possible_edges - (pred_edges | true_edges))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Setup for edge type based precision and recall
        old_nodes_pred = [n for n in pred_graph.nodes() if n in edgebank_pred.keys()]
        new_nodes_pred = [n for n in pred_graph.nodes() if n not in edgebank_pred.keys()]
        old_nodes_true = [n for n in true_graph.nodes() if n in edgebank_true.keys()]
        new_nodes_true = [n for n in true_graph.nodes() if n not in edgebank_true.keys()]

        # print(sorted(new_nodes_pred))
        # print(sorted(new_nodes_true))
        
        # Compute for oo
        common_nodes_oo = set(old_nodes_pred).intersection(old_nodes_true)
        if len(common_nodes_oo) == 0:
            print("No common old-old nodes")
        pred_sub_oo = pred_graph.subgraph(common_nodes_oo)
        true_sub_oo = true_graph.subgraph(common_nodes_oo)
        pred_edges_oo = set([edge for edge in pred_sub_oo.edges() if edge[1] in edgebank_pred.get(edge[0], [])])
        true_edges_oo = set([edge for edge in true_sub_oo.edges() if edge[1] in edgebank_true.get(edge[0], [])])
        
        possible_edges_oo = set((u, v) for u in common_nodes_oo for v in common_nodes_oo if u != v)
        
        tp_oo = len(pred_edges_oo & true_edges_oo)
        fp_oo = len(pred_edges_oo - true_edges_oo)
        fn_oo = len(true_edges_oo - pred_edges_oo)
        tn_oo = len(possible_edges_oo - (pred_edges_oo | true_edges_oo))
        precision_oo = tp_oo / (tp_oo + fp_oo) if (tp_oo + fp_oo) > 0 else 0.0
        recall_oo = tp_oo / (tp_oo + fn_oo) if (tp_oo + fn_oo) > 0 else 0.0
        
        
        # Compute for oon
        common_nodes_oon = set(old_nodes_pred).intersection(old_nodes_true)
        pred_sub_oon = pred_graph.subgraph(common_nodes_oon)
        true_sub_oon  = true_graph.subgraph(common_nodes_oon)
        pred_edges_oon = {
            edge for edge in pred_sub_oon.edges()
            if edge[0] in edgebank_pred and edge[1] not in edgebank_pred[edge[0]]
        }
        true_edges_oon = {
            edge for edge in true_sub_oon.edges()
            if edge[0] in edgebank_pred and edge[1] not in edgebank_true[edge[0]]
        }
        
        possible_edges_oon = {
        (u, v) for u in common_nodes_oon for v in common_nodes_oon if u != v
        }
        
        tp_oon = len(pred_edges_oon & true_edges_oon)
        fp_oon = len(pred_edges_oon - true_edges_oon)
        fn_oon = len(true_edges_oon - pred_edges_oon)
        tn_oon = len(possible_edges_oon - (pred_edges_oon | true_edges_oon))
        precision_oon= tp_oon / (tp_oon + fp_oon) if (tp_oon + fp_oon) > 0 else 0.0
        recall_oon = tp_oon / (tp_oon + fn_oon) if (tp_oon + fn_oon) > 0 else 0.0
        
        old_set_pred = set(old_nodes_pred)
        new_set_pred = set(new_nodes_pred)
        old_set_true = set(old_nodes_true)
        new_set_true = set(new_nodes_true)

        # Get edges between type 0 and type 1 in pred_graph
        pred_edges_on = {
            (u, v) for u, v in pred_graph.edges()
            if (u in old_set_pred and v in new_set_pred) or (u in new_set_pred and v in old_set_pred)
        }

        # Get edges between type 0 and type 1 in true_graph
        true_edges_on = {
            (u, v) for u, v in true_graph.edges()
            if (u in old_set_true and v in new_set_true) or (u in new_set_true and v in old_set_true)
        }

        # Optional: If node sets differ between graphs, intersect nodes
        common_nodes = set(pred_graph.nodes()) & set(true_graph.nodes())
        pred_edges_on = {(u, v) for (u, v) in pred_edges_on if u in common_nodes and v in common_nodes}
        true_edges_on = {(u, v) for (u, v) in true_edges_on if u in common_nodes and v in common_nodes}
        possible_edges_on = set((u, v) for u in common_nodes for v in common_nodes if u != v)

        # Compute precision and recall
        tp_on = len(pred_edges_on & true_edges_on)
        fp_on = len(pred_edges_on - true_edges_on)
        fn_on = len(true_edges_on - pred_edges_on)
        tn_on = len(possible_edges_on - (pred_edges_on | true_edges_on))
        precision_on = tp_on / (tp_on + fp_on) if (tp_on + fp_on) > 0 else 0.0
        recall_on = tp_on / (tp_on + fn_on) if (tp_on + fn_on) > 0 else 0.0
        
        # Compute for edge type nn
        common_nodes_nn = set(new_nodes_pred).intersection(new_nodes_true)
        if len(common_nodes_nn) == 0:
            print("No common new-new nodes")
        pred_sub_nn = pred_graph.subgraph(common_nodes_nn)
        true_sub_nn = true_graph.subgraph(common_nodes_nn)
        pred_edges_nn = set(pred_sub_nn.edges())
        true_edges_nn = set(true_sub_nn.edges())
        possible_edges_nn = set((u, v) for u in common_nodes_nn for v in common_nodes_nn if u != v)
        
        tp_nn = len(pred_edges_nn & true_edges_nn)
        fp_nn = len(pred_edges_nn - true_edges_nn)
        fn_nn = len(true_edges_nn - pred_edges_nn)
        tn_nn = len(possible_edges_nn - (pred_edges_nn | true_edges_nn))
        precision_nn = tp_nn / (tp_nn + fp_nn) if (tp_nn + fp_nn) > 0 else 0.0
        recall_nn = tp_nn / (tp_nn + fn_nn) if (tp_nn + fn_nn) > 0 else 0.0
        
        pred_nodes_overall = set(pred_graph.nodes())
        true_nodes_overall = set(true_graph.nodes())
        
        correct_nodes_old = len(set(old_nodes_pred) & set(old_nodes_true))
        correct_nodes_new = len(set(new_nodes_pred) & set(new_nodes_true))
        correct_nodes_overall = len(pred_nodes_overall & true_nodes_overall)
        
        res = {
            'Graph Number': graph_num,
            'precision overall': precision,
            'recall overall': recall,
            'tp overall': tp, 
            'fp overall': fp,
            'tn overall': tn,
            'fn overall': fn,
            'precision oo': precision_oo,
            'recall oo': recall_oo,
            'tp oo': tp_oo, 
            'fp oo': fp_oo,
            'tn oo': tn_oo,
            'fn oo': fn_oo,
            'precision oon': precision_oon,
            'recall oon': recall_oon,
            'tp oon': tp_oon, 
            'fp oon': fp_oon,
            'tn oon': tn_oon,
            'fn oon': fn_oon,
            'precision on': precision_on,
            'recall on': recall_on,
            'tp on': tp_on, 
            'fp on': fp_on,
            'tn on': tn_on,
            'fn on': fn_on,
            'precision nn': precision_nn,
            'recall nn': recall_nn,
            'tp nn': tp_nn, 
            'fp nn': fp_nn,
            'tn nn': tn_nn,
            'fn nn': fn_nn,
            'Correct Node IDs': len(common_nodes) / len(true_graph.nodes()),
            'Correct Old Node IDs': correct_nodes_old,
            'Precision Old IDs': correct_nodes_old / len(old_nodes_pred) if old_nodes_pred else 0.0,
            'Recall Old IDs': correct_nodes_old / len(old_nodes_true) if old_nodes_true else 0.0,
            'Correct New Node IDs': correct_nodes_new,
            'Precision New IDs': correct_nodes_new / len(new_nodes_pred) if new_nodes_pred else 0.0,
            'Recall New IDs': correct_nodes_new / len(new_nodes_true) if new_nodes_true else 0.0,
            'Correct Overall IDs': correct_nodes_overall,
            'Precision Overall IDs': correct_nodes_overall / len(pred_nodes_overall) if pred_nodes_overall else 0.0,
            'Recall Overall IDs': correct_nodes_overall / len(true_nodes_overall) if true_nodes_overall else 0.0,
        }
        
        return res
    
    
    def evaluateGraphletKernel(self, pred_graph: nx.DiGraph, true_graph: nx.DiGraph):
        # Must be undirected
        pred_graph_undirected = pred_graph.to_undirected()
        true_graph_undirected = true_graph.to_undirected()
        
        # Must convert the graphs to a usable format
        def to_grakel_format(graph):
            edges = list(graph.edges())
            return Graph(edges)
        
        gk = GraphKernel(kernel=["graphlet_sampling"], normalize=False)  # For computing the kernels (False means discrete counts of subgraphs)
        
        # Compute the kernels
        feature_vectors = gk.fit_transform([to_grakel_format(pred_graph_undirected), to_grakel_format(true_graph_undirected)])
        pred_kernel, true_kernel = feature_vectors[0], feature_vectors[1]
        
        return pred_kernel, true_kernel, wasserstein_distance(pred_kernel, true_kernel)
    
    
    def evaluateOrca(self, pred_graph: nx.DiGraph, true_graph: nx.DiGraph):
        # Graphs must be undirected for orca
        pred_graph_undirected = pred_graph.to_undirected()
        true_graph_undirected = true_graph.to_undirected()
        
        # Check the binaries path
        pred_kernel = get_five_node_graphlet_vector(pred_graph_undirected)
        true_kernel = get_five_node_graphlet_vector(true_graph_undirected)
        
        return pred_kernel, true_kernel, wasserstein_distance(pred_kernel, true_kernel)
        
    
    def evaluateTopER(self, A, B, pred_label, true_label, graph_num=0):
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
    
    
    # Animate the graphs for visualization
    def create_animation(self, predicted_graphs, target_graphs, output_file="graph_animation.gif"):
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
        fig.suptitle("Predicted Graph (Left) vs True Graph (Right)", fontsize=16)
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

        #plt.show()
    
    
    def __calculateAvgDegree(self, graph: nx.DiGraph):
        try:
            degrees = dict(graph.degree())
            avg_degree = sum(degrees.values()) / graph.number_of_nodes()
            return avg_degree 
        
        except:
            return float('inf')
        
        
    def __calculateUniqueDegrees(self, graph: nx.DiGraph):
        try:
            degrees = dict(graph.degree())
            unique_degrees = set(degrees.values())
            return len(unique_degrees)
        
        except:
            return float('inf')
        
        
    def __calculateBetweenness(self, graph: nx.DiGraph):
        try:
            return np.mean(list(nx.betweenness_centrality(graph).values()))
        
        except:
            return float('inf')
        
        
    def __calculateCloseness(self, graph: nx.DiGraph):
        try:
            return np.mean(list(nx.closeness_centrality(graph).values()))
        
        except:
            return float('inf')
        
        
    def __calculateDegreeCentrality(self, graph: nx.DiGraph):
        try:
            return np.mean(list(nx.degree_centrality(graph).values()))
        
        except:
            return float('inf')
        
        
    def __calculateAssortivity(self, graph: nx.DiGraph):
        try:
            return nx.degree_assortativity_coefficient(graph)
            
        except:
            return float('inf')
        
        
    def __calculateClustering(self, graph: nx.DiGraph):
        try:
            return nx.average_clustering(graph.to_undirected())
        
        except:
            return float('inf')
        
        
    def __calculateDensity(self, graph: nx.DiGraph):
        try:
            return nx.density(graph)
        
        except:
            return float('inf')
        
        
    def __calculateDiameter(self, graph: nx.DiGraph):
        try:
            # Since the graph must be strongly connected for this to work
            if nx.is_strongly_connected(graph):
                return nx.diameter(graph)
            else:
                largest_cc = max(nx.strongly_connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                return nx.diameter(subgraph)
        
        except:
            return float('inf')
        
        
    def __countMotifs(self, graph: nx.DiGraph):
        try:
            return sum(nx.triangles(graph.to_undirected()).values()) // 3
        except:
            return float('inf')
        
        
    def __countCliques(self, graph: nx.DiGraph):
        try:
            return len(list(nx.find_cliques(graph.to_undirected())))
        except:
            return float('inf')
        
        
    def __countCycles(self, graph: nx.DiGraph):
        try:
            return sum(1 for _ in nx.simple_cycles(graph))
        
        except:
            return float('inf')
        
        
    def __countWeakComponents(self, graph: nx.DiGraph):
        try:
            return len(list(nx.weakly_connected_components(graph)))
        
        except:
            return float('inf')
        
        
    def __countStrongComponents(self, graph: nx.DiGraph):
        try:
            return len(list(nx.strongly_connected_components(graph)))
        
        except:
            return float('inf')
        
    
    def __calculateEigenvalues(self, graph: nx.Graph, num_values=5):
        matrix = nx.to_numpy_array(graph)
        eigenvals = np.linalg.eigvals(matrix).real
        
        top_k_vals = sorted(eigenvals, key=lambda x: abs(x), reverse=True)[:num_values]
        
        return top_k_vals

    def kl_divergence_graphs(self, G1, G2, mode="in"):
        def get_degree_distribution(graph, mode="in"):
            if mode == "in":
                degrees = [deg for _, deg in graph.in_degree()]
            elif mode == "out":
                degrees = [deg for _, deg in graph.out_degree()]
            elif mode == "total":
                degrees = [graph.in_degree(n) + graph.out_degree(n) for n in graph.nodes()]
            else:
                raise ValueError("mode must be 'in', 'out', or 'total'")

            return Counter(degrees)
        
        dist1 = get_degree_distribution(G1, mode=mode)
        dist2 = get_degree_distribution(G2, mode=mode)

        all_degrees = sorted(set(dist1) | set(dist2))
        
        P = np.array([dist1.get(d, 0) for d in all_degrees], dtype=float)
        Q = np.array([dist2.get(d, 0) for d in all_degrees], dtype=float)

        epsilon = 1e-10
        P += epsilon
        Q += epsilon
        P /= P.sum()
        Q /= Q.sum()

        kl = np.sum(rel_entr(P, Q))
        return kl