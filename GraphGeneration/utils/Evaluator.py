import networkx as nx
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
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
    
    def evaluateSingleStructure(self, graph: nx.Graph, graph_num=1):
        # print('Generating statistics for one graph')
        if graph.number_of_nodes() == 0:
            print('GRAPH HAS NO NODES')
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
            'Number of Triangles': sum(nx.triangles(graph).values()) // 3,  # Since each node is counted 3 times per triangle, we divide by 3
            'Number of Connected Components': nx.number_connected_components(graph),
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
            # 'precision overall': precision,
            # 'recall overall': recall,
            # 'tp overall': tp, 
            # 'fp overall': fp,
            # 'tn overall': tn,
            # 'fn overall': fn,
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
            # 'precision on': precision_on,
            # 'recall on': recall_on,
            # 'tp on': tp_on, 
            # 'fp on': fp_on,
            # 'tn on': tn_on,
            # 'fn on': fn_on,
            # 'precision nn': precision_nn,
            # 'recall nn': recall_nn,
            # 'tp nn': tp_nn, 
            # 'fp nn': fp_nn,
            # 'tn nn': tn_nn,
            # 'fn nn': fn_nn,
            # 'Correct Node IDs': len(common_nodes) / len(true_graph.nodes()),
            # 'Correct Old Node IDs': correct_nodes_old,
            # 'Precision Old IDs': correct_nodes_old / len(old_nodes_pred) if old_nodes_pred else 0.0,
            # 'Recall Old IDs': correct_nodes_old / len(old_nodes_true) if old_nodes_true else 0.0,
            # 'Correct New Node IDs': correct_nodes_new,
            # 'Precision New IDs': correct_nodes_new / len(new_nodes_pred) if new_nodes_pred else 0.0,
            # 'Recall New IDs': correct_nodes_new / len(new_nodes_true) if new_nodes_true else 0.0,
            # 'Correct Overall IDs': correct_nodes_overall,
            # 'Precision Overall IDs': correct_nodes_overall / len(pred_nodes_overall) if pred_nodes_overall else 0.0,
            # 'Recall Overall IDs': correct_nodes_overall / len(true_nodes_overall) if true_nodes_overall else 0.0,
        }
        
        return res
    
    
    def evaluateEdgesNew(self, pred_graph, true_graph, old_true, old_pred):
        """
        pred_graph: NxN numpy or tensor of predicted probabilities
        true_graph: NxN numpy or tensor of 0/1
        old_true:   iterable of old node IDs in the *true* graph
        old_pred:   iterable of old node IDs in the *predicted* graph

        Returns dict of metrics, including:
        - ROC (old-old only)
        - AP for all 6 categories
        - precision, recall, F1 for all 6 categories
        - counts of invalid edges
        """
        old_true = set(old_true)
        old_pred = set(old_pred)
        old_nodes = sorted(list(old_true.intersection(old_pred)))

        N = len(true_graph.nodes)
        all_nodes = list(range(N))
        new_nodes = sorted(list(set(all_nodes) - set(old_nodes)))

        invalid_edge_count = 0

        def collect_pairs(A, B):
            y_true, y_pred = [], []
            true_counts, false_counts = 0, 0
            for i in A:
                for j in B:
                    if i == j:
                        continue
                    # If either node doesn't exist in TRUE graph → invalid edge
                    if i not in true_graph.nodes or j not in true_graph.nodes:
                        nonlocal invalid_edge_count
                        invalid_edge_count += 1
                        y_true.append(0)
                        y_pred.append(int(pred_graph.has_edge(i, j)))
                        continue

                    label = int(true_graph.has_edge(i, j))
                    pred = int(pred_graph.has_edge(i, j))

                    y_true.append(label)
                    y_pred.append(pred)

                    if pred == 1:
                        if label == 1:
                            true_counts += 1
                        else:
                            false_counts += 1
            return np.array(y_true), np.array(y_pred), true_counts, false_counts

        # -------------------------- Groups --------------------------
        y_true_oo, y_pred_oo, tp_oo, fp_oo = collect_pairs(old_nodes, old_nodes)
        y_true_on, y_pred_on, tp_on, fp_on = collect_pairs(old_nodes, new_nodes)
        y_true_nn, y_pred_nn, tp_nn, fp_nn = collect_pairs(new_nodes, new_nodes)

        # Split old-old by true label
        mask_bank = y_true_oo == 1
        mask_nobank = y_true_oo == 0

        y_true_oobank = y_true_oo[mask_bank]
        y_pred_oobank = y_pred_oo[mask_bank]

        y_true_oonobank = y_true_oo[mask_nobank]
        y_pred_oonobank = y_pred_oo[mask_nobank]

        # ------------------------ Metrics ---------------------------
        def safe_roc(y, p):
            return float("nan") if len(np.unique(y)) < 2 else roc_auc_score(y, p)

        def safe_ap(y, p):
            return float("nan") if len(y) == 0 else average_precision_score(y, p)

        def pr_metrics(y, p):
            if len(y) == 0 or np.sum(y) == 0:
                return 0.0, 0.0, 0.0
            preds = (p > 0.5).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
            return prec, rec, f1

        # ------------------------ Build Output -----------------------
        results = {
            # ROC (old-old only)
            "roc_o-o_nobank": safe_roc(y_true_oonobank, y_pred_oonobank),
            "roc_o-o_bank": safe_roc(y_true_oobank, y_pred_oobank),
            "roc_o-o_overall": safe_roc(y_true_oo, y_pred_oo),

            # AP
            "ap_o-o_nobank": safe_ap(y_true_oonobank, y_pred_oonobank),
            "ap_o-o_bank": safe_ap(y_true_oobank, y_pred_oobank),
            "ap_o-n": safe_ap(y_true_on, y_pred_on),
            "ap_n-n": safe_ap(y_true_nn, y_pred_nn),
            "ap_o-o_overall": safe_ap(y_true_oo, y_pred_oo),
            "ap_all": safe_ap(
                np.concatenate([y_true_oo, y_true_on, y_true_nn]),
                np.concatenate([y_pred_oo, y_pred_on, y_pred_nn])
            ),

            # Precision / Recall / F1
            "prf_o-o_nobank": pr_metrics(y_true_oonobank, y_pred_oonobank),
            "prf_o-o_bank": pr_metrics(y_true_oobank, y_pred_oobank),
            "prf_o-n": pr_metrics(y_true_on, y_pred_on),
            "prf_n-n": pr_metrics(y_true_nn, y_pred_nn),
            "prf_o-o_overall": pr_metrics(y_true_oo, y_pred_oo),
            "prf_all": pr_metrics(
                np.concatenate([y_true_oo, y_true_on, y_true_nn]),
                np.concatenate([y_pred_oo, y_pred_on, y_pred_nn])
            ),

            # Counts of invalid edges
            "invalid_pred_edge_count": invalid_edge_count,

            # True / False edges in pred
            "tp_o-o": tp_oo,
            "fp_o-o": fp_oo,
            "tp_o-n": tp_on,
            "fp_o-n": fp_on,
            "tp_n-n": tp_nn,
            "fp_n-n": fp_nn,
        }

        return results
        
    
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
    
    
    def evaluateOrca(self, pred_graph, true_graph):
        # Graphs must be undirected for orca
        pred_graph_undirected = pred_graph.to_undirected()
        true_graph_undirected = true_graph.to_undirected()
        
        # Check the binaries path
        pred_kernel = get_five_node_graphlet_vector(pred_graph_undirected)
        true_kernel = get_five_node_graphlet_vector(true_graph_undirected)
        
        return pred_kernel, true_kernel, wasserstein_distance(pred_kernel, true_kernel)
        
    
    def evaluateTopER(self, A, B, graph_num=0):
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
        
        except Exception as e:
            print(f"Error in __calculateAvgDegree: {e}")
            return float('inf')
        
        
    def __calculateUniqueDegrees(self, graph: nx.DiGraph):
        try:
            degrees = dict(graph.degree())
            unique_degrees = set(degrees.values())
            return len(unique_degrees)
        
        except Exception as e:
            print(f"Error in __calculateUniqueDegrees: {e}")
            return float('inf')
        
        
    def __calculateBetweenness(self, graph: nx.DiGraph):
        try:
            return np.mean(list(nx.betweenness_centrality(graph).values()))
        
        except Exception as e:
            print(f"Error in __calculateBetweenness: {e}")
            return float('inf')
        
        
    def __calculateCloseness(self, graph: nx.DiGraph):
        try:
            return np.mean(list(nx.closeness_centrality(graph).values()))
        
        except Exception as e:
            print(f"Error in __calculateCloseness: {e}")
            return float('inf')
        
        
    def __calculateDegreeCentrality(self, graph: nx.DiGraph):
        try:
            return np.mean(list(nx.degree_centrality(graph).values()))
        
        except Exception as e:
            print(f"Error in __calculateDegreeCentrality: {e}")
            return float('inf')
        
        
    def __calculateAssortivity(self, graph: nx.Graph):
        try:
            return nx.degree_assortativity_coefficient(graph)
            
        except Exception as e:
            print(f"Error in __calculateAssortivity: {e}")
            return float('inf')
        
        
    def __calculateClustering(self, graph: nx.DiGraph):
        try:
            return nx.average_clustering(graph.to_undirected())
        
        except Exception as e:
            print(f"Error in __calculateClustering: {e}")
            return -1.0
        
    
    def __calculateDensity(self, graph: nx.DiGraph):
        try:
            return nx.density(graph)
        
        except Exception as e:
            print(f"Error in __calculateDensity: {e}")
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
        
        except Exception as e:
            # print(f"Error in __countStrongComponents: {e}")
            return float('inf')
        
    
    def __calculateEigenvalues(self, graph: nx.Graph, num_values=5):
        if graph.number_of_nodes() == 0:
            return np.zeros(num_values)
        if graph.number_of_nodes() < num_values:
            # Pad with zeros
            return np.zeros(num_values)
        matrix = nx.to_numpy_array(graph)
        try:
            eigenvals = np.linalg.eigvals(matrix).real
        except:
            return np.zeros(num_values)
        
        top_k_vals = sorted(eigenvals, key=lambda x: abs(x), reverse=True)[:num_values]
        
        return top_k_vals

    def kl_divergence_graphs(self, G1, G2, mode="in"):
        def get_degree_distribution(graph, mode="in"):
            if mode == "in":
                degrees = [deg for _, deg in graph.in_degree()]
            elif mode == "out":
                degrees = [deg for _, deg in graph.out_degree()]
            elif mode == "total":
                # degrees = [graph.in_degree(n) + graph.out_degree(n) for n in graph.nodes()]
                degrees = [graph.degree(n) for n in graph.nodes()]
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
    
    
    def evaluate_node_selection(self, sorted_nodes_pred, sorted_nodes_true, graph_num=0):
        
        # Unpack for readability
        pred_new_nodes = set(sorted_nodes_pred.get('new_nodes', set()))
        pred_old_nodes = set(sorted_nodes_pred.get('old_nodes', set()))
        true_new_nodes = set(sorted_nodes_true.get('new_nodes', set()))
        true_old_nodes = set(sorted_nodes_true.get('old_nodes', set()))

        # --- New nodes ---
        new_correct = pred_new_nodes & true_new_nodes
        new_precision = len(new_correct) / len(pred_new_nodes) if pred_new_nodes else 0
        new_recall = len(new_correct) / len(true_new_nodes) if true_new_nodes else 0
        new_f1 = (2 * new_precision * new_recall / (new_precision + new_recall)
                if (new_precision + new_recall) > 0 else 0)

        new_node_accuracy = len(new_correct) / len(true_new_nodes) if true_new_nodes else 0

        # --- Old nodes ---
        old_correct = pred_old_nodes & true_old_nodes
        old_precision = len(old_correct) / len(pred_old_nodes) if pred_old_nodes else 0
        old_recall = len(old_correct) / len(true_old_nodes) if true_old_nodes else 0
        old_f1 = (2 * old_precision * old_recall / (old_precision + old_recall)
                if (old_precision + old_recall) > 0 else 0)
        
        old_node_accuracy = len(old_correct) / len(true_old_nodes) if true_old_nodes else 0
        
        # --- Combined nodes ---
        pred_all = pred_new_nodes | pred_old_nodes
        true_all = true_new_nodes | true_old_nodes
        all_correct = pred_all & true_all
        all_precision = len(all_correct) / len(pred_all) if pred_all else 0
        all_recall = len(all_correct) / len(true_all) if true_all else 0
        all_f1 = (2 * all_precision * all_recall / (all_precision + all_recall)
                if (all_precision + all_recall) > 0 else 0)

        return {
            'Graph_Num': graph_num,
            'Precision_New': new_precision,
            'Recall_New': new_recall,
            'F1_New': new_f1,
            'Accuracy_New': new_node_accuracy,
            'Precision_Old': old_precision,
            'Recall_Old': old_recall,
            'F1_Old': old_f1,
            'Accuracy_Old': old_node_accuracy,
            'Precision_All': all_precision,
            'Recall_All': all_recall,
            'F1_All': all_f1,
        }

        
    def evaluate_graph_edges(self, pred_graph, true_graph, is_directed=False, graph_num=0, edgebank=None):
        # 1. Get the Node Universe (Union of both graphs)
        pred_nodes = set(pred_graph.nodes())
        true_nodes = set(true_graph.nodes())
        all_nodes = list(pred_nodes | true_nodes)
        num_total_nodes = len(all_nodes)
        
        # 2. Standardize Edge Sets
        if is_directed:
            pred_edges = set(pred_graph.edges())
            true_edges = set(true_graph.edges())
        else:
            pred_edges = {tuple(sorted(e)) for e in pred_graph.edges()}
            true_edges = {tuple(sorted(e)) for e in true_graph.edges()}

        if edgebank is not None:
            pred_bank = {e for e in pred_edges if e[1] in edgebank.get(e[0], [])}
            pred_nobank = pred_edges - pred_bank

            true_bank = {e for e in true_edges if e[1] in edgebank.get(e[0], [])}
            true_nobank = true_edges - true_bank

            # Bank Metrics
            tp_bank = len(pred_bank & true_bank)
            fp_bank = len(pred_bank - true_bank)
            fn_bank = len(true_bank - pred_bank)

            # No-Bank Metrics
            tp_nobank = len(pred_nobank & true_nobank)
            fp_nobank = len(pred_nobank - true_nobank)
            fn_nobank = len(true_nobank - pred_nobank)

            # Observation Spaces for TN calculation
            if is_directed:
                total_possible = num_total_nodes * (num_total_nodes - 1)
            else:
                total_possible = (num_total_nodes * (num_total_nodes - 1)) // 2

            # Count potential bank edges (pairs existing in historical edgebank involving current nodes)
            bank_possible_count = sum(len(neighbors) for node, neighbors in edgebank.items() if node in all_nodes)
            if not is_directed:
                bank_possible_count = bank_possible_count // 2
            
            nobank_possible_count = total_possible - bank_possible_count

            # True Negatives
            tn_bank = bank_possible_count - (tp_bank + fp_bank + fn_bank)
            tn_nobank = nobank_possible_count - (tp_nobank + fp_nobank + fn_nobank)

            # Ratios
            precision_bank = tp_bank / (tp_bank + fp_bank) if (tp_bank + fp_bank) > 0 else 0
            recall_bank = tp_bank / (tp_bank + fn_bank) if (tp_bank + fn_bank) > 0 else 0
            accuracy_bank = (tp_bank + tn_bank) / bank_possible_count if bank_possible_count > 0 else 0
            f1_bank = 2 * precision_bank * recall_bank / (precision_bank + recall_bank) if (precision_bank + recall_bank) > 0 else 0

            precision_nobank = tp_nobank / (tp_nobank + fp_nobank) if (tp_nobank + fp_nobank) > 0 else 0
            recall_nobank = tp_nobank / (tp_nobank + fn_nobank) if (tp_nobank + fn_nobank) > 0 else 0
            accuracy_nobank = (tp_nobank + tn_nobank) / nobank_possible_count if nobank_possible_count > 0 else 0
            f1_nobank = 2 * precision_nobank * recall_nobank / (precision_nobank + recall_nobank) if (precision_nobank + recall_nobank) > 0 else 0
        else:
            # 3. Intersection and Differences
            tp = len(pred_edges & true_edges)  # True Positives
            fp = len(pred_edges - true_edges)  # False Positives
            fn = len(true_edges - pred_edges)  # False Negatives
            
            # 4. Accuracy Logic: The "Observation Space"
            # To calculate Accuracy, we need TN (True Negatives).
            # We define the space as all edges that COULD have been predicted between existing nodes.
            if is_directed:
                possible_edges_count = num_total_nodes * (num_total_nodes - 1)
            else:
                possible_edges_count = (num_total_nodes * (num_total_nodes - 1)) // 2
            
            # TN is every possible edge that was NOT predicted and DOES NOT exist
            tn = possible_edges_count - (tp + fp + fn)

            # 5. Metric Calculations
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            # Accuracy = (TP + TN) / (TP + TN + FP + FN)
            accuracy = (tp + tn) / possible_edges_count if possible_edges_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if edgebank is not None:
            results = {
                "Graph_Num": graph_num,
                "TP_bank": tp_bank,
                "FP_bank": fp_bank,
                "TN_bank": tn_bank,
                "FN_bank": fn_bank,
                "TP_nobank": tp_nobank,
                "FP_nobank": fp_nobank,
                "TN_nobank": tn_nobank,
                "FN_nobank": fn_nobank,
                "Precision_bank": precision_bank,
                "Recall_bank": recall_bank,
                "Accuracy_bank": accuracy_bank,                
                "Precision_nobank": precision_nobank,
                "Recall_nobank": recall_nobank,
                "Accuracy_nobank": accuracy_nobank,
                "F1_bank": f1_bank,
                "F1_nobank": f1_nobank
            }
        else:
            results = {
                "Graph_Num": graph_num,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "Accuracy": accuracy,
                "F1": f1
            }

        return results
    
    
    def write_kl_results(self, path, value, graph_num):
        with open(path, "a") as f:
            f.write(f"{graph_num + 1}, {value:.6f}\n")
            f.flush()