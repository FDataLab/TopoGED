import argparse
import math
import numpy as np 
import networkx as nx
import random
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sympy import S
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
import os
import pathlib
from pathlib import Path
import re
import sys
import yaml
import pickle 
import matplotlib.pyplot as plt
import gc
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from GraphGeneration.utils.Evaluator import Evaluator
from load_data import load_data, generate_training_data_cached, generate_validation_data_cached, generate_negative_edges
from create_sub_graphs import create_nn_graph, create_on_graph
from GraphGeneration.utils.ablation_utils import ablationSetup

from torch.utils.data import DataLoader

# Import Loss fn
# from GraphGeneration.scripts.composite_graphlet_loss_fn import GraphletLoss
# from GraphGeneration.utils.estimate_graphlet import run_graphlet_estimate   
# TODO Rename these ^^^

from utils.embedding_methods.degree import EmbedDegree
from GraphGeneration.scripts.load_data import load_data
from process_data import modifyGraphIds, build_edgebanks_from_start

import warnings

# Silence the specific Scikit-Learn "No positive class" warning
warnings.filterwarnings("ignore", message="No positive class found in y_true")
warnings.filterwarnings("ignore", category=RuntimeWarning)

num_test_graphs = {
    'CollegeMsg': 28,
    'mathoverflow': 29,
    'networkadex': 45,
    'networkaeternity': 36,
    'networkaion': 30,
    'networkaragon': 52,
    'networkbancor': 48,
    'networkcentra': 40,
    'networkcoindash': 42,
    'networkcindicator': 34,
    'networkdgd': 109,
    'networkiconomi': 83,
    'Reddit_B': 61,
    'tgbl-wiki': 5,
}


import warnings
from sklearn.exceptions import UndefinedMetricWarning

class EvaluateGraphs():
    def __init__(self, args, threshold=None, prefix=None, data_path=None):
        self.evaluator = Evaluator()
        self.args = args
        self.all_results = []
        
        self.dataset = args.dataset if args is not None else None
        self.embedding_method = args.embedding_method if args is not None else None
        self.use_ma = getattr(args, 'use_ma', None)
        self.is_directed=False
        
        if prefix == '' and data_path == '':
            return
        
        if prefix:
            prefix = prefix
        elif threshold:
            prefix = f"GraphGeneration/scripts/results/{self.dataset}/{self.embedding_method}{f'_ma{self.use_ma}' if self.use_ma else ''}_lr001_5back_{self.embedding_method}_threshold{threshold}_oldonly/"
        else:
            # prefix = f"GraphGeneration/scripts/results/{self.dataset}/{self.embedding_method}{f'_ma{self.use_ma}' if self.use_ma else ''}_lr001_5back_oobankchanges_oldonly/"
            # prefix = f"GraphGeneration/scripts/results/{self.dataset}/{self.embedding_method}{f'_ma{self.use_ma}' if self.use_ma else ''}_lr001_5back_learnedparams_oldonly/"
            prefix = f"GraphGeneration/scripts/results/{self.dataset}/{self.embedding_method}{f'_ma{self.use_ma}' if self.use_ma else ''}_lr001_5back_existchanges_oldonly/"
        
        
        self.file_visualization_path = os.path.join(prefix, "file_visualization")
        self.structure_dir = os.path.join(prefix, "structure")
        self.edge_eval_dir = os.path.join(prefix, "edge_evaluation")
        self.kernel_dir = os.path.join(prefix, "kernel_visualization")

        # Load the data
        if data_path:
            self.data_path = data_path
        elif self.embedding_method == "hks":
            self.data_path = os.path.join(f"data/output/constructed_graphs/{self.dataset}_topoGED_embedding_mlpEncodingConcat_embeddingType{self.embedding_method}_{self.embedding_method}{f'_{self.use_ma}' if self.use_ma else ''}", f"{self.embedding_method}_constructed_graphs_{self.dataset}.pkl")
        elif self.embedding_method == "Node2Vec":
            self.data_path = os.path.join(f"data/output/constructed_graphs/{self.dataset}_topoGED_embedding_mlpEncodingConcat_embeddingType{self.embedding_method}{f'_{self.use_ma}' if self.use_ma else ''}_lr001", f"{self.embedding_method}_constructed_graphs_{self.dataset}.pkl")
        elif threshold and self.embedding_method != "htgn":
            self.data_path = f"data/output/constructed_graphs/benchmarking/{self.dataset}_topoGED_embeddingDegree_{self.embedding_method}_threshold{threshold}/GCN_constructed_graphs_{self.dataset}.pkl"
        elif threshold:
            self.data_path = f"data/output/constructed_graphs/{self.dataset}_topoGED_embeddingDegree_{self.embedding_method}_threshold{threshold}/GCN_constructed_graphs_{self.dataset}.pkl"
        else:
            # self.data_path = f"data/output/constructed_graphs/{self.dataset}_topoGED_embeddingDegree_mlpEncodingConcat_embeddingTypeGCN_lr0.001_5back_oobankchanges/GCN_constructed_graphs_{self.dataset}_old_only.pkl"
            # self.data_path = f"data/output/constructed_graphs/{self.dataset}_topoGED_embeddingDegree_mlpEncodingConcat_embeddingTypeGCN_binary_lr0.001_predictednodesFalse_5back_learnedparams/GCN_constructed_graphs_{self.dataset}_old_only.pkl"
            # self.data_path = f"data/output/constructed_graphs/{self.dataset}_topoGED_embeddingDegree_mlpEncodingConcat_embeddingTypeGAT_binary_lr0.001_predictednodesFalse_5back_learnedparams/GAT_constructed_graphs_{self.dataset}_old_only.pkl"
            # self.data_path = f"data/output/constructed_graphs/{self.dataset}_topoGED_embeddingDegree_mlpEncodingConcat_embeddingTypeGAT_lr0.001_5back_oobankchanges/GAT_constructed_graphs_{self.dataset}_old_only.pkl"
            self.data_path = f"data/output/constructed_graphs/{self.dataset}_topoGED_embeddingDegree_mlpEncodingConcat_embeddingTypeGCN_binary_lr0.001_5back_existanceweighted/GCN_constructed_graphs_{self.dataset}_old_only.pkl"
            # self.data_path = f"data/output/constructed_graphs/{self.dataset}_topoGED_embeddingDegree_mlpEncodingConcat_embeddingTypeGAT_binary_lr0.001_5back_existanceweighted/GAT_constructed_graphs_{self.dataset}_old_only.pkl"
        
        
        
        print(self.data_path)
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)

            # 1. Handle Unpacking based on actual data type
            if isinstance(data, tuple):
                # TopoGED style: tuple unpacking
                pred_graphs, _, _, _ = data 
                num_nodes = None 
            elif isinstance(data, dict):
                # Standard style: dictionary access
                pred_graphs = data['graphs']
                num_nodes = data.get('node_count')
            else:
                raise TypeError(f"Unknown data format: {type(data)}")

            # 2. Shared Data Loading Logic
            path_obj = pathlib.Path(self.data_path)
            filename_no_ext = path_obj.stem

            # If it's a TopoGED constructed file, dataset is at the end
            if "constructed_graphs_" in filename_no_ext:
                # This removes the specific prefix and gives you 'Reddit_B'
                extracted_dataset = filename_no_ext.replace("GCN_constructed_graphs_", "")
            else:
                # For Benchmarkers or others, check if the parent folder is more reliable
                # If the filename starts with a known generic word, use the parent's first chunk
                first_chunk = filename_no_ext.split('_')[0]
                if first_chunk.lower() in ['gcn', 'evolvegcn', 'constructed']:
                    extracted_dataset = path_obj.parent.name.split('_')[0]
                else:
                    extracted_dataset = first_chunk

            print(f"Detected Dataset: {extracted_dataset}")
            
            import scipy.sparse as sp
            
            # Check the first graph to determine the format of the whole list
            if len(pred_graphs) > 0 and sp.issparse(pred_graphs[0]):
                print(f"INFO: Scipy sparse matrices detected. Converting to NetworkX...")
                new_nx_graphs = []
                for s_mat in pred_graphs:
                    # Convert to directed or undirected based on your flag
                    if self.is_directed:
                        G = nx.from_scipy_sparse_array(s_mat, create_using=nx.DiGraph)
                    else:
                        G = nx.from_scipy_sparse_array(s_mat, create_using=nx.Graph)
                    G.remove_nodes_from(list(nx.isolates(G)))
                    
                    new_nx_graphs.append(G)
                pred_graphs = new_nx_graphs
            else:
                print("INFO: NetworkX objects detected in pickle.")
            
            if extracted_dataset == "Reddit":
                extracted_dataset = "Reddit_B"
            
            _, _, thresholds, target_graphs = load_data(
                extracted_dataset, '', '', '', 'all', 
                use_predicted=False, num_buckets=10, use_test_style=None
            )
            
            if "topoGED" in str(self.data_path):
                target_graphs, _ = modifyGraphIds(target_graphs, thresholds, 10000)
            
            target_graphs_last = [inner_list[-1] for inner_list in target_graphs if inner_list]
            base_graphs = target_graphs_last[:len(target_graphs_last) - len(pred_graphs)]
            target_graphs = target_graphs_last[-len(pred_graphs):]

            # 3. Organize into old and new nodes
            self.sorted_nodes_pred = []
            self.sorted_nodes_true = []
            old_nodes = set()
            
            for graph in base_graphs:
                old_nodes.update(graph.nodes())

            for g_true, g_pred in zip(target_graphs, pred_graphs):
                curr_nodes_true = set(g_true.nodes())
                curr_nodes_pred = set(g_pred.nodes())

                # Logic for True Graphs
                curr_old_nodes_true = old_nodes.intersection(curr_nodes_true)
                curr_new_nodes_true = curr_nodes_true.difference(old_nodes)
                
                # Logic for Predicted Graphs
                curr_old_nodes_pred = old_nodes.intersection(curr_nodes_pred)
                curr_new_nodes_pred = curr_nodes_pred.difference(old_nodes)

                self.sorted_nodes_pred.append({'old_nodes': curr_old_nodes_pred, 'new_nodes': curr_new_nodes_pred})
                self.sorted_nodes_true.append({'old_nodes': curr_old_nodes_true, 'new_nodes': curr_new_nodes_true})

                # Update running knowledge of nodes
                old_nodes.update(curr_nodes_true)
                    
            print(f'Num pred graphs: {len(pred_graphs)}')
            print(f'Num true graphs: {len(target_graphs)}')
            print(f'Num sorted nodes pred: {len(self.sorted_nodes_pred)}')
            print(f'Num sorted nodes true: {len(self.sorted_nodes_true)}')
                    

        except Exception as e:
            print(f"ERROR: Failed to load pickle file at {self.data_path}. Skipping. Error: {e}")
            # Raise a custom flag or exception so the main loop knows to skip this instance
            self.loaded_successfully = False
            return 
        
        self.loaded_successfully = True
            
        
        # Safety for later eval if needed
        if not self.is_directed:
            self.pred_graphs = [g.to_undirected() if g.is_directed() else g for g in pred_graphs]
            self.true_graphs = [g.to_undirected() if g.is_directed() else g for g in target_graphs]
            print("INFO: Evaluation mode set to UNDIRECTED. Converted all DiGraphs.")
        else:
            self.pred_graphs = [g.to_directed() if not g.is_directed() else g for g in pred_graphs]
            self.true_graphs = [g.to_directed() if not g.is_directed() else g for g in target_graphs]
            print("INFO: Evaluation mode set to DIRECTED. Converted all Graphs.")
            
        


            
        
    def _setup_clean_files(self, directories, file_paths):
        """
        Utility function for emptying files if they exist
        We don't want to repeatedly add trials to the csvs/txt files after many runs
        Params:
            directories (list): The directories to create
            file_paths (list): The file paths we plan on storing data in
            
        Returns:
            None
        """
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Delete if they exist
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
                
        

    def evaluate(self, pred_graph, true_graph, sorted_nodes_pred, sorted_nodes_true, graph_num):
        """
        Evaluate the constructed graph across various metrics including:
            - Graphlet kernels
            - Its TopER vector
            - Precision, Recall, F1 score for various subgraphs
            - The nodes we added to the constructed graph
            - The graph's structure
            
        Params:
            pred_graph (nx.Graph/nx.DiGraph): The graph we just finished constructing
            true_graph (nx.Graph/nx.DiGraph): The target graph
            sorted_nodes_pred (dict{str: list}): A dictionary storing lists of ids for 'old_nodes' and 'new_nodes' for pred_graph
            sorted_nodes_true (dict{str: list}): A dictionary storing lists of ids for 'old_nodes' and 'new_nodes' for true_graph
            
        Returns:
            None
        """
        # Setup (clearing files if needed)
        
        if graph_num == 0:
            directories = [
                self.file_visualization_path,
                self.structure_dir,
                self.edge_eval_dir,
                self.kernel_dir
            ]
                        
            target_files = [
                f"{self.edge_eval_dir}/old_nodes_only.csv",
                f"{self.edge_eval_dir}/new_nodes_only.csv",
                f"{self.edge_eval_dir}/on_edges_only.csv",
                f"{self.edge_eval_dir}/all_edges.csv",
                f"{self.edge_eval_dir}/real_edge_eval.csv",
                f"{self.structure_dir}/node_evaluation.csv",
                f"{self.structure_dir}/structure_true.csv",
                f"{self.structure_dir}/structure_pred.csv",
                f"{self.structure_dir}/toper_eval.csv",
                f"{self.kernel_dir}/kernel_pred.csv",
                f"{self.kernel_dir}/kernel_true.csv",
                f"{self.kernel_dir}/kernel_diff.csv",
                f"{self.file_visualization_path}/kl_results_old_nodes.txt",
                f"{self.file_visualization_path}/kl_results_nn.txt",
                f"{self.file_visualization_path}/kl_results_on.txt",
                f"{self.file_visualization_path}/kl_results_overall.txt",
            ]
            
            self._setup_clean_files(directories, target_files)
            self.edgebank = {}  # Used in evaluation later
        
        
        # Evaluate the correctness of nodes that we have chosen (old, new, together)
        results_node_evaluation = self.evaluator.evaluate_node_selection(sorted_nodes_pred, sorted_nodes_true, graph_num=graph_num)
        
        # Evaluate the graph of just old nodes (edge types: oo, oon)
        pred_old_nodes_graph = pred_graph.subgraph(sorted_nodes_pred['old_nodes']).copy()
        true_old_nodes_graph = true_graph.subgraph(sorted_nodes_true['old_nodes']).copy()
        
        old_nodes_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_old_nodes_graph, true_old_nodes_graph, mode="total")
            
        self.evaluator.write_kl_results(path=rf'{self.file_visualization_path}/kl_results_old_nodes.txt', 
                                        value=old_nodes_kl_divergence_results, graph_num=graph_num)
        
        # Evaluate the AUC here
        results_old_nodes_edges = self.evaluator.evaluate_graph_edges(pred_old_nodes_graph, true_old_nodes_graph, is_directed=self.is_directed, graph_num=graph_num, edgebank=self.edgebank)
        
        # Evaluate the graph of just new nodes (edge types: nn)
        pred_nn_graph = pred_graph.subgraph(sorted_nodes_pred['new_nodes']).copy()
        true_nn_graph = true_graph.subgraph(sorted_nodes_true['new_nodes']).copy()

        nn_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_nn_graph, true_nn_graph, mode="total")
            
        self.evaluator.write_kl_results(path=rf'{self.file_visualization_path}/kl_results_nn.txt', 
                                        value=nn_kl_divergence_results, graph_num=graph_num)
        
        # Want to evaluate AUC of these
        results_nn_edges = self.evaluator.evaluate_graph_edges(pred_nn_graph, true_nn_graph, is_directed=self.is_directed, graph_num=graph_num)
        
        # Evaluate the graph of just edge type on
        pred_on_graph = create_on_graph(sorted_nodes_pred["new_nodes"], sorted_nodes_pred["old_nodes"], pred_graph.copy(), is_directed=self.is_directed)
        true_on_graph = create_on_graph(sorted_nodes_true["new_nodes"], sorted_nodes_true["old_nodes"], true_graph.copy(), is_directed=self.is_directed)
        
        on_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_on_graph, true_on_graph, mode="total")
            
        self.evaluator.write_kl_results(path=rf'{self.file_visualization_path}/kl_results_on.txt', 
                                        value=on_kl_divergence_results, graph_num=graph_num)
            
        # Evaluate the AUC here
        results_on_edges = self.evaluator.evaluate_graph_edges(pred_on_graph, true_on_graph, is_directed=self.is_directed, graph_num=graph_num)
        
        # Evaluate the graph of shared nodes (all edge types)
        results_all_edges = self.evaluator.evaluate_graph_edges(pred_graph, true_graph, is_directed=self.is_directed, graph_num=graph_num)
        
        # Evaluate the graphs in terms of structure
        results_true_structure = self.evaluator.evaluateSingleStructure(true_graph, graph_num=graph_num)
        results_pred_structure = self.evaluator.evaluateSingleStructure(pred_graph, graph_num=graph_num)
        
        # Evaluate the graph in terms of kl divergence
        overall_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_graph, true_graph, mode="total")

        self.evaluator.write_kl_results(path=rf'{self.file_visualization_path}/kl_results_overall.txt', 
                                        value=overall_kl_divergence_results, graph_num=graph_num)
        
        # Evaluate the graph in terms of its TopER vector (Degree)
        embedder = EmbedDegree(include_weights=False)

        # Make the TopER embedding
        if pred_graph.number_of_nodes() == 0:
            # If the graph is empty, return a zero vector of size 20
            pred_toper = np.zeros(20)
            print(f"INFO: Pred graph {graph_num} is empty. Using zero vector for TopER.")
        else:
            # Existing logic
            pred_embedding, _, _ = embedder.process_graphs_for_embeddings([pred_graph])
            pred_toper = pred_embedding[0]
            
        true_embedding, _, _ = embedder.process_graphs_for_embeddings([true_graph])
        true_toper = true_embedding[0]

        results_toper_diff = self.evaluator.evaluateTopER(pred_toper, true_toper, graph_num=graph_num)  # Get the difference
            
        # Evaluate the graph in terms of graphlet kernels
        # pred_kernel, true_kernel, distance = self.evaluator.evaluateOrca(pred_graph, true_graph, )
        pred_kernel, true_kernel, distance = np.zeros(11), np.zeros(11), np.zeros(11)  # Placeholder for now (it takes too long)
        
        edge_results_real = self.evaluator.evaluateEdgesNew(pred_graph, true_graph, sorted_nodes_true["old_nodes"], sorted_nodes_pred["old_nodes"])
        
        # Update the edgebank
        for u, v in true_graph.edges():
            self.edgebank.setdefault(u, set()).add(v)
            if not self.is_directed:
                self.edgebank.setdefault(v, set()).add(u)
        
        # Write results
        for data, path in [
            (results_old_nodes_edges, f"{self.edge_eval_dir}/old_nodes_only.csv"),
            (results_nn_edges, f"{self.edge_eval_dir}/new_nodes_only.csv"),
            (results_on_edges, f"{self.edge_eval_dir}/on_edges_only.csv"),
            (results_all_edges, f"{self.edge_eval_dir}/all_edges.csv"),
            (edge_results_real, f"{self.edge_eval_dir}/real_edge_eval.csv"),
            (results_node_evaluation, f"{self.structure_dir}/node_evaluation.csv"),
            (results_true_structure, f"{self.structure_dir}/structure_true.csv"),
            (results_pred_structure, f"{self.structure_dir}/structure_pred.csv"),
            (results_toper_diff, f"{self.structure_dir}/toper_eval.csv"),
            (pred_kernel, f"{self.kernel_dir}/kernel_pred.csv"),
            (true_kernel, f"{self.kernel_dir}/kernel_true.csv"),
            (distance, f"{self.kernel_dir}/kernel_diff.csv"),
        ]:
            write_mode = 'w' if graph_num == 0 else 'a'
            
            pd.DataFrame([data]).to_csv(
                path, 
                mode=write_mode, 
                header=(graph_num == 0), # Only write header on the first snapshot
                index=False
            )
            
            
    def run(self):
        for i, (pred_graph, true_graph, sorted_nodes_pred, sorted_nodes_true) in enumerate(zip(self.pred_graphs, self.true_graphs, self.sorted_nodes_pred, self.sorted_nodes_true)):
            # print(f"Evaluating graph {i}...")
            self.evaluate(pred_graph, true_graph, sorted_nodes_pred, sorted_nodes_true, graph_num=i)
            
        print(self.data_path)
        
    def makeTable(self, df, path, caption, label, table_type=None):
        path_obj = Path(path)
        is_structure = "structure" in str(path).lower()
        
        min_best_cols = [
            'Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 
            'Clustering Coefficient', 'Density', 'Num Triangles', 'Descriptor Norm',
            'Median Extra Nodes', 'Median Missing Nodes', 
            'Median Extra Edges',  'Median Missing Edges', 'Num o-n Predicted', 'Num n-n Predicted', 'New Nodes Predicted'
        ]

        # 1. Column Ordering
        bench_indices = []
        edgebank_idx = -1
        topo_idx = -1
        processed_names = {}

        for i, m in enumerate(df.index):
            m_str = str(m)
            if m_str == "TopoGED Edgebank":
                processed_names[i] = r"\shortstack{TopoGED \\ Edgebank}"
                edgebank_idx = i
            elif m_str == "TopoGED":
                processed_names[i] = "TopoGED"
                topo_idx = i
            elif r"True $\Phi(\widehat{\mathcal{G}}_t)$, True Probs" in m_str:
                processed_names[i] = r"\shortstack{True $\Phi(\widehat{\mathcal{G}}_t)$ \\ True Probs}"
                bench_indices.append(i)
            else:
                name = m_str.replace('TopoGED - ', '').replace('_', r'\_')
                processed_names[i] = name
                bench_indices.append(i)

        final_order = bench_indices + ([edgebank_idx] if edgebank_idx != -1 else []) + ([topo_idx] if topo_idx != -1 else [])
        ordered_df = df.iloc[final_order]
        final_names = [processed_names[idx] for idx in final_order]

        # 2. COLUMN SETUP FIX: Removing the 'ill' and fixing rule spanning
        num_benchmarks = len(bench_indices)
        num_variants = (1 if edgebank_idx != -1 else 0) + (1 if topo_idx != -1 else 0)
        
        # Standardize the setup for clean horizontal lines
        # We use @{\extracolsep{\fill}} once at the start to force the horizontal rules to the target width
        col_setup = "l " + "c " * num_benchmarks + "| " + "c " * num_variants
        latex = [r"\begin{tabular}{ l c c c c c c c}"]

        # 3. Header Row
        latex.append(r"\toprule")
        latex.append(r"\toprule")
        header_row = [r"\textbf{Metric}"] + [fr"\textbf{{{name}}}" for name in final_names]
        latex.append(" & ".join(header_row) + r" \\ \midrule")

        # 4. Data Rows
        for col_name in df.columns:
            if ordered_df[col_name].apply(lambda x: str(x).lower() in ['nan', 'none', '']).all():
                continue

            row_cells = [col_name.replace('_', r'\_')]
            
            def get_num(x):
                try: return float(x)
                except: return np.nan

            col_data = ordered_df[col_name].apply(get_num).dropna()
            
            # Determine best and second best for bold/underline
            if not col_data.empty:
                if col_name in min_best_cols:
                    sorted_vals = np.sort(col_data.abs().unique())
                else:
                    sorted_vals = np.sort(col_data.unique())[::-1]
                best_val = sorted_vals[0] if len(sorted_vals) > 0 else None
                second_best = sorted_vals[1] if len(sorted_vals) > 1 else None
            else:
                best_val, second_best = None, None

            for _, row in ordered_df.iterrows():
                val_raw = str(row[col_name])
                val_num = get_num(val_raw)

                if val_raw == "CHAL":
                    row_cells.append(r"\text{CHAL}")
                    continue
                if np.isnan(val_num):
                    row_cells.append("$OOM$")
                    continue
                
                check_val = abs(val_num) if col_name in min_best_cols else val_num
                display_num = str(val_raw)

                if best_val is not None and np.isclose(check_val, best_val, atol=1e-5) and display_num != "0.00":
                    row_cells.append(r"$\mathbf{" + display_num + "}$")
                elif second_best is not None and np.isclose(check_val, second_best, atol=1e-5) and display_num != "0.00":
                    row_cells.append(r"$\underline{" + display_num + "}$")
                else:
                    row_cells.append(f"${display_num}$")
                    
            latex.append(" & ".join(row_cells) + r" \\")

            # Insert the horizontal divider for the structure table
            if is_structure and col_name == "Num Triangles":
                latex.append(r"\midrule")

        latex.append(r"\bottomrule")
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        
        latex.append(r"\begin{center}")
        latex.append(r"\vspace{-2pt}") # Tighten space to the table slightly
        
        if table_type == "nodes":
            latex.append(r"{\small For all metrics higher is better. \par}")
        elif table_type == "structure":
            latex.append(r"{\small For all metrics closer to 0 is better. \par}")
        elif table_type == "edges":
            latex.append(r"{\small For \texttt{Num o-n Predicted} and \texttt{Num n-n Predicted} closer to 0 is better; for others, higher is better. \par}")
            
        # The legend (bold/underline) - \par is cleaner than \\ here
        latex.append(r"{\small \textbf{Bold} indicates best, the second-best is \underline{underlined}. \par}")
        latex.append(r"\end{center}")

        
        # Save
        txt_path = path_obj.with_suffix('.txt')
        os.makedirs(txt_path.parent, exist_ok=True)
        with open(txt_path, 'w') as f:
            f.write("\n".join(latex))
            
            
    def collect_data_for_heatmap(self, df, dataset, old_only, threshold=None, lr=None):      
        valid_metrics = [
            'Assortativity Coefficient', 'Clustering Coefficient', 'Degree Centrality', 'Density', 
            'Num Triangles', 'Num n-n Predicted', 'Num o-n Predicted',
            'oo-bank F1', 'oo-bank Precision', 'oo-bank Recall', 'oo-nobank F1', 'oo-nobank Precision', 'oo-nobank Recall']
        for model_name in df.index:
            if "edgebank" in str(model_name).lower():
                continue
            for metric_name in df.columns:
                if metric_name in valid_metrics:
                    val = df.at[model_name, metric_name]
                    
                    try:
                        score_val = float(val) if str(val).lower() != "chal" else np.nan
                    except (ValueError, TypeError):
                        score_val = np.nan
                    
                    self.all_results.append({
                        'Dataset': dataset,
                        'Model': model_name,
                        'Metric': metric_name,
                        'Threshold': threshold,
                        'lr': lr,
                        'Score': score_val,
                        'OldOnly': old_only
                    })
                    
    
    def create_heatmaps(self, file_path, curr_models, old_only, target_threshold=None, lr=None):
        # 1. Load base data and filter for current view/threshold
        df_base = pd.DataFrame(self.all_results)
        # df_filtered = df_base[
        #     (df_base['OldOnly'] == old_only) & 
        #     (df_base['Threshold'] == target_threshold) &
        #     (df_base['lr'] == lr)
        # ].copy()        
        print(df_base.head())
        
        df_filtered = df_base.copy()

        if df_filtered.empty:
            print(f"Skipping: No data matches Threshold {target_threshold} and OldOnly {old_only}")
            return

        # Primary methods to iterate through
        methods = ['oobankchanges']
        benchmarks = ['HTGN', 'ROLAND', 'VGRNN', 'TGCN', 'GCLSTM', 'EvolveGCN']
        
        methods = ['TopoGED'] 
        benchmarks = ['HTGN', 'ROLAND', 'VGRNN', 'TGCN', 'GCLSTM', 'EvolveGCN']
        
        for method in methods:
            curr_file_path = os.path.join(file_path, f"{method.replace('.', '')}")
            os.makedirs(curr_file_path, exist_ok=True)

            # --- NEW FILTER: Explicitly exclude Edgebank from the heatmap data ---
            # We only keep benchmarks and the ONE specific method trial
            models_to_keep = benchmarks + [method]
            
            # This ensures that even if 'TopoGED Edgebank' is in self.all_results,
            # it is dropped before the heatmap pivot.
            curr_df = df_filtered[df_filtered['Model'].isin(models_to_keep)].copy()
            
            # Double check: remove anything with "Edgebank" in the name just in case
            curr_df = curr_df[~curr_df['Model'].str.contains('Edgebank', case=False, na=False)]
            
            if curr_df.empty:
                continue
                        
            node_view = "OldNodes" if old_only else "AllNodes"
            # suffix = f"{node_view}_threshold{target_threshold}_lr{lr}"
            suffix = f"{node_view}"
            
            # Metrics where a lower value is better
            min_best_cols = [
                'Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 
                'Assortativity Coefficient', 'Clustering Coefficient', 'Density', 
                'Num Triangles', 'Descriptor Norm', 
                'Median Extra Nodes', 'Median Missing Nodes', 
                'Median Extra Edges', 'Median Missing Edges', 'Num o-n Predicted', 'Num n-n Predicted', 'New Nodes Predicted'
            ]

            # Adjust score so that the "highest" value is always the winner
            def adjust_score(row):
                if row['Metric'] in min_best_cols:
                    return -abs(row['Score']) 
                return row['Score']

            curr_df['Adj_Score'] = curr_df.apply(adjust_score, axis=1)

            # ---------------------------------------------------------
            # HEATMAP 1: MODEL DOMINANCE (WITH SPECIFIC TIE RULES)
            # ---------------------------------------------------------
            
            # 1. Determine winners using the requested tolerance
            curr_df['Max_Score'] = curr_df.groupby(['Dataset', 'Metric'])['Adj_Score'].transform('max')
            is_winner = np.isclose(curr_df['Adj_Score'], curr_df['Max_Score'], atol=0)
            
            # Group winners into lists to check for ties
            winning_groups = curr_df[is_winner].groupby(['Dataset', 'Metric'])['Model'].apply(list).reset_index(name='Winners')
            
            def determine_winner_label(winners_list):
                if len(winners_list) == 1:
                    return winners_list[0]
                
                # Tie logic: method vs benchmarks
                if method in winners_list:
                    return method # Will be Light Pink
                else:
                    return winners_list[0]

            winning_groups['Final_Winner'] = winning_groups['Winners'].apply(determine_winner_label)
            
            # Pivot for dominance heatmap
            pivot_dom = winning_groups.pivot(index='Dataset', columns='Metric', values='Final_Winner')
            pivot_dom = pivot_dom[sorted(pivot_dom.columns)]
            # pivot_dom.to_csv(os.path.join(curr_file_path, f"filtered_results_threshold{target_threshold}_{'oldonly' if old_only else 'allnodes'}.csv"))
            pivot_dom.to_csv(os.path.join(curr_file_path, f"filtered_results.csv"))

            # ---------------------------------------------------------
            # NEW: HEATMAP 2: BEST VALUES (RAW SCORES)
            # ---------------------------------------------------------
            
            # Capture the winning scores and revert signs for 'min is best' columns
            winning_values = curr_df.groupby(['Dataset', 'Metric'])['Max_Score'].max().reset_index(name='Best_Value')

            def revert_score(row):
                if row['Metric'] in min_best_cols:
                    return abs(row['Best_Value'])
                return row['Best_Value']

            winning_values['Final_Value'] = winning_values.apply(revert_score, axis=1)

            # Pivot for numerical values
            pivot_vals = winning_values.pivot(index='Dataset', columns='Metric', values='Final_Value')
            pivot_vals = pivot_vals[sorted(pivot_vals.columns)]
            
            # Save the numerical values CSV
            # val_csv_name = f"best_values_threshold{target_threshold}_lr{lr}_{'oldonly' if old_only else 'allnodes'}.csv"
            val_csv_name = f"best_values.csv"
            pivot_vals.to_csv(os.path.join(curr_file_path, val_csv_name))

            # ---------------------------------------------------------
            # 2. Define the palette and labels for Plotting
            # ---------------------------------------------------------
            
            heatmap_labels = benchmarks + [method, 'Method Tie', 'Benchmark Tie']
            
            # Color Construction
            benchmark_colors = sns.color_palette("Blues_d", len(benchmarks))
            method_color = '#F45DEB'  
            pink_color = (1.0, 0.75, 0.8)   # Light Pink for Method Tie
            gray_color = (0.8, 0.8, 0.8)    # Gray for Benchmark Tie
            
            custom_colors = list(benchmark_colors) + [method_color, pink_color, gray_color]
            cmap_dom = sns.color_palette(custom_colors)

            # Map labels to indices for plotting
            label_to_idx = {lbl: i for i, lbl in enumerate(heatmap_labels)}
            idx_dom = pivot_dom.replace(label_to_idx).infer_objects(copy=False).astype(float)

            # ---------------------------------------------------------
            # 3. Plotting
            # ---------------------------------------------------------
            plt.figure(figsize=(24, 10))
            ax1 = sns.heatmap(idx_dom, cmap=cmap_dom, linewidths=.8, linecolor='white', cbar_kws={"shrink": 0.8})
            
            # Formatting the colorbar
            colorbar = ax1.collections[0].colorbar
            r = colorbar.vmax - colorbar.vmin
            n = len(heatmap_labels)
            colorbar.set_ticks([colorbar.vmin + r/(2*n) + i*r/n for i in range(n)])
            colorbar.set_ticklabels(heatmap_labels)
            plt.setp(colorbar.ax.get_yticklabels(), weight='bold')

            plt.xticks(rotation=45, ha='right', fontsize=12, weight='bold')
            plt.yticks(fontsize=12, weight='bold')
            plt.title(f"Model Dominance: {method} vs Benchmarks ({node_view})", fontsize=16, weight='bold')
            
            plt.savefig(os.path.join(curr_file_path, f"dominance_heatmap_{suffix}.png"), bbox_inches='tight', dpi=300)
            plt.close()
            # ---------------------------------------------------------
            # FLATTENED HEATMAPS (For Appendix)
            # ---------------------------------------------------------
            # datasets = [
            #     'CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion', 'networkaeternity', 'networkaragon', 'networkbancor', 'networkcentra', 
            #     'networkcoindash', 'networkiconomi', 'networkcindicator', 'networkdgd', 'Reddit_B', 'tgbl-wiki',
            # ]
            
            # for dataset in datasets:
            #     new_curr_df = curr_df.copy()
            #     new_curr_df = new_curr_df[new_curr_df['Dataset'] == dataset]
            #     # --- FIX 2: Use new_curr_df throughout to avoid KeyErrors and Data Mismatches ---
            #     def get_relative(group):
            #         best, worst = group.max(), group.min()
            #         return (group - worst) / (best - worst) if best != worst else pd.Series(1.0, index=group.index)

            #     new_curr_df['Relative_Score'] = new_curr_df.groupby('Task')['Adj_Score'].transform(get_relative)
                
            #     # --- FIX 3: Use pivot_table to handle duplicate entries safely ---
            #     pivot_rel = new_curr_df.pivot_table(index='Task', columns='Model', values='Relative_Score', aggfunc='mean')
            #     pivot_rel = pivot_rel.reindex(columns=models_to_keep)

            #     plt.figure(figsize=(12, 50))
            #     sns.heatmap(pivot_rel, cmap="YlGnBu", cbar_kws={'label': 'Relative Rank'})
            #     plt.savefig(os.path.join(curr_file_path, f"{dataset}_relative_heatmap_appendix_{suffix}.png"), bbox_inches='tight', dpi=150)

            #     # Z-Score
            #     group_stats = new_curr_df.groupby('Task')['Adj_Score']
            #     new_curr_df['Z_Score'] = (new_curr_df['Adj_Score'] - group_stats.transform('mean')) / group_stats.transform('std')
                
            #     pivot_z = new_curr_df.pivot_table(index='Task', columns='Model', values='Z_Score', aggfunc='mean').fillna(0)
            #     pivot_z = pivot_z.reindex(columns=models_to_keep)

            #     plt.figure(figsize=(12, 50))
            #     sns.heatmap(pivot_z, cmap="RdBu_r", center=0)
            #     plt.savefig(os.path.join(curr_file_path, f"{dataset}_zscore_heatmap_appendix_{suffix}.png"), bbox_inches='tight', dpi=150)
            #     plt.close('all')
    

    def construct_ablation_tables(self, datasets):
        ablation_map = {
            'Ablation 7': 'False Probs',
            'Ablation 8': r'False $\Phi(\widehat{\mathcal{G}}_t)$',
            'Ablation 9': r'True $\Phi(\widehat{\mathcal{G}}_t)$, True Probs'
        }


        ablation_modes = [7, 8, 9]
        lr = 0.001
        for dataset in datasets:
            data = []
            for ablation in ablation_modes:
                data_path = f'data/output/ablation/constructed_graphs/{dataset}_topoGED_embedding_mlpEncodingConcat_embeddingTypeGCN_lr0.001_5back_oobankchanges_zeros_sampling_predvalsTrue_edgebank_default_VectorTypeV-EWMA_ablation{ablation}/GCN_constructed_graphs_{dataset}.pkl'
                prefix = f'GraphGeneration/scripts/results/{dataset}/ablations/final/{dataset}_lr{lr}_ablation{ablation}_tmp_V-EWMA/'

                if os.path.exists(data_path) and not os.path.exists(prefix):
                    try:
                        evaluator = EvaluateGraphs(args=None, prefix=prefix, data_path=data_path)
                        evaluator.run()
                    except Exception as e:
                        print(f"Skipping Eval for {dataset} Ablation {ablation}: {e}")
                else:
                    print(f'Skipping eval for {data_path}')
                    
                data.append(prefix)
            
            data.append(f"data/output/results/TopoGED_{dataset}_zeros_0.001_default_V-EWMA_sampling_tmp/")  # The true method
            
            # Setup rows
            rows = [f'Ablation {d}' for d in ablation_modes] + ['TopoGED']
            node_columns = ['Precision Nodes', 'Recall Nodes', 'F1 Nodes', 'Precision Old Nodes', 'Recall Old Nodes', 'F1 Old Nodes', 'New Nodes Predicted']
            structure_columns = ['Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 'Clustering Coefficient',
                    'Density', 'Num Triangles', 'Descriptor Norm', 'Median Extra Nodes', 'Median Missing Nodes', 'Median Extra Edges', 'Median Missing Edges']
            edge_columns = ['oo-bank Precision', "oo-bank Recall", "oo-bank F1", 
                            "oo-nobank Precision", "oo-nobank Recall", "oo-nobank F1",
                            "Num o-n Predicted", "Num n-n Predicted",
                            "Edge Precision", "Edge Recall", "Edge F1"]
        
            node_df = pd.DataFrame(np.nan, index=rows, columns=node_columns)
            structure_df = pd.DataFrame(np.nan, index=rows, columns=structure_columns)
            edge_df = pd.DataFrame(np.nan, index=rows, columns=edge_columns)
            num_eval_graphs = num_test_graphs[dataset]
            
            # Compute Metrics for each dataset
            print(f'Generating ablation table for dataset: {dataset}')
            for directory in data:
                print(directory)
                
                dir_str = str(directory).lower()
                curr_ablation = re.search(r'ablation(\d+)', dir_str)
                
                if curr_ablation:
                    model = f'Ablation {curr_ablation.group(1)}'
                elif "sampling" in dir_str:
                    model = 'TopoGED'
                else:
                    continue
                    
                
                if not os.path.exists(directory):
                    print(f"Data from: {directory} does not exist")
                    structure_df.at[model, 'Avg Node Degree'] = np.nan
                    structure_df.at[model, 'Unique Degree Count'] = np.nan
                    structure_df.at[model, 'Degree Centrality'] = np.nan
                    structure_df.at[model, 'Assortativity Coefficient'] = np.nan
                    structure_df.at[model, 'Clustering Coefficient'] = np.nan
                    structure_df.at[model, 'Density'] = np.nan
                    structure_df.at[model, 'Num Triangles'] = np.nan
                    
                    structure_df.at[model, 'Median Extra Nodes'] = np.nan
                    
                    structure_df.at[model, 'Median Missing Nodes'] = np.nan
                    
                    structure_df.at[model, 'Median Extra Edges'] = np.nan
                    
                    structure_df.at[model, 'Median Missing Edges'] = np.nan
                    structure_df.at[model, 'Descriptor Norm'] = np.nan
                    
                    node_df.at[model, 'Precision Old Nodes'] = np.nan
                    node_df.at[model, 'Recall Old Nodes'] = np.nan
                    node_df.at[model, 'F1 Old Nodes'] = np.nan      
                    
                    node_df.at[model, 'Precision Nodes'] = np.nan
                    node_df.at[model, 'Recall Nodes'] = np.nan
                    node_df.at[model, 'F1 Nodes'] = np.nan        
                    
                    
                    edge_df.at[model, 'Edge Precision'] = np.nan
                    edge_df.at[model, 'Edge Recall'] = np.nan
                    edge_df.at[model, 'Edge F1'] = np.nan
                    
                    node_df.at[model, 'New Nodes Predicted'] = np.nan
                    
                    edge_df.at[model, 'oo-bank Precision'] = np.nan
                    edge_df.at[model, 'oo-bank Recall'] = np.nan
                    edge_df.at[model, 'oo-bank F1'] = np.nan       
                            
                    edge_df.at[model, 'oo-nobank Precision'] = np.nan
                    # Handles typo
                    edge_df.at[model, 'oo-nobank Recall'] = np.nan
                    edge_df.at[model, 'oo-nobank F1'] = np.nan
                
                    edge_df.at[model, 'Num o-n Predicted'] = np.nan
                    edge_df.at[model, 'Num n-n Predicted'] = np.nan
                        
                    continue    
                
                dir_path = Path(directory)
                # Take the tail since we only eval test graphs here
                files = {
                    'node_eval': str(dir_path / 'structure' / 'node_evaluation.csv'),
                    'true_structure': str(dir_path / 'structure' / 'structure_true.csv'),
                    'pred_structure': str(dir_path / 'structure' / 'structure_pred.csv'),
                    'toper_eval': str(dir_path / 'structure' / 'toper_eval.csv'),
                    'all_edges': str(dir_path / 'edge_evaluation' / 'all_edges.csv'),
                    'old_only': str(dir_path / 'edge_evaluation' / 'old_nodes_only.csv'),
                    'on_only': str(dir_path / 'edge_evaluation' / 'on_edges_only.csv'),
                    'new_only': str(dir_path / 'edge_evaluation' / 'new_nodes_only.csv')
                }

                # Load each file and check length
                dfs = {}
                for key, path in files.items():
                    temp_df = pd.read_csv(path)
                    if len(temp_df) < num_eval_graphs:
                        print(f"ERROR: {path} has {len(temp_df)} rows, expected {num_eval_graphs}")
                    dfs[key] = temp_df.tail(num_eval_graphs)

                # Map back to your variables
                node_eval = dfs['node_eval']
                true_structure = dfs['true_structure']
                pred_structure = dfs['pred_structure']
                toper_eval = dfs['toper_eval']
                all_edges = dfs['all_edges']
                old_only = dfs['old_only']
                on_only = dfs['on_only']
                new_only = dfs['new_only']

                # Add metrics
                def get_mean(df, col):
                    return df[col].replace([np.inf, -np.inf], np.nan).mean()

                # Calculate Percent Error: (Pred - True) / True
                metrics_map = {
                    'Avg Node Degree': 'Average Node Degree',
                    'Unique Degree Count': 'Unique Degree Count',
                    'Degree Centrality': 'Degree Centrality',
                    'Assortativity Coefficient': 'Assortivity Coefficient',
                    'Clustering Coefficient': 'Clustering Coefficient',
                    'Density': 'Density',
                    'Num Triangles': 'Number of Triangles'
                }

                for df_col, csv_col in metrics_map.items():
                    t_vals = true_structure[csv_col]
                    p_vals = pred_structure[csv_col]
                    
                    denom = t_vals.copy()
                    denom[denom == 0] = 1.0
                    
                    relative_errors = (p_vals - t_vals) / denom
                    
                    structure_df.at[model, df_col] = relative_errors.mean()
                    
                    
                extra_node_slice = toper_eval[toper_eval['node_diff_10'] > 0]['node_diff_10']
                structure_df.at[model, 'Median Extra Nodes'] = extra_node_slice.median() if not extra_node_slice.empty else 0
                
                missing_node_slice = toper_eval[toper_eval['node_diff_10'] < 0]['node_diff_10']
                structure_df.at[model, 'Median Missing Nodes'] = abs(missing_node_slice.median()) if not missing_node_slice.empty else 0
                
                extra_edge_slice = toper_eval[toper_eval['edge_diff_10'] > 0]['edge_diff_10']
                structure_df.at[model, 'Median Extra Edges'] = extra_edge_slice.median() if not extra_edge_slice.empty else 0
                
                missing_edge_slice = toper_eval[toper_eval['edge_diff_10'] < 0]['edge_diff_10']
                structure_df.at[model, 'Median Missing Edges'] = abs(missing_edge_slice.median()) if not missing_edge_slice.empty else 0
                structure_df.at[model, 'Descriptor Norm'] = toper_eval['l2_norm'].mean()
                
                node_df.at[model, 'Precision Old Nodes'] = node_eval['Precision_Old'].mean()
                node_df.at[model, 'Recall Old Nodes'] = node_eval['Recall_Old'].mean()
                node_df.at[model, 'F1 Old Nodes'] = node_eval['F1_Old'].mean()        
                
                node_df.at[model, 'Precision Nodes'] = node_eval['Precision_All'].mean()
                node_df.at[model, 'Recall Nodes'] = node_eval['Recall_All'].mean()
                node_df.at[model, 'F1 Nodes'] = node_eval['F1_All'].mean()          
                
                
                edge_df.at[model, 'Edge Precision'] = all_edges['Precision'].mean()
                edge_df.at[model, 'Edge Recall'] = all_edges['Recall'].mean()
                edge_df.at[model, 'Edge F1'] = all_edges['F1'].mean() 
                
                denom = node_eval['Num_New_True'].copy().astype(float)
                denom[denom == 0] = 1.0
                percent_diff_new_nodes = (node_eval['Num_New_Predicted'] - node_eval['Num_New_True']) / denom
                node_df.at[model, 'New Nodes Predicted'] = percent_diff_new_nodes.mean()
                
                edge_df.at[model, 'oo-bank Precision'] = old_only['Precision_bank'].mean()
                edge_df.at[model, 'oo-bank Recall'] = old_only['Recall_bank'].mean()
                edge_df.at[model, 'oo-bank F1'] = old_only['F1_bank'].mean()          
                        
                edge_df.at[model, 'oo-nobank Precision'] = old_only['Precision_nobank'].mean()
                # Handles typo
                recall_col_nobank = 'Recall_nobank' if 'Recall_nobank' in old_only.columns else 'Recall_bank_nobank'
                edge_df.at[model, 'oo-nobank Recall'] = old_only[recall_col_nobank].mean()
                edge_df.at[model, 'oo-nobank F1'] = old_only['F1_nobank'].mean()
            
                # Positive means model under predicted, negative means model over predicted
                num_true_on = on_only['TP'] + on_only['FN']
                num_pred_on = on_only['TP'] + on_only['FP']

                denom_on = num_true_on.copy()
                denom_on[denom_on == 0] = 1.0 # Force denominator to 1 if true is 0

                percent_diff_on = (num_pred_on - num_true_on) / denom_on
                edge_df.at[model, 'Num o-n Predicted'] = percent_diff_on.mean()

                # --- n-n Edges ---
                num_true_nn = new_only['TP'] + new_only['FN']
                num_pred_nn = new_only['TP'] + new_only['FP']

                denom_nn = num_true_nn.copy()
                denom_nn[denom_nn == 0] = 1.0

                percent_diff_nn = (num_pred_nn - num_true_nn) / denom_nn
                edge_df.at[model, 'Num n-n Predicted'] = percent_diff_nn.mean()

            
            int_cols = [
                'Median Extra Nodes ', 'Median Missing Nodes', 
                'Median Extra Edges',  'Median Missing Edges',
            ]

            # Apply "CHAL" string to any edge types where there weren't enough nodes to evaluate fairly
            threshold = -1  # Can modify
            for model in rows:
                # Check F1 Old Nodes for bank/nobank edge metrics
                f1_old = float(node_df.at[model, 'F1 Old Nodes'])
                if pd.notnull(f1_old) and f1_old < threshold:
                    cols_to_chal = [
                        'oo-bank Precision', 'oo-bank Recall', 'oo-bank F1',
                        'oo-nobank Precision', 'oo-nobank Recall', 'oo-nobank F1'
                    ]
                    for c in cols_to_chal:
                        edge_df.at[model, c] = "CHAL"

                # Check F1 Nodes for overall edge metrics
                f1_all = float(node_df.at[model, 'F1 Nodes'])
                if pd.notnull(f1_all) and f1_all < threshold:
                    cols_to_chal = ['Edge Precision', 'Edge Recall', 'Edge F1']
                    for c in cols_to_chal:
                        edge_df.at[model, c] = "CHAL"
            
            # 2. Apply formatting to each DataFrame
            for df in [node_df, structure_df, edge_df]:
                for col in df.columns:
                    def format_cell(x):
                        if x == "CHAL": return "CHAL" # Pass through
                        if pd.isnull(x): return np.nan
                        try:
                            val = float(x)
                            if col in int_cols:
                                return f"{int(round(val))}"
                            return f"{val:.2f}"
                        except:
                            return str(x)
                    df[col] = df[col].apply(format_cell)

            file_path = f'data/output/figures/ablation_tables_vewma/'
            os.makedirs(file_path, exist_ok=True)
            structure_output_path = os.path.join(file_path, f"{dataset}_structure_table.png")
            edge_output_path = os.path.join(file_path, f"{dataset}_edge_table.png")
            node_output_path = os.path.join(file_path, f"{dataset}_node_table.png")
                
            
            node_df.rename(index=ablation_map, inplace=True)
            structure_df.rename(index=ablation_map, inplace=True)
            edge_df.rename(index=ablation_map, inplace=True)
            
                
            self.makeTable(node_df, node_output_path, caption=f"Node Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_node_evaluation", table_type="nodes")
            self.makeTable(structure_df, structure_output_path, caption=f"Structure Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_structure_evaluation", table_type="structure")
            self.makeTable(edge_df, edge_output_path, caption=f"Edge Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_edge_evaluation", table_type="edges")



                        
    def make_heatmaps(self):
        # Base for displaying  (REMOVE SETTINGS THAT ARENT AS GOOD FROM LR)
        # models = ['oobankchanges_lr0.01', 'oobankchanges_lr0.001', 'htgn_lr{lr}_threshold{threshold}', 'ROLAND_lr{lr}_threshold{threshold}', 
        #           'VGAE_lr{lr}_threshold{threshold}', 'TGCN_lr{lr}_threshold{threshold}', 'GCLSTM_lr{lr}_threshold{threshold}', 'EvolveGCN_lr{lr}_threshold{threshold}']
        
        # for threshold in [0.8, 0.85]:
        #     for lr in [0.001, 0.01]:
        #         models = [f'oobankchanges_lr{lr}', f'htgn_lr{lr}_threshold{threshold}', f'ROLAND_lr{lr}_threshold{threshold}', 
        #           f'VGAE_lr{lr}_threshold{threshold}', f'TGCN_lr{lr}_threshold{threshold}', f'GCLSTM_lr{lr}_threshold{threshold}', f'EvolveGCN_lr{lr}_threshold{threshold}']
        
        #         # self.create_heatmaps('data/output/figures/evaluation_heatmaps_OldOnly_new/', models, old_only=True, target_threshold=threshold)
        #         self.create_heatmaps(f'data/output/figures/evaluation_heatmaps_AllNodes_new_lr{lr}_threshold{threshold}/', models, old_only=False, target_threshold=threshold, lr=lr)
        self.create_heatmaps('data/output/figures/evaluation_heatmaps_Best/', [], old_only=False, target_threshold=None)
            
            
    def sensitivity_analysis_days(self, datasets):
        days_values = [1, 3, 5, 7, 14, 30]
        lr = 0.001
        for dataset in datasets:
            data = []
            for days_back in days_values:
                data_path = f'data/output/sensitivity_analysis/days/constructed_graphs/{dataset}_topoGED_embedding_mlpEncodingConcat_embeddingTypeGCN_lr0.001_5back_oobankchanges_zeros_sampling_predvalsTrue_edgebank_default_VectorTypeV-EWMA_len10_days{days_back}back/GCN_constructed_graphs_{dataset}.pkl'
                prefix = f'GraphGeneration/scripts/results/{dataset}/sensitivity_analysis/days/{dataset}_lr{lr}_{days_back}back_tmp_V-EWMA/'

                # Base method
                if days_back == 5:
                    data_path = Path(f"data/output/constructed_graphs/{dataset}_topoGED_embedding_mlpEncodingConcat_embeddingTypeGCN_lr0.001_5back_oobankchanges_zeros_sampling_predvalsTrue_tmp_edgebank_default_VectorTypeV-EWMA/GCN_constructed_graphs_{dataset}.pkl")

                if os.path.exists(data_path) and not os.path.exists(prefix):
                    try:
                        evaluator = EvaluateGraphs(args=None, prefix=prefix, data_path=data_path)
                        evaluator.run()
                    except Exception as e:
                        print(f"Skipping Eval for {dataset} {days_back}back: {e}")
                else:
                    print(f'Skipping eval for {data_path}')
                    
                data.append(prefix)
            
            # Setup rows
            rows = [f'TopoGED - {d} days' for d in days_values]
            node_columns = ['Precision Nodes', 'Recall Nodes', 'F1 Nodes', 'Precision Old Nodes', 'Recall Old Nodes', 'F1 Old Nodes', 'New Nodes Predicted']
            structure_columns = ['Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 'Clustering Coefficient',
                    'Density', 'Num Triangles', 'Descriptor Norm', 'Median Extra Nodes', 'Median Missing Nodes', 'Median Extra Edges', 'Median Missing Edges']
            edge_columns = ['oo-bank Precision', "oo-bank Recall", "oo-bank F1", 
                            "oo-nobank Precision", "oo-nobank Recall", "oo-nobank F1",
                            "Num o-n Predicted", "Num n-n Predicted",
                            "Edge Precision", "Edge Recall", "Edge F1"]
        
            node_df = pd.DataFrame(np.nan, index=rows, columns=node_columns)
            structure_df = pd.DataFrame(np.nan, index=rows, columns=structure_columns)
            edge_df = pd.DataFrame(np.nan, index=rows, columns=edge_columns)
            num_eval_graphs = num_test_graphs[dataset]
            
            # Compute Metrics for each dataset
            print(f'Generating sensitivity analysis days table for dataset: {dataset}')
            for directory in data:
                print(directory)
                
                dir_str = str(directory)
                curr_days_back = re.search(r'(\d+)back', dir_str)
                if curr_days_back:
                    # MATCH THIS EXACTLY to the 'rows' definition below
                    model = f'TopoGED - {curr_days_back.group(1)} days' 
                else:
                    continue
                    
                
                if not os.path.exists(directory):
                    print(f"Data from: {directory} does not exist")
                    structure_df.at[model, 'Avg Node Degree'] = np.nan
                    structure_df.at[model, 'Unique Degree Count'] = np.nan
                    structure_df.at[model, 'Degree Centrality'] = np.nan
                    structure_df.at[model, 'Assortativity Coefficient'] = np.nan
                    structure_df.at[model, 'Clustering Coefficient'] = np.nan
                    structure_df.at[model, 'Density'] = np.nan
                    structure_df.at[model, 'Num Triangles'] = np.nan
                    
                    structure_df.at[model, 'Median Extra Nodes'] = np.nan
                    
                    structure_df.at[model, 'Median Missing Nodes'] = np.nan
                    
                    structure_df.at[model, 'Median Extra Edges'] = np.nan
                    
                    structure_df.at[model, 'Median Missing Edges'] = np.nan
                    structure_df.at[model, 'Descriptor Norm'] = np.nan
                    
                    node_df.at[model, 'Precision Old Nodes'] = np.nan
                    node_df.at[model, 'Recall Old Nodes'] = np.nan
                    node_df.at[model, 'F1 Old Nodes'] = np.nan      
                    
                    node_df.at[model, 'Precision Nodes'] = np.nan
                    node_df.at[model, 'Recall Nodes'] = np.nan
                    node_df.at[model, 'F1 Nodes'] = np.nan        
                    
                    
                    edge_df.at[model, 'Edge Precision'] = np.nan
                    edge_df.at[model, 'Edge Recall'] = np.nan
                    edge_df.at[model, 'Edge F1'] = np.nan
                    
                    node_df.at[model, 'New Nodes Predicted'] = np.nan
                    
                    edge_df.at[model, 'oo-bank Precision'] = np.nan
                    edge_df.at[model, 'oo-bank Recall'] = np.nan
                    edge_df.at[model, 'oo-bank F1'] = np.nan       
                            
                    edge_df.at[model, 'oo-nobank Precision'] = np.nan
                    # Handles typo
                    edge_df.at[model, 'oo-nobank Recall'] = np.nan
                    edge_df.at[model, 'oo-nobank F1'] = np.nan
                
                    edge_df.at[model, 'Num o-n Predicted'] = np.nan
                    edge_df.at[model, 'Num n-n Predicted'] = np.nan
                        
                    continue    
                
                dir_path = Path(directory)
                # Take the tail since we only eval test graphs here
                files = {
                    'node_eval': str(dir_path / 'structure' / 'node_evaluation.csv'),
                    'true_structure': str(dir_path / 'structure' / 'structure_true.csv'),
                    'pred_structure': str(dir_path / 'structure' / 'structure_pred.csv'),
                    'toper_eval': str(dir_path / 'structure' / 'toper_eval.csv'),
                    'all_edges': str(dir_path / 'edge_evaluation' / 'all_edges.csv'),
                    'old_only': str(dir_path / 'edge_evaluation' / 'old_nodes_only.csv'),
                    'on_only': str(dir_path / 'edge_evaluation' / 'on_edges_only.csv'),
                    'new_only': str(dir_path / 'edge_evaluation' / 'new_nodes_only.csv')
                }

                # Load each file and check length
                dfs = {}
                for key, path in files.items():
                    temp_df = pd.read_csv(path)
                    if len(temp_df) < num_eval_graphs:
                        print(f"ERROR: {path} has {len(temp_df)} rows, expected {num_eval_graphs}")
                    dfs[key] = temp_df.tail(num_eval_graphs)

                # Map back to your variables
                node_eval = dfs['node_eval']
                true_structure = dfs['true_structure']
                pred_structure = dfs['pred_structure']
                toper_eval = dfs['toper_eval']
                all_edges = dfs['all_edges']
                old_only = dfs['old_only']
                on_only = dfs['on_only']
                new_only = dfs['new_only']

                # Add metrics
                def get_mean(df, col):
                    return df[col].replace([np.inf, -np.inf], np.nan).mean()

                # Calculate Percent Error: (Pred - True) / True
                metrics_map = {
                    'Avg Node Degree': 'Average Node Degree',
                    'Unique Degree Count': 'Unique Degree Count',
                    'Degree Centrality': 'Degree Centrality',
                    'Assortativity Coefficient': 'Assortivity Coefficient',
                    'Clustering Coefficient': 'Clustering Coefficient',
                    'Density': 'Density',
                    'Num Triangles': 'Number of Triangles'
                }

                for df_col, csv_col in metrics_map.items():
                    t_vals = true_structure[csv_col]
                    p_vals = pred_structure[csv_col]
                    
                    denom = t_vals.copy()
                    denom[denom == 0] = 1.0
                    
                    relative_errors = (p_vals - t_vals) / denom
                
                    structure_df.at[model, df_col] = relative_errors.mean()
                    
                    
                extra_node_slice = toper_eval[toper_eval['node_diff_10'] > 0]['node_diff_10']
                structure_df.at[model, 'Median Extra Nodes'] = extra_node_slice.median() if not extra_node_slice.empty else 0
                
                missing_node_slice = toper_eval[toper_eval['node_diff_10'] < 0]['node_diff_10']
                structure_df.at[model, 'Median Missing Nodes'] = abs(missing_node_slice.median()) if not missing_node_slice.empty else 0
                
                extra_edge_slice = toper_eval[toper_eval['edge_diff_10'] > 0]['edge_diff_10']
                structure_df.at[model, 'Median Extra Edges'] = extra_edge_slice.median() if not extra_edge_slice.empty else 0
                
                missing_edge_slice = toper_eval[toper_eval['edge_diff_10'] < 0]['edge_diff_10']
                structure_df.at[model, 'Median Missing Edges'] = abs(missing_edge_slice.median()) if not missing_edge_slice.empty else 0
                structure_df.at[model, 'Descriptor Norm'] = toper_eval['l2_norm'].mean()
                
                node_df.at[model, 'Precision Old Nodes'] = node_eval['Precision_Old'].mean()
                node_df.at[model, 'Recall Old Nodes'] = node_eval['Recall_Old'].mean()
                node_df.at[model, 'F1 Old Nodes'] = node_eval['F1_Old'].mean()        
                
                node_df.at[model, 'Precision Nodes'] = node_eval['Precision_All'].mean()
                node_df.at[model, 'Recall Nodes'] = node_eval['Recall_All'].mean()
                node_df.at[model, 'F1 Nodes'] = node_eval['F1_All'].mean()          
                
                
                edge_df.at[model, 'Edge Precision'] = all_edges['Precision'].mean()
                edge_df.at[model, 'Edge Recall'] = all_edges['Recall'].mean()
                edge_df.at[model, 'Edge F1'] = all_edges['F1'].mean() 
                
                denom = node_eval['Num_New_True'].copy().astype(float)
                denom[denom == 0] = 1.0
                percent_diff_new_nodes = (node_eval['Num_New_Predicted'] - node_eval['Num_New_True']) / denom
                node_df.at[model, 'New Nodes Predicted'] = percent_diff_new_nodes.mean()
                
                edge_df.at[model, 'oo-bank Precision'] = old_only['Precision_bank'].mean()
                edge_df.at[model, 'oo-bank Recall'] = old_only['Recall_bank'].mean()
                edge_df.at[model, 'oo-bank F1'] = old_only['F1_bank'].mean()          
                        
                edge_df.at[model, 'oo-nobank Precision'] = old_only['Precision_nobank'].mean()
                # Handles typo
                recall_col_nobank = 'Recall_nobank' if 'Recall_nobank' in old_only.columns else 'Recall_bank_nobank'
                edge_df.at[model, 'oo-nobank Recall'] = old_only[recall_col_nobank].mean()
                edge_df.at[model, 'oo-nobank F1'] = old_only['F1_nobank'].mean()
            
                # Positive means model under predicted, negative means model over predicted
                num_true_on = on_only['TP'] + on_only['FN']
                num_pred_on = on_only['TP'] + on_only['FP']

                denom_on = num_true_on.copy()
                denom_on[denom_on == 0] = 1.0 # Force denominator to 1 if true is 0

                percent_diff_on = (num_pred_on - num_true_on) / denom_on
                edge_df.at[model, 'Num o-n Predicted'] = percent_diff_on.mean()

                # --- n-n Edges ---
                num_true_nn = new_only['TP'] + new_only['FN']
                num_pred_nn = new_only['TP'] + new_only['FP']

                denom_nn = num_true_nn.copy()
                denom_nn[denom_nn == 0] = 1.0

                percent_diff_nn = (num_pred_nn - num_true_nn) / denom_nn
                edge_df.at[model, 'Num n-n Predicted'] = percent_diff_nn.mean()

            
            int_cols = [
                'Median Extra Nodes ', 'Median Missing Nodes', 
                'Median Extra Edges', 'Median Missing Edges',
            ]

            # Apply "CHAL" string to any edge types where there weren't enough nodes to evaluate fairly
            threshold = -1  # Can modify
            for model in rows:
                # Check F1 Old Nodes for bank/nobank edge metrics
                f1_old = float(node_df.at[model, 'F1 Old Nodes'])
                if pd.notnull(f1_old) and f1_old < threshold:
                    cols_to_chal = [
                        'oo-bank Precision', 'oo-bank Recall', 'oo-bank F1',
                        'oo-nobank Precision', 'oo-nobank Recall', 'oo-nobank F1'
                    ]
                    for c in cols_to_chal:
                        edge_df.at[model, c] = "CHAL"

                # Check F1 Nodes for overall edge metrics
                f1_all = float(node_df.at[model, 'F1 Nodes'])
                if pd.notnull(f1_all) and f1_all < threshold:
                    cols_to_chal = ['Edge Precision', 'Edge Recall', 'Edge F1']
                    for c in cols_to_chal:
                        edge_df.at[model, c] = "CHAL"
            
            # 2. Apply formatting to each DataFrame
            for df in [node_df, structure_df, edge_df]:
                for col in df.columns:
                    def format_cell(x):
                        if x == "CHAL": return "CHAL" # Pass through
                        if pd.isnull(x): return np.nan
                        try:
                            val = float(x)
                            if col in int_cols:
                                return f"{int(round(val))}"
                            return f"{val:.2f}"
                        except:
                            return str(x)
                    df[col] = df[col].apply(format_cell)

            file_path = f'data/output/figures/sensitivity_analysis_days_tables_vewma/'
            os.makedirs(file_path, exist_ok=True)
            structure_output_path = os.path.join(file_path, f"{dataset}_structure_table.png")
            edge_output_path = os.path.join(file_path, f"{dataset}_edge_table.png")
            node_output_path = os.path.join(file_path, f"{dataset}_node_table.png")
                
            self.makeTable(node_df, node_output_path, caption=f"Node Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_node_evaluation", table_type="nodes")
            self.makeTable(structure_df, structure_output_path, caption=f"Structure Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_structure_evaluation", table_type="structure")
            self.makeTable(edge_df, edge_output_path, caption=f"Edge Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_edge_evaluation", table_type="edges")


        
    
    def sensitivity_analysis_len(self, datasets):
        toper_lens = [5, 10, 20, 50]
        lr = 0.001
        for dataset in datasets:
            data = []
            for toper_len in toper_lens:
                data_path = f'data/output/sensitivity_analysis/toper_lens/constructed_graphs/{dataset}_topoGED_embedding_mlpEncodingConcat_embeddingTypeGCN_lr0.001_5back_oobankchanges_zeros_sampling_predvalsTrue_edgebank_default_VectorTypeV-EWMA_len{toper_len}_days5back/GCN_constructed_graphs_{dataset}.pkl'
                prefix = f'GraphGeneration/scripts/results/{dataset}/sensitivity_analysis/lens/{dataset}_lr{lr}_{toper_len}len_tmp_V-EWMA/'
                
                # Base method
                if toper_len == 10:
                    data_path = Path(f"data/output/constructed_graphs/{dataset}_topoGED_embedding_mlpEncodingConcat_embeddingTypeGCN_lr0.001_5back_oobankchanges_zeros_sampling_predvalsTrue_tmp_edgebank_default_VectorTypeV-EWMA/GCN_constructed_graphs_{dataset}.pkl")

                if os.path.exists(data_path) and not os.path.exists(prefix):
                    try:
                        evaluator = EvaluateGraphs(args=None, prefix=prefix, data_path=data_path)
                        evaluator.run()
                    except Exception as e:
                        print(f"Skipping Eval for {dataset} {toper_len}len: {e}")
                else:
                    print(f'Skipping eval for {data_path}')
                    
                data.append(prefix)
            
            # Setup rows
            rows = [f'TopoGED - Len {d}' for d in toper_lens]
            node_columns = ['Precision Nodes', 'Recall Nodes', 'F1 Nodes', 'Precision Old Nodes', 'Recall Old Nodes', 'F1 Old Nodes', 'New Nodes Predicted']
            structure_columns = ['Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 'Clustering Coefficient',
                    'Density', 'Num Triangles', 'Descriptor Norm', 'Median Extra Nodes', 'Median Missing Nodes', 'Median Extra Edges', 'Median Missing Edges']
            edge_columns = ['oo-bank Precision', "oo-bank Recall", "oo-bank F1", 
                            "oo-nobank Precision", "oo-nobank Recall", "oo-nobank F1",
                            "Num o-n Predicted", "Num n-n Predicted",
                            "Edge Precision", "Edge Recall", "Edge F1"]
        
            node_df = pd.DataFrame(np.nan, index=rows, columns=node_columns)
            structure_df = pd.DataFrame(np.nan, index=rows, columns=structure_columns)
            edge_df = pd.DataFrame(np.nan, index=rows, columns=edge_columns)
            num_eval_graphs = num_test_graphs[dataset]
            
            # Compute Metrics for each dataset
            print(f'Generating sensitivity analysis days table for dataset: {dataset}')
            for directory in data:
                print(directory)
                
                dir_str = str(directory)
                curr_len = re.search(r'(\d+)len', dir_str)
                if curr_len:
                    # MATCH THIS EXACTLY to the 'rows' definition below
                    model = f'TopoGED - Len {curr_len.group(1)}' 
                else:
                    continue
                    
                
                if not os.path.exists(directory):
                    print(f"Data from: {directory} does not exist")
                    structure_df.at[model, 'Avg Node Degree'] = np.nan
                    structure_df.at[model, 'Unique Degree Count'] = np.nan
                    structure_df.at[model, 'Degree Centrality'] = np.nan
                    structure_df.at[model, 'Assortativity Coefficient'] = np.nan
                    structure_df.at[model, 'Clustering Coefficient'] = np.nan
                    structure_df.at[model, 'Density'] = np.nan
                    structure_df.at[model, 'Num Triangles'] = np.nan
                    
                    structure_df.at[model, 'Median Extra Nodes'] = np.nan
                    
                    structure_df.at[model, 'Median Missing Nodes'] = np.nan
                    
                    structure_df.at[model, 'Median Extra Edges'] = np.nan
                    
                    structure_df.at[model, 'Median Missing Edges'] = np.nan
                    structure_df.at[model, 'Descriptor Norm'] = np.nan
                    
                    node_df.at[model, 'Precision Old Nodes'] = np.nan
                    node_df.at[model, 'Recall Old Nodes'] = np.nan
                    node_df.at[model, 'F1 Old Nodes'] = np.nan      
                    
                    node_df.at[model, 'Precision Nodes'] = np.nan
                    node_df.at[model, 'Recall Nodes'] = np.nan
                    node_df.at[model, 'F1 Nodes'] = np.nan        
                    
                    
                    edge_df.at[model, 'Edge Precision'] = np.nan
                    edge_df.at[model, 'Edge Recall'] = np.nan
                    edge_df.at[model, 'Edge F1'] = np.nan
                    
                    node_df.at[model, 'New Nodes Predicted'] = np.nan
                    
                    edge_df.at[model, 'oo-bank Precision'] = np.nan
                    edge_df.at[model, 'oo-bank Recall'] = np.nan
                    edge_df.at[model, 'oo-bank F1'] = np.nan       
                            
                    edge_df.at[model, 'oo-nobank Precision'] = np.nan
                    # Handles typo
                    edge_df.at[model, 'oo-nobank Recall'] = np.nan
                    edge_df.at[model, 'oo-nobank F1'] = np.nan
                
                    edge_df.at[model, 'Num o-n Predicted'] = np.nan
                    edge_df.at[model, 'Num n-n Predicted'] = np.nan
                        
                    continue    
                
                dir_path = Path(directory)
                # Take the tail since we only eval test graphs here
                files = {
                    'node_eval': str(dir_path / 'structure' / 'node_evaluation.csv'),
                    'true_structure': str(dir_path / 'structure' / 'structure_true.csv'),
                    'pred_structure': str(dir_path / 'structure' / 'structure_pred.csv'),
                    'toper_eval': str(dir_path / 'structure' / 'toper_eval.csv'),
                    'all_edges': str(dir_path / 'edge_evaluation' / 'all_edges.csv'),
                    'old_only': str(dir_path / 'edge_evaluation' / 'old_nodes_only.csv'),
                    'on_only': str(dir_path / 'edge_evaluation' / 'on_edges_only.csv'),
                    'new_only': str(dir_path / 'edge_evaluation' / 'new_nodes_only.csv')
                }

                # Load each file and check length
                dfs = {}
                for key, path in files.items():
                    temp_df = pd.read_csv(path)
                    if len(temp_df) < num_eval_graphs:
                        print(f"ERROR: {path} has {len(temp_df)} rows, expected {num_eval_graphs}")
                    dfs[key] = temp_df.tail(num_eval_graphs)

                # Map back to your variables
                node_eval = dfs['node_eval']
                true_structure = dfs['true_structure']
                pred_structure = dfs['pred_structure']
                toper_eval = dfs['toper_eval']
                all_edges = dfs['all_edges']
                old_only = dfs['old_only']
                on_only = dfs['on_only']
                new_only = dfs['new_only']

                # Add metrics
                def get_mean(df, col):
                    return df[col].replace([np.inf, -np.inf], np.nan).mean()

                # Calculate Percent Error: (Pred - True) / True
                metrics_map = {
                    'Avg Node Degree': 'Average Node Degree',
                    'Unique Degree Count': 'Unique Degree Count',
                    'Degree Centrality': 'Degree Centrality',
                    'Assortativity Coefficient': 'Assortivity Coefficient',
                    'Clustering Coefficient': 'Clustering Coefficient',
                    'Density': 'Density',
                    'Num Triangles': 'Number of Triangles'
                }

                for df_col, csv_col in metrics_map.items():
                    t_vals = true_structure[csv_col]
                    p_vals = pred_structure[csv_col]
                    
                    denom = t_vals.copy()
                    denom[denom == 0] = 1.0
                    
                    relative_errors = (p_vals - t_vals) / denom
                    
                    structure_df.at[model, df_col] = relative_errors.mean()
                    
                extra_node_slice = toper_eval[toper_eval['node_diff_10'] > 0]['node_diff_10']
                structure_df.at[model, 'Median Extra Nodes'] = extra_node_slice.median() if not extra_node_slice.empty else 0
                
                missing_node_slice = toper_eval[toper_eval['node_diff_10'] < 0]['node_diff_10']
                structure_df.at[model, 'Median Missing Nodes'] = abs(missing_node_slice.median()) if not missing_node_slice.empty else 0
                
                extra_edge_slice = toper_eval[toper_eval['edge_diff_10'] > 0]['edge_diff_10']
                structure_df.at[model, 'Median Extra Edges'] = extra_edge_slice.median() if not extra_edge_slice.empty else 0
                
                missing_edge_slice = toper_eval[toper_eval['edge_diff_10'] < 0]['edge_diff_10']
                structure_df.at[model, 'Median Missing Edges'] = abs(missing_edge_slice.median()) if not missing_edge_slice.empty else 0
                structure_df.at[model, 'Descriptor Norm'] = toper_eval['l2_norm'].mean()
                
                node_df.at[model, 'Precision Old Nodes'] = node_eval['Precision_Old'].mean()
                node_df.at[model, 'Recall Old Nodes'] = node_eval['Recall_Old'].mean()
                node_df.at[model, 'F1 Old Nodes'] = node_eval['F1_Old'].mean()        
                
                node_df.at[model, 'Precision Nodes'] = node_eval['Precision_All'].mean()
                node_df.at[model, 'Recall Nodes'] = node_eval['Recall_All'].mean()
                node_df.at[model, 'F1 Nodes'] = node_eval['F1_All'].mean()          
                
                
                edge_df.at[model, 'Edge Precision'] = all_edges['Precision'].mean()
                edge_df.at[model, 'Edge Recall'] = all_edges['Recall'].mean()
                edge_df.at[model, 'Edge F1'] = all_edges['F1'].mean() 
                
                denom = node_eval['Num_New_True'].copy().astype(float)
                denom[denom == 0] = 1.0
                percent_diff_new_nodes = (node_eval['Num_New_Predicted'] - node_eval['Num_New_True']) / denom
                node_df.at[model, 'New Nodes Predicted'] = percent_diff_new_nodes.mean()
                
                edge_df.at[model, 'oo-bank Precision'] = old_only['Precision_bank'].mean()
                edge_df.at[model, 'oo-bank Recall'] = old_only['Recall_bank'].mean()
                edge_df.at[model, 'oo-bank F1'] = old_only['F1_bank'].mean()          
                        
                edge_df.at[model, 'oo-nobank Precision'] = old_only['Precision_nobank'].mean()
                # Handles typo
                recall_col_nobank = 'Recall_nobank' if 'Recall_nobank' in old_only.columns else 'Recall_bank_nobank'
                edge_df.at[model, 'oo-nobank Recall'] = old_only[recall_col_nobank].mean()
                edge_df.at[model, 'oo-nobank F1'] = old_only['F1_nobank'].mean()
            
                # Positive means model under predicted, negative means model over predicted
                num_true_on = on_only['TP'] + on_only['FN']
                num_pred_on = on_only['TP'] + on_only['FP']

                denom_on = num_true_on.copy()
                denom_on[denom_on == 0] = 1.0 # Force denominator to 1 if true is 0

                percent_diff_on = (num_pred_on - num_true_on) / denom_on
                edge_df.at[model, 'Num o-n Predicted'] = percent_diff_on.mean()

                # --- n-n Edges ---
                num_true_nn = new_only['TP'] + new_only['FN']
                num_pred_nn = new_only['TP'] + new_only['FP']

                denom_nn = num_true_nn.copy()
                denom_nn[denom_nn == 0] = 1.0

                percent_diff_nn = (num_pred_nn - num_true_nn) / denom_nn
                edge_df.at[model, 'Num n-n Predicted'] = percent_diff_nn.mean()

            
            int_cols = [
                'Median Extra Nodes', 'Median Missing Nodes', 
                'Median Extra Edges', 'Median Missing Edges',
            ]

            # Apply "CHAL" string to any edge types where there weren't enough nodes to evaluate fairly
            threshold = -1  # Can modify
            for model in rows:
                # Check F1 Old Nodes for bank/nobank edge metrics
                f1_old = float(node_df.at[model, 'F1 Old Nodes'])
                if pd.notnull(f1_old) and f1_old < threshold:
                    cols_to_chal = [
                        'oo-bank Precision', 'oo-bank Recall', 'oo-bank F1',
                        'oo-nobank Precision', 'oo-nobank Recall', 'oo-nobank F1'
                    ]
                    for c in cols_to_chal:
                        edge_df.at[model, c] = "CHAL"

                # Check F1 Nodes for overall edge metrics
                f1_all = float(node_df.at[model, 'F1 Nodes'])
                if pd.notnull(f1_all) and f1_all < threshold:
                    cols_to_chal = ['Edge Precision', 'Edge Recall', 'Edge F1']
                    for c in cols_to_chal:
                        edge_df.at[model, c] = "CHAL"
            
            # 2. Apply formatting to each DataFrame
            for df in [node_df, structure_df, edge_df]:
                for col in df.columns:
                    def format_cell(x):
                        if x == "CHAL": return "CHAL" # Pass through
                        if pd.isnull(x): return np.nan
                        try:
                            val = float(x)
                            if col in int_cols:
                                return f"{int(round(val))}"
                            return f"{val:.2f}"
                        except:
                            return str(x)
                    df[col] = df[col].apply(format_cell)

            file_path = f'data/output/figures/sensitivity_analysis_lens_tables_vewma/'
            os.makedirs(file_path, exist_ok=True)
            structure_output_path = os.path.join(file_path, f"{dataset}_structure_table.png")
            edge_output_path = os.path.join(file_path, f"{dataset}_edge_table.png")
            node_output_path = os.path.join(file_path, f"{dataset}_node_table.png")
                
            self.makeTable(node_df, node_output_path, caption=f"Node Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_node_evaluation", table_type="nodes")
            self.makeTable(structure_df, structure_output_path, caption=f"Structure Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_structure_evaluation", table_type="structure")
            self.makeTable(edge_df, edge_output_path, caption=f"Edge Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_edge_evaluation", table_type="edges")


    def construct_nn_on_tables(self):
        datasets = [
            'CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion', 'networkaeternity', 'networkaragon', 'networkbancor', 'networkcentra', 
            'networkcoindash', 'networkiconomi', 'networkcindicator', 'networkdgd', 'Reddit_B', 'tgbl-wiki',
        ]  
        
        method = 'oobankchanges'
        lr = 0.001
        res_df = pd.DataFrame(index=datasets, columns=['True_on', 'Pred_on', 'True_nn', 'Pred_nn', '# Graphs'])
        
        for dataset in datasets:
            data_path = f'GraphGeneration/scripts/results/{dataset}/{method}_lr{lr}_5back/edge_evaluation/'
            
            on_path = os.path.join(data_path, 'on_edges_only.csv')
            nn_path = os.path.join(data_path, 'new_nodes_only.csv')
            on_df = pd.read_csv(on_path)
            nn_df = pd.read_csv(nn_path)
            
            tp_on = on_df['TP']
            fp_on = on_df['FP']
            fn_on = on_df['FN']
            true_col_on = tp_on + fn_on
            pred_col_on = tp_on + fp_on
            avg_true_on = true_col_on.mean()
            std_true_on = true_col_on.std()
            avg_pred_on = pred_col_on.mean()
            std_pred_on = pred_col_on.std()
            
            tp_nn = nn_df['TP']
            fp_nn = nn_df['FP']
            fn_nn = nn_df['FN']
            true_col_nn = tp_nn + fn_nn
            pred_col_nn = tp_nn + fp_nn
            avg_true_nn = true_col_nn.mean()
            std_true_nn = true_col_nn.std()
            avg_pred_nn = pred_col_nn.mean()
            std_pred_nn = pred_col_nn.std()
            
            num_graphs = on_df.shape[0]
            
            res_df.at[dataset, 'True_on'] = fr"{avg_true_on:.1f} \pm {std_true_on:.1f}"
            res_df.at[dataset, 'Pred_on'] = fr"{avg_pred_on:.1f} \pm {std_pred_on:.1f}"
            res_df.at[dataset, 'True_nn'] = fr"{avg_true_nn:.1f} \pm {std_true_nn:.1f}"
            res_df.at[dataset, 'Pred_nn'] = fr"{avg_pred_nn:.1f} \pm {std_pred_nn:.1f}"
            res_df.at[dataset, '# Graphs'] = str(num_graphs)

        latex = []

        # Column setup: l (dataset) | c c (o-n) | c c (n-n) | c (n)
        latex.append(r"\begin{tabular}{l cc cc c}")
        latex.append(r"\toprule")
        
        # Spanning Headers
        latex.append(r" & \multicolumn{2}{c}{o-n} & \multicolumn{2}{c}{n-n} & \\")
        latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
        
        # Sub-headers
        latex.append(r"dataset & # True & # Pred & # True & # Pred & # Graphs \\ \midrule")
        
        for dataset, row in res_df.iterrows():
            clean_name = dataset.replace('network', '').replace('_', r'\_')
            row_str = f"{clean_name} & ${row['True_on']}$ & ${row['Pred_on']}$ & ${row['True_nn']}$ & ${row['Pred_nn']}$ & {row['# Graphs']} \\\\"
            latex.append(row_str)
            
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        
        output_path = f'data/output/figures/on_nn_comparison/'
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, 'res_table.txt')
        with open(output_path, 'w') as f:
            f.write("\n".join(latex))    
            
            
    def formTables(self, data, file_path, dataset, old_only_status):
        rows = ['TopoGED', 'HTGN', 'ROLAND', 'VGRNN', 'TGCN', 'GCLSTM', 'EvolveGCN']
        node_columns = ['Precision Nodes', 'Recall Nodes', 'F1 Nodes', 'Precision Old Nodes', 'Recall Old Nodes', 'F1 Old Nodes', 'New Nodes Predicted']
        structure_columns = ['Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 'Clustering Coefficient',
                   'Density', 'Num Triangles', 'Descriptor Norm', 'Median Extra Nodes', 'Median Missing Nodes', 'Median Extra Edges', 'Median Missing Edges']
        if not old_only_status:
            edge_columns = ['oo-bank Precision', "oo-bank Recall", "oo-bank F1", 
                            "oo-nobank Precision", "oo-nobank Recall", "oo-nobank F1",
                            "Num o-n Predicted", "Num n-n Predicted",
                            "Edge Precision", "Edge Recall", "Edge F1"]
            edge_columns = ['oo-bank Precision', 'oo-bank Recall', 'oo-bank F1',
                            'oo-nobank Precision', 'oo-nobank Recall', 'oo-nobank F1',
                            'Edge Precision', 'Edge Recall', 'Edge F1']
        
        node_df = pd.DataFrame(np.nan, index=rows, columns=node_columns)
        structure_df = pd.DataFrame(np.nan, index=rows, columns=structure_columns)
        edge_df = pd.DataFrame(np.nan, index=rows, columns=edge_columns)
        num_eval_graphs = num_test_graphs[dataset]
        
        # Compute Metrics for each dataset
        for directory in data:
            print(directory)
            # Get name of strategy
            dir_str = str(directory).lower()
            
            # Check the path to see if this is our TopoGED variant
            if "topoged" in dir_str:
                if "sampling" in dir_str:
                    model = "TopoGED"
                else:
                    model = "TopoGED Edgebank"
            else:
                # Standard benchmarkers (ROLAND, HTGN, etc.)
                model = Path(directory).parent.name
                
            
            if not os.path.exists(directory):
                print(f"Data from: {directory} does not exist")
                structure_df.at[model, 'Avg Node Degree'] = np.nan
                structure_df.at[model, 'Unique Degree Count'] = np.nan
                structure_df.at[model, 'Degree Centrality'] = np.nan
                structure_df.at[model, 'Assortativity Coefficient'] = np.nan
                structure_df.at[model, 'Clustering Coefficient'] = np.nan
                structure_df.at[model, 'Density'] = np.nan
                structure_df.at[model, 'Num Triangles'] = np.nan
                
                structure_df.at[model, 'Median Extra Nodes'] = np.nan
                
                structure_df.at[model, 'Median Missing Nodes'] = np.nan
                
                structure_df.at[model, 'Median Extra Edges'] = np.nan
                
                structure_df.at[model, 'Median Missing Edges'] = np.nan
                structure_df.at[model, 'Descriptor Norm'] = np.nan
                
                node_df.at[model, 'Precision Old Nodes'] = np.nan
                node_df.at[model, 'Recall Old Nodes'] = np.nan
                node_df.at[model, 'F1 Old Nodes'] = np.nan      
                
                node_df.at[model, 'Precision Nodes'] = np.nan
                node_df.at[model, 'Recall Nodes'] = np.nan
                node_df.at[model, 'F1 Nodes'] = np.nan        
                
                
                edge_df.at[model, 'Edge Precision'] = np.nan
                edge_df.at[model, 'Edge Recall'] = np.nan
                edge_df.at[model, 'Edge F1'] = np.nan
                
                node_df.at[model, 'New Nodes Predicted'] = np.nan
                
                edge_df.at[model, 'oo-bank Precision'] = np.nan
                edge_df.at[model, 'oo-bank Recall'] = np.nan
                edge_df.at[model, 'oo-bank F1'] = np.nan       
                        
                edge_df.at[model, 'oo-nobank Precision'] = np.nan
                # Handles typo
                edge_df.at[model, 'oo-nobank Recall'] = np.nan
                edge_df.at[model, 'oo-nobank F1'] = np.nan
            
                if not old_only_status:                    
                    edge_df.at[model, 'Num o-n Predicted'] = np.nan
                    edge_df.at[model, 'Num n-n Predicted'] = np.nan
                    
                continue    
            
            
            # Take the tail since we only eval test graphs here
            files = {
                'node_eval': str(directory / 'structure' / 'node_evaluation.csv'),
                'true_structure': str(directory / 'structure' / 'structure_true.csv'),
                'pred_structure': str(directory / 'structure' / 'structure_pred.csv'),
                'toper_eval': str(directory / 'structure' / 'toper_eval.csv'),
                'all_edges': str(directory / 'edge_evaluation' / 'all_edges.csv'),
                'old_only': str(directory / 'edge_evaluation' / 'old_nodes_only.csv'),
                'on_only': str(directory / 'edge_evaluation' / 'on_edges_only.csv'),
                'new_only': str(directory / 'edge_evaluation' / 'new_nodes_only.csv')
            }

            # Load each file and check length
            dfs = {}
            for key, path in files.items():
                temp_df = pd.read_csv(path)
                if len(temp_df) < num_eval_graphs:
                    print(f"ERROR: {path} has {len(temp_df)} rows, expected {num_eval_graphs}")
                dfs[key] = temp_df.tail(num_eval_graphs)

            # Map back to your variables
            node_eval = dfs['node_eval']
            true_structure = dfs['true_structure']
            pred_structure = dfs['pred_structure']
            toper_eval = dfs['toper_eval']
            all_edges = dfs['all_edges']
            old_only = dfs['old_only']
            on_only = dfs['on_only']
            new_only = dfs['new_only']

            # Add metrics
            def get_mean(df, col):
                return df[col].replace([np.inf, -np.inf], np.nan).mean()

            # Calculate Percent Error: (Pred - True) / True
            metrics_map = {
                'Avg Node Degree': 'Average Node Degree',
                'Unique Degree Count': 'Unique Degree Count',
                'Degree Centrality': 'Degree Centrality',
                'Assortativity Coefficient': 'Assortivity Coefficient',
                'Clustering Coefficient': 'Clustering Coefficient',
                'Density': 'Density',
                'Num Triangles': 'Number of Triangles'
            }

            for df_col, csv_col in metrics_map.items():
                t_vals = true_structure[csv_col]
                p_vals = pred_structure[csv_col]
                
                denom = t_vals.copy()
                denom[denom == 0] = 1.0
                
                relative_errors = (p_vals - t_vals) / denom
                
                structure_df.at[model, df_col] = relative_errors.mean()
                
                
            extra_node_slice = toper_eval[toper_eval['node_diff_10'] > 0]['node_diff_10']
            structure_df.at[model, 'Median Extra Nodes'] = extra_node_slice.median() if not extra_node_slice.empty else 0
            
            missing_node_slice = toper_eval[toper_eval['node_diff_10'] < 0]['node_diff_10']
            structure_df.at[model, 'Median Missing Nodes'] = abs(missing_node_slice.median()) if not missing_node_slice.empty else 0
            
            extra_edge_slice = toper_eval[toper_eval['edge_diff_10'] > 0]['edge_diff_10']
            structure_df.at[model, 'Median Extra Edges'] = extra_edge_slice.median() if not extra_edge_slice.empty else 0
            
            missing_edge_slice = toper_eval[toper_eval['edge_diff_10'] < 0]['edge_diff_10']
            structure_df.at[model, 'Median Missing Edges'] = abs(missing_edge_slice.median()) if not missing_edge_slice.empty else 0
            structure_df.at[model, 'Descriptor Norm'] = toper_eval['l2_norm'].mean()
            
            node_df.at[model, 'Precision Old Nodes'] = node_eval['Precision_Old'].mean()
            node_df.at[model, 'Recall Old Nodes'] = node_eval['Recall_Old'].mean()
            node_df.at[model, 'F1 Old Nodes'] = node_eval['F1_Old'].mean()        
            
            node_df.at[model, 'Precision Nodes'] = node_eval['Precision_All'].mean()
            node_df.at[model, 'Recall Nodes'] = node_eval['Recall_All'].mean()
            node_df.at[model, 'F1 Nodes'] = node_eval['F1_All'].mean()          
            
            
            edge_df.at[model, 'Edge Precision'] = all_edges['Precision'].mean()
            edge_df.at[model, 'Edge Recall'] = all_edges['Recall'].mean()
            edge_df.at[model, 'Edge F1'] = all_edges['F1'].mean() 
            
            denom = node_eval['Num_New_True'].copy().astype(float)
            denom[denom == 0] = 1.0
            percent_diff_new_nodes = (node_eval['Num_New_Predicted'] - node_eval['Num_New_True']) / denom
            node_df.at[model, 'New Nodes Predicted'] = percent_diff_new_nodes.mean()
            
            edge_df.at[model, 'oo-bank Precision'] = old_only['Precision_bank'].mean()
            edge_df.at[model, 'oo-bank Recall'] = old_only['Recall_bank'].mean()
            edge_df.at[model, 'oo-bank F1'] = old_only['F1_bank'].mean()          
                    
            edge_df.at[model, 'oo-nobank Precision'] = old_only['Precision_nobank'].mean()
            # Handles typo
            recall_col_nobank = 'Recall_nobank' if 'Recall_nobank' in old_only.columns else 'Recall_bank_nobank'
            edge_df.at[model, 'oo-nobank Recall'] = old_only[recall_col_nobank].mean()
            edge_df.at[model, 'oo-nobank F1'] = old_only['F1_nobank'].mean()
        
            if not old_only_status:                    
                # Positive means model under predicted, negative means model over predicted
                num_true_on = on_only['TP'] + on_only['FN']
                num_pred_on = on_only['TP'] + on_only['FP']

                denom_on = num_true_on.copy()
                denom_on[denom_on == 0] = 1.0 # Force denominator to 1 if true is 0

                percent_diff_on = (num_pred_on - num_true_on) / denom_on
                edge_df.at[model, 'Num o-n Predicted'] = percent_diff_on.mean()

                # --- n-n Edges ---
                num_true_nn = new_only['TP'] + new_only['FN']
                num_pred_nn = new_only['TP'] + new_only['FP']

                denom_nn = num_true_nn.copy()
                denom_nn[denom_nn == 0] = 1.0

                percent_diff_nn = (num_pred_nn - num_true_nn) / denom_nn
                edge_df.at[model, 'Num n-n Predicted'] = percent_diff_nn.mean()

        
        int_cols = [
            'Median Extra Nodes', 'Median Missing Nodes', 
            'Median Extra Edges', 'Median Missing Edges',
        ]

        # Apply "CHAL" string to any edge types where there weren't enough nodes to evaluate fairly
        threshold = -1  # Can modify
        for model in rows:
            # Check F1 Old Nodes for bank/nobank edge metrics
            f1_old = float(node_df.at[model, 'F1 Old Nodes'])
            if pd.notnull(f1_old) and f1_old < threshold:
                cols_to_chal = [
                    'oo-bank Precision', 'oo-bank Recall', 'oo-bank F1',
                    'oo-nobank Precision', 'oo-nobank Recall', 'oo-nobank F1'
                ]
                for c in cols_to_chal:
                    edge_df.at[model, c] = "CHAL"

            # Check F1 Nodes for overall edge metrics
            f1_all = float(node_df.at[model, 'F1 Nodes'])
            if pd.notnull(f1_all) and f1_all < threshold:
                cols_to_chal = ['Edge Precision', 'Edge Recall', 'Edge F1']
                for c in cols_to_chal:
                    edge_df.at[model, c] = "CHAL"
        
        # 2. Apply formatting to each DataFrame
        for df in [node_df, structure_df, edge_df]:
            for col in df.columns:
                def format_cell(x):
                    if x == "CHAL": return "CHAL" # Pass through
                    if pd.isnull(x): return np.nan
                    try:
                        val = float(x)
                        if col in int_cols:
                            return f"{int(round(val))}"
                        return f"{val:.2f}"
                    except:
                        return str(x)
                df[col] = df[col].apply(format_cell)

            
        structure_output_path = os.path.join(file_path, f"{dataset}_structure_table.png")
        edge_output_path = os.path.join(file_path, f"{dataset}_edge_table.png")
        node_output_path = os.path.join(file_path, f"{dataset}_node_table.png")
            
        self.makeTable(node_df, node_output_path, caption=f"Node Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_node_evaluation", table_type="nodes")
        self.makeTable(structure_df, structure_output_path, caption=f"Structure Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_structure_evaluation", table_type="structure")
        self.makeTable(edge_df, edge_output_path, caption=f"Edge Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_edge_evaluation", table_type="edges")

        # curr_threshold = float(file_path.split('Threshold')[-1].split('_')[0])
        curr_threshold = 0
        self.collect_data_for_heatmap(node_df, dataset, old_only_status, curr_threshold)
        self.collect_data_for_heatmap(structure_df, dataset, old_only_status, curr_threshold)
        self.collect_data_for_heatmap(edge_df, dataset, old_only_status, curr_threshold)


    def generate_figures(self, datasets):
        # Professional Styling
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['figure.dpi'] = 300
        sns.set_context("paper", font_scale=1.4)
        sns.set_style("whitegrid", {'axes.grid': True, 'grid.alpha': 0.4})
        
        lr = 0.001
        
        # Comprehensive Metric Mapping
        metric_map = {
            'Edge F1': ('edge_evaluation', 'all_edges.csv', 'F1'),
            'oo-bank F1': ('edge_evaluation', 'old_nodes_only.csv', 'F1_bank'),
            'oo-nobank F1': ('edge_evaluation', 'old_nodes_only.csv', 'F1_nobank'),
            'Precision Nodes': ('structure', 'node_evaluation.csv', 'Precision_All'),
            'Recall Nodes': ('structure', 'node_evaluation.csv', 'Recall_All'),
            'F1 Nodes': ('structure', 'node_evaluation.csv', 'F1_All'),
            'Descriptor Norm': ('structure', 'toper_eval.csv', 'l2_norm'),
            'Degree Centrality': ('structure', 'structure_pred.csv', 'Degree Centrality'),
            'Density': ('structure', 'structure_pred.csv', 'Density'),
            'Clustering Coefficient': ('structure', 'structure_pred.csv', 'Clustering Coefficient')
        }

        output_root = Path('data/output/figures/individual_pdf_plots/')
        output_root.mkdir(parents=True, exist_ok=True)

        for dataset in datasets:
            n_test = num_test_graphs.get(dataset, 1)
            print(f"Generating individual plots for: {dataset}")

            for metric_name, (folder, filename, col) in metric_map.items():
                # Sanitize metric name for filenames
                safe_metric = metric_name.replace(' ', '_').replace('-', '_')
                
                # --- 1. TopoER Length Sweep Plot ---
                len_vals, len_scores = [], []
                for length in [5, 10, 20, 50]:
                    path = Path(f'GraphGeneration/scripts/results/{dataset}/sensitivity_analysis/lens/{dataset}_lr{lr}_{length}len/') / folder / filename
                    if path.exists():
                        val = pd.read_csv(path)[col].tail(n_test).mean()
                        len_vals.append(length)
                        len_scores.append(val)
                
                if len_vals:
                    plt.figure(figsize=(5, 4))
                    plt.plot(len_vals, len_scores, marker='o', color='#2980b9', lw=2, markersize=8)
                    plt.xlabel('$n$')
                    plt.ylabel(metric_name)
                    plt.tight_layout()
                    plt.savefig(output_root / f"{dataset}_{safe_metric}_TopER_Length.pdf")
                    plt.close()

                # --- 2. Days Back Sweep Plot ---
                days_vals, days_scores = [], []
                for days in [1, 3, 5, 7, 14, 30]:
                    path = Path(f'GraphGeneration/scripts/results/{dataset}/sensitivity_analysis/days/{dataset}_lr{lr}_{days}back/') / folder / filename
                    if path.exists():
                        val = pd.read_csv(path)[col].tail(n_test).mean()
                        days_vals.append(days)
                        days_scores.append(val)

                if days_vals:
                    plt.figure(figsize=(5, 4))
                    plt.plot(days_vals, days_scores, marker='s', color='#2980b9', lw=2, markersize=8)
                    plt.xlabel('$k$')
                    plt.ylabel(metric_name)
                    plt.tight_layout()
                    plt.savefig(output_root / f"{dataset}_{safe_metric}_Sweep_Days.pdf")
                    plt.close()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Evaluation Script")

    # Add arguments
    parser.add_argument("--dataset", type=str, default="CollegeMsg",
                        help="Name of the dataset to use")
    parser.add_argument("--embedding_method", type=str, required=False, help="The node embedding method")
    parser.add_argument("--use_ma", type=bool, required=False, help="Whether to use moving average, only supported for Node2Vec and HKS")

    # Parse arguments
    args = parser.parse_args()
        
    # Form the list of input graph files

    datasets = [
        'CollegeMsg', 
        'mathoverflow', 
        'networkadex', 
        'networkaion', 
        'networkaeternity', 
        'networkaragon', 
        'networkbancor', 
        'networkcentra', 
        'networkcoindash', 
        'networkiconomi', 
        'networkcindicator', 
        'networkdgd', 
        'Reddit_B', 
        'tgbl-wiki'
    ]
    models = ['EvolveGCN', 'GCLSTM', 'HTGN', 'ROLAND', 'TGCN', 'VGRNN']
    all_inputs = []
    all_prefixes = []

    
    output_base = Path('data/output/results/')
    for dataset in datasets:
        paths = [
            f"data/output/predicted/ROLAND/{dataset}_128_0.001_3_5_undirected_threshold_5xCap.pkl",
            f"data/output/predicted/VGRNN/{dataset}_64_32_2_0.001_GIN_undirected_threshold_5xCap.pkl",
            f"data/output/predicted/TGCN/{dataset}_128_0.01_0.001_undirected_threshold_5xCap.pkl",
            # f"data/output/predicted/SFDyG/{dataset}_hgcn_7_4_0.0001_1024_100_0_threshold_5xCap.pkl",
            f"data/output/predicted/HTGN/{dataset}_0.001_256_128_0.5_undirected_threshold_5xCap.pkl",
            f"data/output/predicted/EvolveGCN/{dataset}_egcn_o_128_64_32_0.01_0.0001_undirected_threshold_5xCap.pkl",
            f"data/output/predicted/GCLSTM/{dataset}_0.0001_128_2_0.001_undirected_threshold_5xCap.pkl"
        ]
        
        for path_str in paths:
            search_file = pathlib.Path(path_str)
            
            model_name = search_file.parts[3] if len(search_file.parts) > 3 else "UnknownModel"
            
            if search_file.exists():
                all_inputs.append(str(search_file))
                
                prefix_path = output_base / model_name / f"{search_file.stem}_tmp"
                
                all_prefixes.append(str(prefix_path))
            else:
                print(f"Skipping: Could not find pkl for {dataset} in {model_name}")
    
    # Add topoged paths to the evaluation
    topoged_base = Path('data/output/constructed_graphs/')
    topoged_output_base = Path('data/output/results/')

    for edgebank_style in ['default']:
        for node_embedding_type in ['zeros']:
            for lr in [0.001]:
                for toper_type in ['False']:
                    for dataset in datasets:
                        # Constructing as a Path object is cleaner
                        pred_val_logic = "False" if toper_type == 'True' else "True"
                        vector_type_logic = "TrueVals" if toper_type == 'True' else "V-EWMA"

                        relative_path = Path(
                            f"{dataset}_topoGED_embedding_mlpEncodingConcat_embeddingTypeGCN_"
                            f"lr{lr}_5back_oobankchanges_{node_embedding_type}_sampling_"
                            f"predvals{pred_val_logic}_tmp_edgebank_{edgebank_style}_"
                            f"VectorType{vector_type_logic}"
                        ) / f"GCN_constructed_graphs_{dataset}.pkl"
                        
                        target_file = topoged_base / relative_path
                        
                        if target_file.exists():
                            all_inputs.append(str(target_file))
                            prefix_path = topoged_output_base / f"TopoGED_{dataset}_{node_embedding_type}_{lr}_{edgebank_style}_{vector_type_logic}_sampling_tmp"
                            all_prefixes.append(str(prefix_path))
                        else:
                            print(f"Skipping: File not found at {target_file}")
                        
    # Now, evaluate each            
    import gc
    for prefix, data_path in zip(all_prefixes, all_inputs):
        if os.path.exists(prefix):
            print(f'Skipping evaluation for {data_path} since output already exists at {prefix}')
            continue
        
        print(f"Processing: {prefix}")
        
        evaluator = EvaluateGraphs(args=None, prefix=prefix, data_path=data_path)
    
        if hasattr(evaluator, 'loaded_successfully') and not evaluator.loaded_successfully:
            print(f"Skipping evaluation for {data_path} due to load error.")
            continue
        
        evaluator.run()
        del evaluator
        gc.collect()
    

    evaluator = EvaluateGraphs(args=None, prefix='', data_path='')
    table_path = Path('data/output/latex_tables/')
    table_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists

            
    for edgebank_style in ['default']:
        for node_embedding_type in ['zeros']:
            for lr_topo in [0.001]: # Renamed to avoid conflict with model lr
                for toper_type in ['False']:
                    for dataset in datasets:
                            # 1. Check for the specific TopoGED variant
                        if toper_type == 'True':
                            pred_val_str = "False"
                            vector_type_str = "TrueVals"
                        else:
                            pred_val_str = "True"
                            vector_type_str = "V-EWMA"

                        topo_folder = (
                            f"{dataset}_topoGED_embedding_mlpEncodingConcat_embeddingTypeGCN_"
                            f"lr{lr_topo}_5back_oobankchanges_{node_embedding_type}_sampling_"
                            f"predvals{pred_val_str}_tmp_edgebank_{edgebank_style}_VectorType{vector_type_str}"
                        )
                        relative_path = Path(topo_folder) / f"GCN_constructed_graphs_{dataset}.pkl"
                        target_file = topoged_base / relative_path
                        
                        if target_file.exists():
                            # Define the prefix for THIS specific TopoGED metrics folder
                            current_topoged_prefix = topoged_output_base / f"TopoGED_{dataset}_{node_embedding_type}_{lr_topo}_{edgebank_style}_{'V-EWMA' if toper_type == 'False' else 'TrueVals'}_sampling_tmp"
                            current_topoged_prefix.mkdir(parents=True, exist_ok=True)
                            
                            # 2. Collect Benchmarker Baselines for this Dataset
                            # We use the exact prefix logic you provided
                            curr_dataset_baselines = []
                            
                            benchmarker_paths = [
                                f"data/output/predicted/ROLAND/{dataset}_128_0.001_3_5_undirected_threshold_5xCap.pkl",
                                f"data/output/predicted/VGRNN/{dataset}_64_32_2_0.001_GIN_undirected_threshold_5xCap.pkl",
                                f"data/output/predicted/TGCN/{dataset}_128_0.01_0.001_undirected_threshold_5xCap.pkl",
                                # f"data/output/predicted/SFDyG/{dataset}_hgcn_7_4_0.0001_1024_100_0_threshold_5xCap.pkl",
                                f"data/output/predicted/HTGN/{dataset}_0.001_256_128_0.5_undirected_threshold_5xCap.pkl",
                                f"data/output/predicted/EvolveGCN/{dataset}_egcn_o_128_64_32_0.01_0.0001_undirected_threshold_5xCap.pkl",
                                f"data/output/predicted/GCLSTM/{dataset}_0.0001_128_2_0.001_undirected_threshold_5xCap.pkl"
                            ]

                            for b_path in benchmarker_paths:
                                b_file = pathlib.Path(b_path)
                                if b_file.exists():
                                    b_model_name = b_file.parts[3]
                                    b_prefix = output_base / b_model_name / f"{b_file.stem}_tmp"
                                    curr_dataset_baselines.append(b_prefix)
                            
                            # 3. Combine Baselines + the current TopoGED variant
                            # extra_path = Path(f"data/output/results/TopoGED_{dataset}_zeros_0.001_default_V-EWMA")

                            curr_table_inputs = curr_dataset_baselines + [current_topoged_prefix]

                            # 4. Generate the unique table
                            table_filename = f"Table_{dataset}_TopoGED_{node_embedding_type}_lr{lr_topo}_{edgebank_style}_{'V-EWMA' if toper_type == 'False' else 'TrueVals'}_sampling"
                            table_filename = f"final_tables/"
                            final_table_path = table_path / table_filename

                            print(f"Generating Table: {table_filename}")
                            evaluator.formTables(
                                curr_table_inputs, 
                                final_table_path, 
                                dataset, 
                                old_only_status=False
                            )
                        else:
                            print(f"Skipping variant: {target_file.name} not found.")
            
    evaluator.construct_ablation_tables(datasets)
    evaluator.sensitivity_analysis_len(datasets)
    evaluator.sensitivity_analysis_days(datasets)
    evaluator.generate_figures(datasets)

    # Finally, run heatmaps
# The code is calling a function `make_heatmaps()` on an object named `evaluator` to generate
# heatmaps.
    evaluator.make_heatmaps()

