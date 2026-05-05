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

# Import all node embedding methods
from torch.utils.data import DataLoader

# Import Loss fn
# from GraphGeneration.scripts.composite_graphlet_loss_fn import GraphletLoss
# from GraphGeneration.utils.estimate_graphlet import run_graphlet_estimate   
# TODO Rename these ^^^

from utils.embedding_methods.degree import EmbedDegree
from GraphGeneration.scripts.load_data import load_data

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
            self.data_path = os.path.join(f"data/output/constructed_graphs/{self.dataset}_topoGED_embedding_mlpEncodingConcat_embeddingType{self.embedding_method}_{self.embedding_method}{f"_{self.use_ma}" if self.use_ma else ""}", f"{self.embedding_method}_constructed_graphs_{self.dataset}.pkl")
        elif self.embedding_method == "Node2Vec":
            self.data_path = os.path.join(f"data/output/constructed_graphs/{self.dataset}_topoGED_embedding_mlpEncodingConcat_embeddingType{self.embedding_method}{f"_{self.use_ma}" if self.use_ma else ""}_lr001", f"{self.embedding_method}_constructed_graphs_{self.dataset}.pkl")
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
            
            _, _, _, target_graphs = load_data(
                extracted_dataset, '', '', '', 'all', 
                use_predicted=False, num_buckets=10, use_test_style=None
            )
            
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
        
    
    def makeTable(self, df, path, caption, label):
        path = Path(path)
        # min_best_cols = ['Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 'Clustering Coefficient',
        # 'Density', 'Num Triangles', 'TopER L2 Norm', 'Extra Nodes Placed', 'Median Extra Nodes Placed', 'Missing Nodes', 'Median Missing Nodes', 'Extra Edges', 'Median Extra Edges', 'Missing Edges', 'Median Missing Edges']
        # fig, ax = plt.subplots(figsize=(28, 7), dpi=300)
        # ax.axis('off')
        
        # # Use the rounded DF for both display and comparison
        # # to ensure row_val == best_val works correctly
        # df_display = df.copy()
        
        # tbl = ax.table(cellText=df_display.values, 
        #             colLabels=df_display.columns, 
        #             rowLabels=df_display.index, 
        #             loc='center', 
        #             cellLoc='center')
        
        # tbl.auto_set_font_size(False)
        # tbl.set_fontsize(9)
        # tbl.scale(1.0, 3.0)
        
        # for j, col_name in enumerate(df_display.columns):
        #     column_data = df_display[col_name]
        #     valid_data = column_data.replace([np.inf, -np.inf], np.nan).dropna()
            
        #     if valid_data.empty:
        #         continue
                
        #     if col_name in min_best_cols:
        #         best_val = valid_data.iloc[(valid_data.abs() - 0).argsort()[:1]].values[0]
        #     else:
        #         best_val = valid_data.max()
                
        #     for i, row_val in enumerate(column_data):
        #         # Check for equality with a small epsilon or use rounded values
        #         if np.isclose(row_val, best_val, atol=1e-5):
        #             # i + 1 targets the data row (row 0 is header)
        #             # j targets the data column
        #             cell = tbl[i + 1, j]
        #             cell.set_facecolor('#E0E0E0') 
        #             cell.set_text_props(weight='bold') 

        # # Bold the Header and Row Labels
        # for (row, col), cell in tbl.get_celld().items():
        #     if row == 0 or col == -1:
        #         cell.set_text_props(weight='bold')
        
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # plt.savefig(path, bbox_inches='tight')
        # plt.close(fig)
            
        min_best_cols = [
            'Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 
            'Clustering Coefficient', 'Density', 'Num Triangles', 'TopER L2 Norm', 'Extra Nodes Placed', 
            'Median Extra Nodes Placed', 'Missing Nodes', 'Median Missing Nodes', 'Extra Edges', 
            'Median Extra Edges', 'Missing Edges', 'Median Missing Edges',
        ]

        # 1. Process Headers and Identify Groups
        processed_methods = []
        base_topo_idx = -1
        for i, m in enumerate(df.index):
            m_str = str(m)
            if m_str.startswith("oobankchanges") and "ablation" not in m_str.lower():
                name = "TopoGED"
                base_topo_idx = i
            elif "ablation" in m_str.lower():
                num_match = re.search(r'\d+', m_str)
                ab_num = num_match.group(0) if num_match else "?"
                # Stack the headers: TopoGED on top, Ablation X on bottom
                name = r"\makecell{TopoGED \\ Ablation " + ab_num + "}"
            else:
                name = m_str.replace('_', r'\_')
            processed_methods.append(name)

        # 2. Reorder: All columns except the last two, then the second-to-last, then the last (Base TopoGED)
        ablation_indices = [i for i, name in enumerate(processed_methods) if "Ablation" in name]
        other_indices = [i for i in range(len(processed_methods)) if i != base_topo_idx and i not in ablation_indices]
        
        # We ensure Base TopoGED is the absolute last index
        # The list 'others' will contain everything that isn't the final TopoGED column
        others = ablation_indices + other_indices
        final_order = others + [base_topo_idx]
        
        ordered_df = df.iloc[final_order]
        final_names = [processed_methods[i] for i in final_order]

        # 3. Start LaTeX generation
        latex = []
        latex.append(r"\begin{table*}[t]")
        latex.append(r"\centering")
        latex.append(r"\small")
        latex.append(r"\setlength{\tabcolsep}{0pt}") 

        # 4. Vertical Bar Logic (Center-Aligned)
        # l: Metric Name
        # c * (n-2): All columns except the last two
        # c | c: The last two columns separated by a bar
        num_total_data_cols = len(final_names) 

        # 2. Determine the number of columns BEFORE the vertical bar
        # Metric Name (1) + All data columns except the last one (n-1)
        # Total columns = l + (n-1) + |c = n + 1 columns total.
        num_leading_data_cols = num_total_data_cols - 1

        # 3. Construct the raw string with the fixed centering logic
        # We use r"" to avoid the syntax warnings you saw earlier.
        col_setup = "l " + "c " * (num_leading_data_cols - 1) + r"c @{\extracolsep{\fill}\hspace{1.5em}}|@{\extracolsep{\fill}\hspace{1.5em}} c"

        # 4. Generate the tabular* line
        latex.append(r"\begin{tabular*}{0.85\textwidth}{@{\extracolsep{\fill}} " + col_setup + " @{}}")

        # Header Row
        header_row = ["Metric"] + final_names
        latex.append(" & ".join(header_row) + r" \\ \midrule")

        # 5. Data Rows
        for col_name in df.columns:
            # Check if column is entirely empty/NaN strings
            if ordered_df[col_name].apply(lambda x: str(x).lower() in ['nan', 'none', '']).all():
                continue

            row_cells = [col_name.replace('_', r'\_')]
            
            # Helper to safely get numeric value for ranking from potential strings
            def get_num(x):
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return np.nan

            col_data = ordered_df[col_name].apply(get_num).replace([np.inf, -np.inf], np.nan).dropna()
            
            # Ranking Logic
            if col_name in min_best_cols:
                sorted_vals = np.sort(col_data.abs().unique())
                get_check_val = lambda x: abs(x)
            else:
                sorted_vals = np.sort(col_data.unique())[::-1]
                get_check_val = lambda x: x

            best_val = sorted_vals[0] if len(sorted_vals) > 0 else None
            second_best = sorted_vals[1] if len(sorted_vals) > 1 else None

            for _, row in ordered_df.iterrows():
                val_raw = row[col_name]
                val_num = get_num(val_raw)

                if np.isnan(val_num):
                    row_cells.append("$nan$")
                    continue
                
                check_val = get_check_val(val_num)
                # Use val_raw because it already contains your 2-decimal or integer formatting
                display_num = str(val_raw)

                if best_val is not None and np.isclose(check_val, best_val, atol=1e-5):
                    row_cells.append(r"$\mathbf{" + display_num + "}$")
                elif second_best is not None and np.isclose(check_val, second_best, atol=1e-5):
                    row_cells.append(r"$\underline{" + display_num + "}$")
                else:
                    row_cells.append(f"${display_num}$")
                    
            latex.append(" & ".join(row_cells) + r" \\")

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular*}")
        
        # Caption and Label at bottom
        latex.append(r"\caption{" + caption + "}")
        latex.append(r"\label{" + label + "}")
        latex.append(r"\end{table*}")

        # Save
        txt_path = path.with_suffix('.txt')

        os.makedirs(txt_path.parent, exist_ok=True)
        with open(txt_path, 'w') as f:
            f.write("\n".join(latex))
            
    def collect_data_for_heatmap(self, df, dataset, old_only, threshold=None, lr=None):      
        valid_metrics = [
            'Assortativity Coefficient', 'Clustering Coefficient', 'Degree Centrality', 'Density', 'Extra Edges', 
            'Extra Nodes Placed', 'Missing Edges', 'Missing Nodes', 'Num Triangles', 'Num n-n Predicted', 'Num o-n Predicted',
            'o-o-bank F1', 'o-o-bank Precision', 'o-o-bank Recall', 'o-o-nobank F1', 'o-o-nobank Precision', 'o-o-nobank Recall']
        for model_name in df.index:
            for metric_name in df.columns:
                if metric_name in valid_metrics:
                    val = df.at[model_name, metric_name]
                    self.all_results.append({
                        'Dataset': dataset,
                        'Model': model_name,
                        'Metric': metric_name,
                        'Threshold': threshold,
                        'lr': lr,
                        'Score': float(val),
                        'OldOnly': old_only
                    })
                    
    
    def create_heatmaps(self, file_path, curr_models, old_only, target_threshold=None, lr=None):
        # 1. Load base data and filter for current view/threshold
        df_base = pd.DataFrame(self.all_results)
        df_filtered = df_base[
            (df_base['OldOnly'] == old_only) & 
            (df_base['Threshold'] == target_threshold) &
            (df_base['lr'] == lr)
        ].copy()        
        df_filtered = df_base.copy()

        if df_filtered.empty:
            print(f"Skipping: No data matches Threshold {target_threshold} and OldOnly {old_only}")
            return

        # Primary methods to iterate through
        methods = ['oobankchanges']
        benchmarks = ['htgn', 'ROLAND', 'VGAE', 'TGCN', 'GCLSTM', 'EvolveGCN']
        
        for method in methods:
            curr_file_path = os.path.join(file_path, f"{method.replace('.', '')}")
            os.makedirs(curr_file_path, exist_ok=True)

            # Filter for models relevant to this specific loop
            models_to_keep = benchmarks + [method]
            curr_df = df_filtered[df_filtered['Model'].isin(models_to_keep)].copy()
            
            if curr_df.empty:
                continue
                        
            node_view = "OldNodes" if old_only else "AllNodes"
            # suffix = f"{node_view}_threshold{target_threshold}_lr{lr}"
            suffix = f"{node_view}"
            
            # Metrics where a lower value is better
            min_best_cols = [
                'Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 
                'Assortativity Coefficient', 'Clustering Coefficient', 'Density', 
                'Num Triangles', 'TopER L2 Norm', 'Extra Nodes Placed', 
                'Median Extra Nodes Placed', 'Missing Nodes', 'Median Missing Nodes', 
                'Extra Edges', 'Median Extra Edges', 'Missing Edges', 'Median Missing Edges'
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
    

    def construct_ablation_tables(self):
        datasets = [
            'CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion', 'networkaeternity', 'networkaragon', 'networkbancor', 'networkcentra', 
            'networkcoindash', 'networkiconomi', 'networkcindicator', 'networkdgd', 'Reddit_B', 'tgbl-wiki',
        ]
        for dataset in datasets:
            print(dataset)
            for lr in [0.001]:
                # for setting in ['learnedparams', 'oobankchanges']:
                for setting in ['oobankchanges']:
                    for old_only_status in [False]:
                        file_path = f'data/output/figures/ablation_study_tables_new_random/' 
                        rows = [f'{setting}']
                        # rows.extend(f'{setting}_ablation{i}' for i in range(1, 7))
                        rows.extend(f'{setting}_ablation{i}' for i in range(7, 10))
                        
                        # Get prefixes
                        data = []
                        # Base prefix
                        
                        data.append(f'GraphGeneration/scripts/results/{dataset}/{setting}_lr{lr}_5back{'_oldonly' if old_only_status else ''}/')
                        
                        # Ablation prefixes
                        # for i in range(1, 7):
                        for i in range(7, 10):
                            data.append(f'GraphGeneration/scripts/results/{dataset}/ablation/{setting}_lr{lr}_5back_ablation{i}{'_old_only' if old_only_status else ''}/')
                        
                        node_columns = ['Precision Nodes', 'Recall Nodes', 'F1 Nodes', 'Precision Old Nodes', 'Recall Old Nodes', 'F1 Old Nodes', 'Precision New Nodes', 'Recall New Nodes', 'F1 New Nodes']
                        structure_columns = ['Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 'Clustering Coefficient', 
                                    'Density', 'Num Triangles', 'TopER L2 Norm', 'Extra Nodes Placed', 'Median Extra Nodes Placed', 'Missing Nodes', 'Median Missing Nodes', 'Extra Edges', 'Median Extra Edges', 'Missing Edges',
                                    'Median Missing Edges']
                        if not old_only_status:
                            edge_columns = ['o-o-bank Precision', 'o-o-bank Recall', 'o-o-bank F1',
                                            'o-o-nobank Precision', 'o-o-nobank Recall', 'o-o-nobank F1',
                                            'Num o-n Predicted', 'Num n-n Predicted'
                                            'Edge Precision', 'Edge Recall', 'Edge F1']
                        else:
                            edge_columns = ['o-o-bank Precision', 'o-o-bank Recall', 'o-o-bank F1',
                                            'o-o-nobank Precision', 'o-o-nobank Recall', 'o-o-nobank F1',
                                            'Edge Precision', 'Edge Recall', 'Edge F1']
                        
                        node_df = pd.DataFrame(np.nan, index=rows, columns=node_columns)
                        structure_df = pd.DataFrame(np.nan, index=rows, columns=structure_columns)
                        edge_df = pd.DataFrame(np.nan, index=rows, columns=edge_columns)
                        num_eval_graphs = num_test_graphs[dataset]
                        
                        # Compute Metrics for each dataset
                        for directory in data:
                            if not os.path.exists(directory):
                                print(f"Data from: {directory} does not exist")
                                continue
                            # Get name of strategy
                            folder_name = directory.strip('/').split('/')[-1] 
                            
                            if 'ablation' in folder_name:
                                parts = folder_name.split('_')
                                model = f"{parts[0]}_{parts[3]}"
                            else:
                                model = folder_name.split('_')[0].strip()                            
                            
                            # Take the tail since we only eval test graphs here
                            node_eval = pd.read_csv(directory + 'structure/node_evaluation.csv').tail(num_eval_graphs)
                            true_structure = pd.read_csv(directory + 'structure/structure_true.csv').tail(num_eval_graphs)
                            pred_structure = pd.read_csv(directory + 'structure/structure_pred.csv').tail(num_eval_graphs)
                            toper_eval = pd.read_csv(directory + 'structure/toper_eval.csv').tail(num_eval_graphs)
                            
                            all_edges = pd.read_csv(directory + 'edge_evaluation/all_edges.csv').tail(num_eval_graphs)
                            old_only = pd.read_csv(directory + 'edge_evaluation/old_nodes_only.csv').tail(num_eval_graphs)
                            on_only = pd.read_csv(directory + 'edge_evaluation/on_edges_only.csv').tail(num_eval_graphs)
                            new_only = pd.read_csv(directory + 'edge_evaluation/new_nodes_only.csv').tail(num_eval_graphs)

                            # Add metrics
                            structure_df.at[model, 'Avg Node Degree'] = true_structure['Average Node Degree'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Average Node Degree'].replace([np.inf, -np.inf], np.nan).mean()
                            structure_df.at[model, 'Unique Degree Count'] = true_structure['Unique Degree Count'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Unique Degree Count'].replace([np.inf, -np.inf], np.nan).mean()
                            structure_df.at[model, 'Degree Centrality'] = true_structure['Degree Centrality'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Degree Centrality'].replace([np.inf, -np.inf], np.nan).mean()
                            structure_df.at[model, 'Assortativity Coefficient'] = true_structure['Assortivity Coefficient'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Assortivity Coefficient'].replace([np.inf, -np.inf], np.nan).mean()
                            structure_df.at[model, 'Clustering Coefficient'] = true_structure['Clustering Coefficient'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Clustering Coefficient'].replace([np.inf, -np.inf], np.nan).mean()
                            structure_df.at[model, 'Density'] = true_structure['Density'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Density'].replace([np.inf, -np.inf], np.nan).mean()
                            structure_df.at[model, 'Num Triangles'] = true_structure['Number of Triangles'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Number of Triangles'].replace([np.inf, -np.inf], np.nan).mean()
                                            
                                            
                            extra_node_slice = toper_eval[toper_eval['node_diff_10'] > 0]['node_diff_10']
                            structure_df.at[model, 'Extra Nodes Placed'] = extra_node_slice.sum() if not extra_node_slice.empty else 0
                            structure_df.at[model, 'Median Extra Nodes Placed'] = extra_node_slice.median() if not extra_node_slice.empty else 0
                            
                            missing_node_slice = toper_eval[toper_eval['node_diff_10'] < 0]['node_diff_10']
                            structure_df.at[model, 'Missing Nodes'] = abs(missing_node_slice.sum()) if not missing_node_slice.empty else 0
                            structure_df.at[model, 'Median Missing Nodes'] = abs(missing_node_slice.median()) if not missing_node_slice.empty else 0
                            
                            extra_edge_slice = toper_eval[toper_eval['edge_diff_10'] > 0]['edge_diff_10']
                            structure_df.at[model, 'Extra Edges'] = extra_edge_slice.sum() if not extra_edge_slice.empty else 0
                            structure_df.at[model, 'Median Extra Edges'] = extra_edge_slice.median() if not extra_edge_slice.empty else 0
                            
                            missing_edge_slice = toper_eval[toper_eval['edge_diff_10'] < 0]['edge_diff_10']
                            structure_df.at[model, 'Missing Edges'] = abs(missing_edge_slice.sum()) if not missing_edge_slice.empty else 0
                            structure_df.at[model, 'Median Missing Edges'] = abs(missing_edge_slice.median()) if not missing_edge_slice.empty else 0
                            structure_df.at[model, 'TopER L2 Norm'] = toper_eval['l2_norm'].mean()
                            
                            node_df.at[model, 'Precision Old Nodes'] = node_eval['Precision_Old'].mean()
                            node_df.at[model, 'Recall Old Nodes'] = node_eval['Recall_Old'].mean()
                            node_df.at[model, 'F1 Old Nodes'] = node_eval['F1_Old'].mean()        
                            
                            node_df.at[model, 'Precision Nodes'] = node_eval['Precision_All'].mean()
                            node_df.at[model, 'Recall Nodes'] = node_eval['Recall_All'].mean()
                            node_df.at[model, 'F1 Nodes'] = node_eval['F1_All'].mean()          
                            
                            
                            edge_df.at[model, 'Edge Precision'] = all_edges['Precision'].mean()
                            edge_df.at[model, 'Edge Recall'] = all_edges['Recall'].mean()
                            edge_df.at[model, 'Edge F1'] = all_edges['F1'].mean() 
                            
                            node_df.at[model, 'Precision New Nodes'] = node_eval['Precision_New'].mean()
                            node_df.at[model, 'Recall New Nodes'] = node_eval['Recall_New'].mean()
                            node_df.at[model, 'F1 New Nodes'] = node_eval['F1_New'].mean()
                            
                            edge_df.at[model, 'o-o-bank Precision'] = old_only['Precision_bank'].mean()
                            edge_df.at[model, 'o-o-bank Recall'] = old_only['Recall_bank'].mean()
                            edge_df.at[model, 'o-o-bank F1'] = old_only['F1_bank'].mean()          
                                
                            edge_df.at[model, 'o-o-nobank Precision'] = old_only['Precision_nobank'].mean()
                            recall_col_nobank = 'Recall_nobank' if 'Recall_nobank' in old_only.columns else 'Recall_bank_nobank'
                            edge_df.at[model, 'o-o-nobank Recall'] = old_only[recall_col_nobank].mean()
                            edge_df.at[model, 'o-o-nobank F1'] = old_only['F1_nobank'].mean()
                            
                            if not old_only_status:                    
                                edge_df.at[model, 'Num o-n Predicted'] = on_only['TP'].sum() + on_only['FP'].sum() + on_only['FN'].sum() + on_only['TN'].sum()
                                edge_df.at[model, 'Num n-n Predicted'] = new_only['TP'].sum() + new_only['FP'].sum() + new_only['FN'].sum() + new_only['TN'].sum()

                        
                        int_cols = [
                            'Extra Nodes Placed', 
                            'Median Extra Nodes Placed', 'Missing Nodes', 'Median Missing Nodes', 
                            'Extra Edges', 'Median Extra Edges', 'Missing Edges', 'Median Missing Edges',
                            'Num o-n Predicted', 'Num n-n Predicted'
                        ]

                        # 2. Apply formatting to each DataFrame
                        for df in [node_df, structure_df, edge_df]:
                            for col in df.columns:
                                if col in int_cols:
                                    # Round and convert to integer string
                                    df[col] = df[col].apply(lambda x: f"{int(round(x))}" if pd.notnull(x) else "0")
                                else:
                                    # Format as 2-decimal float string
                                    df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "0.00")
                        

                        structure_output_path = os.path.join(file_path, f"{dataset}_structure_table.png")
                        edge_output_path = os.path.join(file_path, f"{dataset}_edge_table.png")
                        node_output_path = os.path.join(file_path, f"{dataset}_node_table.png")

                        min_best_cols = ['Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 'Clustering Coefficient', 
                                'Density', 'Num Triangles', 'TopER L2 Norm', 'Extra Nodes Placed', 'Median Extra Nodes Placed', 'Missing Nodes', 'Median Missing Nodes', 'Extra Edges', 'Median Extra Edges', 'Missing Edges', 'Median Missing Edges']
                        # Not used, but for clarity still here
                        max_best_cols = ['Precision Nodes', 'Recall Nodes', 'F1 Nodes', 'Precision Old Nodes', 'Recall Old Nodes', 'F1 Old Nodes', 'Edge Precision', 'Edge Recall', 'Edge F1', 
                                            'Precision New Nodes', 'Recall New Nodes', 'F1 New Nodes',
                                            'o-o Precision', 'o-o Recall', 'o-o F1',
                                            'Num o-n Predicted', 'Num n-n Predicted']
                        

                        ablation_map = {
                            'ablation7': '-TopER',
                            'ablation8': '-probs',
                            'ablation9': '-TopER -probs'
                        }

                        target_dfs = [node_df, structure_df, edge_df]

                        for i in range(len(target_dfs)):
                            # Create a local map for the current dataframe's actual index labels
                            current_rename = {}
                            for label in target_dfs[i].index:
                                for suffix, friendly_name in ablation_map.items():
                                    if suffix in str(label):
                                        current_rename[label] = friendly_name
                            
                            # Apply the rename to the index
                            target_dfs[i].index = target_dfs[i].index.map(lambda x: current_rename.get(x, x))
                        
                        self.makeTable(node_df, node_output_path, caption=f"Node Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_node_evaluation")
                        self.makeTable(structure_df, structure_output_path, caption=f"Structure Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_structure_evaluation")
                        self.makeTable(edge_df, edge_output_path, caption=f"Edge Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_edge_evaluation")
                        gc.collect()

                        
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
            

    def sensitivity_analysis(self):
        datasets = [
            'CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion',
            'networkaeternity', 'networkaragon', 'networkbancor', 'networkcentra',
            'networkcoindash', 'networkiconomi', 'networkcindicator', 'networkdgd',
            'Reddit_B', 'tgbl-wiki']
        
        figure_path = 'data/output/figures/sensitivity_analysis_new/'
        bucket_values = [5, 10, 15, 20]
        for lr in [0.001]:
            for dataset in datasets:
                metrics = {'Degree Centrality': []}
                
                for num_buckets in bucket_values:
                    data_path = f'data/output/sensitivity_analysis/constructed_graphs/{dataset}_topoGED_embeddingDegree_mlpEncodingConcat_embeddingTypeGCN_lr{lr}_5back_oobankchanges_predvalsTrue_len{num_buckets}/GCN_constructed_graphs_{dataset}.pkl'
                    prefix = f'GraphGeneration/scripts/results/{dataset}/sensitivity_analysis/{dataset}_lr{lr}_5back_len_{num_buckets}/'

                    if os.path.exists(data_path) and not os.path.exists(prefix):
                        try:
                            evaluator = EvaluateGraphs(args=None, prefix=prefix, data_path=data_path)
                            evaluator.run()
                        except Exception as e:
                            print(f"Skipping Eval for {dataset} B{num_buckets}: {e}")
                    else:
                        print(f'Skipping eval for {data_path}')
                    
                    try:
                        true_structure = pd.read_csv(prefix + 'structure/structure_true.csv')
                        pred_structure = pd.read_csv(prefix + 'structure/structure_pred.csv')
                        
                        for metric in metrics.keys():
                            t_mean = true_structure[metric].replace([np.inf, -np.inf], np.nan).mean()
                            p_mean = pred_structure[metric].replace([np.inf, -np.inf], np.nan).mean()
                            metrics[metric].append(t_mean - p_mean)
                    except FileNotFoundError:
                        print(f"Data missing for {dataset} at {num_buckets} buckets.")
                        metrics[metric].append(None) # Keep list length consistent

                for metric in metrics.keys():
                    y_values = [v for v in metrics[metric] if v is not None]
                    x_values = [bucket_values[i] for i, v in enumerate(metrics[metric]) if v is not None]

                    if not y_values: continue

                    # figsize=(8, 6) is exactly 4:3 ratio
                    plt.figure(figsize=(8, 6), constrained_layout=True)
                    
                    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', linewidth=2, label='Error Delta')
                    plt.axhline(0, color='red', linestyle='--', alpha=0.5, label='Ideal (0)')
                    
                    plt.title(f'Sensitivity Analysis: {metric}\nDataset: {dataset}', fontsize=14)
                    plt.xlabel('Number of Buckets ($K$)', fontsize=12)
                    plt.ylabel(r'Difference ($\mu_{true} - \mu_{pred}$)', fontsize=12)
                    
                    plt.xticks(bucket_values)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()
                    
                    os.makedirs(figure_path, exist_ok=True)
                    save_name = os.path.join(figure_path, f'{dataset}_{metric.replace(" ", "_")}_sensitivity.png')
                    
                    plt.savefig(save_name, dpi=300) 
                    plt.close()
    
    
    def learned_vs_oobank(self, dataset, lr=0.001):
        out_path = f'data/output/figures/method_comparison/'
        os.makedirs(out_path, exist_ok=True)
        
        # Define the metric we are extracting
        comparison_metric = 'O-O-bank F1'
        methods = ['learnedParams', 'oobankchanges']
        
        # 1. Collect Data
        data_dict = {}
        for method in methods:
            data_path = f'GraphGeneration/scripts/results/{dataset}/{method}_lr{lr}_5back/'
            # Note: Using structure_pred as per your snippet logic
            try:
                curr_path = os.path.join(data_path, 'edge_evaluation/old_nodes_only.csv')
                print(curr_path)
                pred_df = pd.read_csv(curr_path)
                # Extract mean and round for clean LaTeX display
                val = pred_df['F1_bank'].mean()
                data_dict[method] = round(val, 2)
            except Exception as e:
                print(f"Could not load data for {method}: {e}")
                data_dict[method] = np.nan

        # 2. Structure DataFrame for the Table
        # Row = Metric, Columns = Methods
        df = pd.DataFrame([data_dict], index=[comparison_metric])
        
        # 3. LaTeX Generation Setup
        final_names = methods # These become your column headers
        ordered_df = df
        caption = f"Method Comparison for {dataset}"
        label = f"tab:comp_{dataset}"
        
        latex = []
        latex.append(r"\begin{table*}[t]")
        latex.append(r"\centering")
        latex.append(r"\small")
        latex.append(r"\setlength{\tabcolsep}{0pt}") 

        # Column setup: Metric name (l) + Method 1 (c) + Vertical Bar + Method 2 (c)
        # Result: l c | c
        col_setup = r"l c @{\extracolsep{\fill}\hspace{1.5em}}|@{\extracolsep{\fill}\hspace{1.5em}} c"
        latex.append(r"\begin{tabular*}{0.85\textwidth}{@{\extracolsep{\fill}} " + col_setup + " @{}}")

        # Header Row
        header_row = ["Metric"] + [m.replace('_', r'\_') for m in final_names]
        latex.append(" & ".join(header_row) + r" \\ \midrule")

        # 4. Data Row Logic
        for metric_name in ordered_df.index:
            row_cells = [metric_name.replace('_', r'\_')]
            
            # Determine ranking (Bold best, Underline second)
            row_vals = ordered_df.loc[metric_name].values
            valid_vals = row_vals[~np.isnan(row_vals)]
            sorted_vals = np.sort(valid_vals)[::-1] # Higher F1 is better
            
            best_val = sorted_vals[0] if len(sorted_vals) > 0 else None
            second_best = sorted_vals[1] if len(sorted_vals) > 1 else None

            for val_num in row_vals:
                if np.isnan(val_num):
                    row_cells.append("$nan$")
                    continue
                
                display_num = f"{val_num:.3f}"
                
                if best_val is not None and np.isclose(val_num, best_val):
                    row_cells.append(r"$\mathbf{" + display_num + "}$")
                elif second_best is not None and np.isclose(val_num, second_best):
                    row_cells.append(r"$\underline{" + display_num + "}$")
                else:
                    row_cells.append(f"${display_num}$")
                    
            latex.append(" & ".join(row_cells) + r" \\")

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular*}")
        latex.append(r"\caption{" + caption + "}")
        latex.append(r"\label{" + label + "}")
        latex.append(r"\end{table*}")

        # 5. Save output
        txt_path = os.path.join(out_path, f'{dataset}_comparison.txt')
        with open(txt_path, 'w') as f:
            f.write("\n".join(latex))
        
        return "\n".join(latex)

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
        latex.append(r"\begin{table*}[t]")
        latex.append(r"\centering\small")
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
        latex.append(r"\caption{Comparison of Number of True vs Predicted edges for Old-New (o-n) and New-New (n-n) nodes.}")
        latex.append(r"\end{table*}")
        
        output_path = f'data/output/figures/on_nn_comparison/'
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, 'res_table.txt')
        with open(output_path, 'w') as f:
            f.write("\n".join(latex))    
            
            
    def formTables(self, data, file_path, dataset, old_only_status):
        rows = ['TopoGED', 'HTGN', 'ROLAND', 'VGRNN', 'TGCN', 'GCLSTM', 'EvolveGCN', 'SFDyG']
        node_columns = ['Precision Nodes', 'Recall Nodes', 'F1 Nodes', 'Precision Old Nodes', 'Recall Old Nodes', 'F1 Old Nodes', 'Precision New Nodes', 'Recall New Nodes', 'F1 New Nodes']
        structure_columns = ['Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 'Clustering Coefficient',
                   'Density', 'Num Triangles', 'TopER L2 Norm', 'Extra Nodes Placed', 'Median Extra Nodes Placed', 'Missing Nodes', 'Median Missing Nodes', 'Extra Edges', 'Median Extra Edges', 'Missing Edges', 'Median Missing Edges']
        if not old_only_status:
            edge_columns = ['o-o-bank Precision', "o-o-bank Recall", "o-o-bank F1", 
                            "o-o-nobank Precision", "o-o-nobank Recall", "o-o-nobank F1",
                            "Num o-n Predicted", "Num n-n Predicted",
                            "Edge Precision", "Edge Recall", "Edge F1"]
            edge_columns = ['o-o-bank Precision', 'o-o-bank Recall', 'o-o-bank F1',
                            'o-o-nobank Precision', 'o-o-nobank Recall', 'o-o-nobank F1',
                            'Edge Precision', 'Edge Recall', 'Edge F1']
        
        node_df = pd.DataFrame(np.nan, index=rows, columns=node_columns)
        structure_df = pd.DataFrame(np.nan, index=rows, columns=structure_columns)
        edge_df = pd.DataFrame(np.nan, index=rows, columns=edge_columns)
        num_eval_graphs = num_test_graphs[dataset]
        
        # Compute Metrics for each dataset
        for directory in data:
            print(directory)
            # Get name of strategy
            folder_name = Path(directory).name 
            model = folder_name.split('_')[0].strip()
            
            if not os.path.exists(directory):
                print(f"Data from: {directory} does not exist")
                structure_df.at[model, 'Avg Node Degree'] = np.nan
                structure_df.at[model, 'Unique Degree Count'] = np.nan
                structure_df.at[model, 'Degree Centrality'] = np.nan
                structure_df.at[model, 'Assortativity Coefficient'] = np.nan
                structure_df.at[model, 'Clustering Coefficient'] = np.nan
                structure_df.at[model, 'Density'] = np.nan
                structure_df.at[model, 'Num Triangles'] = np.nan
                
                structure_df.at[model, 'Extra Nodes Placed'] = np.nan
                structure_df.at[model, 'Median Extra Nodes Placed'] = np.nan
                
                structure_df.at[model, 'Missing Nodes'] = np.nan
                structure_df.at[model, 'Median Missing Nodes'] = np.nan
                
                structure_df.at[model, 'Extra Edges'] = np.nan
                structure_df.at[model, 'Median Extra Edges'] = np.nan
                
                structure_df.at[model, 'Missing Edges'] = np.nan
                structure_df.at[model, 'Median Missing Edges'] = np.nan
                structure_df.at[model, 'TopER L2 Norm'] = np.nan
                
                node_df.at[model, 'Precision Old Nodes'] = np.nan
                node_df.at[model, 'Recall Old Nodes'] = np.nan
                node_df.at[model, 'F1 Old Nodes'] = np.nan      
                
                node_df.at[model, 'Precision Nodes'] = np.nan
                node_df.at[model, 'Recall Nodes'] = np.nan
                node_df.at[model, 'F1 Nodes'] = np.nan        
                
                
                edge_df.at[model, 'Edge Precision'] = np.nan
                edge_df.at[model, 'Edge Recall'] = np.nan
                edge_df.at[model, 'Edge F1'] = np.nan
                
                node_df.at[model, 'Precision New Nodes'] = np.nan
                node_df.at[model, 'Recall New Nodes'] = np.nan
                node_df.at[model, 'F1 New Nodes'] = np.nan
                
                edge_df.at[model, 'o-o-bank Precision'] = np.nan
                edge_df.at[model, 'o-o-bank Recall'] = np.nan
                edge_df.at[model, 'o-o-bank F1'] = np.nan       
                        
                edge_df.at[model, 'o-o-nobank Precision'] = np.nan
                # Handles typo
                edge_df.at[model, 'o-o-nobank Recall'] = np.nan
                edge_df.at[model, 'o-o-nobank F1'] = np.nan
            
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
            structure_df.at[model, 'Avg Node Degree'] = true_structure['Average Node Degree'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Average Node Degree'].replace([np.inf, -np.inf], np.nan).mean()
            structure_df.at[model, 'Unique Degree Count'] = true_structure['Unique Degree Count'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Unique Degree Count'].replace([np.inf, -np.inf], np.nan).mean()
            structure_df.at[model, 'Degree Centrality'] = true_structure['Degree Centrality'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Degree Centrality'].replace([np.inf, -np.inf], np.nan).mean()
            structure_df.at[model, 'Assortativity Coefficient'] = true_structure['Assortivity Coefficient'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Assortivity Coefficient'].replace([np.inf, -np.inf], np.nan).mean()
            structure_df.at[model, 'Clustering Coefficient'] = true_structure['Clustering Coefficient'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Clustering Coefficient'].replace([np.inf, -np.inf], np.nan).mean()
            structure_df.at[model, 'Density'] = true_structure['Density'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Density'].replace([np.inf, -np.inf], np.nan).mean()
            structure_df.at[model, 'Num Triangles'] = true_structure['Number of Triangles'].replace([np.inf, -np.inf], np.nan).mean() - pred_structure['Number of Triangles'].replace([np.inf, -np.inf], np.nan).mean()
            
            extra_node_slice = toper_eval[toper_eval['node_diff_10'] > 0]['node_diff_10']
            structure_df.at[model, 'Extra Nodes Placed'] = extra_node_slice.sum() if not extra_node_slice.empty else 0
            structure_df.at[model, 'Median Extra Nodes Placed'] = extra_node_slice.median() if not extra_node_slice.empty else 0
            
            missing_node_slice = toper_eval[toper_eval['node_diff_10'] < 0]['node_diff_10']
            structure_df.at[model, 'Missing Nodes'] = abs(missing_node_slice.sum()) if not missing_node_slice.empty else 0
            structure_df.at[model, 'Median Missing Nodes'] = abs(missing_node_slice.median()) if not missing_node_slice.empty else 0
            
            extra_edge_slice = toper_eval[toper_eval['edge_diff_10'] > 0]['edge_diff_10']
            structure_df.at[model, 'Extra Edges'] = extra_edge_slice.sum() if not extra_edge_slice.empty else 0
            structure_df.at[model, 'Median Extra Edges'] = extra_edge_slice.median() if not extra_edge_slice.empty else 0
            
            missing_edge_slice = toper_eval[toper_eval['edge_diff_10'] < 0]['edge_diff_10']
            structure_df.at[model, 'Missing Edges'] = abs(missing_edge_slice.sum()) if not missing_edge_slice.empty else 0
            structure_df.at[model, 'Median Missing Edges'] = abs(missing_edge_slice.median()) if not missing_edge_slice.empty else 0
            structure_df.at[model, 'TopER L2 Norm'] = toper_eval['l2_norm'].mean()
            
            node_df.at[model, 'Precision Old Nodes'] = node_eval['Precision_Old'].mean()
            node_df.at[model, 'Recall Old Nodes'] = node_eval['Recall_Old'].mean()
            node_df.at[model, 'F1 Old Nodes'] = node_eval['F1_Old'].mean()        
            
            node_df.at[model, 'Precision Nodes'] = node_eval['Precision_All'].mean()
            node_df.at[model, 'Recall Nodes'] = node_eval['Recall_All'].mean()
            node_df.at[model, 'F1 Nodes'] = node_eval['F1_All'].mean()          
            
            
            edge_df.at[model, 'Edge Precision'] = all_edges['Precision'].mean()
            edge_df.at[model, 'Edge Recall'] = all_edges['Recall'].mean()
            edge_df.at[model, 'Edge F1'] = all_edges['F1'].mean() 
            
            node_df.at[model, 'Precision New Nodes'] = node_eval['Precision_New'].mean()
            node_df.at[model, 'Recall New Nodes'] = node_eval['Recall_New'].mean()
            node_df.at[model, 'F1 New Nodes'] = node_eval['F1_New'].mean()
            
            edge_df.at[model, 'o-o-bank Precision'] = old_only['Precision_bank'].mean()
            edge_df.at[model, 'o-o-bank Recall'] = old_only['Recall_bank'].mean()
            edge_df.at[model, 'o-o-bank F1'] = old_only['F1_bank'].mean()          
                    
            edge_df.at[model, 'o-o-nobank Precision'] = old_only['Precision_nobank'].mean()
            # Handles typo
            recall_col_nobank = 'Recall_nobank' if 'Recall_nobank' in old_only.columns else 'Recall_bank_nobank'
            edge_df.at[model, 'o-o-nobank Recall'] = old_only[recall_col_nobank].mean()
            edge_df.at[model, 'o-o-nobank F1'] = old_only['F1_nobank'].mean()
        
            if not old_only_status:                    
                edge_df.at[model, 'Num o-n Predicted'] = on_only['TP'].sum() + on_only['FP'].sum() + on_only['FN'].sum() + on_only['TN'].sum()
                edge_df.at[model, 'Num n-n Predicted'] = new_only['TP'].sum() + new_only['FP'].sum() + new_only['FN'].sum() + new_only['TN'].sum()

        
        int_cols = [
            'Extra Nodes Placed', 
            'Median Extra Nodes Placed', 'Missing Nodes', 'Median Missing Nodes', 
            'Extra Edges', 'Median Extra Edges', 'Missing Edges', 'Median Missing Edges',
            'Num o-n Predicted', 'Num n-n Predicted'
        ]

        # 2. Apply formatting to each DataFrame
        for df in [node_df, structure_df, edge_df]:
            for col in df.columns:
                if col in int_cols:
                    # ONLY format if the value is actually a number, else keep as np.nan
                    df[col] = df[col].apply(lambda x: f"{int(round(x))}" if pd.notnull(x) else np.nan)
                else:
                    # ONLY format if the value is actually a number, else keep as np.nan
                    df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else np.nan)

            
        structure_output_path = os.path.join(file_path, f"{dataset}_structure_table.png")
        edge_output_path = os.path.join(file_path, f"{dataset}_edge_table.png")
        node_output_path = os.path.join(file_path, f"{dataset}_node_table.png")
            
        self.makeTable(node_df, node_output_path, caption=f"Node Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_node_evaluation")
        self.makeTable(structure_df, structure_output_path, caption=f"Structure Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_structure_evaluation")
        self.makeTable(edge_df, edge_output_path, caption=f"Edge Evaluation Metrics for Dataset: {dataset}", label=f"tab:{dataset}_edge_evaluation")

        # curr_threshold = float(file_path.split('Threshold')[-1].split('_')[0])
        curr_threshold = 0
        self.collect_data_for_heatmap(node_df, dataset, old_only_status, curr_threshold)
        self.collect_data_for_heatmap(structure_df, dataset, old_only_status, curr_threshold)
        self.collect_data_for_heatmap(edge_df, dataset, old_only_status, curr_threshold)
        
    def evaluateTopoGED(self, file_path=Path('/mnt/d/Downloads/TopogedResults/LatexTables/TopogedComparisons/'), old_only_status=False):
        topoged_output_base = Path('/mnt/d/Downloads/TopogedResults/Topoged/results/')
        datasets = [
            'CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion', 'networkaeternity', 
            'networkaragon', 'networkbancor', 'networkcentra', 'networkcoindash', 
            'networkiconomi', 'networkcindicator', 'networkdgd', 'Reddit_B', 'tgbl-wiki'
        ]
        
        for dataset in datasets:
            rows, combo_names, data = [], [], []
            
            # 1. Path Construction (Ensure this matches your actual folder naming convention)
            for edgebank_style in ['default', 'frequency', 'shuffle']:
                for node_embedding_type in ['zeros', 'random', 'degree_average']:
                    for lr in [0.001, 0.01]:
                        for toper_type in ['VECM', 'VAR', 'AES']:
                            folder_name = f"TopoGED_{dataset}_{node_embedding_type}_{lr}_{edgebank_style}_{toper_type}"
                            rows.append(folder_name)
                            combo_names.append(folder_name)
                            data.append(topoged_output_base / folder_name)

            # 2. Column Definitions
            node_columns = ['Precision Nodes', 'Recall Nodes', 'F1 Nodes', 'Precision Old Nodes', 'Recall Old Nodes', 'F1 Old Nodes', 'Precision New Nodes', 'Recall New Nodes', 'F1 New Nodes']
            structure_columns = ['Avg Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortativity Coefficient', 'Clustering Coefficient',
                                'Density', 'Num Triangles', 'TopER L2 Norm', 'Extra Nodes Placed', 'Median Extra Nodes Placed', 'Missing Nodes', 'Median Missing Nodes', 'Extra Edges', 'Median Extra Edges', 'Missing Edges', 'Median Missing Edges']
            
            edge_columns = ['o-o-bank Precision', 'o-o-bank Recall', 'o-o-bank F1', 
                            'o-o-nobank Precision', 'o-o-nobank Recall', 'o-o-nobank F1',
                            'Edge Precision', 'Edge Recall', 'Edge F1']
            
            if not old_only_status:
                edge_columns += ["Num o-n Predicted", "Num n-n Predicted"]

            # Initialize DataFrames
            node_df = pd.DataFrame(np.nan, index=rows, columns=node_columns)
            structure_df = pd.DataFrame(np.nan, index=rows, columns=structure_columns)
            edge_df = pd.DataFrame(np.nan, index=rows, columns=edge_columns)
            
            num_eval_graphs = num_test_graphs[dataset]

            # 3. Processing Loop
            for directory, combo_name in zip(data, combo_names):
                if not directory.exists():
                    print(f"Skipping: {directory} (Not Found)")
                    continue

                # File Map
                files = {
                    'node_eval': directory / 'structure' / 'node_evaluation.csv',
                    'true_struct': directory / 'structure' / 'structure_true.csv',
                    'pred_struct': directory / 'structure' / 'structure_pred.csv',
                    'toper_eval': directory / 'structure' / 'toper_eval.csv',
                    'all_edges': directory / 'edge_evaluation' / 'all_edges.csv',
                    'old_only': directory / 'edge_evaluation' / 'old_nodes_only.csv',
                    'on_only': directory / 'edge_evaluation' / 'on_edges_only.csv',
                    'new_only': directory / 'edge_evaluation' / 'new_nodes_only.csv'
                }

                try:
                    dfs = {k: pd.read_csv(v).tail(num_eval_graphs) for k, v in files.items() if v.exists()}
                    
                    # Structural Metrics (Differences)
                    s_true, s_pred = dfs['true_struct'], dfs['pred_struct']
                    mapping = {
                        'Avg Node Degree': ('Average Node Degree', 'Average Node Degree'),
                        'Unique Degree Count': ('Unique Degree Count', 'Unique Degree Count'),
                        'Degree Centrality': ('Degree Centrality', 'Degree Centrality'),
                        'Assortativity Coefficient': ('Assortivity Coefficient', 'Assortivity Coefficient'),
                        'Clustering Coefficient': ('Clustering Coefficient', 'Clustering Coefficient'),
                        'Density': ('Density', 'Density'),
                        'Num Triangles': ('Number of Triangles', 'Number of Triangles')
                    }
                    for col, (t_col, p_col) in mapping.items():
                        structure_df.at[combo_name, col] = s_true[t_col].mean() - s_pred[p_col].mean()

                    # TopER specific metrics
                    te = dfs['toper_eval']
                    structure_df.at[combo_name, 'Extra Nodes Placed'] = te[te['node_diff_10'] > 0]['node_diff_10'].sum()
                    structure_df.at[combo_name, 'Missing Nodes'] = abs(te[te['node_diff_10'] < 0]['node_diff_10'].sum())
                    structure_df.at[combo_name, 'Extra Edges'] = te[te['edge_diff_10'] > 0]['edge_diff_10'].sum()
                    structure_df.at[combo_name, 'Missing Edges'] = abs(te[te['edge_diff_10'] < 0]['edge_diff_10'].sum())
                    structure_df.at[combo_name, 'TopER L2 Norm'] = te['l2_norm'].mean()

                    # Node & Edge Metrics
                    ne, ae, oo = dfs['node_eval'], dfs['all_edges'], dfs['old_only']
                    node_df.loc[combo_name, ['Precision Nodes', 'Recall Nodes', 'F1 Nodes']] = [ne['Precision_All'].mean(), ne['Recall_All'].mean(), ne['F1_All'].mean()]
                    edge_df.loc[combo_name, ['Edge Precision', 'Edge Recall', 'Edge F1']] = [ae['Precision'].mean(), ae['Recall'].mean(), ae['F1'].mean()]
                    
                    # Handle typos in your CSV column names
                    recall_nobank = 'Recall_nobank' if 'Recall_nobank' in oo.columns else 'Recall_bank_nobank'
                    edge_df.loc[combo_name, ['o-o-bank Precision', 'o-o-bank Recall', 'o-o-bank F1']] = [oo['Precision_bank'].mean(), oo['Recall_bank'].mean(), oo['F1_bank'].mean()]
                    edge_df.loc[combo_name, ['o-o-nobank Precision', 'o-o-nobank Recall', 'o-o-nobank F1']] = [oo['Precision_nobank'].mean(), oo[recall_nobank].mean(), oo['F1_nobank'].mean()]

                    if not old_only_status:
                        edge_df.at[combo_name, 'Num o-n Predicted'] = dfs['on_only'][['TP','FP','FN','TN']].sum().sum()
                        edge_df.at[combo_name, 'Num n-n Predicted'] = dfs['new_only'][['TP','FP','FN','TN']].sum().sum()

                except Exception as e:
                    print(f"Error processing {combo_name}: {e}")

            # 4. Collect Heatmap Data BEFORE String Formatting
            self.collect_data_for_heatmap(node_df, dataset, old_only_status, 0)
            self.collect_data_for_heatmap(structure_df, dataset, old_only_status, 0)
            self.collect_data_for_heatmap(edge_df, dataset, old_only_status, 0)

            # 5. Final Formatting & Table Generation
            int_cols = ['Extra Nodes Placed', 'Missing Nodes', 'Extra Edges', 'Missing Edges', 'Num o-n Predicted', 'Num n-n Predicted']
            for df, p in [(node_df, 'node'), (structure_df, 'structure'), (edge_df, 'edge')]:
                formatted_df = df.copy()
                for col in formatted_df.columns:
                    if col in int_cols:
                        formatted_df[col] = formatted_df[col].apply(lambda x: f"{int(round(x))}" if pd.notnull(x) else "N/A")
                    else:
                        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                
                file_path = Path(file_path) 

                # Now when you do this, out_path is guaranteed to be a Path object
                out_path = file_path / f"{dataset}_{p}_table.png"

                # The function call remains the same, but the object passed is now correct
                self.makeTable(formatted_df, out_path, 
                            caption=f"{p.capitalize()} Metrics: {dataset}", 
                            label=f"tab:{dataset}_{p}")
            # curr_threshold = float(file_path.split('Threshold')[-1].split('_')[0])
            # curr_threshold = 0
            # self.collect_data_for_heatmap(node_df, dataset, old_only_status, curr_threshold)
            # self.collect_data_for_heatmap(structure_df, dataset, old_only_status, curr_threshold)
            # self.collect_data_for_heatmap(edge_df, dataset, old_only_status, curr_threshold)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Evaluation Script")

    # Add arguments
    parser.add_argument("--dataset", type=str, default="CollegeMsg",
                        help="Name of the dataset to use")
    parser.add_argument("--embedding_method", type=str, required=False, help="The node embedding method")
    parser.add_argument("--use_ma", type=bool, required=False, help="Whether to use moving average, only supported for Node2Vec and HKS")

    # Parse arguments
    args = parser.parse_args()
    # benchmarkers = ['htgn', 'VGAE', 'EvolveGCN', 'ROLAND', 'GCLSTM', 'TGCN']
    # #for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]:
    # if args.embedding_method in benchmarkers:
    #     for threshold in [0.6, 0.7, 0.8, 0.85]:
    #         evaluator = EvaluateGraphs(args, threshold=threshold)
    #         print("Starting evaluation...")
    #         evaluator.run()
    # else:
    #     evaluator = EvaluateGraphs(args)
    #     print("Starting evaluation...")
    #     evaluator.run()
        
    # Form the list of input graph files

    # datasets = [
    #     'CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion', 'networkaeternity', 
    #     'networkaragon', 'networkbancor', 'networkcentra', 'networkcoindash', 
    #     'networkiconomi', 'networkcindicator', 'networkdgd', 'Reddit_B', 'tgbl-wiki'
    # ]
    datasets = [
        'CollegeMsg'
    ]
    models = ['EvolveGCN', 'GCLSTM', 'HTGN', 'ROLAND', 'SFDyG', 'TGCN', 'VGRNN']

    model_template_part = {
        'EvolveGCN': '_egcn_h_64_32_32_0.001_0.0001_undirected_threshold.pkl',
        'GCLSTM': '_0.001_128_2_0.001_undirected_threshold.pkl',
        'HTGN': '_0.001_32_16_1.0_att_undirected_threshold.pkl',
        'ROLAND': '_64_0.001_2_5_undirected_threshold.pkl',
        'SFDyG': '_hgat_10_2_64_0.001_undirected_threshold.pkl',
        'TGCN': '_100_0.001_0.0015_undirected_threshold.pkl',
        'VGRNN': '_32_16_1_0.001_GCN_undirected_threshold.pkl'
    }

    # WSL uses /mnt/d/ to access your D: drive
    input_base = Path('/mnt/d/Downloads/TopogedResults/Benchmarkers/graphs/')
    output_base = Path('/mnt/d/Downloads/TopogedResults/Benchmarkers/results/')

    all_inputs = []
    all_prefixes = []

    # for model in models:
    #     search_path = input_base / model 
        
    #     if not search_path.exists():
    #         print(f"Warning: Directory {search_path} does not exist.")
    #         continue

    #     # Get all pkl files once for this model folder
    #     pkl_files = list(search_path.glob("*.pkl"))
        
    #     for dataset in datasets:
    #         target_file = None
    #         string_name = f"{dataset}{model_template_part[model]}"
            
    #         search_file = search_path / string_name
            
    #         if search_file.exists():
    #             all_inputs.append(str(search_file))
    #             folder_name = f"{model}_{string_name.removesuffix('.pkl')}"
    #             prefix_path = output_base / folder_name  # remove the pkl part to be a folder name
    #             all_prefixes.append(str(prefix_path))
    #         else:
    #             print(f"Skipping: Could not find pkl for {dataset} in {model}")
    
    # Add topoged paths to the evaluation
    topoged_base = Path('/mnt/d/Downloads/TopogedResults/Topoged/graphs/')
    topoged_output_base = Path('/mnt/d/Downloads/TopogedResults/Topoged/results/')
    datasets = [
        'CollegeMsg', 'mathoverflow', 'networkadex', 'networkaion', 'networkaeternity', 
        'networkaragon', 'networkbancor', 'networkcentra', 'networkcoindash', 
        'networkiconomi', 'networkcindicator', 'networkdgd', 'Reddit_B', 'tgbl-wiki'
    ]
    # for edgebank_style in ['default', 'frequency', 'shuffle']:
    #     for node_embedding_type in ['zeros', 'random', 'degree_average']:
    #         for lr in [0.001, 0.01]:
    #             for toper_type in ['VECM', 'VAR', 'AES']:
    #                 for dataset in datasets:
    #                     # Constructing as a Path object is cleaner
    #                     relative_path = Path(f"{dataset}_topoGED_embeddingDegree_mlpEncodingConcat_embeddingTypeGCN_lr{lr}_5back_oobankchanges_predvalsTrue_newnodestrategy{node_embedding_type}_edgebank_{toper_type}_{edgebank_style}") / f"GCN_constructed_graphs_{dataset}.pkl"
                        
    #                     target_file = topoged_base / relative_path
                        
    #                     if target_file.exists():
    #                         all_inputs.append(str(target_file))
    #                         prefix_path = topoged_output_base / f"TopoGED_{dataset}_{node_embedding_type}_{lr}_{edgebank_style}_{toper_type}"
    #                         all_prefixes.append(str(prefix_path))
    #                     else:
    #                         print(f"Skipping: File not found at {target_file}")
                        
    # Now, evaluate each            
    import gc
    for prefix, data_path in zip(all_prefixes, all_inputs):
        print(f"Processing: {prefix}")
        
        if not os.path.exists(prefix):
            evaluator = EvaluateGraphs(args=None, prefix=prefix, data_path=data_path)
        
            if hasattr(evaluator, 'loaded_successfully') and not evaluator.loaded_successfully:
                print(f"Skipping evaluation for {data_path} due to load error.")
                continue
            
            evaluator.run()
            del evaluator
            gc.collect()
    
        else:
            print(f'Already processed data for path: {prefix}')
        # Manually trigger garbage collection
        gc.collect()

    evaluator = EvaluateGraphs(args=None, prefix='', data_path='')
    table_path = Path('/mnt/d/Downloads/TopogedResults/Benchmarkers/LatexTables/')
    table_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    
    # Run evaluations in the morning
    # Assemble table code while they run
    # Need to compare topoged models against each other

    for dataset in datasets:
        curr_dataset_data_all = []
        
        for model in models:
            full_name_with_ext = f"{model}_{dataset}{model_template_part[model]}"
            folder_name = Path(full_name_with_ext).stem
            prefix_path = output_base / folder_name
            curr_dataset_data_all.append(prefix_path)
        
        
        
        # best_topoged = topoged_output_base / f"{model}_{dataset}_{node_embedding_type}_{lr}_{edgebank_style}"  # Find which one we are using
        
        # Now that the list is full of this dataset's model results, make the table
        if curr_dataset_data_all:
            print(f"Generating tables for {dataset}...")
            evaluator.formTables(curr_dataset_data_all, table_path, dataset, old_only_status=False)
            
    evaluator.evaluateTopoGED()

    # Finally, run heatmaps
    # evaluator.make_heatmaps()


