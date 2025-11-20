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
import sys
import yaml
import pickle 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from GraphGeneration.utils.Evaluator import Evaluator
from load_data import load_data, generate_training_data_cached, generate_validation_data_cached, generate_negative_edges
from create_sub_graphs import create_nn_graph, create_on_graph

# Import all node embedding methods
from torch.utils.data import DataLoader

# Import Loss fn
# from GraphGeneration.scripts.composite_graphlet_loss_fn import GraphletLoss
# from GraphGeneration.utils.estimate_graphlet import run_graphlet_estimate   
# TODO Rename these ^^^

from utils.embedding_methods.degree import EmbedDegree


import warnings
from sklearn.exceptions import UndefinedMetricWarning

class EvaluateGraphs():
    def __init__(self, args):
        self.evaluator = Evaluator()
        self.args = args
        
        self.dataset = args.dataset
        self.embedding_method = args.embedding_method
        self.use_ma = getattr(args, 'use_ma', None)
        self.is_directed=False
        
        prefix = f"GraphGeneration/scripts/results/{self.dataset}/{self.embedding_method}{f'_ma{self.use_ma}' if self.use_ma else ''}/"
        self.file_visualization_path = os.path.join(prefix, "file_visualization")
        self.structure_dir = os.path.join(prefix, "structure")
        self.edge_eval_dir = os.path.join(prefix, "edge_evaluation")
        self.kernel_dir = os.path.join(prefix, "kernel_visualization")

        # Load the data
        if self.embedding_method == "hks":
            data_path = os.path.join(f"data/output/constructed_graphs/{self.dataset}_topoGED_embedding_mlpEncodingConcat_embeddingType{self.embedding_method}_{self.embedding_method}{f"_{self.use_ma}" if self.use_ma else ""}", f"{self.embedding_method}_constructed_graphs_{self.dataset}.pkl")
        elif self.embedding_method == "Node2Vec":
            data_path = os.path.join(f"data/output/constructed_graphs/{self.dataset}_topoGED_embedding_mlpEncodingConcat_embeddingType{self.embedding_method}{f"_{self.use_ma}" if self.use_ma else ""}_oobankchanges", f"{self.embedding_method}_constructed_graphs_{self.dataset}.pkl")
        else:
            data_path = os.path.join(f"data/output/constructed_graphs/{self.dataset}_topoGED_embedding_mlpEncodingConcat_embeddingType{self.embedding_method}{f"_{self.use_ma}" if self.use_ma else ""}_oobankchanges", f"{self.embedding_method}_constructed_graphs_{self.dataset}.pkl")
        data_path = "data/output/constructed_graphs/networkadex_topoGED_embeddingDegree_mlpEncodingConcat_embeddingTypeGCN_True/Node2Vec_constructed_graphs_networkadex.pkl"
        print(data_path)
        with open(data_path, 'rb') as f:
            self.pred_graphs, self.true_graphs, self.sorted_nodes_pred, self.sorted_nodes_true = pickle.load(f)
        
        
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
        
        
        # Evaluate the correctness of nodes that we have chosen (old, new, together)
        results_node_evaluation = self.evaluator.evaluate_node_selection(sorted_nodes_pred, sorted_nodes_true, graph_num=graph_num)
        
        # Evaluate the graph of just old nodes (edge types: oo, oon)
        pred_old_nodes_graph = pred_graph.subgraph(sorted_nodes_pred['old_nodes']).copy()
        true_old_nodes_graph = true_graph.subgraph(sorted_nodes_true['old_nodes']).copy()
        
        old_nodes_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_old_nodes_graph, true_old_nodes_graph, mode="total")
            
        self.evaluator.write_kl_results(path=rf'{self.file_visualization_path}/kl_results_old_nodes.txt', 
                                        value=old_nodes_kl_divergence_results, graph_num=graph_num)
        
        # Evaluate the AUC here
        results_old_nodes_edges = self.evaluator.evaluate_graph_edges(pred_old_nodes_graph, true_old_nodes_graph, is_directed=self.is_directed, graph_num=graph_num)
        
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
        pred_kernel, true_kernel, distance = self.evaluator.evaluateOrca(pred_graph, true_graph, )
                        
        # Write results
        for data, path in [
            (results_old_nodes_edges, f"{self.edge_eval_dir}/old_nodes_only.csv"),
            (results_nn_edges, f"{self.edge_eval_dir}/new_nodes_only.csv"),
            (results_on_edges, f"{self.edge_eval_dir}/on_edges_only.csv"),
            (results_all_edges, f"{self.edge_eval_dir}/all_edges.csv"),
            (results_node_evaluation, f"{self.structure_dir}/node_evaluation.csv"),
            (results_true_structure, f"{self.structure_dir}/structure_true.csv"),
            (results_pred_structure, f"{self.structure_dir}/structure_pred.csv"),
            (results_toper_diff, f"{self.structure_dir}/toper_eval.csv"),
            (pred_kernel, f"{self.kernel_dir}/kernel_pred.csv"),
            (true_kernel, f"{self.kernel_dir}/kernel_true.csv"),
            (distance, f"{self.kernel_dir}/kernel_diff.csv"),
        ]:
            pd.DataFrame([data]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
            
            
    def run(self):
        for i, (pred_graph, true_graph, sorted_nodes_pred, sorted_nodes_true) in enumerate(zip(self.pred_graphs, self.true_graphs, self.sorted_nodes_pred, self.sorted_nodes_true)):
            print(f"Evaluating graph {i}...")
            self.evaluate(pred_graph, true_graph, sorted_nodes_pred, sorted_nodes_true, graph_num=i)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Evaluation Script")

    # Add arguments
    parser.add_argument("--dataset", type=str, default="CollegeMsg",
                        help="Name of the dataset to use")
    parser.add_argument("--embedding_method", type=str, required=True, help="The node embedding method")
    parser.add_argument("--use_ma", type=bool, required=False, help="Whether to use moving average, only supported for Node2Vec and HKS")

    # Parse arguments
    args = parser.parse_args()
    evaluator = EvaluateGraphs(args)
    print("Starting evaluation...")
    evaluator.run()