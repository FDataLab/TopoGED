import os
import pandas as pd

class EvaluateConstructedGraphs():
    def __init__(self, dataset):
        self.dataset = dataset
    
    
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

    # ======================= Evaluate =======================
    def evaluate(self, pred_graph, true_graph, sorted_nodes_pred, sorted_nodes_true):
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
        
        if self.current_target_snapshot == self.starting_graph:
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
                f"{self.file_visualization_path}/{self.dataset}/{encoder_config['encoder_model']['nodeEmbeddingType']}/kl_results_old_nodes.txt",
                f"{self.file_visualization_path}/{self.dataset}/{encoder_config['encoder_model']['nodeEmbeddingType']}/kl_results_nn.txt",
                f"{self.file_visualization_path}/{self.dataset}/{encoder_config['encoder_model']['nodeEmbeddingType']}/kl_results_on.txt",
                f"{self.file_visualization_path}/{self.dataset}/{encoder_config['encoder_model']['nodeEmbeddingType']}/kl_results_overall.txt",
            ]
            
            self._setup_clean_files(directories, target_files)
        
        
        # Evaluate the correctness of nodes that we have chosen (old, new, together)
        results_node_evaluation = self.evaluator.evaluate_node_selection(sorted_nodes_pred, sorted_nodes_true, self.current_target_old_nodes, graph_num=self.current_target_snapshot)
        
        # Evaluate the graph of just old nodes (edge types: oo, oon)
        pred_old_nodes_graph = pred_graph.subgraph(sorted_nodes_pred['old_nodes']).copy()
        true_old_nodes_graph = true_graph.subgraph(sorted_nodes_true['old_nodes']).copy()
        
        old_nodes_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_old_nodes_graph, true_old_nodes_graph, mode="total")
            
        self.evaluator.write_kl_results(path=rf'{self.file_visualization_path}/{self.dataset}/{encoder_config["encoder_model"]["nodeEmbeddingType"]}/kl_results_old_nodes.txt', 
                                        value=old_nodes_kl_divergence_results, graph_num=self.current_target_snapshot)
        
        # Evaluate the AUC here
        results_old_nodes_edges = self.evaluator.evaluate_graph_edges(pred_old_nodes_graph, true_old_nodes_graph, is_directed=self.is_directed, graph_num=self.current_target_snapshot)
        
        # Evaluate the graph of just new nodes (edge types: nn)
        pred_nn_graph = pred_graph.subgraph(sorted_nodes_pred['new_nodes']).copy()
        true_nn_graph = true_graph.subgraph(sorted_nodes_true['new_nodes']).copy()

        nn_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_nn_graph, true_nn_graph, mode="total")
            
        self.evaluator.write_kl_results(path=rf'{self.file_visualization_path}/{self.dataset}/{encoder_config["encoder_model"]["nodeEmbeddingType"]}/kl_results_nn.txt', 
                                        value=nn_kl_divergence_results, graph_num=self.current_target_snapshot)
        
        # Want to evaluate AUC of these
        results_nn_edges = self.evaluator.evaluate_graph_edges(pred_nn_graph, true_nn_graph, is_directed=self.is_directed, graph_num=self.current_target_snapshot)
        
        # Evaluate the graph of just edge type on
        pred_on_graph = create_on_graph(sorted_nodes_pred["new_nodes"], sorted_nodes_pred["old_nodes"], pred_graph.copy(), is_directed=self.is_directed)
        true_on_graph = create_on_graph(sorted_nodes_true["new_nodes"], sorted_nodes_true["old_nodes"], true_graph.copy(), is_directed=self.is_directed)
        
        on_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_on_graph, true_on_graph, mode="total")
            
        self.evaluator.write_kl_results(path=rf'{self.file_visualization_path}/{self.dataset}/{encoder_config["encoder_model"]["nodeEmbeddingType"]}/kl_results_on.txt', 
                                        value=on_kl_divergence_results, graph_num=self.current_target_snapshot)
            
        # Evaluate the AUC here
        results_on_edges = self.evaluator.evaluate_graph_edges(pred_on_graph, true_on_graph, is_directed=self.is_directed, graph_num=self.current_target_snapshot)
        
        # Evaluate the graph of shared nodes (all edge types)
        results_all_edges = self.evaluator.evaluate_graph_edges(pred_graph, true_graph, is_directed=self.is_directed, graph_num=self.current_target_snapshot)
        
        # Evaluate the graphs in terms of structure
        results_true_structure = self.evaluator.evaluateSingleStructure(true_graph, graph_num=self.current_target_snapshot)
        results_pred_structure = self.evaluator.evaluateSingleStructure(pred_graph, graph_num=self.current_target_snapshot)
        
        # Evaluate the graph in terms of kl divergence
        overall_kl_divergence_results = self.evaluator.kl_divergence_graphs(pred_graph, true_graph, mode="total")

        self.evaluator.write_kl_results(path=rf'{self.file_visualization_path}/{self.dataset}/{encoder_config["encoder_model"]["nodeEmbeddingType"]}/kl_results_overall.txt', 
                                        value=overall_kl_divergence_results, graph_num=self.current_target_snapshot)
        
        # Evaluate the graph in terms of its TopER vector (Degree)
        embedder = EmbedDegree(include_weights=False)

        # Make the TopER embedding
        pred_embedding, _, _ = embedder.process_graphs_for_embeddings([pred_graph])
        pred_toper = pred_embedding[0]
        true_embedding, _, _ = embedder.process_graphs_for_embeddings([true_graph])
        true_toper = true_embedding[0]

        results_toper_diff = self.evaluator.evaluateTopER(pred_toper, true_toper, graph_num=self.current_target_snapshot)  # Get the difference
            
        # Evaluate the graph in terms of graphlet kernels
        pred_kernel, true_kernel, distance = self.evaluator.evaluateOrca(pred_graph, true_graph)
                        
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
            
            
    def load_graphs(self):
        pass 
    
    
    def run(self):
        pass 
    
    # Need this later
    """
    sorted_nodes_pred = {k: set(v) for k, v in node_types.items()}  # We get this from the construction; cast it to sets
            sorted_nodes_true = {
                'old_nodes': self.current_target_old_nodes & set(self.target_graphs[i].nodes()), 
                'new_nodes': set(self.target_graphs[i].nodes()) - self.current_target_old_nodes
            }
    """
    
    
if __name__ == "__main__":
    evaluator = EvaluateConstructedGraphs(dataset='CollegeMsg')
    evaluator.run()