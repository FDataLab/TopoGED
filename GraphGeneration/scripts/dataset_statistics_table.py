import numpy as np
import networkx as nx
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from GraphGeneration.scripts.load_data import load_data
from process_data import modifyGraphIds, build_edgebanks_from_start

if __name__ == "__main__":
    datasets = ["CollegeMsg", "mathoverflow", "networkadex", "networkaeternity", "networkaion", "networkaragon", "networkbancor", "networkcentra", "networkcindicator", "networkcoindash", "networkdgd", "networkiconomi", "Reddit_B", "tgbl-wiki"]

    results = []
    
    latex = []
    
    latex.append(r"\begin{tabular}{ l c c c c c c c c c c}")
    latex.append(r"\toprule")
    latex.append(r"\toprule")
    latex.append(r"""\textbf{Dataset} & 
        \shortstack{\textbf{Total} \\ \textbf{$|V|$}} & 
        \shortstack{\textbf{Num} \\ \textbf{Graphs}} & 
        \shortstack{\textbf{Avg} \\ \textbf{$|V|$}} & 
        \shortstack{\textbf{Avg} \\ \textbf{$|E|$}} & 
        \shortstack{\textbf{Avg \%} \\ \textbf{Old}} & 
        \shortstack{\textbf{Avg \%} \\ \textbf{New}} & 
        \shortstack{\textbf{Avg \%} \\ $\mathcal{E}^{\mathrm{oo-b}}$} & 
        \shortstack{\textbf{Avg \%} \\ $\mathcal{E}^{\mathrm{oo-nb}}$} & 
        \shortstack{\textbf{Avg \%} \\ $\mathcal{E}^{\mathrm{o-n}}$} & 
        \shortstack{\textbf{Avg \%} \\ $\mathcal{E}^{\mathrm{n-n}}$}"""
    ) 
    latex.append(r"\\ \midrule")

    for dataset in datasets:
        probabilities, graph_descriptions, thresholds, target_graphs = load_data(dataset, "", "", "", 'all', False, 10, use_test_style=None)
        target_graphs, _ = modifyGraphIds(target_graphs, thresholds, 10000)
        
        curr_results = {
            "node_count": None,
            "avg_node_count": None,
            "avg_edge_count": None,
            "num_graphs": None,
            "avg_prob_old_nodes": None,
            "avg_prob_new_nodes": None,
            "avg_prob_oobank": None,
            "avg_prob_oonobank": None,
            "avg_prob_on": None,
            "avg_prob_nn": None
        } 
        
        nodes = set()
        number_of_edges = 0
        number_of_nodes = 0
        num_graphs = len(target_graphs)
        
        for curr_graph in target_graphs:
            graph = curr_graph[-1]
            nodes.update(graph.nodes())
            number_of_edges += graph.number_of_edges()
            number_of_nodes += graph.number_of_nodes()
            
        curr_results["node_count"] = len(nodes)
        curr_results["avg_node_count"] = int(number_of_nodes / num_graphs)
        curr_results["avg_edge_count"] = int(number_of_edges / num_graphs)
        curr_results["num_graphs"] = num_graphs
        
        prob_array = np.array(probabilities)
    
        # Calculate the average across the first axis (vertical average)
        # result shape: (6,)
        avg_probs = np.mean(prob_array, axis=0)

        # Assign to your dictionary
        curr_results["avg_prob_old_nodes"] = avg_probs[0]
        curr_results["avg_prob_new_nodes"] = avg_probs[1]
        curr_results["avg_prob_oobank"]    = avg_probs[2]
        curr_results["avg_prob_oonobank"] = avg_probs[5]
        curr_results["avg_prob_on"]       = avg_probs[4]
        curr_results["avg_prob_nn"]       = avg_probs[3]
        
        latex.append(f"{dataset} & {curr_results['node_count']} & {curr_results['num_graphs']} & {curr_results['avg_node_count']} & {curr_results['avg_edge_count']} & {curr_results['avg_prob_old_nodes']:.2f} & {curr_results['avg_prob_new_nodes']:.2f} & {curr_results['avg_prob_oobank']:.2f} & {curr_results['avg_prob_oonobank']:.2f} & {curr_results['avg_prob_on']:.2f} & {curr_results['avg_prob_nn']:.2f} \\\\")    
    
    latex.append(r"\bottomrule")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    
    final_latex = "\n".join(latex)

    # Define the output path
    out_path = "latex_tables/dataset_stats.tex"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write to file
    with open(out_path, 'w') as f:
        f.write(final_latex)

    print(f"LaTeX table saved to {out_path}")