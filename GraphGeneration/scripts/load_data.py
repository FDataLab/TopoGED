import pandas as pd 
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader


def load_data(dataset, strategy, embedding, mlpEncoding, embedOld, trainingStyle, embeddingType):
    my_loader = Loader()
    
    # Construct csv
    structure_pred_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/structure_pred.csv'
    structure_true_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/structure_true.csv'
    structure_diff_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/structure_diff.csv'
    kernel_pred_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/kernel_pred.csv'
    kernel_true_file_path = f'GraphGeneration/output/results/kernel/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/kernel_true.csv'
    edge_file_path = f'GraphGeneration/output/results/structure/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/edge_analysis.csv'
    topER_file_path = f'GraphGeneration/output/results/topER/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/toper_diff.csv'
    animation_path = f'GraphGeneration/output/results/animations/{dataset}/model_gen_retrain_{strategy}_embedding{embedding}_mlpEncoding{mlpEncoding}_embedOld{embedOld}_trainingStyle{trainingStyle}_embeddingType{embeddingType}/pred_vs_true.mp4'

    # Create file paths if needed
    for path in [structure_pred_file_path, structure_true_file_path, structure_diff_file_path, kernel_pred_file_path, 
                kernel_true_file_path, edge_file_path, topER_file_path, animation_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    columns_structure = ['Graph Number', 'Average Node Degree', 'Unique Degree Count', 'Degree Centrality', 'Assortivity Coefficient',
            'Clustering Coefficient', 'Density', 'Number of Weakly Connected Components',
            'Number of Strongly Connected Components', 'Number of Nodes', 'Number of Edges',
            'Eigenvalue_1', 'Eigenvalue_2', 'Eigenvalue_3', 'Eigenvalue_4', 'Eigenvalue_5', ]
    removed = ['Betweenness Centrality', 'Closeness Centrality', 'Number of Cliques', 'Diameter', 'Number of 3-Motifs',  'Number of Cycles', ]

    # Write the header and empty content
    pd.DataFrame(columns=columns_structure).to_csv(structure_pred_file_path, index=False)
    pd.DataFrame(columns=columns_structure).to_csv(structure_true_file_path, index=False)
    pd.DataFrame(columns=columns_structure + ['Kernel Distance']).to_csv(structure_diff_file_path, index=False)

    columns_edges = ['Graph Number', 'precision overall', 'recall overall', 'tp_overall', 'fp_overall','tn_overall','fn_overall', 'precision oo', 'recall oo', 'tp_oo', 'fp_oo','tn_oo','fn_oo', 'precision oon', 'recall oon', 'tp_oon', 'fp_oon','tn_oon','fn_oon',  'precision on', 'recall on', 'tp_on', 'fp_on','tn_on','fn_on', 'precision nn', 'recall nn', 'tp_nn', 'fp_nn','tn_nn','fn_nn', 
                        'Correct Node IDs', 'Correct Old Node IDs', 'Precision Old IDs', 'Recall Old IDs',  'Correct New Node IDs', 'Precision New IDs', 
                        'Recall New IDs', 'Correct Overall IDs', 'Precision Overall IDs', 'Recall Overall IDs']

    # Write the header and empty content
    pd.DataFrame(columns=columns_edges).to_csv(edge_file_path, index=False)

    columns_kernel = ['Subgraph 1', 'Subgraph 2', 'Subgraph 3', 'Subgraph 4']

    # Write the header and empty content
    pd.DataFrame(columns=columns_kernel).to_csv(kernel_pred_file_path, index=False)
    pd.DataFrame(columns=columns_kernel).to_csv(kernel_true_file_path, index=False)

    # Load probabilities
    probabilities_df = my_loader.load_data(type='probabilities', dataset=dataset, activation='')
    probabilities = probabilities_df.values.tolist()

    # Load all features, thresholds, and target subgraphs
    features, _ = my_loader.load_data(dataset, activation='Degree', type='features', include_weights=True)
    thresholds = my_loader.load_data(dataset, activation='Degree', type='thresholds', include_weights=True)
    target_graphs = my_loader.load_data(dataset, activation='Degree', type='subgraphs', include_weights=False)
    
    return probabilities, features, thresholds, target_graphs