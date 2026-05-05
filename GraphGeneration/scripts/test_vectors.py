from load_data import load_data, generate_training_data_cached, generate_validation_data_cached, generate_negative_edges
from utils.loader import Loader 
from utils.embedding_methods.degree import EmbedDegree
days_back_val = 'all' 
datasets = ["CollegeMsg", "mathoverflow", "networkadex", "networkaeternity", "networkaion", "networkaragon", "networkbancor", "networkcentra", "networkcindicator", "networkcoindash", "networkdgd", "networkiconomi", "Reddit_B", "tgbl-wiki"]

for dataset in datasets:
    for vec_type in ['VECM', 'VAR', 'AES']:
        print(f'[INFO] TESTING ON DATASET: {dataset}')

        probabilities, graph_descriptions, thresholds, target_graphs = load_data(dataset, "Degree", "Concat", "GCN", days_back_val, True, 10, use_test_style=vec_type)
        true_probabilities, true_graph_descriptions, true_thresholds, true_target_graphs = load_data(dataset, "Degree", "Concat", "GCN", days_back_val, False, 10, use_test_style=None)
        loader = Loader()
        fresh_graphs = loader.read_edges(dataset)
        embedder = EmbedDegree(num_buckets=10, include_weights=False)
        fresh_embeddings, _, _ = embedder.process_graphs_for_embeddings(fresh_graphs)

        num_test_graphs = int(len(probabilities) * 0.15)
        
        # Prep data
        test_probs = probabilities[-num_test_graphs:]
        test_graph_descriptions = graph_descriptions[-num_test_graphs:]
        test_thresholds = thresholds[-num_test_graphs:]
        test_target_graphs_tmp = target_graphs[-num_test_graphs:]
        test_target_graphs = [test_target_graphs_tmp[i][-1] for i in range(len(test_target_graphs_tmp))]  # Unpack
        true_graph_descriptions = true_graph_descriptions[-num_test_graphs:]
        fresh_embeddings_test = fresh_embeddings[-num_test_graphs:]
        fresh_graphs_test = fresh_graphs[-num_test_graphs:]
        
        for i, (probs, graph_description, threshold, target_graph) in enumerate(zip(test_probs, test_graph_descriptions, test_thresholds, test_target_graphs)):
            curr_description = list(map(int, graph_description))
            true_curr_description = list(map(int, true_graph_descriptions[i]))
            print(curr_description)
            print(fresh_embeddings_test[i])
            if curr_description[-1] <= 0 or curr_description[-2] <= 0:
                print(f'[WARNING] Graphdescription at target {i} has no nodes/edges: {curr_description}')
                

