import os
import sys

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.loader import Loader
import argparse
from GraphGeneration.scripts.load_data import load_data
import matplotlib.pyplot as plt

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
probabilities, features, thresholds, target_graphs = load_data(args.dataset, args.strategy, args.embedding, args.mlpEncoding, args.embedOld, args.trainingStyle, args.embeddingType)
print(probabilities)
def visualize_edge_type_counts(dataset, max_snapshot=30):
    # Limit to available snapshots
    max_snapshot = min(max_snapshot, len(probabilities))

    # Create the sliced dataframe
    df = {
        'snapshot': [i for i in range(max_snapshot)],
        'OO-bank edges': [probabilities[i][2] for i in range(max_snapshot)],
        'OO-nobank edges': [probabilities[i][5] for i in range(max_snapshot)],
        'ON edges': [probabilities[i][4] for i in range(max_snapshot)],
        'NN edges': [probabilities[i][3] for i in range(max_snapshot)],
    }

    snapshots = df['snapshot']
    width = 0.2  # width of each bar
    x = np.arange(len(snapshots))  # base positions for each snapshot

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, df['OO-bank edges'], width=width, label='OO-bank', color='#4CAF50')
    plt.bar(x - 0.5 * width, df['OO-nobank edges'], width=width, label='OO-nobank', color='#2196F3')
    plt.bar(x + 0.5 * width, df['ON edges'], width=width, label='ON', color='#FFC107')
    plt.bar(x + 1.5 * width, df['NN edges'], width=width, label='NN', color='#F44336')

    # Labels and title
    plt.xlabel("Snapshot")
    plt.ylabel("Edge Count")
    plt.title(f"{dataset} Edge Type Counts per Snapshot")
    plt.xticks(x, snapshots)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()


    
visualize_edge_type_counts(args.dataset)