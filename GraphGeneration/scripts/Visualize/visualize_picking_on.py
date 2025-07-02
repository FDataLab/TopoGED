import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

def visualize_kl_results(dataset, embeddingType):
    # Path to the .txt file
    file_path = rf"GraphGeneration\scripts\Visualize\{dataset}\{embeddingType}\picking_nodes_on.txt"

    # Read the file like CSV
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Strip column names in case of whitespace
    df.columns = df.columns.str.strip()
    df = df[:27]
    bar_width = 0.25
    index = np.arange(len(df['snapshot']))

    plt.figure(figsize=(10, 6))
    plt.bar(index, df['precison_on'], bar_width, label='Precision')
    plt.bar(index + bar_width, df['recall_on'], bar_width, label='Recall')
    plt.bar(index + 2 * bar_width, df['f1_on'], bar_width, label='F1 Score')

    plt.xlabel('Snapshot')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1 Score over Snapshots')
    plt.xticks(index + bar_width, df['snapshot'])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f'GraphGeneration/scripts/Visualize/{dataset}/{embeddingType}/picking_nodes_on.png')
    plt.tight_layout()
    plt.show()

    
visualize_kl_results("networkaion", "GCLSTM")