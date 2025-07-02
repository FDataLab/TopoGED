import pandas as pd
import matplotlib.pyplot as plt

def visualize_kl_results(dataset, embeddingType, edgetype):
    # Path to the .txt file
    file_path = rf"GraphGeneration\scripts\Visualize\{dataset}\{embeddingType}\kl_results_{edgetype}.txt"

    # Read the file like CSV
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Strip column names in case of whitespace
    df.columns = df.columns.str.strip()

    # Define bar colors: green if < 1, else blue
    bar_colors = ['green' if val < 1 else '#C2CCFD' for val in df['kl-divergence']]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(df['snapshot'], df['kl-divergence'], color=bar_colors, edgecolor='black')

    # Labels and title
    plt.xlabel("Snapshot")
    plt.ylabel("KL-Divergence")
    plt.title("KL-Divergence of Degree Distribution per Snapshot")
    plt.xticks(df['snapshot'])
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Save and show
    plt.savefig(f'GraphGeneration/scripts/Visualize/{dataset}/{embeddingType}/{dataset}_kl_results_{edgetype}.png')
    plt.tight_layout()
    plt.show()

    
visualize_kl_results("networkaion", "GCLSTM", 'on')