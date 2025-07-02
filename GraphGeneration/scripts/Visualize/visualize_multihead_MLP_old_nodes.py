import matplotlib.pyplot as plt
import re
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

def visualize_multihead_MLP_performance(dataset, embeddingType):
    log_path = rf"GraphGeneration\scripts\Visualize\{dataset}\{embeddingType}\multiheadMLP_performance.txt"
    # GraphGeneration\scripts\Visualize\CollegeMsg\GCLSTM\multiheadMLP_performance.txt
    # Store metrics per edge type
    loss_dict = defaultdict(list)
    auc_dict = defaultdict(list)

    with open(log_path, "r") as f:
        for line in f:
            # Pattern match each log line
            match = re.match(
                r"Epoch (\d+) \| Edge Type: ([\w\-]+) \| Train Loss: ([\d\.e\-]+) \| Train AUCROC (.+)", line
            )
            if match:
                epoch = int(match.group(1))
                edge_type = match.group(2)
                loss = float(match.group(3))
                auc_raw = match.group(4)

                if edge_type == "n-n" or edge_type == "o-n":
                    continue
                
                # Handle "inf" or "nan"
                if auc_raw.lower() == 'inf':
                    auc = float('inf')
                elif auc_raw.lower() == 'nan':
                    auc = float('nan')
                else:
                    auc = float(auc_raw)

                loss_dict[edge_type].append(loss)
                auc_dict[edge_type].append(auc)



    fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column

    # --- Plot Loss ---
    ax1 = axes[0]
    for edge_type in loss_dict:
        loss_at_5 = loss_dict[edge_type][4::5]
        ax1.plot(range(3, 3 + len(loss_at_5)), loss_at_5, label=f"{edge_type} Loss")

    # ax1b.plot(prob_new_nodes_sampled, '-o', color='purple', label='Prob NN')

    ax1.set_title("Train Loss per Edge Type")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # --- Plot AUC ---
    ax2 = axes[1]
    max_len = 0
    for edge_type in auc_dict:
        auc_at_5 = auc_dict[edge_type][4::5]
        x_vals = range(3, 3 + len(auc_at_5))
        ax2.plot(x_vals, auc_at_5, label=f"{edge_type} AUC")
        max_len = max(max_len, len(auc_at_5))

    # Set x-axis ticks once, outside the loop
    ax2.set_xticks(range(3, 3 + max_len))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ax2b.plot(prob_new_nodes_sampled, '-o', color='purple', label='Prob NN')

    ax2.set_title("Train AUCROC per Edge Type")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("AUC-ROC")
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    plt.savefig(f'GraphGeneration\scripts\Visualize\{dataset}\{embeddingType}\{dataset}_multiheadMLP_performance.png')
    plt.tight_layout()
    plt.show()
    

visualize_multihead_MLP_performance("networkaion", "GCLSTM")