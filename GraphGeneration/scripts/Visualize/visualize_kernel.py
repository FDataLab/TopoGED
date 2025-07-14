import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

def visualize_graphlet_diff_linechart(kernel_pred_file, kernel_true_file, dataset):
    kernel_pred = pd.read_csv(kernel_pred_file)
    kernel_true = pd.read_csv(kernel_true_file)

    assert kernel_pred.shape == kernel_true.shape, "Prediction and ground truth shapes must match"
    
    diff = kernel_pred.values - kernel_true.values  # shape: (T, 21)
    num_graphlets = diff.shape[1]
    time_steps = np.arange(diff.shape[0])

    plt.figure(figsize=(12, 6))
    
    for i in range(num_graphlets):
        plt.plot(time_steps, diff[:, i], label=f"G{i+1}", alpha=0.8)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Graphlet Count Difference (Pred - True)")
    plt.title(f"Graphlet Count Differences Over Time – {dataset}")
    plt.legend(loc='upper right', fontsize='small', ncol=2, bbox_to_anchor=(1.15, 1.0))
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-1e8, 1e8)
    plt.show()

def visualize_kernel_barplot(kernel_pred_file, kernel_true_file, dataset):
    kernel_pred = pd.read_csv(kernel_pred_file)
    kernel_true = pd.read_csv(kernel_true_file)
    
    num_frames = kernel_pred.shape[0]  # one row per time step

    fig, ax = plt.subplots(figsize=(10, 5))

    def update(frame):
        ax.clear()  # Reset the axes each frame

        true_vec = kernel_true.iloc[frame]
        pred_vec = kernel_pred.iloc[frame]
        num_graphlets = len(true_vec)
        x = np.arange(num_graphlets)

        # Bars: true and predicted
        ax.bar(x - 0.2, true_vec, width=0.4, label='True', alpha=0.7)
        ax.bar(x + 0.2, pred_vec, width=0.4, label='Predicted', alpha=0.7)

        # Update labels and title
        ax.set_title(f"Time Step {frame + 3} - Graphlet Kernel Comparison On oo-Graph {dataset}")
        ax.set_xlabel("Graphlet Type")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels([f"G{i+1}" for i in x], rotation=45)
        ax.legend()
        ax.set_ylim(0, max(max(true_vec), max(pred_vec)) * 1.2)  # dynamic y-limit

    anim = FuncAnimation(fig, update, frames=num_frames, repeat=True, interval=1000)
    anim.save('animation.gif', writer='pillow', fps=2)
    plt.show()

def call_visualize_kernel(dataset, embeddingType):
    pred_path = rf"GraphGeneration\scripts\Visualize\{dataset}\{embeddingType}\kernel_results_pred_oonn.txt"
    true_path = rf"GraphGeneration\scripts\Visualize\{dataset}\{embeddingType}\kernel_results_true_oonn.txt"

    # visualize_graphlet_diff_linechart(pred_path, true_path, dataset)
    visualize_kernel_barplot(pred_path, true_path, dataset)

call_visualize_kernel("mathoverflow", "LSTM")
