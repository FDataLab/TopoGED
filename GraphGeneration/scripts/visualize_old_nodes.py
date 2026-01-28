import os
from json import encoder
from math import e
import random
import numpy as np
import argparse
import os
import sys
from sklearn.metrics import recall_score
from sqlalchemy import all_
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from sympy import use
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_metric_3d(df, metric, save_path):
    """
    Create a 3D scatter plot for a metric vs. decay_factor, alpha, beta.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        df['decay_factor'],
        df['alpha'],
        df['beta'],
        c=df[metric],
        cmap='viridis'
    )

    ax.set_xlabel('Decay Factor')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('Beta')
    ax.set_title(f'3D Scatter: {metric}')

    cb = plt.colorbar(sc)
    cb.set_label(metric)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{metric}_3d_plot.png"), dpi=300)
    plt.close()


def generate_all_plots(csv_path, output_dir):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Create directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Metrics to plot
    metrics = ['mean_f1', 'mean_recall', 'mean_precision']

    for metric in metrics:
        plot_metric_3d(df, metric, output_dir)

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    # Example usage:
    csv_path = "GraphGeneration/output/results/old_node_optimization/networkadex_equation_results.csv"              # CHANGE THIS
    output_dir = "GraphGeneration/output/results/old_node_optimization/networkadexplots/"         # CHANGE THIS

    generate_all_plots(csv_path, output_dir)