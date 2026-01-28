import os
import sys

import numpy as np
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.loader import Loader
import argparse
from GraphGeneration.scripts.load_data import load_data
import matplotlib.pyplot as plt
from GraphGeneration.scripts.process_data import modifyGraphIds, build_edgebanks_from_start

# Load YAML config
with open("GraphGeneration/encoder.yaml", "r") as file:
    encoder_config = yaml.safe_load(file)
    print(encoder_config)

# Load all the snapshot true data 
probabilities, graph_descriptions, thresholds, target_graphs = load_data(encoder_config["dataset"], encoder_config["encoder_model"]["addOnFeature"], 
                                                                                                encoder_config["decoder_model"]["encode_links"], encoder_config["encoder_model"]["nodeEmbeddingType"])

# Modify the graph ids to 1,2,3,...
target_graphs, _ = modifyGraphIds(target_graphs, thresholds)

# Build the edgebanks for construction
all_edgebanks = build_edgebanks_from_start(target_graphs) 
print(all_edgebanks[1])
for snapshot in range(len(target_graphs)):
    count = 0
    # Get the last graphs from up to 5 previous snapshots
    prev_graphs = [graph[-1] for graph in target_graphs[max(snapshot - 5, 0): snapshot]]
    prev_edges = set().union(*(graph.edges() for graph in prev_graphs))

    # Count overlapping edges
    for edge in target_graphs[snapshot][-1].edges():
        if edge in prev_edges:
            count += 1

    # Get total edgebank size for current snapshot
    total_edgebank = sum(len(all_edgebanks[snapshot][u]) for u in all_edgebanks[snapshot])

    # Write the result
    with open("graph_analysis/analyze_oobank_count.txt", "a") as f:
        f.write(
            f"Snapshot: {snapshot}, "
            f"EdgeBank: {total_edgebank}, "
            f"#oobank_probs: {probabilities[snapshot][2]}, "
            f"#oobank_re_cal: {count}\n"
        )