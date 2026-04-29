#!/bin/bash

# --- Configuration ---
# SCRIPT_DIR="/home/ronan/Topological-Temporal-GFM"
# cd "$SCRIPT_DIR" || exit
eval "$(conda shell.bash hook)"  # Solves conda errors

GENERAL_OUTPUT="data/output/pipelineResults/running.txt"
CONFIG_FILE="GraphGeneration/encoder.yaml"

DATASETS=("CollegeMsg" "mathoverflow" "networkadex" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B" "tgbl-wiki" "bitcoinotc" "bitcoinalpha" "uci-message" "Hypertext09" "ia-contact" "Enron" "radoslaw" "fb-forum" "HepPH" "HepTH")
DATASETS=("CollegeMsg" "mathoverflow" "networkadex" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B" "tgbl-wiki" "bitcoinotc" "bitcoinalpha" "uci-message" "Hypertext09" "ia-contact" "Enron" "radoslaw" "fb-forum" "HepPH" "HepTH")
LEARNING_RATES=(0.001)
USE_PRED_VALS=("true" "false")


# --- Phase 1: Global Bayesian Searches ---
echo "Starting Bayesian Search Phases..." | tee -a "$GENERAL_OUTPUT"

# python toper/toper_bayesian_search.py 2>&1 | tee -a "data/output/pipelineResults/toper_bayesian_search.txt"
# python toper/generate_toper_from_best.py 2>&1 | tee -a "data/output/pipelineResults/generate_toper_from_best.txt"

# python probs/probability_bayesian_search.py 2>&1 | tee -a "data/output/pipelineResults/probability_bayesian_search.txt"
# python probs/generate_probs_from_best.py 2>&1 | tee -a "data/output/pipelineResults/generate_probs_from_best.txt"
GRAPH_FLAG="--undirected"
# # Our models first
for DS in "${DATASETS[@]}"; do
    echo "  [$(date +%T)] Starting Dataset: $DS" >> "$GENERAL_OUTPUT"

    # Add Topk config here (or just in the model files)

    conda activate evolvegcn_env
    python3 benchmarkers/evolvegcn/main_benchmarker.py --dataset $DS $GRAPH_FLAG 

    conda activate gclstm_env
    python3 benchmarkers/gclstm/main.py --dataset $DS $GRAPH_FLAG

    conda activate htgn_env
    python3 benchmarkers/htgn/script/main.py --dataset $DS $GRAPH_FLAG

    conda activate roland_env
    python3 benchmarkers/roland/run/main_benchmarker.py --dataset $DS $GRAPH_FLAG

    conda activate tgcn_env
    python3 benchmarkers/tgcn/main.py --dataset $DS $GRAPH_FLAG

    conda activate vgrnn_env
    python3 benchmarkers/vgrnn/main_benchmarker.py --dataset $DS $GRAPH_FLAG

    conda activate sfdyg_env
    python3 benchmarkers/sfdyg/benchmarker_main.py --dataset $DS $GRAPH_FLAG
done
