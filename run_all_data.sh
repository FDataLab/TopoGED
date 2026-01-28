#!/bin/bash

# --- Configuration ---
# SCRIPT_DIR="/home/ronan/Topological-Temporal-GFM"
# cd "$SCRIPT_DIR" || exit
GENERAL_OUTPUT="data/output/pipelineResults/running.txt"
CONFIG_FILE="GraphGeneration/encoder.yaml"

DATASETS=("CollegeMsg" "mathoverflow" "networkadex" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B" "tgbl-wiki")
LEARNING_RATES=(0.001)
USE_PRED_VALS=("true")

# Ensure base directories exist
mkdir -p "data/output/pipelineResults"

# Initialize General Output

# --- Dataset Parameter Lookup ---
get_params() {
    case $1 in
        "CollegeMsg")
            echo "0.6365 3.1867"
            ;;
        "mathoverflow")
            echo "0.6084 2.6673"
            ;;
        "networkadex")
            echo "1.8909 2.3180"
            ;;
        "networkaeternity")
            echo "1.5281 2.5341"
            ;;
        "networkaion")
            echo "1.7361 2.4002"
            ;;
        "networkaragon")
            echo "1.9114 2.2486"
            ;;
        "networkbancor")
            echo "1.9292 3.0540"
            ;;
        "networkcentra")
            echo "1.8929 2.1588"
            ;;
        "networkcindicator")
            echo "1.7161 2.2850"
            ;;
        "networkcoindash")
            echo "1.8498 2.2329"
            ;;
        "networkdgd")
            echo "1.6166 3.4156"
            ;;
        "networkiconomi")
            echo "1.8601 3.1801"
            ;;
        "Reddit_B")
            echo "1.3972 25.2509"
            ;;
        "tgbl-wiki")
            echo "0.8789 3.6224"
            ;;
    esac
}

# --- Phase 1: Global Bayesian Searches ---
echo "Starting Bayesian Search Phases..." | tee -a "$GENERAL_OUTPUT"

# python toper/toper_bayesian_search.py 2>&1 | tee -a "data/output/pipelineResults/toper_bayesian_search.txt"
# python toper/generate_toper_from_best.py 2>&1 | tee -a "data/output/pipelineResults/generate_toper_from_best.txt"

# python probs/probability_bayesian_search.py 2>&1 | tee -a "data/output/pipelineResults/probability_bayesian_search.txt"
# python probs/generate_probs_from_best.py 2>&1 | tee -a "data/output/pipelineResults/generate_probs_from_best.txt"

# # Our models first
for LR in "${LEARNING_RATES[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "  [$(date +%T)] Starting Dataset: $DS" >> "$GENERAL_OUTPUT"

        # Get dataset specific params
        PARAMS=$(get_params "$DS")
        BETA=$(echo $PARAMS | cut -d' ' -f1)
        DECAY=$(echo $PARAMS | cut -d' ' -f2)

        # Update YAML for current dataset (Note: using 2 spaces for nested keys)
        sed -i "s/^dataset: .*/dataset: $DS/" "$CONFIG_FILE"
        sed -i "s/^  lr: .*/  lr: 0.001/" "$CONFIG_FILE"
        sed -i "s/^  beta: .*/  beta: $BETA/" "$CONFIG_FILE"
        sed -i "s/^  decay_factor: .*/  decay_factor: $DECAY/" "$CONFIG_FILE"

        sed -i "s/^  lr: .*/  lr: $LR/" "$CONFIG_FILE"

        # for PRED in "${USE_PRED_VALS[@]}"; do
        #     # Update YAML Boolean (lowercase true/false for YAML parser)
        #     sed -i "s/^use_predicted_vals: .*/use_predicted_vals: $PRED/" "$CONFIG_FILE"

        #     CURR_RESULT_DIR="data/output/pipelineResults/$DS/Pred$PRED"
        #     mkdir -p "$CURR_RESULT_DIR"

        #     # python3 GraphGeneration/scripts/topoGED_gnn_implementation.py 

            python3 GraphGeneration/scripts/topoGED_gnn_implementation_oobankchanges.py
        # done

        # python3 GraphGeneration/scripts/topoGED_benchmarker.py --model ROLAND 

        # python3 GraphGeneration/scripts/topoGED_benchmarker.py --model TGCN 

        # python3 GraphGeneration/scripts/topoGED_benchmarker.py --model GCLSTM 

        # python3 GraphGeneration/scripts/topoGED_benchmarker.py --model VGAE 

        # python3 GraphGeneration/scripts/topoGED_benchmarker.py --model EvolveGCN 

        # python3 GraphGeneration/scripts/topoGED_htgn_implementation.py 
    done

    echo "Finished Dataset: $DS"
done




echo "=== Pipeline Complete: $(date) ===" >> "$GENERAL_OUTPUT"
# python GraphGeneration/scripts/evaluate_graphs.py


