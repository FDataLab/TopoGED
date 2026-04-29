#!/bin/bash

# --- Configuration ---
# SCRIPT_DIR="/home/ronan/Topological-Temporal-GFM"
# cd "$SCRIPT_DIR" || exit
# eval "$(conda shell.bash hook)"  # Solves conda errors
source /home/ronan/miniconda3/etc/profile.d/conda.sh
GENERAL_OUTPUT="data/output/pipelineResults/running.txt"
CONFIG_FILE="GraphGeneration/encoder.yaml"

DATASETS=("CollegeMsg" "mathoverflow" "networkadex" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B" "tgbl-wiki" "bitcoinotc" "bitcoinalpha" "uci-message" "Hypertext09" "ia-contact" "Enron" "radoslaw" "fb-forum" "HepPH" "HepTH")
DATASETS=("CollegeMsg" "mathoverflow" "networkadex")
USE_PRED_VALS=("true")


# --- Phase 1: Global Bayesian Searches ---
echo "Starting Bayesian Search Phases..." | tee -a "$GENERAL_OUTPUT"

GRAPH_FLAGS=("--undirected" "")
# Format: "LAYER_1 LAYER_2 CLS"
SIZE_PROFILES=("32 16 16" "64 32 32" "128 64 32")
LEARNING_RATES=(0.001 0.01) 
MODEL_LIST=('egcn_h' 'egcn_o')

conda activate evolvegcn_env

# # Our models first
for DS in "${DATASETS[@]}"; do
    for GRAPH_FLAG in "${GRAPH_FLAGS[@]}"; do
        for SIZE in "${SIZE_PROFILES[@]}"; do
            
            # Unpack the current profile into the three distinct variables
            read -r LAYER_1_FEATS LAYER_2_FEATS CLS_FEATS <<< "$SIZE"
            
            for LR in "${LEARNING_RATES[@]}"; do
                for MODEL in "${MODEL_LIST[@]}"; do
                    
                    python3 benchmarkers/evolvegcn/main_benchmarker.py \
                        --dataset "$DS" $GRAPH_FLAG \
                        --lr "$LR" \
                        --model "$MODEL" \
                        --layer_1_feats "$LAYER_1_FEATS" \
                        --layer_2_feats "$LAYER_2_FEATS" \
                        --cls_feats "$CLS_FEATS"
                        
                done
            done
        done
    done
done
