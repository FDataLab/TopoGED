#!/bin/bash

eval "$(conda shell.bash hook)"

GENERAL_OUTPUT="data/output/pipelineResults/running.txt"
mkdir -p "$(dirname "$GENERAL_OUTPUT")"

DATASETS=("CollegeMsg" "mathoverflow" "networkadex" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B" "tgbl-wiki")

# Start redirection block
{
echo "Run started at: $(date)"

for DS in "${DATASETS[@]}"; do
    echo "STARTING DATASET: $DS"

    # --- EvolveGCN ---
    conda activate evolvegcn_env
    python3 -u benchmarkers/evolvegcn/main_benchmarker.py \
        --dataset "$DS" --undirected \
        --lr 0.01 --model "egcn_o" --layer_1_feats 128 --layer_2_feats 64 --cls_feats 32 --l2_reg 0.0001

    # --- GC-LSTM ---
    conda activate gclstm_env
    python3 -u benchmarkers/gclstm/main.py --dataset "$DS" --undirected --hidden_dim 128 --beta 0.0001 --K 2 --lr 0.001

    # --- HTGN ---
    conda activate htgn_env
    python3 -u benchmarkers/htgn/script/main.py \
        --dataset "$DS" --undirected --lr 0.001 --nhid 256 --nout 128 --curvature 0.5 --aggregation "deg"

    # --- ROLAND ---
    conda activate roland_env  
    python3 -u benchmarkers/roland/run/main_benchmarker.py --dataset "$DS" --undirected --lr 0.001 --num_updates_per_snapshot 5 --hidden_dim 128 --num_layers 3

    # --- TGCN ---
    conda activate tgcn_env
    python3 -u benchmarkers/tgcn/main.py --dataset "$DS" --undirected --hidden_dim 128 --lr 0.01 --lambda_loss 0.001 --window_size 3

    # --- VGRNN ---
    conda activate vgrnn_env
    python3 -u benchmarkers/vgrnn/main_benchmarker.py \
        --dataset "$DS" --undirected --conv "GIN" --z_dim 32 --h_dim 64 --n_layers 2 --eps 0.0000000001 --lr 0.001    

    echo "========================================================================"
    echo "COMPLETED DATASET: $DS"
    echo "========================================================================"
done

echo "Run finished at: $(date)"

} 2>&1 | tee "$GENERAL_OUTPUT"