#!/bin/bash
# SBATCH --account=XXXXXXX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00
#SBATCH --mem=100G
#SBATCH --error=slurm_results/rebuttal_%j.err
#SBATCH --output=slurm_results/rebuttal_%j.out
#SBATCH --job-name=TopoGED_Rebuttal
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=XXXXXXXX
#SBATCH --gres=gpu:1
#SBATCH --partition=highgpu


source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
CONFIG_FILE="GraphGeneration/encoder.yaml"
LOG_BASE_DIR="slurm_results/rebuttal_logs"
mkdir -p "$LOG_BASE_DIR"

DATASETS=("networkadex" "CollegeMsg" "mathoverflow" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B" "tgbl-wiki")


SEEDS=(42 123 456 789 101112)


conda activate temporal_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python probs/generate_probs.py
python descriptors/generate_descriptors.py

# FOR EACH SEED AND DATASET, RUN THE PIPELINE
for SEED in "${SEEDS[@]}"; do
    echo "========================================================================"
    echo "STARTING SEED: $SEED"
    echo "========================================================================"

    for DS in "${DATASETS[@]}"; do

		
        echo "========================================================================"
        echo "STARTING GLOBAL DATASET: $DS | SEED: $SEED"
        echo "========================================================================"

        sed -i "s/^dataset: .*/dataset: $DS/" "$CONFIG_FILE"

        case $DS in
            "CollegeMsg")
                ALPHA=3.885; BETA=8.809; DECAY_FACTOR=0.307 ;;
            "networkadex")
                ALPHA=2.641; BETA=8.941; DECAY_FACTOR=0.852 ;;
            "tgbl-wiki")
                ALPHA=4.671; BETA=9.324; DECAY_FACTOR=0.206 ;;
            "networkaion")
                ALPHA=1.945; BETA=8.753; DECAY_FACTOR=0.598 ;;
            "networkiconomi")
                ALPHA=2.935; BETA=9.701; DECAY_FACTOR=0.367 ;;
            "Reddit_B")
                ALPHA=3.924; BETA=9.959; DECAY_FACTOR=9.958 ;;
            "networkdgd")
                ALPHA=1.319; BETA=9.962; DECAY_FACTOR=0.306 ;;
            "networkcoindash")
                ALPHA=2.974; BETA=9.475; DECAY_FACTOR=0.629 ;;
            "networkcindicator")
                ALPHA=3.737; BETA=9.237; DECAY_FACTOR=0.510 ;;
            "networkcentra")
                ALPHA=3.252; BETA=9.777; DECAY_FACTOR=0.345 ;;
            "networkbancor")
                ALPHA=4.189; BETA=9.511; DECAY_FACTOR=0.637 ;;
            "networkaragon")
                ALPHA=3.529; BETA=9.999; DECAY_FACTOR=0.554 ;;
            "networkaeternity")
                ALPHA=3.530; BETA=9.240; DECAY_FACTOR=0.529 ;;
            "mathoverflow")
                ALPHA=5.935; BETA=7.458; DECAY_FACTOR=0.147 ;;
        esac

        sed -i "s/^ablation: .*/ablation: False/" "$CONFIG_FILE"
        sed -i "s/^sensitivity_analysis: .*/sensitivity_analysis: False/" "$CONFIG_FILE"
        sed -i "s/^use_predicted_vals: .*/use_predicted_vals: False/" "$CONFIG_FILE"
        sed -i "s/^use_test_style: .*/use_test_style: V-EWMA/" "$CONFIG_FILE"

        # 1. EvolveGCN
        LOG_FILE="${LOG_BASE_DIR}/${DS}_evolvegcn_seed${SEED}.log"
        echo "=== Running EvolveGCN ===" | tee "$LOG_FILE"
        conda activate evolvegcn_env
        python3 -u benchmarkers/evolvegcn/main_benchmarker.py \
            --dataset "$DS" --undirected \
            --lr 0.01 --model "egcn_o" \
            --layer_1_feats 128 --layer_2_feats 64 \
            --cls_feats 32 --l2_reg 0.0001 \
            --seed "$SEED" >> "$LOG_FILE" 2>&1

        # 2. GCLSTM
        LOG_FILE="${LOG_BASE_DIR}/${DS}_gclstm_seed${SEED}.log"
        echo "=== Running GCLSTM ===" | tee "$LOG_FILE"
        conda activate gclstm_env
        python3 -u benchmarkers/gclstm/main.py \
            --dataset "$DS" --undirected --hidden_dim 128 \
            --beta 0.0001 --K 2 --lr 0.001 --seed "$SEED" >> "$LOG_FILE" 2>&1

        # 3. HTGN
        LOG_FILE="${LOG_BASE_DIR}/${DS}_htgn_seed${SEED}.log"
        echo "=== Running HTGN ===" | tee "$LOG_FILE"
        conda activate htgn_env
        python3 -u benchmarkers/htgn/script/main.py \
            --dataset "$DS" --undirected --lr 0.001 \
            --hidden_dim 256 --beta 0.001 --K 3 \
            --seed "$SEED" >> "$LOG_FILE" 2>&1

        # 4. ROLAND
        LOG_FILE="${LOG_BASE_DIR}/${DS}_roland_seed${SEED}.log"
        echo "=== Running ROLAND ===" | tee "$LOG_FILE"
        conda activate roland_env
        python3 -u benchmarkers/roland/run/main_benchmarker.py \
            --dataset "$DS" --undirected --lr 0.001 \
            --num_updates_per_snapshot 5 --hidden_dim 128 \
            --num_layers 3 --seed "$SEED" >> "$LOG_FILE" 2>&1

        # 5. TGCN
        LOG_FILE="${LOG_BASE_DIR}/${DS}_tgcn_seed${SEED}.log"
        echo "=== Running TGCN ===" | tee "$LOG_FILE"
        conda activate tgcn_env
        python3 -u benchmarkers/tgcn/main.py \
            --dataset "$DS" --undirected --hidden_dim 128 \
            --lr 0.01 --lambda_loss 0.001 --window_size 3 \
            --seed "$SEED" >> "$LOG_FILE" 2>&1

        # 6. VGRNN
        LOG_FILE="${LOG_BASE_DIR}/${DS}_vgrnn_seed${SEED}.log"
        echo "=== Running VGRNN ===" | tee "$LOG_FILE"
        conda activate vgrnn_env
        python3 -u benchmarkers/vgrnn/main_benchmarker.py \
            --dataset "$DS" --undirected --conv "GIN" \
            --z_dim 32 --h_dim 64 --n_layers 2 \
            --eps 0.0000000001 --lr 0.001 --seed "$SEED" >> "$LOG_FILE" 2>&1

        sed -i "s/^use_predicted_vals: .*/use_predicted_vals: True/" "$CONFIG_FILE"

        conda activate temporal_env

        # 7. TopoGED Basic
        LOG_FILE="${LOG_BASE_DIR}/${DS}_topoged_basic_seed${SEED}_scores.log"
        echo "=== Running TopoGED Basic ===" | tee "$LOG_FILE"
        python3 -u GraphGeneration/scripts/topoGED_gnn_implementation_oobankchanges_sampling.py \
           --dataset "$DS" --alpha $ALPHA --beta $BETA \
           --decay_factor $DECAY_FACTOR --new_node_strategy "random" --seed "$SEED" --ablation False --sensitivity_analysis False --use_predicted_vals True >> "$LOG_FILE" 2>&1

        echo "FINISHED RUN FOR DATASET: $DS | SEED: $SEED"
    done
done

echo "=== ALL BATCH PIPELINES FINISHED ==="