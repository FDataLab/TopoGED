#!/bin/bash
#SBATCH --account=cakcora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --mem=250G
#SBATCH --error=slurm_results/toper_%j.err
#SBATCH --output=slurm_results/toper_%j.out
#SBATCH --job-name=TopoGED_Benchmark
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ro214340@ucf.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=highgpu

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
GENERAL_OUTPUT="data/output/pipelineResults/running.txt"
CONFIG_FILE="GraphGeneration/encoder.yaml"

conda activate temporal_env

#DATASETS=("networkiconomi")
#DATASETS=("Reddit_B")
#DATASETS=("tgbl-wiki")
#DATASETS=("networkdgd")
#DATASETS=("networkcoindash")
#DATASETS=("networkcindicator")
#DATASETS=("networkcentra")
#DATASETS=("networkbancor")
#DATASETS=("networkaragon")
#DATASETS=("networkaion")
#DATASETS=("networkaeternity")
#DATASETS=("networkadex")
#DATASETS=("mathoverflow")
DATASETS=("CollegeMsg")

LEARNING_RATES=(0.001)
USE_PRED_VALS=("true")
EMBEDDING_STYLES=("VECM")



for LR in "${LEARNING_RATES[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "Setting up parameters for $DS"

        # --- PARAMETER MAPPING ---
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



        # Update YAML
        sed -i "s/^dataset: .*/dataset: $DS/" "$CONFIG_FILE"
        sed -i "s/^[[:space:]]*lr: .*/  lr: $LR/" "$CONFIG_FILE"

        for PRED in "${USE_PRED_VALS[@]}"; do
            sed -i "s/^use_predicted_vals: .*/use_predicted_vals: $PRED/" "$CONFIG_FILE"

            for NEWNODETYPE in "degree_average" "random" "zeros"; do
                sed -i "s/^new_node_strategy: .*/new_node_strategy: \"$NEWNODETYPE\"/" "$CONFIG_FILE"
                for EMBEDDING_STYLE in "${EMBEDDING_STYLES[@]}"; do
                    sed -i "s/^use_test_style: .*/use_test_style: \"$EMBEDDING_STYLE\"/" "$CONFIG_FILE"
                    for STYLE in "default"; do
                        sed -i "s/^edgebank_style: .*/edgebank_style: \"$STYLE\"/" "$CONFIG_FILE"
                        
                        echo "Executing: python3 with Alpha=$ALPHA, Beta=$BETA, Decay=$DECAY_FACTOR"
                        python3 -u GraphGeneration/scripts/topoGED_gnn_implementation_oobankchanges_edgebank.py \
                                --dataset "$DS" \
                                --alpha $ALPHA \
                                --beta $BETA \
                                --decay_factor $DECAY_FACTOR
                    done
                done
            done
        done
        echo "Finished Dataset: $DS"
    done
done