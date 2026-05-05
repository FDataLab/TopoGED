#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
GENERAL_OUTPUT="data/output/pipelineResults/running.txt"
CONFIG_FILE="GraphGeneration/encoder.yaml"
DATASETS=("CollegeMsg" "mathoverflow" "networkadex" "Reddit_B" "tgbl-wiki")

# For all models
GRAPH_FLAGS=("--undirected" "")
GRAPH_FLAGS=("--undirected")
LEARNING_RATES=(0.001 0.01) 

# For EvolveGCN
EvolveGCN_SIZE_PROFILES=("32 16 16" "64 32 32" "128 64 32")  # Format: "LAYER_1 LAYER_2 CLS"
EvolveGCN_MODEL_LIST=('egcn_h' 'egcn_o')

# For GCLSTM
GCLSTM_HIDDEN_DIM_LIST=(64 128 256)
GCLSTM_K_LIST=(1 2 3)
GCLSTM_BETAS=(0.0001 0.001)


# For HTGN
HTGN_SIZE_PROFILES=("32 16" "64 32")  # Format: "NHID NOUT"
HTGN_CURVATURES=(0.5 1.0 2.0)
HTGN_AGGREGATIONS=("deg" "att")

# For ROLAND
ROLAND_HIDDEN_DIM_LIST=(32 64 128)
ROLAND_NUM_LAYERS_LIST=(2 3 4)
ROLAND_NUM_UPDATES_PER_SNAPSHOT_LIST=(5 10)

# For SFDyG
SFDyG_MODELS=("hgat" "hgcn")
SFDyG_WINDOWS=(10)
SFDyG_N_NEG_TRAIN_LIST=(1)
SFDyG_N_LAYERS_LIST=(2 3)
SFDyG_N_HIDDEN_LIST=(64 128)
SFDyG_DROPOUT_LIST=(0.1)
SFDyG_WEIGHT_DECAY_LIST=(1e-4)
SFDyG_BIAS_FLAGS=("" "--bias")
SFDyG_BN_FLAGS=("" "--bn") 
SFDyG_HEADS_LIST=(2)
SFDyG_NORM_TYPE_LIST=("snorm")
SFDyG_EPOCHS_LIST=(200)
SFDyG_EVAL_STEPS_LIST=(5)

# For TGCN
TGCN_HIDDEN_DIMS=(64 128)
TGCN_LAMBDA_LOSSES=(0.001 0.0015)
TGCN_WINDOW_SIZES=(5 10 12)

# For VGRNN
VGRNN_CONVS=("GCN" "GIN" "SAGE")
VGRNN_SIZE_PROFILES=(
    "8 16 1"   # Small
    "16 32 2"  # Medium (Standard VGRNN baseline)
    "32 64 2"  # Large
)
VGRNN_EPSS=(0.0000000001)


for DS in "${DATASETS[@]}"; do
    conda activate evolvegcn_env

    # # Our models first
    if [ "$DS" == "networkaeternity" ]; then
        echo "Skipping models that did networkaeternity"
    else
        for GRAPH_FLAG in "${GRAPH_FLAGS[@]}"; do
            for SIZE in "${EvolveGCN_SIZE_PROFILES[@]}"; do
                
                # Unpack the current profile into the three distinct variables
                read -r LAYER_1_FEATS LAYER_2_FEATS CLS_FEATS <<< "$SIZE"
                
                for LR in "${LEARNING_RATES[@]}"; do
                    for MODEL in "${EvolveGCN_MODEL_LIST[@]}"; do
                        
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
    fi

        conda activate gclstm_env

        # # Our models first
        for GRAPH_FLAG in "${GRAPH_FLAGS[@]}"; do
            for HIDDEN_DIM in "${GCLSTM_HIDDEN_DIM_LIST[@]}"; do
                for BETA in "${GCLSTM_BETAS[@]}"; do
                    for K in "${GCLSTM_K_LIST[@]}"; do
                        for LR in "${LEARNING_RATES[@]}"; do
                            python3 benchmarkers/gclstm/main.py --dataset "$DS" $GRAPH_FLAG --hidden_dim "$HIDDEN_DIM" --beta "$BETA" --K "$K" --lr "$LR"
                        done
                    done
                done
            done
        done

        conda activate htgn_env

        # # Our models first
        for GRAPH_FLAG in "${GRAPH_FLAGS[@]}"; do
            for SIZE in "${HTGN_SIZE_PROFILES[@]}"; do
                
                # Unpack the current profile into the two variables
                read -r NHID NOUT <<< "$SIZE"
                
                for CURVATURE in "${HTGN_CURVATURES[@]}"; do
                    for LR in "${LEARNING_RATES[@]}"; do
                        for AGGREGATION in "${HTGN_AGGREGATIONS[@]}"; do
                            
                            python3 benchmarkers/htgn/script/main.py \
                                --dataset "$DS" $GRAPH_FLAG \
                                --lr "$LR" \
                                --nhid "$NHID" \
                                --nout "$NOUT" \
                                --curvature "$CURVATURE" \
                                --aggregation "$AGGREGATION"
                                
                        done
                    done
                done
            done
        done

        conda activate roland_env  

        # # Our models first
        for GRAPH_FLAG in "${GRAPH_FLAGS[@]}"; do
            for HIDDEN_DIM in "${ROLAND_HIDDEN_DIM_LIST[@]}"; do
                for NUM_LAYERS in "${ROLAND_NUM_LAYERS_LIST[@]}"; do
                    for NUM_UPDATES_PER_SNAPSHOT in "${ROLAND_NUM_UPDATES_PER_SNAPSHOT_LIST[@]}"; do
                        for LR in "${LEARNING_RATES[@]}"; do                    
                            python3 benchmarkers/roland/run/main_benchmarker.py --dataset "$DS" $GRAPH_FLAG --lr "$LR" --num_updates_per_snapshot "$NUM_UPDATES_PER_SNAPSHOT" --hidden_dim "$HIDDEN_DIM" --num_layers "$NUM_LAYERS"
                        done
                    done
                done
            done
        done
    

    conda activate sfdyg_env

    # # Our models first
    for MODEL in "${SFDyG_MODELS[@]}"; do
        if [[ "$MODEL" == "hgat" && "$DS" == "CollegeMsg" ]]; then
            continue
        fi
        for UNDIRECTED in "${GRAPH_FLAGS[@]}"; do
            for WINDOW in "${SFDyG_WINDOWS[@]}"; do
            for N_NEG_TRAIN in "${SFDyG_N_NEG_TRAIN_LIST[@]}"; do
                for N_LAYERS in "${SFDyG_N_LAYERS_LIST[@]}"; do
                for N_HIDDEN in "${SFDyG_N_HIDDEN_LIST[@]}"; do
                    for DROPOUT in "${SFDyG_DROPOUT_LIST[@]}"; do
                    for LR in "${LEARNING_RATES[@]}"; do
                        for WEIGHT_DECAY in "${SFDyG_WEIGHT_DECAY_LIST[@]}"; do
                        for BIAS in "${SFDyG_BIAS_FLAGS[@]}"; do
                            for BN in "${SFDyG_BN_FLAGS[@]}"; do
                            for HEADS in "${SFDyG_HEADS_LIST[@]}"; do
                                for NORM_TYPE in "${SFDyG_NORM_TYPE_LIST[@]}"; do
                                for EPOCHS in "${SFDyG_EPOCHS_LIST[@]}"; do
                                    for EVAL_STEPS in "${SFDyG_EVAL_STEPS_LIST[@]}"; do

                                    echo "Running: DS=$DS, Model=$MODEL, Layers=$N_LAYERS, Hidden=$N_HIDDEN, LR=$LR"

                                    python3 benchmarkers/sfdyg/benchmarker_main.py \
                                        --dataset "$DS" \
                                        --model "$MODEL" \
                                        $UNDIRECTED \
                                        --window "$WINDOW" \
                                        --n_neg_train "$N_NEG_TRAIN" \
                                        --n_layers "$N_LAYERS" \
                                        --n_hidden "$N_HIDDEN" \
                                        --dropout "$DROPOUT" \
                                        --lr "$LR" \
                                        --weight_decay "$WEIGHT_DECAY" \
                                        $BIAS \
                                        $BN \
                                        --heads "$HEADS" \
                                        --norm_type "$NORM_TYPE" \
                                        --epochs "$EPOCHS" \
                                        --eval_steps "$EVAL_STEPS"

                                    done
                                done
                                done
                            done
                            done
                        done
                        done
                    done
                    done
                done
                done
            done
            done
        done
    done



    conda activate tgcn_env


    # # Our models first
    for GRAPH_FLAG in "${GRAPH_FLAGS[@]}"; do
        for HIDDEN_DIM in "${TGCN_HIDDEN_DIMS[@]}"; do
            for LR in "${LEARNING_RATES[@]}"; do
                for LAMBDA_LOSS in "${TGCN_LAMBDA_LOSSES[@]}"; do
                    for WINDOW_SIZE in "${TGCN_WINDOW_SIZES[@]}"; do
                        python3 benchmarkers/tgcn/main.py --dataset "$DS" $GRAPH_FLAG --hidden_dim "$HIDDEN_DIM" --lr "$LR" --lambda_loss "$LAMBDA_LOSS" --window_size "$WINDOW_SIZE"
                    done
                done
            done
        done
    done


    conda activate vgrnn_env
    # # Our models first
    for CONV in "${VGRNN_CONVS[@]}"; do
        for UNDIRECTED in "${GRAPH_FLAGS[@]}"; do
            for SIZE in "${VGRNN_SIZE_PROFILES[@]}"; do
                for EPS in "${VGRNN_EPSS[@]}"; do
                    for LR in "${LEARNING_RATES[@]}"; do
                        # Unpack the size profile
                        read -r Z_DIM H_DIM N_LAYERS <<< "$SIZE"
                        
                        echo "Running VGRNN | Conv: $CONV | Size: $SIZE | LR: $LR"

                        python3 benchmarkers/vgrnn/main_benchmarker.py \
                            --dataset "$DS" \
                            $UNDIRECTED \
                            --conv "$CONV" \
                            --z_dim "$Z_DIM" \
                            --h_dim "$H_DIM" \
                            --n_layers "$N_LAYERS" \
                            --eps "$EPS" \
                            --lr "$LR"    
                    done
                done
            done
        done
    done
done