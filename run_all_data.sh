#!/bin/bash

GENERAL_OUTPUT="data/output/pipelineResults/running.txt"
CONFIG_FILE="GraphGeneration/encoder.yaml"

DATASETS=("CollegeMsg" "mathoverflow" "networkadex" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B" "tgbl-wiki" "bitcoinalpha" "bitcoinotc" "fb-forum" "HepPH" "HepTH" "Hypertext09" "ia-contact" "radoslaw" "uci-message" "enron" "tgbl-review" "tgbl-coin")
DATASETS=("CollegeMsg" "mathoverflow" "networkadex" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B" "tgbl-wiki" "bitcoinalpha" "bitcoinotc" "fb-forum" "HepPH" "Hypertext09" "ia-contact" "radoslaw" "uci-message" "enron")
DATASETS=("CollegeMsg" "mathoverflow" "networkadex" "networkaeternity" "networkaion" "networkaragon" "networkbancor" "networkcentra" "networkcindicator" "networkcoindash" "networkdgd" "networkiconomi" "Reddit_B" "tgbl-wiki" "Hypertext09" "uci-message")
DATASETS=("Reddit_B" "tgbl-wiki" "Hypertext09" "uci-message")
LEARNING_RATES=(0.001 0.01)  
USE_PRED_VALS=("true" ) # Removed comma
EMBEDDING_STYLES=("AES" "VECM" "VAR")

mkdir -p "data/output/pipelineResults"

echo "=== Pipeline Started: $(date) ===" > "$GENERAL_OUTPUT"
# python probs/probs_testing.py
# python toper/toper_testing.py

for LR in "${LEARNING_RATES[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "  [$(date +%T)] Starting Dataset: $DS" >> "$GENERAL_OUTPUT"

        # Update YAML (Using single quotes for sed to handle internal double quotes safely)
        sed -i "s/^dataset: .*/dataset: $DS/" "$CONFIG_FILE"
        sed -i "s/^[[:space:]]*lr: .*/  lr: $LR/" "$CONFIG_FILE"

        for PRED in "${USE_PRED_VALS[@]}"; do
            sed -i "s/^use_predicted_vals: .*/use_predicted_vals: $PRED/" "$CONFIG_FILE"

            # Edgebank Style Iterations
            for NEWNODETYPE in "zeros" "random" "degree_average"; do
                sed -i "s/^new_node_strategy: .*/new_node_strategy: \"$NEWNODETYPE\"/" "$CONFIG_FILE"
                for EMBEDDING_STYLE in "${EMBEDDING_STYLES[@]}"; do
                    sed -i "s/^use_test_style: .*/use_test_style: \"$EMBEDDING_STYLE\"/" "$CONFIG_FILE"
                    for STYLE in "default" "frequency" "shuffle"; do
                        sed -i "s/^edgebank_style: .*/edgebank_style: \"$STYLE\"/" "$CONFIG_FILE"
                        python3 GraphGeneration/scripts/topoGED_gnn_implementation_oobankchanges_edgebank.py
                    done

                    # python3 GraphGeneration/scripts/topoGED_gnn_implementation_oobankchanges_base3.py
                done
            done
        done
        echo "Finished Dataset: $DS"
    done
done

echo "=== Pipeline Complete: $(date) ===" >> "$GENERAL_OUTPUT"