#!/bin/bash

choices=("colmap" "vggt")
nums=(1 2 4 8 10 20 30 40 50 75 100 125 150 175 200)
seeds=(43 44)
dataset="bonsai"

GT_PATH="./data/360_v2/$dataset/sparse/0"

echo "--- Starting Full Evaluation for Dataset: $dataset ---"
for choice in "${choices[@]}"; do
    for num in "${nums[@]}"; do
        for seed in "${seeds[@]}"; do
            FOLDER_NAME="${dataset}_8_n${num}_s${seed}"
            PRED_PATH="./${choice}_outputs/${FOLDER_NAME}"
            
            if [ -d "$PRED_PATH" ]; then
                echo "Evaluating $choice with $num images..."
                python evaluation.py \
                    --pred "$PRED_PATH" \
                    --gt "$GT_PATH"
            else
                echo "Skipping: $PRED_PATH (Folder not found)"
            fi
        done
    done
done
python plot_metrics.py --name ${dataset}_8
echo "--- Evaluation Complete ---"