#!/bin/bash

choices=("colmap" "vggt")
dataset="bonsai"
factor=2

GT_PATH="./data/360_v2/$dataset/sparse/0"

echo "--- Starting Full Evaluation for Dataset: $dataset ---"
for choice in "${choices[@]}"; do
    FOLDER_GLOB="${dataset}_${factor}_*"
    PRED_GLOB="./${choice}_outputs/${FOLDER_GLOB}"
    echo "Evaluating $PRED_GLOB."
    python evaluation.py \
        --pred-glob "$PRED_GLOB" \
        --gt "$GT_PATH"
done
python plot_metrics.py --name ${dataset}_${factor}
echo "--- Evaluation Complete ---"