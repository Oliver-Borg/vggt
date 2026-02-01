#!/bin/bash

dataset=$1
factor=$2

python plot_metrics.py --name ${dataset}_${factor} \
    --split_param "conf_thres_value" \
    --x_axis "num_images" \
    --filter "num_images=10,20,50,100;conf_thres_value=1.5"

python plot_metrics.py --name ${dataset}_${factor} \
    --split_param "num_images" \
    --x_axis "conf_thres_value" \
    --filter "num_images=10,20,50,100"

python plot_metrics.py --name ${dataset}_${factor} \
    --split_param "step" \
    --x_axis "conf_thres_value" \
    --filter "num_images=20"