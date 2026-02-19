export CUDA_VISIBLE_DEVICES=3

# nums=(1 2 4 8 10 20 30 40 50 75 100 125 150 175 200)
# nums=(10 20 30 40 50 100)
nums=(50)

# dataset="360_v2"
# scene="bonsai"
# suffix=images_${factor}
# factor=2
dataset="nerf_synthetic"
scene="lego"
suffix=train
factor=1

seeds=(42)  #  43 44
conf_thres_values=(0.0)  #  1.5 2.0 3.0 4.0 5.0
num_points_values=(1000 5000 10000 20000 30000)  #  50000 75000 100000 200000 500000
input=./data/$dataset/$scene/$suffix
sampling_modes=("voxels")  # "random" "confidence" 

for seed in "${seeds[@]}"; do
    for num in "${nums[@]}"; do
        for num_points_value in "${num_points_values[@]}"; do
            for conf_thres_value in "${conf_thres_values[@]}"; do
                for sampling_mode in "${sampling_modes[@]}"; do

                    echo "Running command: python -m reconstruct"
                    echo "  --input $input"
                    echo "  --name \"${scene}_${factor}\""
                    echo "  --choice vggt"
                    echo "  --num_images $num"
                    echo "  --num_points $num_points_value"
                    echo "  --seed $seed"
                    echo "  --conf_thres_value $conf_thres_value"
                    echo "  --sampling_mode $sampling_mode"
                    echo "  --force"

                    python -m reconstruct \
                        --input $input \
                        --name "${scene}_${factor}" \
                        --choice vggt \
                        --num_images $num \
                        --num_points $num_points_value \
                        --seed $seed \
                        --conf_thres_value $conf_thres_value \
                        --sampling_mode $sampling_mode  # \
                        # --force
                done
            done
        done
        python -m reconstruct \
            --input $input \
            --name "${scene}_${factor}" \
            --choice colmap \
            --num_images $num \
            --seed $seed
    done
done
