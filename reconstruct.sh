export CUDA_VISIBLE_DEVICES=2

# nums=(1 2 4 8 10 20 30 40 50 75 100 125 150 175 200)
nums=(10 20 30 40 50 100)
# nums=(20)
dataset="bonsai"
seeds=(42 43 44)
factor=2
conf_thres_values=(1.0 1.5 2.0 3.0 4.0 5.0)
for seed in "${seeds[@]}"; do
    for num in "${nums[@]}"; do
        for conf_thres_value in "${conf_thres_values[@]}"; do
            python -m reconstruct \
                --input ./data/360_v2/$dataset/images_${factor} \
                --name "${dataset}_${factor}" \
                --choice vggt \
                --num_images $num \
                --seed $seed \
                --conf_thres_value $conf_thres_value
            python -m check_sparse ./vggt_outputs/${dataset}_${factor}_n${num}_s${seed}_c${conf_thres_value}/sparse
        done
        python -m reconstruct \
            --input ./data/360_v2/$dataset/images_${factor} \
            --name "${dataset}_${factor}" \
            --choice colmap \
            --num_images $num \
            --seed $seed
        python -m check_sparse ./colmap_outputs/${dataset}_${factor}_n${num}_s${seed}_c5.0/sparse
    done
done
