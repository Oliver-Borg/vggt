export CUDA_VISIBLE_DEVICES=2

choices=("colmap" "vggt")
nums=(1 2 4 8 10 20 30 40 50 75 100 125 150 175 200)
dataset="bonsai"
seeds=(43 44)

for choice in "${choices[@]}"; do
    for num in "${nums[@]}"; do
        for seed in "${seeds[@]}"; do
            python -m reconstruct \
                --input ./data/360_v2/$dataset/images_8 \
                --name "${dataset}_8" \
                --choice $choice \
                --num_images $num \
                --seed $seed
        done
    done
done
