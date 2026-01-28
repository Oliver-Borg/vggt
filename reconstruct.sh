export CUDA_VISIBLE_DEVICES=2

choices=("colmap" "vggt")
nums=(1 2 4 8 10 20 50 100 200)
dataset="bonsai"

for choice in "${choices[@]}"; do
    for num in "${nums[@]}"; do
        python -m reconstruct \
            --input ./data/360_v2/$dataset/images_8 \
            --name "${dataset}_8" \
            --choice $choice \
            --num_images $num
    done
done
