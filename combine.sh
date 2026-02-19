# dataset="360_v2"
# scene="bonsai"
# suffix=images_${factor}
# factor=2

dataset="nerf_synthetic"
scene="lego"
suffix=train
factor=1
GT="./data/nerf_synthetic/lego/transforms_train.json"


num=30
seed=42
conf_thres_value=0.0
num_points_value=30000
input=./data/$dataset/$scene/$suffix
sampling_mode="voxels"  # "random" "confidence" "voxels" "ba"

export CUDA_VISIBLE_DEVICES=0
python -m reconstruct \
    --input $input \
    --name "${scene}_${factor}" \
    --choice vggt \
    --num_images $num \
    --num_points $num_points_value \
    --seed $seed \
    --conf_thres_value $conf_thres_value \
    --sampling_mode $sampling_mode

python -m reconstruct \
    --input $input \
    --name "${scene}_${factor}" \
    --choice colmap \
    --num_images $num \
    --seed $seed

if [ $sampling_mode == "ba" ]; then
    NAME="${scene}_${factor}_n${num}_s${seed}_${sampling_mode}"
else
    NAME="${scene}_${factor}_n${num}_s${seed}_c${conf_thres_value}_p${num_points_value}_${sampling_mode}"
fi


VGGT="./vggt_outputs/$NAME/sparse"
COLMAP="./colmap_outputs/${scene}_${factor}_n${num}_s${seed}/sparse/0"
OUTDIR="./test/$NAME"


IMAGE_DIR="$VGGT/../images"
mkdir -p $OUTDIR

python combine_clouds.py --camera_source $VGGT --point_source $COLMAP --output_dir $OUTDIR/vggt-cam-colmap-pcd/sparse
python combine_clouds.py --camera_source $VGGT --point_source $VGGT --output_dir $OUTDIR/vggt-cam-vggt-pcd/sparse
python combine_clouds.py --camera_source $COLMAP --point_source $COLMAP --output_dir $OUTDIR/colmap-cam-colmap-pcd/sparse
python combine_clouds.py --camera_source $COLMAP --point_source $VGGT --output_dir $OUTDIR/colmap-cam-vggt-pcd/sparse
python combine_clouds.py --camera_source $GT --point_source $COLMAP --output_dir $OUTDIR/gt-cam-colmap-pcd/sparse
python combine_clouds.py --camera_source $GT --point_source $VGGT --output_dir $OUTDIR/gt-cam-vggt-pcd/sparse

cp -r $IMAGE_DIR $OUTDIR/vggt-cam-colmap-pcd/images
cp -r $IMAGE_DIR $OUTDIR/vggt-cam-vggt-pcd/images
cp -r $IMAGE_DIR $OUTDIR/colmap-cam-colmap-pcd/images
cp -r $IMAGE_DIR $OUTDIR/colmap-cam-vggt-pcd/images
cp -r $IMAGE_DIR $OUTDIR/gt-cam-colmap-pcd/images
cp -r $IMAGE_DIR $OUTDIR/gt-cam-vggt-pcd/images