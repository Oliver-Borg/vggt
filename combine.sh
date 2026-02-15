IMAGE_DIR="./vggt_outputs/bonsai_2_n50_s42_c0.0_p30000_voxels/images"
PCD1="./vggt_outputs/bonsai_2_n50_s42_c0.0_p30000_voxels/sparse"
PCD2="./colmap_outputs/bonsai_2_n50_s42_c5.0/sparse/0"
OUTDIR="./test"

mkdir $OUTDIR

python combine_clouds.py --camera_source $PCD1 --point_source $PCD2 --output_dir $OUTDIR/vggt-cam-colmap-pcd/sparse
python combine_clouds.py --camera_source $PCD1 --point_source $PCD1 --output_dir $OUTDIR/vggt-cam-vggt-pcd/sparse
python combine_clouds.py --camera_source $PCD2 --point_source $PCD2 --output_dir $OUTDIR/colmap-cam-colmap-pcd/sparse
python combine_clouds.py --camera_source $PCD2 --point_source $PCD1 --output_dir $OUTDIR/colmap-cam-vggt-pcd/sparse

cp -r $IMAGE_DIR $OUTDIR/vggt-cam-colmap-pcd/images
cp -r $IMAGE_DIR $OUTDIR/vggt-cam-vggt-pcd/images
cp -r $IMAGE_DIR $OUTDIR/colmap-cam-colmap-pcd/images
cp -r $IMAGE_DIR $OUTDIR/colmap-cam-vggt-pcd/images