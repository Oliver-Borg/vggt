# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import time
from typing import Literal, TypedDict
from line_profiler import profile
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

from combine_clouds import save_cameras_json
from reconstruct_args import SAMPLING_MODE
from vggt.dependency.pycolmap_to_np import extract_poses_from_reconstruction

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap
import matplotlib.pyplot as plt

import pyceres


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import project_world_points_to_cam, unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues, uniform_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
from vggt.utils.image_utils import np_rgb

# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(model: VGGT, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def get_gpu_stats(reset: bool = True):
    """Returns the peak allocated and peak reserved memory in MB for the current device."""
    if not torch.cuda.is_available():
        return 0.0, 0.0

    assert torch.cuda.device_count() == 1
    device = torch.device("cuda:0")
    # Allocated: actual tensors
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**2)
    # Reserved: reserved memory
    peak_res = torch.cuda.max_memory_reserved(device) / (1024**2)

    if torch.cuda.is_available() and reset:
        torch.cuda.reset_peak_memory_stats()

    return peak_alloc, peak_res


def demo_fn(args):
    return run_vggt(
        scene_dir=args.scene_dir,
        seed=args.seed,
        use_ba=args.use_ba,
        max_reproj_error=args.max_reproj_error,
        shared_camera=args.shared_camera,
        camera_type=args.camera_type,
        vis_thresh=args.vis_thresh,
        query_frame_num=args.query_frame_num,
        max_query_pts=args.max_query_pts,
        fine_tracking=args.fine_tracking,
        conf_thres_value=args.conf_thres_value,
    )


class VGGTProfiling(TypedDict):
    model_vram_mb: tuple[float, float]
    model_load_t: float
    warmup_vram_mb: tuple[float, float]
    warmup_t: float
    inference_vram_mb: tuple[float, float]
    inference_times: list[float]
    image_load_t: float
    point_cloud_processing_vram_mb: tuple[float, float]
    point_cloud_processing_t: float
    saving_t: float


def save_depths(path: str, depths: np.ndarray, raw_confs: np.ndarray, viz_confs: np.ndarray, camera_names: list[str]):
    # Save the raw depth and confidence maps for visualization
    b = depths.shape[0]
    os.makedirs(os.path.join(path, "depths"), exist_ok=True)
    for i in range(b):
        depth = depths[i]
        raw_conf = raw_confs[i]
        viz_conf = viz_confs[i]
        camera_name = camera_names[i]
        np.save(os.path.join(path, "depths", f"depth_{camera_name}.npy"), depth)
        np.save(os.path.join(path, "depths", f"raw_conf_{camera_name}.npy"), raw_conf)
        np.save(os.path.join(path, "depths", f"depth_conf_{camera_name}.npy"), viz_conf)


model_cache = {}


def load_model() -> VGGT:
    model_key = "vggt"
    if model_key in model_cache:
        return model_cache[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    model_cache[model_key] = model
    return model


def predict_tracks_with_cache(
    images, conf, points_3d, masks, max_query_pts, query_frame_num, keypoint_extractor, fine_tracking, cache_dir, device
):
    track_cache_file = os.path.join(cache_dir, "tracks.pt")
    track_config_file = os.path.join(cache_dir, "tracks_config.json")

    track_config = {
        "max_query_pts": max_query_pts,
        "query_frame_num": query_frame_num,
        "keypoint_extractor": keypoint_extractor,
        "fine_tracking": fine_tracking,
    }

    load_cached_tracks = False
    if track_cache_file and os.path.exists(track_cache_file) and os.path.exists(track_config_file):
        try:
            with open(track_config_file, "r") as f:
                cached_config = json.load(f)
            if cached_config == track_config:
                load_cached_tracks = True
        except Exception as e:
            print(f"Warning: Failed to read track cache config: {e}")

    if load_cached_tracks:
        print(f"Loading cached tracks from {track_cache_file}")
        cached_tracks = torch.load(track_cache_file, map_location=device)
        pred_tracks = cached_tracks["pred_tracks"]
        pred_vis_scores = cached_tracks["pred_vis_scores"]
        pred_confs = cached_tracks["pred_confs"]
        tracked_points_3d = cached_tracks["tracked_points_3d"]
        tracked_points_rgb = cached_tracks["tracked_points_rgb"]
    else:
        pred_tracks, pred_vis_scores, pred_confs, tracked_points_3d, tracked_points_rgb = predict_tracks(
            images,
            conf=conf,
            points_3d=points_3d,
            masks=masks,
            max_query_pts=max_query_pts,
            query_frame_num=query_frame_num,
            keypoint_extractor=keypoint_extractor,
            fine_tracking=fine_tracking,
        )

        if track_cache_file:
            print(f"Saving tracks to {track_cache_file}")
            torch.save(
                {
                    "pred_tracks": pred_tracks,
                    "pred_vis_scores": pred_vis_scores,
                    "pred_confs": pred_confs,
                    "tracked_points_3d": tracked_points_3d,
                    "tracked_points_rgb": tracked_points_rgb,
                },
                track_cache_file,
            )
            with open(track_config_file, "w") as f:
                json.dump(track_config, f)

    return pred_tracks, pred_vis_scores, pred_confs, tracked_points_3d, tracked_points_rgb


def filter_nearest(
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_map: np.ndarray,
    conf_mask: np.ndarray,
    strength: float,
    quorum: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    num_cameras = extrinsic.shape[0]
    height, width = depth_map.shape[1], depth_map.shape[2]
    threshold = quorum

    # Convert depth_map to tensor once for sampling
    depth_map_t = torch.tensor(depth_map, dtype=torch.float32, device=device)

    # Initialize a counter for how many times each point is invalidated
    invalid_counts = np.zeros_like(conf_mask, dtype=np.int32)

    # Loop over each camera
    for i in range(num_cameras):
        # Extract indices of currently valid points to map them back later
        valid_idx = np.where(conf_mask)
        world_points = torch.tensor(points_3d[valid_idx], dtype=torch.float32, device=device)

        if world_points.shape[0] == 0:
            break

        image_points, cam_points = project_world_points_to_cam(
            world_points,
            torch.tensor(extrinsic[i: i + 1], dtype=torch.float32, device=device),
            torch.tensor(intrinsic[i: i + 1], dtype=torch.float32, device=device),
        )
        assert image_points is not None

        # Remove batch dimension: image_points is (N, 2), cam_points is (3, N)
        image_points = image_points.squeeze(0)
        cam_points = cam_points.squeeze(0)

        # 2. Get a mask of valid image_points using image dimensions
        u = image_points[:, 0].round().long()
        v = image_points[:, 1].round().long()

        valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)

        # 3. Calculate depths of all points within frame (Z-coordinate is index 2)
        projected_depths = cam_points[2, :]

        # 4. Sample depth map and find nearer points
        nearer_mask = torch.zeros(world_points.shape[0], dtype=torch.bool, device=device)

        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        # Sample the depth map at the valid projected pixel coordinates
        sampled_depths = depth_map_t[i, v_valid, u_valid].view(-1)

        # Ignore pixels where the depth map has no reading (e.g., depth == 0)
        valid_depth_mask = sampled_depths > 1e-8

        # Check if projected depth is nearer than the depth map (with a small margin to prevent self-occlusion)
        nearer_mask[valid_mask] = (projected_depths[valid_mask] < (sampled_depths - 1e-3) * strength) & valid_depth_mask
        print(nearer_mask.sum().item(), invalid_counts.max())

        # Combine masks and update invalid counts using the mapped indices
        # We count points that are validly projected AND nearer
        invalidated_points = (valid_mask & nearer_mask).cpu().numpy()
        invalid_counts[valid_idx] += invalidated_points
        conf_mask[valid_idx] = invalid_counts[valid_idx] < threshold

    return conf_mask


@profile
def run_vggt(
    scene_dir: str,
    seed: int = 42,
    use_ba: bool = False,
    max_reproj_error: float = 8.0,
    shared_camera: bool = True,
    camera_type: str = "SIMPLE_PINHOLE",
    vis_thresh: float = 0.2,
    query_frame_num: int = 8,
    max_query_pts: int = 4096,
    fine_tracking: bool = True,
    conf_thres_value: float = 5.0,
    num_profiling_runs: int = 0,
    sampling_mode: SAMPLING_MODE = "random",
    num_points: int = 100000,
    model: VGGT | None = None,
    cache_dir: str | None = None,
    save_conf_as_errors: bool = False,
    max_ba_iterations: int = 50,
    near_filtering_strength: float = 0.0,
    near_filtering_quorum: int = 2,
) -> VGGTProfiling:

    # Print configuration
    print(
        "Arguments:",
        {
            "scene_dir": scene_dir,
            "seed": seed,
            "use_ba": use_ba,
            "max_reproj_error": max_reproj_error,
            "shared_camera": shared_camera,
            "camera_type": camera_type,
            "vis_thresh": vis_thresh,
            "query_frame_num": query_frame_num,
            "max_query_pts": max_query_pts,
            "fine_tracking": fine_tracking,
            "conf_thres_value": conf_thres_value,
            "num_profiling_runs": num_profiling_runs,
            "sampling_mode": sampling_mode,
            "num_points": num_points,
            "model": model,
            "cache_dir": cache_dir,
            "save_conf_as_errors": save_conf_as_errors,
            "max_ba_iterations": max_ba_iterations,
            "near_filtering_strength": near_filtering_strength,
            "near_filtering_quorum": near_filtering_quorum,
        },
    )

    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    print(f"Setting seed as: {seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_num = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(f"Using device: {device} ({gpu_num})")
    print(f"Using dtype: {dtype}")

    vggt_fixed_resolution = 518
    img_load_resolution = 518  # TODO try change this back to 1024

    cache_file = os.path.join(cache_dir, "raw_preds.pt") if cache_dir else None

    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached raw predictions from {cache_file}")
        model_load_t1 = time.time()
        model_load_t2 = time.time()
        model_vram_mb = (0.0, 0.0)

        image_load_t1 = time.time()
        cached_data = torch.load(cache_file, map_location=device)
        images = cached_data["images"].to(device)
        original_coords = cached_data["original_coords"].to(device)
        extrinsic = cached_data["extrinsic"]
        intrinsic = cached_data["intrinsic"]
        depth_map = cached_data["depth_map"]
        depth_conf = cached_data["depth_conf"]
        orig_depth_conf = cached_data["orig_depth_conf"]
        masks = cached_data["masks"]
        base_image_path_list = cached_data["base_image_path_list"]
        image_load_t2 = time.time()
        print(f"Cached data loaded in {image_load_t2 - image_load_t1:.2f} seconds")

        warmup_t1 = time.time()
        warmup_t2 = time.time()
        warmup_vram_mb = (0.0, 0.0)
        inf_times = []
        inf_vram_mb = (0.0, 0.0)

    else:
        # Run VGGT for camera and depth estimation
        model_load_t1 = time.time()
        if model is None:
            model = load_model()
        model_load_t2 = time.time()
        model_alloc, model_res = model_vram_mb = get_gpu_stats()
        print(
            f"Model loaded | Peak allocated GPU Mem: {model_alloc:.2f} MB | Peak reserved GPU Mem: {model_res:.2f} MB"
        )

        image_load_t1 = time.time()
        # Get image paths and preprocess them
        image_dir = os.path.join(scene_dir, "images")
        image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
        if len(image_path_list) == 0:
            raise ValueError(f"No images found in {image_dir}")
        base_image_path_list = [os.path.basename(path) for path in image_path_list]

        # Load images and original coordinates
        # Load Image in 1024, while running VGGT with 518
        images, masks, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
        images = images.to(device)
        original_coords = original_coords.to(device)
        print(f"Loaded {len(images)} images from {image_dir}")
        image_load_t2 = time.time()

        # Run VGGT to estimate camera and depth
        # Run with 518x518 images

        # Warmup run
        warmup_t1 = time.time()
        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
        masks = F.interpolate(
            masks.to(torch.uint8), size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="nearest"
        )

        masks = masks.cpu().numpy().transpose(0, 2, 3, 1)
        depth_conf[masks[..., 0] == 0] = 0.0
        orig_depth_conf = depth_conf.copy()
        depth_map[masks == 0] = np.nan
        warmup_t2 = time.time()
        warmup_vram_mb = get_gpu_stats()

        inf_times: list[float] = []
        for _ in range(num_profiling_runs):
            inf_t1 = time.time()
            run_VGGT(model, images, dtype, vggt_fixed_resolution)
            inf_t2 = time.time()
            inf_times.append(inf_t2 - inf_t1)

        inf_alloc, inf_res = inf_vram_mb = get_gpu_stats()
        print(f"Inference run | Peak allocated GPU Mem: {inf_alloc:.2f} MB | Peak reserved GPU Mem: {inf_res:.2f} MB")

        if cache_file:
            print(f"Saving raw predictions to {cache_file}")
            torch.save(
                {
                    "images": images.cpu(),
                    "original_coords": original_coords.cpu(),
                    "extrinsic": extrinsic,
                    "intrinsic": intrinsic,
                    "depth_map": depth_map,
                    "depth_conf": depth_conf,
                    "orig_depth_conf": orig_depth_conf,
                    "masks": masks,
                    "base_image_path_list": base_image_path_list,
                },
                cache_file,
            )

    processing_t1 = time.time()

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    points_3d[masks[..., 0] == 0] = np.nan

    model = None

    with torch.no_grad():
        torch.cuda.empty_cache()

    print(get_gpu_stats())

    points_conf_rgb = None

    if use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = shared_camera

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, tracked_points_3d, tracked_points_rgb = predict_tracks_with_cache(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=max_query_pts,
                query_frame_num=query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=fine_tracking,
                cache_dir=cache_dir,
                device=device,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            tracked_points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=max_reproj_error,
            shared_camera=shared_camera,
            camera_type=camera_type,
            points_rgb=tracked_points_rgb,
            min_inlier_per_frame=16,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions(
            loss_function_scale=1.0,
            refine_extra_params=True,
            refine_extrinsics=True,
            refine_focal_length=True,
            refine_principal_point=False,
        )
        ba_options.solver_options.max_num_iterations = max_ba_iterations
        ba_options.solver_options.function_tolerance = 0.0
        ba_options.solver_options.minimizer_progress_to_stdout = True
        examples = []
        for _ in range(1):
            pycolmap.bundle_adjustment(reconstruction, ba_options)
            examples.append(str([cam.params for cam in dict(reconstruction.cameras).values()][0]))

        print("\n".join(examples))
        reconstruction_resolution = img_load_resolution

        extrinsic, intrinsic = extract_poses_from_reconstruction(
            reconstruction, base_image_path_list, extrinsic, intrinsic, scale
        )

        points_rgb = tracked_points_rgb
        if sampling_mode == "ba":
            points_3d = tracked_points_3d

    if sampling_mode != "ba":
        conf_thres_value = conf_thres_value
        max_points_for_colmap = num_points  # randomly sample 3D points
        if not use_ba:
            shared_camera = False  # in the feedforward manner, we do not support shared camera
            camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        print(
            f"Confidence\tmin: {depth_conf.min():.1f}\tmax: {depth_conf.max():.1f}\tmean: {depth_conf.mean():.1f}. Normalizing..."
        )

        depth_conf = (depth_conf - depth_conf.min()) / (depth_conf.max() - depth_conf.min())

        conf_mask = (depth_conf >= conf_thres_value) & (~np.isnan(points_3d[..., 0]))

        assert conf_thres_value <= 0.0 or ((masks[..., 0] == 0) & conf_mask).sum() == 0

        if near_filtering_strength > 0.0:
            conf_mask = filter_nearest(
                extrinsic, intrinsic, depth_map, conf_mask, near_filtering_strength, near_filtering_quorum
            )

        # at most writing 100000 3d points to colmap reconstruction object
        if sampling_mode == "random" or sampling_mode == "confidence":
            conf_mask = randomly_limit_trues(
                conf_mask, max_points_for_colmap, depth_conf=(depth_conf if sampling_mode == "confidence" else None)
            )
        elif sampling_mode == "voxels":
            conf_mask = uniform_limit_trues(conf_mask, max_points_for_colmap, points_3d, depth_conf)
        elif sampling_mode == "vox3":
            conf_mask = uniform_limit_trues(
                conf_mask, max_points_for_colmap, points_3d, depth_conf, min_grid_occupancy=3
            )

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]
        points_errors = 1 / np.maximum(orig_depth_conf[conf_mask], 1)
        points_conf_rgb = np_rgb(depth_conf[conf_mask] ** 2 * depth_map[conf_mask].flatten())

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
            points_errors=points_errors if save_conf_as_errors else None,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        camera_type=camera_type,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )
    processing_t2 = time.time()
    processing_vram_mb = get_gpu_stats()

    saving_t1 = time.time()

    print(f"Saving reconstruction to {scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)
    save_depths(scene_dir, depth_map[..., 0], orig_depth_conf, depth_conf, base_image_path_list)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(scene_dir, "sparse/points.ply"))
    if points_conf_rgb is not None:
        trimesh.PointCloud(points_3d, colors=points_conf_rgb).export(os.path.join(scene_dir, "sparse/points_conf.ply"))
    saving_t2 = time.time()

    save_cameras_json(reconstruction, Path(scene_dir) / "cameras.json")

    return VGGTProfiling(
        model_vram_mb=model_vram_mb,
        model_load_t=model_load_t2 - model_load_t1,
        warmup_vram_mb=warmup_vram_mb,
        warmup_t=warmup_t2 - warmup_t1,
        inference_vram_mb=inf_vram_mb,
        inference_times=inf_times,
        image_load_t=image_load_t2 - image_load_t1,
        point_cloud_processing_vram_mb=processing_vram_mb,
        point_cloud_processing_t=processing_t2 - processing_t1,
        saving_t=saving_t2 - saving_t1,
    )


def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    camera_type,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size

            # TODO Check if this SIMPLE_PINHOLE and SIMPLE_RADIAL are correct now
            if camera_type == "SIMPLE_PINHOLE":
                pred_params[:3] *= resize_ratio
                real_pp = real_image_size / 2
                pred_params[-2:] = real_pp  # center of the image
            elif camera_type == "SIMPLE_RADIAL":
                pred_params[:3] *= resize_ratio
                real_pp = real_image_size / 2
                pred_params[1:3] = real_pp  # center of the image
            else:
                pred_params = pred_params * resize_ratio
                real_pp = real_image_size / 2
                pred_params[-2:] = real_pp  # center of the image
            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
