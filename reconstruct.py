import argparse
import datetime
import json
import os
from pathlib import Path
from random import shuffle
import random
import shutil
import subprocess
import threading
import time
from typing import Callable, TypedDict, get_args
import numpy as np
from scipy.spatial.distance import cdist
import torch
import cv2
import tqdm
import pycolmap

from pycolmap_utils import get_poses
from check_sparse import check_sparse_folder
from combine_clouds import align_to_world_space, load_cameras, load_point_cloud, load_point_clouds, save_cameras_json
from demo_colmap import VGGTProfiling, run_vggt
from reconstruct_args import CAMERA_TYPE, COLMAP, COLMAP_MODE, COPY_MODE, IMAGE_MODE, SAMPLING_MODE, ReconstructArgs


class GPUMonitor(threading.Thread):
    def __init__(self, delay: float = 0.1):
        # TODO Replace this with nsight or pynvml
        super().__init__()
        self.stopped = False
        self.delay = delay
        self.peak_memory = 0
        self.gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    def run(self):
        while not self.stopped:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={self.gpu_id}"]
                )
                current_mem = int(out.decode("utf-8").strip())
                if current_mem > self.peak_memory:
                    self.peak_memory = current_mem
            except Exception:
                pass
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def run_command(cmd: list[str]):
    """Executes a shell command and ensures it succeeds."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False


class COLMAPProfiling(TypedDict):
    colmap_vram_mb: tuple[float, float]
    colmap_t: float
    failed: bool


def run_colmap_pipeline(
    base_out: str,
    images_path: str,
    db_path: str,
    sparse_path: str,
    low_view_count: bool = False,
    camera_type: CAMERA_TYPE = "SIMPLE_PINHOLE",
    shared_camera: bool = False,
    mask_path: str | None = None,
    random_seed: int = 42,
) -> COLMAPProfiling:
    """Executes the standard COLMAP SfM stages."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    monitor = GPUMonitor(delay=0.1)
    monitor.start()

    t1 = time.time()

    passed = True

    feature_extractor_args = [
        COLMAP,
        "feature_extractor",
        "--database_path",
        db_path,
        "--image_path",
        images_path,
        "--ImageReader.camera_model",
        camera_type,
    ]
    if shared_camera:
        feature_extractor_args.extend(["--ImageReader.single_camera", "1"])

    if mask_path is not None:
        feature_extractor_args.extend(["--ImageReader.mask_path", mask_path])

    matcher_args = [
        COLMAP,
        "exhaustive_matcher",
        "--database_path",
        db_path,
    ]
    if low_view_count:
        matcher_args.extend(
            [
                "--FeatureMatching.guided_matching",
                "1",
                "--TwoViewGeometry.min_num_inliers",
                "10",
                "--TwoViewGeometry.min_inlier_ratio",
                "0.15",
            ]
        )

    mapper_args = [
        COLMAP,
        "mapper",
        "--database_path",
        db_path,
        "--image_path",
        images_path,
        "--output_path",
        sparse_path,
        "--Mapper.random_seed",
        str(random_seed),
    ]
    if low_view_count:
        mapper_args.extend(
            [
                "--Mapper.init_min_tri_angle",
                "1.0",
                "--Mapper.init_min_num_inliers",
                "15",
                "--Mapper.abs_pose_min_num_inliers",
                "8",
                "--Mapper.tri_ignore_two_view_tracks",
                "0",
                "--Mapper.min_model_size",
                "2",
                "--Mapper.init_num_trials",
                "1000",
                "--Mapper.min_num_matches",
                "10",
                "--Mapper.ba_local_max_num_iterations",
                "50",
            ]
        )

    passed = run_command(feature_extractor_args) and run_command(matcher_args) and run_command(mapper_args)
    total_time = time.time() - t1

    monitor.stop()
    monitor.join()

    reconstruction = load_point_cloud(Path(sparse_path))
    save_cameras_json(reconstruction, Path(base_out) / "cameras.json")

    return COLMAPProfiling(
        colmap_vram_mb=(monitor.peak_memory, monitor.peak_memory),
        colmap_t=total_time,
        failed=not passed,
    )


def run_vggt_pipeline(
    base_out: str,
    cache_dir: str | None = None,
    conf_thres_value: float = 0.0,
    sampling_mode: SAMPLING_MODE = "random",
    num_points: int = 100000,
    camera_type: CAMERA_TYPE = "SIMPLE_PINHOLE",
    save_conf_as_errors: bool = False,
    shared_camera: bool = False,
    use_ba: bool = False,
    max_ba_iterations: int = 50,
    near_filtering_strength: float = 0.0,
    near_filtering_quorum: int = 1,
    reconstruct_pose_opt: bool = False,
    optimisation_iterations: int = 0,
    optimisation_neighbourhood: int = 10,
) -> VGGTProfiling:
    """Executes the VGGT transformer-based reconstruction"""
    return run_vggt(
        scene_dir=base_out,
        num_profiling_runs=0,
        use_ba=sampling_mode == "ba" or use_ba,
        camera_type=camera_type,
        conf_thres_value=conf_thres_value,
        sampling_mode=sampling_mode,
        num_points=num_points,
        cache_dir=cache_dir,
        save_conf_as_errors=save_conf_as_errors,
        shared_camera=shared_camera,
        max_ba_iterations=max_ba_iterations,
        near_filtering_strength=near_filtering_strength,
        near_filtering_quorum=near_filtering_quorum,
        reconstruct_pose_opt=reconstruct_pose_opt,
        optimisation_iterations=optimisation_iterations,
        optimisation_neighbourhood=optimisation_neighbourhood,
    )


def save_timing(
    base_out: str,
    input_path: str,
    name: str,
    choice: str,
    num_images: int,
    profiling: VGGTProfiling | COLMAPProfiling,
) -> None:
    stats: list[dict[str, str | int | float | VGGTProfiling | COLMAPProfiling]] = []
    try:
        with open("stats.json", "r") as f:
            stats = json.load(f)
    except Exception:
        stats = []

    stat: dict[str, str | int | float | VGGTProfiling | COLMAPProfiling] = {
        "date": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
        "input_path": input_path,
        "name": name,
        "type": choice,
        "num_images": num_images,
        "profiling": profiling,
    }

    stats.append(stat)

    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    with open(os.path.join(base_out, "stat.json"), "w") as f:
        json.dump(stat, f, indent=4)


def _select_indices(
    coords: np.ndarray,
    num_images: int,
    seed: int,
    farthest: bool = True,
    agg_fn: Callable[..., np.ndarray] = np.mean,
):
    # Farthest Point Sampling
    # TODO Try out normal Farthest Point Sampling
    # TODO Actually parse GT poses and use the real coordinates
    n = int(coords.shape[0])

    selected_indices = [seed % n]  # These are the indices of the indices
    all_indices = np.arange(n)

    while len(selected_indices) < num_images:
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[np.array(selected_indices)] = True
        selected_coords = coords[selected_mask]
        remaining_coords = coords[~selected_mask]
        distances = cdist(selected_coords, remaining_coords, metric="euclidean")
        agg_dists: np.ndarray = agg_fn(distances, axis=0)

        remaining_indices = all_indices[~selected_mask]
        if farthest:
            next_index = int(remaining_indices[agg_dists.argmax()])
        else:
            next_index = int(remaining_indices[agg_dists.argmin()])

        selected_indices.append(next_index)

    return np.array(selected_indices)


def select_indices(indices: np.ndarray, num_images: int, seed: int, farthest: bool = True):
    angles = indices / (indices.max() - 1) * 2 * np.pi  # This assumes points are distributed in a circle
    xs = np.cos(angles)
    ys = np.sin(angles)
    coords = np.stack([xs, ys], axis=1)  # n, 2
    return indices[_select_indices(coords, num_images, seed, farthest)]


assert set(select_indices(np.arange(296), 30, 42)).issubset(select_indices(np.arange(296), 40, 42))


def get_image_list(
    all_images: list[str],
    num_images: int | None,
    seed: int,
    image_mode: IMAGE_MODE,
    pcd: pycolmap.Reconstruction | None = None,
):

    def sorter(name: str):  # TODO This sort may cause issues with lego
        digits = [c for c in name if c.isdigit()]
        return int("".join(digits)) if digits else 0

    # This sort (without sorter) matches the sort used in GSplat
    all_images.sort()

    # This means there shouldn't ever be any overlap between these images and evaluation set
    all_images = [im for i, im in enumerate(all_images) if i % 8 != 0]

    assert num_images is None or num_images <= len(all_images)

    indices = np.arange(len(all_images))

    if not num_images:
        return all_images
    if image_mode == "shuffle":
        random.seed(seed)
        shuffle(all_images)
        all_images = all_images[:num_images]
    elif image_mode == "distributed":
        image_subset = []
        for i in range(num_images):
            k = (int(round(i / num_images * len(all_images))) + seed - 42) % len(all_images)
            image_subset.append(all_images[k])
        all_images = image_subset
    elif image_mode == "mfps":
        selected_indices = select_indices(indices, num_images, seed)
        image_subset = []
        for i in selected_indices:
            image_subset.append(all_images[i])
        all_images = image_subset
    elif image_mode == "farthestpose" or image_mode == "nearestpose":
        assert pcd is not None
        poses = get_poses(pcd)
        coords = np.array([poses[name][:3, 3] for name in all_images])
        farthest = image_mode == "farthestpose"
        assert set(_select_indices(coords, num_images // 2, seed, farthest)).issubset(
            _select_indices(coords, num_images, seed, farthest)
        )
        selected_indices = indices[_select_indices(coords, num_images, seed, farthest)]
        image_subset = []
        for i in selected_indices:
            image_subset.append(all_images[i])
        all_images = image_subset

    return all_images


def check_files(
    base_output: Path, all_images: list[str], require_depth_conf: bool, shared_camera: bool, require_masks: bool = False
) -> bool:
    if not (
        os.path.exists(base_output)
        and os.path.exists(base_output / "images")
        and os.path.exists(base_output / "sparse")
    ):
        return False
    valid = True

    if require_masks and not os.path.exists(base_output / "masks"):
        return False

    images_path = Path(base_output) / "images"
    num_images = len(all_images)
    existing_images = os.listdir(images_path)

    found_recon = False
    try:
        colmap_pcd: pycolmap.Reconstruction = load_point_cloud(base_output / "sparse")
        colmap_image_names = [image.name for image in colmap_pcd.images.values()]
        num_cameras = len(colmap_pcd.cameras)
        found_recon = True
    except Exception:
        colmap_image_names = []
        num_cameras = 0

    cameras_file = base_output / "cameras.json"

    if not os.path.exists(cameras_file):
        try:
            reconstruction = load_point_cloud(base_output / "sparse")
            save_cameras_json(reconstruction, Path(base_output) / "cameras.json")
        except Exception:
            pass

    if os.path.exists(cameras_file):
        with open(cameras_file, "r") as f:
            camera_data = json.load(f)
            camera_data_names = set(camera_data.keys())
    else:
        camera_data_names = set()

    if len(set(all_images) | camera_data_names) > num_images or len(camera_data_names) == 0:
        print("Invalid cameras found in camera json. Forcing reconstruction.")
        valid = False

    if (shared_camera and num_cameras != 1) or (not shared_camera and num_cameras != len(colmap_image_names)):
        print("Invalid number of cameras found in COLMAP database. Forcing reconstruction.")
        valid = False

    if len(set(all_images) & set(existing_images)) != num_images:
        print("Invalid images found in directory. Forcing reconstruction.")
        for image in existing_images:
            os.remove(Path(images_path) / image)
        valid = False

    if len(set(all_images) | set(colmap_image_names)) > num_images and found_recon:
        print("Invalid images found in COLMAP database. Forcing reconstruction.")
        valid = False

    if (
        require_masks
        and len(
            set([name + ".png" for name in all_images])
            & (existing_masks := set(os.listdir(Path(base_output) / "masks")))
        )
        != num_images
    ):
        print("Invalid masks found in directory. Forcing reconstruction.")
        for mask in existing_masks:
            os.remove(Path(base_output) / "masks" / mask)
        valid = False

    if require_depth_conf:
        depths_path = Path(base_output) / "depths"
        if not os.path.exists(depths_path):
            os.makedirs(depths_path)

        all_depth_files = os.listdir(depths_path)

        # np.save(os.path.join(path, "depths", f"depth_{camera_name}.npy"), depth)
        # np.save(os.path.join(path, "depths", f"raw_conf_{camera_name}.npy"), depth_conf)
        # np.save(os.path.join(path, "depths", f"depth_conf_{camera_name}.npy"), viz_conf)

        missing_depths = False

        for camera_name in all_images:
            depth_map = f"depth_{camera_name}.npy"
            raw_conf = f"raw_conf_{camera_name}.npy"
            viz_conf = f"depth_conf_{camera_name}.npy"

            if depth_map not in all_depth_files or viz_conf not in all_depth_files or raw_conf not in all_depth_files:
                missing_depths = True
                valid = False
                break

        if missing_depths:
            for depth_file in all_depth_files:
                os.remove(Path(depths_path) / depth_file)

    return valid


def run_reconstruction(
    args: ReconstructArgs,
):
    os.makedirs(args.cache_dir, exist_ok=True)
    base_out = args.base_out
    sparse_path = os.path.join(base_out, "sparse")
    images_path = os.path.join(base_out, "images")
    masks_path = os.path.join(base_out, "masks")
    db_path = os.path.join(base_out, "database.db")

    input_path = args.input

    input_files: list[str] = os.listdir(input_path)
    all_images: list[str] = list(
        sorted(
            [
                f
                for f in input_files
                if f.lower().endswith((".png", ".jpg", ".jpeg")) and "depth" not in f and "normal" not in f
            ]
        )
    )

    pcd = None

    is_nerf_synthetic = "nerf_synthetic" in Path(input_path).parts

    if args.image_mode == "farthestpose" or args.image_mode == "nearestpose":
        if is_nerf_synthetic:
            pcd = load_cameras(Path(input_path).parent / f"transforms_{Path(input_path).name}.json")
        else:
            pcd = load_cameras(Path(input_path).parent / "sparse")

    all_images = get_image_list(all_images, args.num_images, args.seed, args.image_mode, pcd)

    num_images = len(all_images)

    if args.force or not check_files(
        Path(base_out),
        all_images,
        require_depth_conf=args.require_depth_conf and args.choice == "vggt",
        shared_camera=args.shared_camera,
        require_masks=args.choice == "colmap" and is_nerf_synthetic,
    ):
        if os.path.exists(base_out):
            shutil.rmtree(base_out)

    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    if is_nerf_synthetic:
        os.makedirs(masks_path, exist_ok=True)

    if os.path.exists(os.path.join(base_out, "stat.json")) and not args.force:
        print(Path(base_out), "has already been constructed.\nUse --force to force reconstruction.")
        camera_path = Path(base_out) / "aligned_cameras.json"
        if (
            not os.path.exists(camera_path)
            or camera_path.stat().st_mtime < datetime.datetime(2026, 5, 18, 19, 45, 0).timestamp()
        ):
            # TODO Instead of this save aligned_cams.json and gt_cams.json
            # Then in plot_metrics, we can just read these in
            if is_nerf_synthetic:
                gt_pcd = load_cameras(Path(input_path).parent / f"transforms_{Path(input_path).name}.json")
            else:
                gt_pcd = load_cameras(Path(input_path).parent / "sparse")
            pred_pcds = load_point_clouds(Path(base_out) / "sparse")
            for i, pred_pcd in enumerate(pred_pcds):
                pred_pcd = align_to_world_space(pred_pcd, gt_pcd)
                if i == 0:
                    save_cameras_json(pred_pcd, Path(base_out) / "aligned_cameras.json")
                # TODO Use these to demonstrate partial reconstructions
                save_cameras_json(pred_pcd, Path(base_out) / f"aligned_cameras{i:04d}.json")
            save_cameras_json(gt_pcd, Path(base_out) / "gt_cameras.json")
            print("Saved aligned cameras")
        return

    print(f"Copying {len(all_images)} images from {Path(input_path)} to {Path(images_path)}...")
    for img in all_images:
        img_full_path = os.path.join(input_path, img)

        if is_nerf_synthetic:
            im_bgra = cv2.imread(img_full_path, cv2.IMREAD_UNCHANGED)
            if im_bgra is not None and im_bgra.shape[2] == 4:
                mask = im_bgra[:, :, 3]
                cv2.imwrite(os.path.join(masks_path, f"{img}.png"), mask)

        if args.copy_mode is None:
            shutil.copy2(os.path.join(input_path, img), os.path.join(images_path, img))
        elif args.copy_mode == "crop" or args.copy_mode == "square":
            im = cv2.imread(os.path.join(input_path, img))
            assert im is not None
            h, w = im.shape[:2]
            crop_size = min(h, w) if args.copy_mode == "square" else 518
            if h > crop_size:
                im = im[(h - crop_size) // 2: (h + crop_size) // 2]
            if w > crop_size:
                im = im[:, (w - crop_size) // 2: (w + crop_size) // 2]
            cv2.imwrite(os.path.join(images_path, img), im)

            if is_nerf_synthetic:
                m = cv2.imread(os.path.join(masks_path, f"{img}.png"), cv2.IMREAD_GRAYSCALE)
                if h > crop_size:
                    m = m[(h - crop_size) // 2: (h + crop_size) // 2]
                if w > crop_size:
                    m = m[:, (w - crop_size) // 2: (w + crop_size) // 2]
                cv2.imwrite(os.path.join(masks_path, f"{img}.png"), m)

        elif args.copy_mode == "tiles":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown copy mode {args.copy_mode}")

    t1 = time.time()

    if args.choice == "colmap":
        profiling = run_colmap_pipeline(
            base_out,
            images_path,
            db_path,
            sparse_path,
            low_view_count=args.colmap_mode == "relaxed",
            shared_camera=args.shared_camera,
            mask_path=masks_path if is_nerf_synthetic else None,
            camera_type=args.camera_type,
            random_seed=args.seed,
        )
    elif args.choice == "vggt":
        profiling = run_vggt_pipeline(
            base_out,
            args.cache_dir,
            args.conf_thres_value,
            args.sampling_mode,
            args.num_points,
            args.camera_type,
            args.save_conf_as_errors,
            shared_camera=args.shared_camera,
            use_ba=args.use_ba,
            max_ba_iterations=args.max_ba_iterations,
            near_filtering_strength=args.near_filtering_strength,
            near_filtering_quorum=args.near_filtering_quorum,
            reconstruct_pose_opt=args.reconstruct_pose_opt,
            optimisation_iterations=args.optimisation_iterations,
            optimisation_neighbourhood=args.optimisation_neighbourhood,
        )
    else:
        raise ValueError("Invalid choice")

    total_time = time.time() - t1

    save_timing(base_out, input_path, args.name, args.choice, num_images, profiling)

    print(f"\nPipeline finished. Results saved in: {base_out}")
    print(f"Time: {total_time:.2f}s")

    best_path = check_sparse_folder(sparse_path)

    if best_path is not None and best_path.name != "0" and best_path.name.isnumeric():
        # Swap best path with path 0
        first_path = best_path.parent / "0"
        tmp_path = best_path.parent / "tmp"
        os.rename(first_path, tmp_path)
        os.rename(best_path, first_path)
        os.rename(tmp_path, best_path)

    if is_nerf_synthetic:
        gt_pcd = load_cameras(Path(input_path).parent / f"transforms_{Path(input_path).name}.json")
    else:
        gt_pcd = load_cameras(Path(input_path).parent / "sparse")
    pred_pcds = load_point_clouds(Path(base_out) / "sparse")
    for i, pred_pcd in enumerate(pred_pcds):
        pred_pcd = align_to_world_space(pred_pcd, gt_pcd)
        if i == 0:
            save_cameras_json(pred_pcd, Path(base_out) / "aligned_cameras.json")
        save_cameras_json(pred_pcd, Path(base_out) / f"aligned_cameras{i:04d}.json")
    save_cameras_json(gt_pcd, Path(base_out) / "gt_cameras.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP or VGGT reconstruction pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Execution mode")

    single_parser = subparsers.add_parser("single", help="Run a single reconstruction via CLI arguments")
    single_parser.add_argument("--input", required=True, help="Path to the input images folder")
    single_parser.add_argument("--name", required=True, help="Output folder name (e.g., garden_8)")
    single_parser.add_argument("--choice", choices=["colmap", "vggt"], required=True, help="Pipeline to run")
    single_parser.add_argument("--num_images", type=int, default=None, help="Limit the number of images to process")
    single_parser.add_argument("--seed", type=int, default=42, help="Seed for the random shuffling of images")
    single_parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold for point cloud"
    )
    single_parser.add_argument("--force", action="store_true", help="Force reconstruction")
    single_parser.add_argument(
        "--sampling_mode",
        type=str,
        default="random",
        choices=list(get_args(SAMPLING_MODE)),
        help="Sampling mode for point cloud subsampling",
    )
    single_parser.add_argument("--use_ba", action="store_true", help="Use Bundle Adjustment for VGGT.")
    single_parser.add_argument(
        "--image_mode",
        type=str,
        default="distributed",
        choices=list(get_args(IMAGE_MODE)),
        help="Image selection mode",
    )
    single_parser.add_argument(
        "--copy_mode",
        type=str,
        default=None,
        choices=list(get_args(COPY_MODE)),
        help="Image copy mode",
    )
    single_parser.add_argument(
        "--num_points", type=int, default=100000, help="Number of points to use for reconstruction"
    )
    single_parser.add_argument(
        "--camera_type",
        type=str,
        default="SIMPLE_PINHOLE",
        choices=list(get_args(CAMERA_TYPE)),
        help="Camera type for reconstruction",
    )
    single_parser.add_argument(
        "--colmap_mode",
        type=str,
        default="default",
        choices=list(get_args(COLMAP_MODE)),
        help="COLMAP mode for reconstruction",
    )
    single_parser.add_argument("--require_depth_conf", action="store_true", help="Require depth confidence map")
    single_parser.add_argument("--save_conf_as_errors", action="store_true", help="Save depth confidence map as errors")
    single_parser.add_argument("--shared_camera", action="store_true", help="Share cameras between images")
    single_parser.add_argument("--max_ba_iterations", type=int, default=50, help="Max BA iterations")
    single_parser.add_argument(
        "--reconstruct_pose_opt", action="store_true", help="Use pose optimization during reconstruction"
    )
    single_parser.add_argument(
        "--optimisation_iterations", type=int, default=0, help="Number of optimisation iterations"
    )
    single_parser.add_argument(
        "--optimisation_neighbourhood", type=int, default=10, help="Size of neighbourhood for optimisation"
    )

    batch_parser = subparsers.add_parser("batch", help="Run multiple reconstructions from a JSON config file")
    batch_parser.add_argument(
        "--config_path", required=True, help="Path to a config file containing a list of dictionaries"
    )

    parsed_args = parser.parse_args()

    if parsed_args.command == "single":
        args_dict = vars(parsed_args)
        args_dict.pop("command")
        run_reconstruction(ReconstructArgs(**args_dict))

    elif parsed_args.command == "batch":
        with open(parsed_args.config_path, "r") as f:
            configs = json.load(f)

        for config_dict in tqdm.tqdm(configs):
            run_args = ReconstructArgs(**config_dict)
            # run_reconstruction(run_args)
            try:
                run_reconstruction(run_args)
            except Exception as e:
                print(e)
                continue
