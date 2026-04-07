import argparse
from dataclasses import dataclass
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
from typing import Literal, TypedDict, get_args
import numpy as np
from scipy.spatial.distance import cdist
import torch
import cv2
import tqdm

from check_sparse import check_sparse_folder
from demo_colmap import VGGTProfiling, run_vggt, SAMPLING_MODE


IMAGE_MODE = Literal["shuffle", "distributed", "mfps"]
COLMAP = os.path.expanduser("~/.conda/envs/vggt/bin/colmap")
CAMERA_TYPE = Literal["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]
COPY_MODE = Literal[None, "crop", "square", "tiles"]


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
    base_out: str, images_path: str, db_path: str, sparse_path: str, low_view_count: bool = False
) -> COLMAPProfiling:
    """Executes the standard COLMAP SfM stages."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    monitor = GPUMonitor(delay=0.1)
    monitor.start()

    t1 = time.time()

    passed = True

    matcher_args = [COLMAP, "exhaustive_matcher", "--database_path", db_path]
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

    passed = (
        run_command([COLMAP, "feature_extractor", "--database_path", db_path, "--image_path", images_path])
        and run_command(matcher_args)
        and run_command(mapper_args)
    )
    total_time = time.time() - t1

    monitor.stop()
    monitor.join()

    return COLMAPProfiling(
        colmap_vram_mb=(monitor.peak_memory, monitor.peak_memory),
        colmap_t=total_time,
        failed=not passed,
    )


def run_vggt_pipeline(
    base_out: str,
    conf_thres_value: float = 0.0,
    sampling_mode: SAMPLING_MODE = "random",
    num_points: int = 100000,
    camera_type: CAMERA_TYPE = "SIMPLE_PINHOLE",
) -> VGGTProfiling:
    """Executes the VGGT transformer-based reconstruction"""
    return run_vggt(
        scene_dir=base_out,
        num_profiling_runs=0,
        use_ba=sampling_mode == "ba",
        camera_type=camera_type,
        conf_thres_value=conf_thres_value,
        sampling_mode=sampling_mode,
        num_points=num_points,
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


def select_indices(indices: np.ndarray, num_images: int, seed: int):
    # Mean Farthest Point Sampling
    # TODO Try out normal Farthest Point Sampling
    n = int(indices.shape[0])
    angles = indices / (indices.max() - 1) * 2 * np.pi  # This assumes points are distributed in a circle
    # TODO Actually parse GT poses and use the real coordinates
    xs = np.cos(angles)
    ys = np.sin(angles)
    coords = np.stack([xs, ys], axis=1)  # n, 2

    selected_indices = [seed % n]  # These are the indices of the indices
    all_indices = np.arange(n)

    while len(selected_indices) < num_images:
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[np.array(selected_indices)] = True
        selected_coords = coords[selected_mask]
        remaining_coords = coords[~selected_mask]
        distances = cdist(selected_coords, remaining_coords, metric="euclidean")
        mean_dists: np.ndarray = distances.mean(axis=0)
        assert mean_dists.shape[0] == remaining_coords.shape[0]

        remaining_indices = all_indices[~selected_mask]
        next_index = int(remaining_indices[mean_dists.argmax()])

        selected_indices.append(next_index)

    return indices[np.array(selected_indices)]


assert set(select_indices(np.arange(296), 30, 42)).issubset(select_indices(np.arange(296), 40, 42))


def get_image_list(all_images: list[str], num_images: int | None, seed: int, image_mode: IMAGE_MODE):

    def sorter(name: str):  # TODO This sort may cause issues with lego
        digits = [c for c in name if c.isdigit()]
        return int("".join(digits)) if digits else 0

    all_images.sort()

    # This means there shouldn't ever be any overlap between these images and evaluation set
    all_images = [im for i, im in enumerate(all_images) if i % 8 != 0]

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

    return all_images


def run_reconstruction(
    name: str,
    input_path: str,
    choice: str,
    num_images: int | None,
    seed: int,
    conf_thres_value: float,
    force: bool,
    sampling_mode: SAMPLING_MODE,
    image_mode: IMAGE_MODE,
    copy_mode: COPY_MODE,
    camera_type: CAMERA_TYPE,
    num_points: int = 100000,
):
    name = name.strip("/")
    extra_parts = []
    if num_images:
        extra_parts.append(f"n{num_images}")

    extra_parts.append(f"s{seed}")

    if choice == "vggt":
        if sampling_mode != "ba":
            extra_parts.append(f"c{conf_thres_value}")
            extra_parts.append(f"p{num_points}")
        else:
            extra_parts.append(camera_type.lower().replace("simple_", "m"))

        extra_parts.append(sampling_mode)

    extra_parts.append(image_mode)

    if copy_mode is not None:
        extra_parts.append(copy_mode)

    # name = f"{name}_s{seed}_c{conf_thres_value}_p{num_points}_{sampling_mode}"
    name = name + "_" + "_".join(extra_parts)
    base_out = f"./{choice}_outputs/{name}"
    sparse_path = os.path.join(base_out, "sparse")
    images_path = os.path.join(base_out, "images")
    db_path = os.path.join(base_out, "database.db")

    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    input_files: list[str] = os.listdir(input_path)
    all_images: list[str] = list(sorted([f for f in input_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]))

    all_images = get_image_list(all_images, num_images, seed, image_mode)

    num_images = len(all_images)

    existing_images = os.listdir(images_path)

    if len(set(all_images) & set(existing_images)) != num_images:
        print("Invalid images found in directory. Forcing reconstruction.")
        for image in existing_images:
            os.remove(Path(images_path) / image)
        force = True

    if os.path.exists(os.path.join(base_out, "stat.json")) and not force:
        print(Path(base_out), "has already been constructed.\nUse --force to force reconstruction.")
        return

    print(f"Copying {len(all_images)} images from {Path(input_path)} to {Path(images_path)}...")
    for img in all_images:
        if copy_mode is None:
            shutil.copy2(os.path.join(input_path, img), os.path.join(images_path, img))
        elif copy_mode == "crop" or copy_mode == "square":
            im = cv2.imread(os.path.join(input_path, img))
            assert im is not None
            h, w = im.shape[:2]
            crop_size = min(h, w) if copy_mode == "square" else 518
            if h > crop_size:
                im = im[(h - crop_size) // 2: (h + crop_size) // 2]
            if w > crop_size:
                im = im[:, (w - crop_size) // 2: (w + crop_size) // 2]
            cv2.imwrite(os.path.join(images_path, img), im)

        elif copy_mode == "tiles":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown copy mode {copy_mode}")

    t1 = time.time()

    if choice == "colmap":
        profiling = run_colmap_pipeline(base_out, images_path, db_path, sparse_path, low_view_count=num_images < 50)
    elif choice == "vggt":
        profiling = run_vggt_pipeline(base_out, conf_thres_value, sampling_mode, num_points, camera_type)
    else:
        raise ValueError("Invalid choice")

    total_time = time.time() - t1

    save_timing(base_out, input_path, name, choice, num_images, profiling)

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


@dataclass
class Args:
    name: str
    input: str
    choice: str
    num_images: int | None
    seed: int
    conf_thres_value: float
    sampling_mode: SAMPLING_MODE
    image_mode: IMAGE_MODE
    camera_type: CAMERA_TYPE
    num_points: int
    copy_mode: COPY_MODE = None
    force: bool = False


def main(args: Args):
    run_reconstruction(
        name=args.name,
        input_path=args.input,
        choice=args.choice,
        num_images=args.num_images,
        seed=args.seed,
        conf_thres_value=args.conf_thres_value,
        force=args.force,
        sampling_mode=args.sampling_mode,
        image_mode=args.image_mode,
        copy_mode=args.copy_mode,
        camera_type=args.camera_type,
        num_points=args.num_points,
    )


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

    batch_parser = subparsers.add_parser("batch", help="Run multiple reconstructions from a JSON config file")
    batch_parser.add_argument(
        "--config_path", required=True, help="Path to a config file containing a list of dictionaries"
    )

    parsed_args = parser.parse_args()

    if parsed_args.command == "single":
        args_dict = vars(parsed_args)
        args_dict.pop("command")
        main(Args(**args_dict))

    elif parsed_args.command == "batch":
        with open(parsed_args.config_path, "r") as f:
            configs = json.load(f)

        for config_dict in tqdm.tqdm(configs):
            run_args = Args(**config_dict)
            main(run_args)
