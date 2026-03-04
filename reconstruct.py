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
from typing import Literal, TypedDict, get_args
import torch

from check_sparse import check_sparse_folder
from demo_colmap import VGGTProfiling, run_vggt, SAMPLING_MODE


IMAGE_MODE = Literal["shuffle", "distributed"]


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

    matcher_args = ["colmap", "exhaustive_matcher", "--database_path", db_path]
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
        "colmap",
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
        run_command(["colmap", "feature_extractor", "--database_path", db_path, "--image_path", images_path])
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
) -> VGGTProfiling:
    """Executes the VGGT transformer-based reconstruction"""
    return run_vggt(
        scene_dir=base_out,
        num_profiling_runs=0,
        use_ba=sampling_mode == "ba",
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


def get_image_list(all_images: list[str], num_images: int | None, seed: int, image_mode: IMAGE_MODE):

    def sorter(name: str):
        digits = [c for c in name if c.isdigit()]
        return int("".join(digits)) if digits else 0

    all_images.sort(key=sorter)

    if not num_images:
        return all_images
    if image_mode == "shuffle":
        random.seed(seed)
        shuffle(all_images)
        all_images = all_images[:num_images]
    elif image_mode == "distributed":
        image_subset = []
        for i in range(num_images):
            k = (int(round(i / num_images *  len(all_images))) + seed - 42) % len(all_images)
            image_subset.append(all_images[k])
        all_images = image_subset
    return all_images


def main(
    name: str,
    input_path: str,
    choice: str,
    num_images: int | None,
    seed: int,
    conf_thres_value: float,
    force: bool,
    sampling_mode: SAMPLING_MODE,
    image_mode: IMAGE_MODE,
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
        extra_parts.append(sampling_mode)

    extra_parts.append(image_mode)

    # name = f"{name}_s{seed}_c{conf_thres_value}_p{num_points}_{sampling_mode}"
    name = name + "_" + "_".join(extra_parts)
    base_out = f"./{choice}_outputs/{name}"
    sparse_path = os.path.join(base_out, "sparse")
    images_path = os.path.join(base_out, "images")
    db_path = os.path.join(base_out, "database.db")

    if os.path.exists(os.path.join(base_out, "stat.json")) and not force:
        print(Path(base_out), "has already been constructed.\nUse --force to force reconstruction.")
        return

    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    input_files: list[str] = os.listdir(input_path)
    all_images: list[str] = list(sorted([f for f in input_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]))

    all_images = get_image_list(all_images, num_images, seed, image_mode)

    num_images = len(all_images)

    print(f"Copying {len(all_images)} images to {images_path}...")
    for img in all_images:
        shutil.copy2(os.path.join(input_path, img), os.path.join(images_path, img))

    t1 = time.time()

    if choice == "colmap":
        profiling = run_colmap_pipeline(base_out, images_path, db_path, sparse_path, low_view_count=num_images < 50)
    elif choice == "vggt":
        profiling = run_vggt_pipeline(base_out, conf_thres_value, sampling_mode, num_points)
    else:
        raise ValueError("Invalid choice")

    total_time = time.time() - t1

    save_timing(base_out, input_path, name, choice, num_images, profiling)

    print(f"\nPipeline finished. Results saved in: {base_out}")
    print(f"Time: {total_time:.2f}s")

    check_sparse_folder(sparse_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP or VGGT reconstruction pipeline.")

    parser.add_argument("--input", required=True, help="Path to the input images folder")
    parser.add_argument("--name", required=True, help="Output folder name (e.g., garden_8)")
    parser.add_argument("--choice", choices=["colmap", "vggt"], required=True, help="Pipeline to run")
    parser.add_argument("--num_images", type=int, default=None, help="Limit the number of images to process")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random shuffling of images")
    parser.add_argument("--conf_thres_value", type=float, default=5.0, help="Confidence threshold for point cloud")
    parser.add_argument("--force", action="store_true", help="Force reconstruction")
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="random",
        choices=list(get_args(SAMPLING_MODE)),
        help="Sampling mode for point cloud subsampling",
    )
    parser.add_argument(
        "--image_mode",
        type=str,
        default="distributed",
        choices=list(get_args(IMAGE_MODE)),
        help="Image selection mode",
    )
    parser.add_argument("--num_points", type=int, default=100000, help="Number of points to use for reconstruction")

    args = parser.parse_args()
    main(
        name=args.name,
        input_path=args.input,
        choice=args.choice,
        num_images=args.num_images,
        seed=args.seed,
        conf_thres_value=args.conf_thres_value,
        force=args.force,
        sampling_mode=args.sampling_mode,
        image_mode=args.image_mode,
        num_points=args.num_points,
    )
