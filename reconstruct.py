import argparse
import datetime
import json
import os
from random import shuffle
import random
import shutil
import subprocess
import threading
import time
from typing import TypedDict
import torch

from demo_colmap import VGGTProfiling, run_vggt


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


def run_vggt_pipeline(base_out: str) -> VGGTProfiling:
    """Executes the VGGT transformer-based reconstruction"""
    return run_vggt(scene_dir=base_out, num_profiling_runs=5)


def save_timing(
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


def main():
    parser = argparse.ArgumentParser(description="Run COLMAP or VGGT reconstruction pipeline.")

    parser.add_argument("--input", required=True, help="Path to the input images folder")
    parser.add_argument("--name", required=True, help="Output folder name (e.g., garden_8)")
    parser.add_argument("--choice", choices=["colmap", "vggt"], required=True, help="Pipeline to run")
    parser.add_argument("--num_images", type=int, default=None, help="Limit the number of images to process")

    args = parser.parse_args()

    name = args.name.strip("/")
    if args.num_images:
        name = f"{name}_n{args.num_images}"
    base_out = f"./{args.choice}_outputs/{name}"
    sparse_path = os.path.join(base_out, "sparse")
    images_path = os.path.join(base_out, "images")
    db_path = os.path.join(base_out, "database.db")

    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    input_path: str = args.input
    input_files: list[str] = os.listdir(input_path)
    all_images: list[str] = list(sorted([f for f in input_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]))
    random.seed(42)
    shuffle(all_images)

    if args.num_images:
        all_images = all_images[: args.num_images]
    num_images = len(all_images)

    print(f"Copying {len(all_images)} images to {images_path}...")
    for img in all_images:
        shutil.copy2(os.path.join(input_path, img), os.path.join(images_path, img))

    t1 = time.time()

    if args.choice == "colmap":
        profiling = run_colmap_pipeline(base_out, images_path, db_path, sparse_path, low_view_count=num_images < 50)
    elif args.choice == "vggt":
        profiling = run_vggt_pipeline(base_out)
    else:
        raise ValueError("Invalid choice")

    total_time = time.time() - t1

    save_timing(input_path, name, args.choice, num_images, profiling)

    print(f"\nPipeline finished. Results saved in: {base_out}")
    print(f"Time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
