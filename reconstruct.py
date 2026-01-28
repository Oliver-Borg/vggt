import argparse
import datetime
import json
import os
from random import shuffle
import random
import shutil
import subprocess
import sys
import time
import torch

from demo_colmap import get_gpu_stats, run_vggt


def run_command(cmd: list[str]):
    """Executes a shell command and ensures it succeeds."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def run_colmap_pipeline(base_out: str, images_path: str, db_path: str, sparse_path: str) -> tuple[float, float, float]:
    """Executes the standard COLMAP SfM stages."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t1 = time.time()
    run_command(["colmap", "feature_extractor", "--database_path", db_path, "--image_path", images_path])

    run_command(["colmap", "exhaustive_matcher", "--database_path", db_path])

    run_command(
        ["colmap", "mapper", "--database_path", db_path, "--image_path", images_path, "--output_path", sparse_path]
    )
    total_time = time.time() - t1

    peak_alloc, peak_res = get_gpu_stats()
    return total_time, peak_alloc, peak_res


def run_vggt_pipeline(base_out: str) -> tuple[float, float, float]:
    """Executes the VGGT transformer-based reconstruction and returns (time, peak_gpu_memory_mb)."""

    total_time = run_vggt(scene_dir=base_out)

    peak_alloc, peak_res = get_gpu_stats()
    return total_time, peak_alloc, peak_res


def save_timing(
    input_path: str,
    name: str,
    choice: str,
    num_images: int,
    total_time: float,
    peak_alloc_gpu_mem: float,
    peak_reserved_gpu_mem: float,
) -> None:
    stats: list[dict[str, str | int | float]] = []
    try:
        with open("stats.json", "r") as f:
            stats = json.load(f)
    except Exception:
        stats = []

    stat: dict[str, str | int | float] = {
        "date": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
        "input_path": input_path,
        "name": name,
        "type": choice,
        "num_images": num_images,
        "total_time": round(total_time, 3),
        "peak_alloc_gpu_mem_mb": round(peak_alloc_gpu_mem, 3),
        "peak_reserved_gpu_mem_mb": round(peak_reserved_gpu_mem, 3),
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

    if args.choice == "colmap":
        total_time, peak_alloc, peak_res = run_colmap_pipeline(base_out, images_path, db_path, sparse_path)
    elif args.choice == "vggt":
        total_time, peak_alloc, peak_res = run_vggt_pipeline(base_out)
    else:
        total_time, peak_alloc, peak_res = 0, 0, 0

    save_timing(input_path, name, args.choice, num_images, total_time, peak_alloc, peak_res)

    print(f"\nPipeline finished. Results saved in: {base_out}")
    print(
        f"Time: {total_time:.2f}s | Peak allocated GPU Mem: {peak_alloc:.2f} MB | Peak reserved GPU Mem: {peak_res:.2f} MB"
    )


if __name__ == "__main__":
    main()
