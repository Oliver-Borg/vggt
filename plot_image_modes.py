import argparse
from pathlib import Path
from typing import get_args

from combine_clouds import load_point_cloud
from plot_cameras import plot_dashboard
from reconstruct_args import IMAGE_MODE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot top-down cameras color-coded by sequence membership.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth sparse reconstruction")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--image_counts", type=int, nargs="+", required=True, help="List of image counts")
    parser.add_argument("--seed", type=int, default=42, help="Seed for selection")

    args = parser.parse_args()
    pcd = load_point_cloud(args.gt_path)

    print("Generating dashboards...")
    for image_mode in list(get_args(IMAGE_MODE)):
        out_dir = Path("plots/cameras")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir
            / f"{args.dataset_name}_dashboard_{image_mode}_"
            f"seed_{args.seed}_counts_{'-'.join(map(str, args.image_counts))}.png"
        )
        plot_dashboard(pcd, args.image_counts, args.seed, image_mode, out_path)
