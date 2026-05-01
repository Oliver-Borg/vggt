import argparse
from pathlib import Path
from typing import get_args
from collections import Counter

from matplotlib import pyplot as plt
import matplotlib.colors as clrs

from combine_clouds import load_point_cloud
from plot_cameras import plot_cameras
from pycolmap_utils import get_poses
from reconstruct import get_image_list
from reconstruct_args import IMAGE_MODE


def plot_membership_bars(membership_counts: dict[str, int], image_counts: list[int], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    sorted_counts = sorted(membership_counts.values(), reverse=True)

    ax.bar(range(len(sorted_counts)), sorted_counts, color="skyblue", edgecolor="black", linewidth=0.2)
    ax.set_title("Membership Count per Image Index")
    ax.set_ylabel("Count")
    ax.set_xticks([])
    ax.set_yticks(range(1, len(image_counts) + 1))
    ax.grid(axis="y", linestyle="--", alpha=0.6)


def plot_membership_distribution(membership_counts: dict[str, int], image_counts: list[int], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    dist = Counter(membership_counts.values())
    x_vals = list(range(1, len(image_counts) + 1))
    y_vals = [dist.get(x, 0) for x in x_vals]

    bars = ax.bar(x_vals, y_vals, color="coral", edgecolor="black")
    ax.set_title("Membership Distribution")
    ax.set_xlabel("Membership Value")
    ax.set_ylabel("Number of Cameras")
    ax.set_xticks(x_vals)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def plot_dashboard(pcd, image_counts, seed, image_mode, out_path):
    all_images = [img.name for img in pcd.images.values()]

    membership_counts = {name: 0 for name in all_images}
    for count in image_counts:
        subset = get_image_list(all_images.copy(), count, seed, image_mode, pcd)
        for img_name in subset:
            membership_counts[img_name] += 1

    # Filter out cameras with 0 membership counts
    membership_counts = {k: v for k, v in membership_counts.items() if v > 0}

    all_poses = get_poses(pcd)
    poses = {k: all_poses[k] for k in membership_counts.keys() if k in all_poses}
    max_membership = len(image_counts)
    cmap = plt.get_cmap("coolwarm")
    colors = {k: cmap(v / max_membership)[:3] for k, v in membership_counts.items()}

    fig = plt.figure(figsize=(16, 12))
    ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    ax_bar = plt.subplot2grid((3, 2), (2, 0))
    ax_dist = plt.subplot2grid((3, 2), (2, 1))

    # Set ax_main as active context so plot_cameras hooks into the dashboard correctly
    plt.sca(ax_main)
    title = f"Aligned Top-Down Camera Poses\n(Mode: {image_mode} | Seed: {seed})"
    plot_cameras(poses, colors, title=title)
    norm = clrs.Normalize(vmin=0, vmax=max_membership)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_main)
    cbar.set_label("Set Membership Count", fontsize=10)
    cbar.set_ticks(range(max_membership + 1))

    plot_membership_bars(membership_counts, image_counts, ax=ax_bar)
    plot_membership_distribution(membership_counts, image_counts, ax=ax_dist)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard saved to {out_path}")
    plt.show()


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
            out_dir / f"{args.dataset_name}_dashboard_{image_mode}_"
            f"seed_{args.seed}_counts_{'-'.join(map(str, args.image_counts))}.png"
        )
        plot_dashboard(pcd, args.image_counts, args.seed, image_mode, out_path)
