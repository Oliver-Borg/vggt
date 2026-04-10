import argparse
import numpy as np
import matplotlib.pyplot as plt
import pycolmap
from pathlib import Path
from typing import get_args
from collections import Counter

# Importing functions and types directly from the existing scripts
from reconstruct import get_image_list, IMAGE_MODE
from combine_clouds import load_point_cloud


def plot_cameras(
    pcd: pycolmap.Reconstruction,
    membership_counts: dict[str, int],
    image_counts: list[int],
    seed: int,
    image_mode: IMAGE_MODE,
    ax=None,
):
    xs, ys = [], []
    dxs, dys = [], []
    colors = []
    is_ups = []
    sizes = []

    line_length = 0.3
    norm_dim = 3

    for img_id, image in pcd.images.items():
        name = image.name
        count = membership_counts.get(name, 0)
        if count == 0:
            continue

        # Compatibility block for different pycolmap versions
        try:
            # Newer versions use cam_from_world
            R = image.cam_from_world.rotation.matrix()
            t = image.cam_from_world.translation
        except AttributeError:
            # Older versions use qvec/tvec / rotmat()
            R = image.rotmat()
            t = image.tvec

        # Camera center in world coordinates: -R^T * t
        center = -R.T @ t
        # Look direction in world coordinates: R^T * [0, 0, 1]^T
        look_dir = R.T @ np.array([0, 0, 1])

        # True if looking up (positive Z), False if looking down
        is_ups.append(look_dir[2] > 0)
        look_dir[:norm_dim] = look_dir[:norm_dim] / np.linalg.norm(look_dir[:norm_dim]) * line_length
        look_dir[2] = 0.0

        xs.append(center[0])
        ys.append(center[1])  # Projecting onto Top-down X-Y plane

        dxs.append(look_dir[0])
        dys.append(look_dir[1])
        colors.append(count)

        # Scale the size dynamically based on membership count
        sizes.append(20 + count * 20)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()

    max_membership = len(image_counts)
    cmap = "coolwarm"

    xs_arr, ys_arr, colors_arr, sizes_arr, is_ups_arr = map(np.array, [xs, ys, colors, sizes, is_ups])
    mask_up, mask_down = is_ups_arr, ~is_ups_arr

    scatter = None

    # Scatter plots for the camera centers looking DOWN (o)
    if np.any(mask_down):
        scatter = ax.scatter(
            xs_arr[mask_down],
            ys_arr[mask_down],
            c=colors_arr[mask_down],
            cmap=cmap,
            vmin=0,
            vmax=max_membership,
            s=sizes_arr[mask_down],
            zorder=2,
            edgecolors="k",
            marker="o",
        )

    # Scatter plots for the camera centers looking UP (x)
    if np.any(mask_up):
        sc_up = ax.scatter(
            xs_arr[mask_up],
            ys_arr[mask_up],
            c=colors_arr[mask_up],
            cmap=cmap,
            vmin=0,
            vmax=max_membership,
            s=sizes_arr[mask_up],
            zorder=2,
            linewidths=1.5,
            marker="x",
        )
        if scatter is None:
            scatter = sc_up

    ax.quiver(xs, ys, dxs, dys, colors, cmap=cmap, angles="xy", scale_units="xy", scale=0.8, width=0.003, zorder=1)

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Set Membership Count", fontsize=10)
        cbar.set_ticks(range(max_membership + 1))

    ax.set_title(f"Top-Down Camera Poses\n(Mode: {image_mode} | Seed: {seed})", fontsize=12)
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.6)


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

    fig = plt.figure(figsize=(16, 12))
    ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    ax_bar = plt.subplot2grid((3, 2), (2, 0))
    ax_dist = plt.subplot2grid((3, 2), (2, 1))

    plot_cameras(pcd, membership_counts, image_counts, seed, image_mode, ax=ax_main)
    plot_membership_bars(membership_counts, image_counts, ax=ax_bar)
    plot_membership_distribution(membership_counts, image_counts, ax=ax_dist)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot top-down cameras color-coded by sequence membership.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth sparse reconstruction")
    parser.add_argument("--image_counts", type=int, nargs="+", required=True, help="List of image counts")
    parser.add_argument("--seed", type=int, default=42, help="Seed for selection")

    args = parser.parse_args()
    pcd = load_point_cloud(args.gt_path)

    print("Generating dashboards...")
    for image_mode in list(get_args(IMAGE_MODE)):
        out_dir = Path("plots/cameras")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir / f"dashboard_{image_mode}_seed_{args.seed}_counts_{'-'.join(map(str, args.image_counts))}.png"
        )
        plot_dashboard(pcd, args.image_counts, args.seed, image_mode, out_path)
