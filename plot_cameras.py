import argparse
import numpy as np
import matplotlib.pyplot as plt
import pycolmap
from pathlib import Path
from typing import get_args

from reconstruct import get_image_list, IMAGE_MODE
from combine_clouds import load_point_cloud


def plot_cameras(
    pcd: pycolmap.Reconstruction, image_counts: list[int], seed: int, image_mode: IMAGE_MODE, out_path: Path
):
    # 1. Gather all base images from the point cloud to evaluate
    all_images = [img.name for img in pcd.images.values()]

    # 2. Count memberships across all specified sizes
    membership_counts = {name: 0 for name in all_images}

    for count in image_counts:
        # Pass a copy of the list to avoid mutating the original sorting sequence mid-loop
        subset = get_image_list(all_images.copy(), count, seed, image_mode, pcd)
        for img_name in subset:
            membership_counts[img_name] += 1

    # 3. Extract camera poses and look-directions for plotting
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

    # 4. Render the plot
    plt.figure(figsize=(12, 10))

    max_membership = len(image_counts)
    cmap = "coolwarm"

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    colors_arr = np.array(colors)
    sizes_arr = np.array(sizes)
    is_ups_arr = np.array(is_ups)

    mask_up = is_ups_arr
    mask_down = ~is_ups_arr

    scatter = None

    # Scatter plots for the camera centers looking DOWN (o)
    if np.any(mask_down):
        scatter = plt.scatter(
            xs_arr[mask_down],
            ys_arr[mask_down],
            c=colors_arr[mask_down],
            cmap=cmap,
            vmin=0,
            vmax=max_membership,
            s=sizes_arr[mask_down],
            zorder=2,
            edgecolors="k",
            linewidths=0.5,
            marker="o",
        )

    # Scatter plots for the camera centers looking UP (x)
    if np.any(mask_up):
        scatter_up = plt.scatter(
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
            scatter = scatter_up

    # Quiver plot to simulate 2D frustum direction
    plt.quiver(xs, ys, dxs, dys, colors, cmap=cmap, angles="xy", scale_units="xy", scale=0.8, width=0.003, zorder=1)

    if scatter is not None:
        cbar = plt.colorbar(scatter)
        cbar.set_label("Set Membership Count", fontsize=12)
        cbar.set_ticks(range(max_membership + 1))

    plt.title(f"Top-Down Camera Poses\n(Mode: {image_mode} | Seed: {seed})", fontsize=14)
    plt.xlabel("World X", fontsize=12)
    plt.ylabel("World Y", fontsize=12)
    plt.axis("equal")  # Prevents distortion of the frustum directions
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save the plot
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot top-down cameras color-coded by sequence membership.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth sparse reconstruction")
    parser.add_argument(
        "--image_counts", type=int, nargs="+", required=True, help="List of image counts to process (e.g. 10 20 50)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random shuffling of images")

    args = parser.parse_args()

    # Generate serialized arguments for the filename
    counts_str = "-".join(map(str, args.image_counts))

    print(f"Loading point cloud from {args.gt_path}...")
    pcd = load_point_cloud(args.gt_path)

    print("Calculating subset memberships and generating plot...")
    for image_mode in list(get_args(IMAGE_MODE)):
        serialized_args = f"mode_{image_mode}_seed_{args.seed}_counts_{counts_str}"

        out_dir = Path("plots/cameras")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{serialized_args}.png"
        plot_cameras(pcd, args.image_counts, args.seed, image_mode, out_path)
