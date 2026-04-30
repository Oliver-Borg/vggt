import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from pathlib import Path
from collections import Counter

from cam_alignment import get_alignment_rotation
from cam_utils import get_poses
from reconstruct import get_image_list
from combine_clouds import align_to_world_space, load_point_cloud


def plot_cameras(
    poses: dict[str, np.ndarray],
    colors: dict[str, tuple[float, float, float, float]] | dict[str, tuple[float, float, float]],
    title: str | None = None,
):
    ax = plt.gca()
    poses_list = list(poses.values())

    if not poses_list:
        return ax

    R_align = get_alignment_rotation(np.array(poses_list))

    xs, ys = [], []
    dxs, dys = [], []
    colors_list = []
    is_ups = []

    line_length = 0.1
    norm_dim = 3

    for name, c2w in poses.items():
        # Apply alignment
        # Camera center in world coordinates: c2w[:3, 3]
        center = R_align @ c2w[:3, 3]
        # Look direction in world coordinates: Z-axis of c2w
        look_dir = R_align @ c2w[:3, 2]

        # True if looking up (positive Z), False if looking down
        is_ups.append(look_dir[2] > 0)
        look_dir[:norm_dim] = look_dir[:norm_dim] / np.linalg.norm(look_dir[:norm_dim]) * line_length
        look_dir[2] = 0.0

        xs.append(center[0])
        ys.append(center[1])  # Projecting onto Top-down X-Y plane

        dxs.append(look_dir[0])
        dys.append(look_dir[1])
        colors_list.append(colors.get(name, (0, 0, 0)))  # Default to black if missing

    xs_arr, ys_arr, colors_arr, is_ups_arr = map(np.array, [xs, ys, colors_list, is_ups])

    world_width = np.ptp(xs_arr)
    world_height = np.ptp(ys_arr)

    max_size = max(world_width, world_height)

    dxs = [dx / max_size for dx in dxs]
    dys = [dy / max_size for dy in dys]

    mask_up, mask_down = is_ups_arr, ~is_ups_arr

    # Scatter plots for the camera centers looking DOWN (o)
    if np.any(mask_down):
        ax.scatter(
            xs_arr[mask_down],
            ys_arr[mask_down],
            c=colors_arr[mask_down],
            s=40,
            zorder=2,
            edgecolors="k",
            marker="o",
        )

    # Scatter plots for the camera centers looking UP (x)
    if np.any(mask_up):
        ax.scatter(
            xs_arr[mask_up],
            ys_arr[mask_up],
            c=colors_arr[mask_up],
            s=40,
            zorder=2,
            linewidths=1.5,
            marker="x",
        )

    ax.quiver(xs, ys, dxs, dys, color=colors_list, angles="xy", scale_units="xy", scale=0.5, width=0.003, zorder=1)

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.6)

    return ax


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


def plot_extrinsics(gt_path: Path, pred_path: Path, output_path: Path, image_names: list[str] | None = None):
    gt_pcd = load_point_cloud(gt_path)
    pred_pcd = load_point_cloud(pred_path)

    pred_pcd = align_to_world_space(pred_pcd, gt_pcd)

    gt_poses = get_poses(gt_pcd)
    pred_poses = get_poses(pred_pcd)

    combined_poses: dict[str, np.ndarray] = {}
    for k, v in gt_poses.items():
        if image_names is not None and k not in image_names:
            continue
        combined_poses["gt_" + k] = v

    for k, v in pred_poses.items():
        if image_names is not None and k not in image_names:
            continue
        combined_poses["pred_" + k] = v

    # Red for pred, blue for GT
    colours = {k: (1.0, 0.0, 0.0, 0.5) if k.startswith("gt_") else (0.0, 0.0, 1.0, 0.5) for k in combined_poses.keys()}

    fig = plt.figure(figsize=(16, 12))
    ax_main = plt.subplot2grid((1, 1), (0, 0))
    plt.sca(ax_main)
    title = "Aligned Top-Down Camera Poses"

    plot_cameras(combined_poses, colours, title=title)
    plt.tight_layout()

    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    png_path = output_path / "camera_alignment.png"
    pdf_path = output_path / "camera_alignment.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Dashboard saved to {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot top-down cameras color-coded by sequence membership.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth sparse reconstruction")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the ground truth sparse reconstruction")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the plot")
    parser.add_argument("--num_images", type=int, required=True, help="Num images from GT")

    args = parser.parse_args()

    plot_extrinsics(
        gt_path=Path(args.gt_path),
        pred_path=Path(args.pred_path),
        output_path=Path(args.output_path),
        image_names=None,
    )
