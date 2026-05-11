import argparse
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pathlib import Path

from .cam_alignment import get_alignment_rotation
from .cam_utils import load_poses_from_json


@dataclass
class CameraSeries:
    label: str
    colour: tuple[float, float, float] | tuple[float, float, float, float]
    poses: dict[str, np.ndarray]


def plot_cameras(
    poses: dict[str, np.ndarray],
    colors: dict[str, tuple[float, float, float, float]] | dict[str, tuple[float, float, float]],
    title: str | None = None,
    linked_poses: dict[str, list[str]] | None = None,
):
    ax = plt.gca()
    poses_list = list(poses.values())

    if not poses_list:
        return ax

    R_align = get_alignment_rotation(np.array(poses_list))

    aligned_centers = {name: R_align @ c2w[:3, 3] for name, c2w in poses.items()}

    if linked_poses:
        for link_group in linked_poses.values():
            if len(link_group) > 1:
                first_pose_name = link_group[0]
                if first_pose_name not in aligned_centers:
                    continue
                first_center = aligned_centers[first_pose_name]
                for other_pose_name in link_group[1:]:
                    if other_pose_name not in aligned_centers:
                        continue
                    other_center = aligned_centers[other_pose_name]
                    ax.plot(
                        [first_center[0], other_center[0]],
                        [first_center[1], other_center[1]],
                        color="gray",
                        linestyle="--",
                        linewidth=0.5,
                        zorder=0,
                    )

    xs, ys = [], []
    dxs, dys = [], []
    colors_list = []
    is_ups = []

    line_length = 0.1
    norm_dim = 3

    for name, c2w in poses.items():
        # Apply alignment
        # Camera center in world coordinates: c2w[:3, 3]
        center = aligned_centers[name]
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

    dxs = [dx * max_size for dx in dxs]
    dys = [dy * max_size for dy in dys]

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

    ax.quiver(xs, ys, dxs, dys, color=colors_list, angles="xy", scale_units="xy", scale=0.5, width=0.0015, zorder=1)

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.axis("equal")
    ax.margins(0.02)
    ax.grid(True, linestyle="--", alpha=0.6)

    return ax


def plot_extrinsics(
    gt_series: CameraSeries,
    series_list: list[CameraSeries],
    output_path: Path,
    image_names: list[str] | None = None,
    max_cols: int = 4,
):
    num_series = len(series_list)
    if num_series == 0:
        return

    cols = min(num_series, max_cols)
    rows = (num_series + cols - 1) // cols

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    for idx, series in enumerate(series_list):
        ax = plt.subplot(rows, cols, idx + 1)
        plt.sca(ax)

        combined_poses: dict[str, np.ndarray] = {}
        colours = {}
        linked_poses = defaultdict(list)

        # Add GT series
        for k, v in gt_series.poses.items():
            if image_names is not None and k not in image_names:
                continue
            combined_poses[gt_series.label + k] = v
            colours[gt_series.label + k] = gt_series.colour
            linked_poses[k].append(gt_series.label + k)

        # Add the current series being compared
        for k, v in series.poses.items():
            if image_names is not None and k not in image_names:
                continue
            combined_poses[series.label + k] = v
            colours[series.label + k] = series.colour
            linked_poses[k].append(series.label + k)

        plot_cameras(combined_poses, colours, linked_poses=linked_poses)

        # Create legend handles for the GT and the current series subplot
        legend_elements = []
        if any(k.startswith(gt_series.label) for k in combined_poses):
            legend_elements.append(
                lines.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=gt_series.label,
                    markerfacecolor=gt_series.colour,
                    markersize=10,
                    markeredgecolor="k",
                )
            )
        if any(k.startswith(series.label) for k in combined_poses):
            legend_elements.append(
                lines.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=series.label,
                    markerfacecolor=series.colour,
                    markersize=10,
                    markeredgecolor="k",
                )
            )

        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper right", frameon=True, shadow=True)

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
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth json cameras")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the ground truth json cameras")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the plot")
    parser.add_argument("--num_images", type=int, required=True, help="Num images from GT")

    args = parser.parse_args()

    assert args.gt_path.endswith(".json")
    assert args.pred_path.endswith(".json")

    gt_series = CameraSeries(
        "gt_",
        (0.0, 0.0, 1.0, 0.5),
        load_poses_from_json(Path(args.gt_path)),
    )
    pred_series = CameraSeries(
        "pred_",
        (1.0, 0.0, 0.0, 0.5),
        load_poses_from_json(Path(args.pred_path)),
    )

    plot_extrinsics(
        gt_series=gt_series,
        series_list=[pred_series],
        output_path=Path(args.output_path),
        image_names=None,
    )
