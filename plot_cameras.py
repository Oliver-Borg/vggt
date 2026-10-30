import argparse
from collections import defaultdict
from dataclasses import dataclass
from line_profiler import profile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.colors as clrs
import matplotlib.patches as patches
from pathlib import Path

from .cam_alignment import get_alignment_rotation
from .cam_utils import load_image_sizes_from_json, load_intrinsics_from_json, load_poses_from_json


@dataclass
class CameraSeries:
    label: str
    colour: tuple[float, float, float] | tuple[float, float, float, float]
    poses: dict[str, np.ndarray]
    params: dict[str, np.ndarray]
    image_sizes: dict[str, tuple[int, int]]
    gt_poses: dict[str, np.ndarray]
    gt_params: dict[str, np.ndarray]
    gt_image_sizes: dict[str, tuple[int, int]]
    image_names: list[str]
    image_name_key: str = "default"


def get_camera_triangle(
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    size: float = 1.0,
):
    """
    Create top down frustum. Return three points for the triangle.

    Args:
        pose: (4, 4) Camera-to-World matrix.
        intrinsics: (3, 3) Pinhole intrinsics matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        image_size: (W, H) tuple of image dimensions.
    """
    W, H = image_size
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 1. Define the 4 corners of the image plane + the camera center (0,0,0)
    # The depth of the frustum base is determined by 'size' (z = size)
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy

    # Corners in pixel coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    corners_pix = np.array([[0, 0], [W, 0], [W, H], [0, H]])

    corners_cam = np.zeros((5, 3))
    # Point 0 is the camera center
    corners_cam[0] = [0, 0, 0]

    # Points 1-4 are the image corners projected to depth = size
    corners_cam[1:, 0] = (corners_pix[:, 0] - cx) * size / fx
    corners_cam[1:, 1] = (corners_pix[:, 1] - cy) * size / fy
    corners_cam[1:, 2] = size

    # 2. Transform to world coordinates
    # Apply rotation (3x3) and translation (3x1)
    pts_world = (c2w[:3, :3] @ corners_cam.T + c2w[:3, 3:4]).T
    return pts_world[:3]


def _get_global_alignment(
    series_list: list[CameraSeries],
) -> np.ndarray | None:
    """Gathers all poses and computes a single alignment rotation."""
    all_poses_for_alignment = []
    for series in series_list:
        all_poses_for_alignment.extend(
            v for k, v in series.gt_poses.items() if series.image_names is None or k in series.image_names
        )
        all_poses_for_alignment.extend(
            v for k, v in series.poses.items() if series.image_names is None or k in series.image_names
        )

    if not all_poses_for_alignment:
        return None

    return get_alignment_rotation(np.array(all_poses_for_alignment))


def _calculate_all_errors(
    series_list: list[CameraSeries],
    R_align_global: np.ndarray,
) -> tuple[defaultdict, float]:
    """Calculates all RTEs for all series given a global alignment."""
    all_errors = defaultdict(dict)
    global_max_rte = 0.0

    for series in series_list:
        series_errors = {}
        # Align GT centers once for this specific series
        gt_centers_aligned = {
            k: R_align_global @ v[:3, 3]
            for k, v in series.gt_poses.items()
            if series.image_names is None or k in series.image_names
        }

        for k, v in series.poses.items():
            if k not in gt_centers_aligned:
                continue

            pred_center_aligned = R_align_global @ v[:3, 3]
            gt_center_aligned = gt_centers_aligned[k]
            rte = np.linalg.norm(pred_center_aligned - gt_center_aligned)

            series_errors[k] = rte
            global_max_rte = max(global_max_rte, rte)
        all_errors[series.label] = series_errors

    return all_errors, global_max_rte


@profile
def plot_cameras(
    poses: dict[str, np.ndarray],
    intrinsics: dict[str, np.ndarray],
    image_sizes: dict[str, tuple[int, int]],
    colors: dict[str, tuple[float, float, float, float] | tuple[float, float, float] | None],
    title: str | None = None,
    linked_poses: dict[str, list[str]] | None = None,
    use_error: bool = False,
    gt_label: str | None = None,
    max_error_override: float | None = None,
    R_align_override: np.ndarray | None = None,
    errors_override: dict[str, float] | None = None,
    use_arrows: bool = True,
):
    ax = plt.gca()
    poses_list = list(poses.values())

    if not poses_list:
        return ax

    if R_align_override is not None:
        R_align = R_align_override
    else:
        R_align = get_alignment_rotation(np.array(poses_list))

    aligned_centers = {name: R_align @ c2w[:3, 3] for name, c2w in poses.items()}

    frustum_points = {
        name: get_camera_triangle(c2w, intrinsics[name], image_sizes[name]) for name, c2w in poses.items()
    }

    if use_error and linked_poses and gt_label:
        errors = {}
        if errors_override is not None:
            errors = errors_override
        else:
            # For each linked group, calculate RTE if not provided
            for link_group in linked_poses.values():
                if len(link_group) < 2:
                    continue

                gt_pose_name = next((name for name in link_group if name.startswith(gt_label)), None)
                pred_pose_names = [name for name in link_group if not name.startswith(gt_label)]

                if gt_pose_name and gt_pose_name in aligned_centers:
                    gt_center = aligned_centers[gt_pose_name]
                    errors[gt_pose_name] = 0.0  # GT error is 0
                    for pred_name in pred_pose_names:
                        if pred_name in aligned_centers:
                            pred_center = aligned_centers[pred_name]
                            rte = np.linalg.norm(pred_center - gt_center)
                            errors[pred_name] = rte

        if errors:
            max_error = (
                max_error_override if max_error_override is not None else (max(errors.values()) if errors else 1.0)
            )
            max_error = max(max_error, 1e-9)  # Ensure max_error is positive for LogNorm

            min_error = 1e-3  # Smallest value for log scale, avoids log(0)

            cmap = plt.get_cmap("jet")

            # Use LogNorm for a logarithmic color scale
            norm = clrs.LogNorm(vmin=min_error, vmax=max_error)

            # Create a color map based on errors
            error_colors = {name: cmap(norm(error)) if error > 0 else cmap(0.0) for name, error in errors.items()}
            for name, color in colors.items():
                if color is None and name in error_colors:
                    colors[name] = error_colors[name]

            # Add a colorbar to the plot
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
            cbar.set_label("Translation Error (RTE)")

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
    triangles = []
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
        frustum_list = list(frustum_points[name])
        for i in range(len(frustum_list)):
            frustum_list[i] = R_align @ frustum_list[i]
        assert (frustum_list[0] == center).all()
        aligned_frustum_points = np.array(frustum_list)

        # True if looking up (positive Z), False if looking down
        is_ups.append(look_dir[2] > 0)
        look_dir[:norm_dim] = look_dir[:norm_dim] / np.linalg.norm(look_dir[:norm_dim]) * line_length
        look_dir[2] = 0.0

        triangles.append(aligned_frustum_points[:, :2])

        xs.append(center[0])
        ys.append(center[1])  # Projecting onto Top-down X-Y plane

        dxs.append(look_dir[0])
        dys.append(look_dir[1])
        colors_list.append(colors.get(name) or (0.5, 0.5, 0.5, 1.0))  # Default to gray if missing

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
            edgecolors="none",
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

    if use_arrows:
        ax.quiver(xs, ys, dxs, dys, color=colors_list, angles="xy", scale_units="xy", scale=0.5, width=0.0015, zorder=1)
    else:
        for triangle, color in zip(triangles, colors_list):
            opa_color = (*color[:3], 0.5)
            ax.add_patch(patches.Polygon(triangle, color=opa_color, fill=True, zorder=1))

    if title is not None:
        ax.set_title(title, fontsize=12)

    ax.axis("equal")
    ax.margins(0.02)
    ax.grid(True, linestyle="--", alpha=0.6)

    return ax


@profile
def plot_extrinsics(
    groups_data: list[tuple[str, list[CameraSeries]]],
    output_path: Path,
    max_cols: int = 3,
    use_error_colors: bool = False,
    max_error_override: float | None = None,
    stack_datasets_horizontally: bool = True,
):
    if not groups_data:
        return

    group_layouts = []
    for name, series_list in groups_data:
        num_series = len(series_list)
        cols = min(num_series, max_cols)
        rows = (num_series + cols - 1) // cols if cols > 0 else 1
        group_layouts.append((rows, cols))

    # Calculate overall grid dimensions based on stacking preference
    if stack_datasets_horizontally:
        total_rows = max(r for r, c in group_layouts) if group_layouts else 1
        total_cols = sum(c for r, c in group_layouts) if group_layouts else 1
    else:
        total_rows = sum(r for r, c in group_layouts) if group_layouts else 1
        total_cols = max(c for r, c in group_layouts) if group_layouts else 1

    fig = plt.figure(figsize=(5 * total_cols, 4.5 * total_rows))

    if use_error_colors:
        min_error_for_norm = 1e-3  # Matching plot_cameras
        cmap = plt.get_cmap("jet")

    gt_label = "GT_"
    gt_colour = (0.5, 0.5, 0.5, 1.0)

    row_offset = 0
    col_offset = 0

    for g_idx, (group_name, series_list) in enumerate(groups_data):
        r_g, c_g = group_layouts[g_idx]
        if len(series_list) == 0:
            continue

        R_align_global = _get_global_alignment(series_list)
        if use_error_colors and R_align_global is not None:
            all_errors, group_max_rte = _calculate_all_errors(series_list, R_align_global)
            if max_error_override is not None:
                group_max_rte = max_error_override
        else:
            all_errors, group_max_rte = defaultdict(dict), 0.0

        if use_error_colors:
            norm = clrs.LogNorm(vmin=min_error_for_norm, vmax=max(group_max_rte, min_error_for_norm + 1e-9))

        for idx, series in enumerate(series_list):
            r = idx // c_g
            c = idx % c_g

            # Determine the subplot index in the global grid
            if stack_datasets_horizontally:
                ax_idx = r * total_cols + col_offset + c + 1
            else:
                ax_idx = (row_offset + r) * total_cols + c + 1

            ax = plt.subplot(total_rows, total_cols, ax_idx)
            plt.sca(ax)

            combined_poses: dict[str, np.ndarray] = {}
            combined_intrinsics: dict[str, np.ndarray] = {}
            combined_image_sizes: dict[str, tuple[int, int]] = {}
            colours = {}
            linked_poses = defaultdict(list)

            # Add GT series
            for k, v in series.gt_poses.items():
                if series.image_names is not None and k not in series.image_names:
                    continue
                combined_poses[gt_label + k] = v
                combined_intrinsics[gt_label + k] = series.gt_params[k]
                combined_image_sizes[gt_label + k] = series.gt_image_sizes[k]
                colours[gt_label + k] = gt_colour
                linked_poses[k].append(gt_label + k)

            # Add the current series being compared
            for k, v in series.poses.items():
                if series.image_names is not None and k not in series.image_names:
                    continue
                combined_poses[series.label + k] = v
                combined_intrinsics[series.label + k] = series.params[k]
                combined_image_sizes[series.label + k] = series.image_sizes[k]
                if use_error_colors:
                    colours[series.label + k] = None
                else:
                    colours[series.label + k] = series.colour
                linked_poses[k].append(series.label + k)

            # Prepare errors for the current subplot
            subplot_errors = {}
            if use_error_colors:
                pred_errors_by_key = all_errors.get(series.label, {})
                for k, rte in pred_errors_by_key.items():
                    subplot_errors[series.label + k] = rte
                # Add GT errors (which are 0)
                for k in pred_errors_by_key:
                    if (gt_label + k) in combined_poses:
                        subplot_errors[gt_label + k] = 0.0

            plot_cameras(
                combined_poses,
                combined_intrinsics,
                combined_image_sizes,
                colours,
                linked_poses=linked_poses,
                use_error=use_error_colors,
                gt_label=gt_label,
                max_error_override=group_max_rte if use_error_colors else None,
                R_align_override=R_align_global,
                errors_override=subplot_errors if use_error_colors else None,
            )

            # Calculate and display metrics on the subplot
            local_rtes = list(all_errors.get(series.label, {}).values())

            if local_rtes:
                avg_rte = np.mean(local_rtes)
                num_aligned = len(local_rtes)
                text_str = f"Avg. RTE: {avg_rte:.4f}\nNum Aligned: {num_aligned}"
                ax.text(
                    0.95,
                    0.95,
                    text_str,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                )

            # Create legend handles for the GT and the current series subplot
            legend_elements = []
            if any(k.startswith(gt_label) for k in combined_poses):
                legend_elements.append(
                    lines.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label="Ground Truth",
                        markerfacecolor=gt_colour,
                        markersize=10,
                        markeredgecolor="k",
                    )
                )
            if any(k.startswith(series.label) for k in combined_poses):
                if use_error_colors:
                    local_max_rte = max(local_rtes) if local_rtes else 0.0
                    pred_marker_face_color = cmap(norm(local_max_rte)) if local_max_rte > 0 else cmap(0.0)
                else:
                    pred_marker_face_color = series.colour

                legend_elements.append(
                    lines.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=series.label,
                        markerfacecolor=pred_marker_face_color,
                        markersize=10,
                        markeredgecolor="k",
                    )
                )
            else:
                legend_elements.append(
                    lines.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=series.label + "\nNot Found",
                        markerfacecolor="black",
                        markersize=10,
                        markeredgecolor="k",
                    )
                )

            if legend_elements:
                ax.legend(
                    handles=legend_elements,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    ncol=1,
                    frameon=True,
                    fancybox=True,
                    shadow=False,
                    framealpha=0.9,
                )

        if stack_datasets_horizontally:
            col_offset += c_g
        else:
            row_offset += r_g

    plt.tight_layout(rect=(0, 0.05, 1, 1))

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    jpg_path = output_path / "camera_alignment.jpg"
    pdf_path = output_path / "camera_alignment.pdf"

    plt.savefig(jpg_path, dpi=100, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=100)
    print(f"Dashboard saved to {jpg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot top-down cameras color-coded by sequence membership.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth json cameras")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the ground truth json cameras")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the plot")
    parser.add_argument("--num_images", type=int, required=True, help="Num images from GT")

    args = parser.parse_args()

    assert args.gt_path.endswith(".json")
    assert args.pred_path.endswith(".json")

    gt_poses = load_poses_from_json(Path(args.gt_path))
    gt_intrinsics = load_intrinsics_from_json(Path(args.gt_path))
    gt_imsizes = load_image_sizes_from_json(Path(args.gt_path))

    pred_poses = load_poses_from_json(Path(args.pred_path))
    pred_intrinsics = load_intrinsics_from_json(Path(args.pred_path))
    pred_imsizes = load_image_sizes_from_json(Path(args.pred_path))

    pred_series = CameraSeries(
        "pred_",
        (1.0, 0.0, 0.0, 0.5),
        pred_poses,
        pred_intrinsics,
        pred_imsizes,
        gt_poses,
        gt_intrinsics,
        gt_imsizes,
        list(gt_poses.keys()),
    )

    plot_extrinsics(
        groups_data=[("default", [pred_series], None)],
        output_path=Path(args.output_path),
        use_error_colors=True,
    )
