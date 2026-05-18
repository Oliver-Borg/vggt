import os
from pathlib import Path


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import pycolmap

from .cam_alignment import get_alignment_rotation

chrome_path = Path("~/work/git/gsplat/chrome/chrome-linux64/chrome").expanduser()
os.environ["BROWSER_PATH"] = str(chrome_path)


def load_pcd_from_sparse(sparse_dir: Path):
    """Loads 3D points, colors, and camera poses from a COLMAP sparse reconstruction directory."""
    if (sparse_dir / "0").exists():
        sparse_dir = sparse_dir / "0"
    if not sparse_dir.exists():
        return None, None, None

    if hasattr(pycolmap, "Reconstruction"):
        try:
            recon = pycolmap.Reconstruction(str(sparse_dir))
            points = recon.points3D
            if points is None or len(points) == 0:
                return None, None, None
            xyz = np.array([p.xyz for p in points.values()])
            rgb = np.array([p.color for p in points.values()])

            c2ws = []
            images = sorted(recon.images.values(), key=lambda img: img.name)
            for img in images:
                if hasattr(img, "cam_from_world"):
                    w2c = img.cam_from_world.matrix()
                    c2w = np.linalg.inv(w2c)
                else:
                    R = img.qvec2rotmat()
                    t = img.tvec
                    c2w = np.eye(4)
                    c2w[:3, :3] = R.T
                    c2w[:3, 3] = -R.T @ t
                c2ws.append(c2w)

            print(f"Loaded {len(xyz)} points from {sparse_dir}")
            return xyz, rgb, np.array(c2ws)
        except Exception as e:
            print(f"Failed to load reconstruction from {sparse_dir}: {e}")

    else:
        try:
            manager = pycolmap.SceneManager(str(sparse_dir))
            manager.load_cameras()
            manager.load_images()
            manager.load_points3D()

            points = manager.points3D
            if points is None or len(points) == 0:
                return None, None, None

            xyz = manager.points3D.astype(np.float32)
            rgb = manager.point3D_colors.astype(np.uint8)

            imdata = manager.images
            c2ws = []
            image_names = []
            bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

            for k in imdata:
                im = imdata[k]
                rot = im.R()
                trans = im.tvec.reshape(3, 1)
                w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
                c2w = np.linalg.inv(w2c)

                c2ws.append(c2w)
                image_names.append(im.name)

            c2ws = np.stack(c2ws, axis=0)

            # Sort poses based on image names (matching the Parser implementation)
            inds = np.argsort(image_names)
            c2ws = c2ws[inds]

            print(f"Loaded {len(xyz)} points from {sparse_dir}")
            return xyz, rgb, c2ws
        except Exception as e:
            print(f"Failed to load reconstruction from {sparse_dir}: {e}")

    return None, None, None


def plot_point_clouds(
    df: pd.DataFrame,
    dest_base: Path,
    x_axis: str | None = None,
    max_cols: int = 3,
    remove_ceiling: bool = False,
    view_mode: str = "isometric",
):
    """Plots point clouds from COLMAP sparse directories using Plotly (Headless HTML)."""
    dest_base = Path(dest_base)
    dest_base.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    configs = []
    point_clouds = []

    for i, (idx, row) in enumerate(df.iterrows()):
        file_path = row.get("sfm_file_path", "")
        if not file_path or pd.isna(file_path):
            continue

        p = Path(file_path)
        if p.name == "eval_results.json":
            config_name = (
                str(row.get("plot_series", p.parent.name))
                .replace("colmap", "COLMAP")
                .replace("vggt", "VGGT")
                .replace("gt", "GT")
                .replace("combined", "Combined")
            )
            if x_axis and pd.notna(row.get(x_axis)):
                config_name = f"{config_name} | {x_axis}={row.get(x_axis)}"

            sparse_dir = p.parent / "sparse"
            xyz, rgb, c2ws = load_pcd_from_sparse(sparse_dir)

            if xyz is not None and c2ws is not None and len(c2ws) > 0:
                # 1. Align upward (using same utility as plot_cameras.py)
                R_align_up = get_alignment_rotation(c2ws)

                # 2. Get first camera's aligned position
                c1_center_aligned = R_align_up @ c2ws[0][:3, 3]

                # 3. Compute rotation around Z-axis so first camera is on positive Y-axis
                theta = np.arctan2(c1_center_aligned[1], c1_center_aligned[0])
                alpha = np.pi / 2.0 - theta

                R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])

                # Combined rotation matrix
                R_total = R_z @ R_align_up

                # Apply rotation to point cloud
                xyz_aligned = xyz @ R_total.T

                # Calculate the minimum bounding box of the camera positions
                cam_centers = c2ws[:, :3, 3]
                cam_centers_aligned = cam_centers @ R_total.T
                scene_center = np.mean(cam_centers_aligned, axis=0)

                c_x_min, c_x_max = cam_centers_aligned[:, 0].min(), cam_centers_aligned[:, 0].max()
                c_y_min, c_y_max = cam_centers_aligned[:, 1].min(), cam_centers_aligned[:, 1].max()
                c_z_min, c_z_max = cam_centers_aligned[:, 2].min(), cam_centers_aligned[:, 2].max()

                # Make the camera bounds a perfect cube to maximize square subplot space
                cam_max_range = max(c_x_max - c_x_min, c_y_max - c_y_min, c_z_max - c_z_min)
                cam_half = (cam_max_range / 2) * 1.1  # 10% padding

                cam_c_x = (c_x_max + c_x_min) / 2
                cam_c_y = (c_y_max + c_y_min) / 2
                cam_c_z = (c_z_max + c_z_min) / 2

                cam_bounds = [
                    cam_c_x - cam_half,
                    cam_c_x + cam_half,
                    cam_c_y - cam_half,
                    cam_c_y + cam_half,
                    cam_c_z - cam_half,
                    cam_c_z + cam_half,
                ]

                # Calculate a 95% bounding box around the scene center to drop massive outliers
                distances = np.linalg.norm(xyz_aligned - scene_center, axis=1)
                threshold = np.percentile(distances, 95)
                mask_95 = distances <= threshold
                xyz_95 = xyz_aligned[mask_95]

                local_bounds = [
                    xyz_95[:, 0].min(),
                    xyz_95[:, 0].max(),
                    xyz_95[:, 1].min(),
                    xyz_95[:, 1].max(),
                    xyz_95[:, 2].min(),
                    xyz_95[:, 2].max(),
                ]

                # Optional filtering: remove points above the highest camera (c_z_max)
                if remove_ceiling:
                    mask = xyz_aligned[:, 2] <= c_z_max
                    xyz_filtered = xyz_aligned[mask]
                    rgb_filtered = rgb[mask]
                else:
                    xyz_filtered = None
                    rgb_filtered = None

                configs.append(config_name)
                point_clouds.append((xyz_aligned, rgb, cam_bounds, xyz_filtered, rgb_filtered, local_bounds))

    if not configs:
        print("No point clouds found to plot.")
        return

    # Calculate global main bounding box across all 95% local bounds
    all_local_bounds = np.array([pc[5] for pc in point_clouds])
    x_min, x_max = all_local_bounds[:, 0].min(), all_local_bounds[:, 1].max()
    y_min, y_max = all_local_bounds[:, 2].min(), all_local_bounds[:, 3].max()
    z_min, z_max = all_local_bounds[:, 4].min(), all_local_bounds[:, 5].max()

    # Force the global bounding box to be a perfect cube
    main_max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    main_half = main_max_range / 2
    main_c_x = (x_max + x_min) / 2
    main_c_y = (y_max + y_min) / 2
    main_c_z = (z_max + z_min) / 2

    x_min_cube, x_max_cube = main_c_x - main_half, main_c_x + main_half
    y_min_cube, y_max_cube = main_c_y - main_half, main_c_y + main_half
    z_min_cube, z_max_cube = main_c_z - main_half, main_c_z + main_half

    num_series = len(configs)
    cols = min(num_series, max_cols)
    original_rows = (num_series + cols - 1) // cols

    multiplier = 3 if remove_ceiling else 2
    total_rows = original_rows * multiplier

    # Build the subplot specifications and titles
    specs = []
    subplot_titles = []
    for r in range(original_rows):
        # Spec and Titles for the Main Views row
        specs.append([{"type": "scene"} for _ in range(cols)])
        for c in range(cols):
            idx = r * cols + c
            subplot_titles.append(configs[idx] if idx < num_series else "")

        # Spec and Titles for the Zoom Views row
        specs.append([{"type": "scene"} for _ in range(cols)])
        for c in range(cols):
            idx = r * cols + c
            subplot_titles.append(configs[idx] + " (Zoom)" if idx < num_series else "")

        # Spec and Titles for the No Ceiling Views row
        if remove_ceiling:
            specs.append([{"type": "scene"} for _ in range(cols)])
            for c in range(cols):
                idx = r * cols + c
                subplot_titles.append(configs[idx] + " (No Ceiling)" if idx < num_series else "")

    # Initialize Plotly subplots for 3D scenes
    fig = make_subplots(rows=total_rows, cols=cols, specs=specs, subplot_titles=subplot_titles, vertical_spacing=0.08)

    # Set up the camera perspective based on view_mode parameter
    if view_mode == "top_down":
        camera_dict = dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0))
    else:
        camera_dict = dict(eye=dict(x=1.5, y=1.5, z=1.0))

    for idx, (xyz, rgb, cam_bounds, xyz_filt, rgb_filt, local_bounds) in enumerate(point_clouds):
        orig_r = idx // cols
        c = (idx % cols) + 1

        r_main = (orig_r * multiplier) + 1
        r_zoom = (orig_r * multiplier) + 2
        r_filt = (orig_r * multiplier) + 3 if remove_ceiling else None

        # Safety cap for browser memory
        max_points = 110000
        if len(xyz) > max_points:
            indices = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[indices]
            rgb = rgb[indices]

        # Plotly requires colors as a list of strings: 'rgb(R, G, B)'
        colors = [f"rgb({color[0]},{color[1]},{color[2]})" for color in rgb]

        scatter_main = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=colors, opacity=0.9),
            name=configs[idx],
        )

        # Add traces to main view and zoom view
        fig.add_trace(scatter_main, row=r_main, col=c)
        fig.add_trace(scatter_main, row=r_zoom, col=c)

        # Handle filtered trace if requested
        if remove_ceiling and xyz_filt is not None:
            if len(xyz_filt) > max_points:
                indices_filt = np.random.choice(len(xyz_filt), max_points, replace=False)
                xyz_filt = xyz_filt[indices_filt]
                rgb_filt = rgb_filt[indices_filt]

            colors_filt = [f"rgb({color[0]},{color[1]},{color[2]})" for color in rgb_filt]
            scatter_filt = go.Scatter3d(
                x=xyz_filt[:, 0],
                y=xyz_filt[:, 1],
                z=xyz_filt[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=colors_filt, opacity=0.9),
                name=configs[idx] + " (Filtered)",
            )
            fig.add_trace(scatter_filt, row=r_filt, col=c)

            # Configure filtered view axes (use global cubic bounds for easy comparison with main view)
            fig.update_scenes(
                dict(
                    xaxis=dict(visible=False, range=[x_min_cube, x_max_cube]),
                    yaxis=dict(visible=False, range=[y_min_cube, y_max_cube]),
                    zaxis=dict(visible=False, range=[z_min_cube, z_max_cube]),
                    aspectmode="cube",  # Forces perfect square fit
                    bgcolor="black",
                    camera=camera_dict,
                ),
                row=r_filt,
                col=c,
            )

        # Configure main view axes (global 95% cubic bounding box)
        fig.update_scenes(
            dict(
                xaxis=dict(visible=False, range=[x_min_cube, x_max_cube]),
                yaxis=dict(visible=False, range=[y_min_cube, y_max_cube]),
                zaxis=dict(visible=False, range=[z_min_cube, z_max_cube]),
                aspectmode="cube",  # Forces perfect square fit
                bgcolor="black",
                camera=camera_dict,
            ),
            row=r_main,
            col=c,
        )

        # Configure zoomed view axes (camera cubic bounding box)
        fig.update_scenes(
            dict(
                xaxis=dict(visible=False, range=[cam_bounds[0], cam_bounds[1]]),
                yaxis=dict(visible=False, range=[cam_bounds[2], cam_bounds[3]]),
                zaxis=dict(visible=False, range=[cam_bounds[4], cam_bounds[5]]),
                aspectmode="cube",  # Forces perfect square fit
                bgcolor="black",
                camera=camera_dict,
            ),
            row=r_zoom,
            col=c,
        )

    # ---------------------------------------------------------
    # Draw 2D white borders around each subplot cell
    # ---------------------------------------------------------
    shapes = []
    for i in range(1, total_rows * cols + 1):
        scene_name = f"scene{i}" if i > 1 else "scene"
        if scene_name in fig.layout:
            domain = fig.layout[scene_name].domain
            if domain and domain.x and domain.y:
                shapes.append(
                    dict(
                        type="rect",
                        xref="paper",
                        yref="paper",
                        x0=domain.x[0],
                        y0=domain.y[0],
                        x1=domain.x[1],
                        y1=domain.y[1],
                        line=dict(color="white", width=2),
                    )
                )

    # Layout sizing: height and width are strictly proportional (500x500 per cell) to guarantee square output
    fig.update_layout(
        height=500 * total_rows,
        width=500 * cols,
        showlegend=False,
        margin=dict(l=10, r=10, b=10, t=50),
        paper_bgcolor="black",
        font=dict(family="Computer Modern, Times New Roman, serif", color="white"),
        shapes=shapes,
    )

    # Enlarge the subplot titles
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=24)

    # 1. Save to interactive HTML (as a backup)
    out_html = dest_base / "point_clouds.html"
    fig.write_html(out_html)

    # 2. Automatically generate a high-res PNG for LaTeX
    out_png = dest_base / "point_clouds.png"
    print("Generating high-res PNG via Kaleido...")

    # Adjust safe pixel dimensions strictly to the square grid ratio (800x800 per cell)
    safe_width = 800 * cols
    safe_height = 800 * total_rows

    fig.write_image(out_png, width=safe_width, height=safe_height, scale=2)

    print(f"Static point clouds saved to {out_png}")
