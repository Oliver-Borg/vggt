from pathlib import Path

import pandas as pd
import numpy as np
import pycolmap
import matplotlib.pyplot as plt
import open3d as o3d
import scienceplots

from .cam_alignment import get_alignment_rotation

plt.style.use(["science", "grid"])

textwidth = 7.00697
aspect_ratio = 1.0
scale = 4.0
# Total figure width
width = textwidth * scale

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "dejavusans",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "lines.linewidth": 0.75,
        "lines.markersize": 6.0,
        "patch.linewidth": 0.5,
        "axes.linewidth": 0.4,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        # Integrated options for black-on-white plots
        "text.color": "black",
        "axes.labelcolor": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def get_lookat_c2w(target: np.ndarray, eye: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """
    Creates a 4x4 camera-to-world matrix looking at a target point from an eye position.
    Follows the OpenCV coordinate convention (X right, Y down, Z forward).
    """
    forward = target - eye
    norm_f = np.linalg.norm(forward)
    if norm_f < 1e-6:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / norm_f

    right = np.cross(forward, up)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-6:  # If looking strictly along the UP vector (e.g., Top-Down)
        right = np.array([1.0, 0.0, 0.0])
        down = np.cross(forward, right)
        down = down / np.linalg.norm(down)
    else:
        right = right / norm_r
        down = np.cross(forward, right)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


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


def _render_o3d_image(xyz: np.ndarray, rgb: np.ndarray, c2w: np.ndarray, K: np.ndarray, W: int, H: int) -> np.ndarray:
    """Helper method to render a point cloud using Open3D from a specific viewpoint."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb.dtype == np.uint8:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    else:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    # Use OffscreenRenderer for headless EGL-based rendering
    render = o3d.visualization.rendering.OffscreenRenderer(W, H)
    render.scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))

    # Set up material to use simple vertex colors without lighting
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0

    render.scene.add_geometry("pcd", pcd, mat)

    # Open3D's camera setup expects intrinsics (3x3) and extrinsics (world-to-camera 4x4)
    w2c = np.linalg.inv(c2w)
    render.setup_camera(K, w2c, W, H)

    # Capture the image
    img = np.asarray(render.render_to_image())

    # Clean up to prevent EGL context leaks across many renders
    render.scene.remove_geometry("pcd")

    # Image might be returned with an alpha channel depending on the Open3D version,
    # dropping it if present so it matches the expected HxWx3 format
    if img.shape[-1] == 4:
        img = img[..., :3]

    # Open3D images are returned in uint8 [0, 255], scale back to float [0, 1] for matplotlib
    return img.astype(np.float32) / 255.0


def plot_point_clouds(
    df: pd.DataFrame,
    dest_base: Path,
    x_axis: str | None = None,
    max_cols: int = 5,
    remove_ceiling: bool = True,
    view_mode: str = "isometric",
):
    """Projects point clouds to 2D images using Open3D rendering and Matplotlib."""
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
                # 1. Align upward
                R_align_up = get_alignment_rotation(c2ws)
                c1_center_aligned = R_align_up @ c2ws[0][:3, 3]
                theta = np.arctan2(c1_center_aligned[1], c1_center_aligned[0])
                alpha = np.pi / 2.0 - theta
                R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])

                R_total = R_z @ R_align_up
                xyz_aligned = xyz @ R_total.T

                # Calculate camera centers and their bounding box
                cam_centers = c2ws[:, :3, 3]
                cam_centers_aligned = cam_centers @ R_total.T
                scene_center = np.mean(cam_centers_aligned, axis=0)

                c_x_min, c_x_max = cam_centers_aligned[:, 0].min(), cam_centers_aligned[:, 0].max()
                c_y_min, c_y_max = cam_centers_aligned[:, 1].min(), cam_centers_aligned[:, 1].max()
                c_z_min, c_z_max = cam_centers_aligned[:, 2].min(), cam_centers_aligned[:, 2].max()

                # Make the camera bounds a perfect cube
                cam_max_range = max(c_x_max - c_x_min, c_y_max - c_y_min, c_z_max - c_z_min)
                cam_c_x = (c_x_max + c_x_min) / 2
                cam_c_y = (c_y_max + c_y_min) / 2
                cam_c_z = (c_z_max + c_z_min) / 2
                cam_target = np.array([cam_c_x, cam_c_y, cam_c_z])

                # Calculate 95% bounding box for global view
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

                # Calculate the aligned C2W matrix for the first camera
                c1_aligned = np.eye(4)
                c1_aligned[:3, :3] = R_total @ c2ws[0][:3, :3]
                c1_aligned[:3, 3] = R_total @ c2ws[0][:3, 3]

                # Optional filtering: remove points above the highest camera (c_z_max)
                if remove_ceiling:
                    mask = xyz_aligned[:, 2] <= c_z_max
                    xyz_filtered = xyz_aligned[mask]
                    rgb_filtered = rgb[mask]
                else:
                    xyz_filtered = None
                    rgb_filtered = None

                configs.append(config_name)
                point_clouds.append(
                    (xyz_aligned, rgb, cam_target, cam_max_range, xyz_filtered, rgb_filtered, local_bounds, c1_aligned)
                )

    if not configs:
        print("No point clouds found to plot.")
        return

    num_series = len(configs)
    cols = min(num_series, max_cols)
    original_rows = (num_series + cols - 1) // cols
    multiplier = 3 if remove_ceiling else 2
    total_rows = original_rows * multiplier

    # Calculate proportional figure height to maintain roughly square subplots
    fig_height = (width / cols) * total_rows * aspect_ratio

    fig, axes = plt.subplots(total_rows, cols, figsize=(width, fig_height), dpi=200, squeeze=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.2)

    # Clean the background for all grid cells (hiding empty ones)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Virtual Camera Intrinsic Properties
    W, H = 1200, 1200
    K = np.array([[W, 0.0, W / 2], [0.0, H, H / 2], [0.0, 0.0, 1.0]])

    if view_mode == "top_down":
        dir_vec = np.array([0.0, 0.0, 1.0])
        up_vec = np.array([0.0, 1.0, 0.0])
    else:  # isometric
        dir_vec = np.array([1.0, 1.0, 1.0])
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        up_vec = np.array([0.0, 0.0, 1.0])

    print(f"Generating 2D Projections for {num_series} point clouds...")

    for idx, (xyz, rgb, cam_target, cam_max_range, xyz_filt, rgb_filt, local_bounds, c1_aligned) in enumerate(
        point_clouds
    ):
        orig_r = idx // cols
        c = idx % cols

        r_main = orig_r * multiplier
        r_zoom = orig_r * multiplier + 1
        r_filt = orig_r * multiplier + 2 if remove_ceiling else None

        # Calculate main target and range from local bounds instead of global bounds
        x_min, x_max, y_min, y_max, z_min, z_max = local_bounds
        main_max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        main_target = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2])

        # Determine camera eye distance based on 800px focal length to fit the requested range
        # f_view = FOV_distance ~= range * 1.2 padding
        D_main = main_max_range * 1.2

        c2w_main = get_lookat_c2w(main_target, main_target + dir_vec * D_main, up=up_vec)

        # Render from the perspective of the first camera instead of zoomed look-at
        c2w_zoom = c1_aligned

        # Fetch configured spine width from rcParams
        spine_width = plt.rcParams.get("axes.linewidth", 0.4)

        # Render Main Image
        img_main = _render_o3d_image(xyz, rgb, c2w_main, K, W, H)
        ax = axes[r_main, c]
        ax.imshow(img_main)
        ax.set_title(configs[idx], pad=10)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(spine_width)

        # Render Zoomed Image
        img_zoom = _render_o3d_image(xyz, rgb, c2w_zoom, K, W, H)
        ax = axes[r_zoom, c]
        ax.imshow(img_zoom)
        ax.set_title(f"{configs[idx]}\n(Zoom)", pad=10)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(spine_width)

        # Render Filtered Image (if requested)
        if remove_ceiling and xyz_filt is not None:
            img_filt = _render_o3d_image(xyz_filt, rgb_filt, c2w_main, K, W, H)
            ax = axes[r_filt, c]
            ax.imshow(img_filt)
            ax.set_title(f"{configs[idx]}\n(No Ceiling)", pad=10)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(spine_width)

    out_png = dest_base / "point_clouds.png"
    plt.savefig(out_png, bbox_inches="tight")
    out_pdf = dest_base / "point_clouds.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Static 2D projections fully rendered and saved to {out_png}")
