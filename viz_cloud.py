import os
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import trimesh
import cv2
from typing import Optional, List, Union

from evaluation import get_poses, umeyama_alignment, calculate_metrics, get_point_cloud, get_intrinsics
from viz import create_frustum_traces


def create_point_cloud_trace(
    points: np.ndarray, color: Union[str, np.ndarray], name: str, subsample: int = 1
) -> go.Scatter3d:
    """Creates a Scatter3d trace for a point cloud with subsampling for performance."""
    # Subsample points to avoid freezing the browser (Plotly can struggle with >50k points)
    pts_sub = points[::subsample]

    # Handle color: if it's an array (RGB/Confidence), subsample and format for Plotly
    if isinstance(color, np.ndarray):
        col_sub = color[::subsample]
        # Ensure we just take the first 3 channels (RGB) and cast to integer for formatting
        col_sub = col_sub[:, :3].astype(int)
        marker_color = [f"rgb({r},{g},{b})" for r, g, b in col_sub]
    else:
        marker_color = color

    return go.Scatter3d(
        x=pts_sub[:, 0],
        y=pts_sub[:, 1],
        z=pts_sub[:, 2],
        mode="markers",
        marker=dict(size=3.0, color=marker_color, opacity=0.6),
        name=name,
    )


def project_points(points_3d, c2w, intrinsics):
    """Projects 3D points into 2D image coordinates."""
    # w2c transform
    w2c = np.linalg.inv(c2w)
    pts_cam = (w2c[:3, :3] @ points_3d.T).T + w2c[:3, 3]

    # Perspective projection
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    x = (pts_cam[:, 0] * fx / pts_cam[:, 2]) + cx
    y = (pts_cam[:, 1] * fy / pts_cam[:, 2]) + cy

    # Return coords and depth values (instead of just mask)
    return np.stack([x, y], axis=1), pts_cam[:, 2]


# def project_points(points_3d, c2w, intrinsics):
#     """Projects 3D points into 2D image coordinates with k1 distortion."""
#     # world to camera transform
#     w2c = np.linalg.inv(c2w)
#     pts_cam = (w2c[:3, :3] @ points_3d.T).T + w2c[:3, 3]

#     # Standard perspective projection
#     z = pts_cam[:, 2]
#     u = pts_cam[:, 0] / z
#     v = pts_cam[:, 1] / z

#     # Apply Simple Radial Distortion (k1)
#     if intrinsics.get("k1", 0) != 0:
#         r2 = u**2 + v**2
#         radial = 1 + intrinsics["k1"] * r2
#         u *= radial
#         v *= radial

#     # Scale by focal length and offset by principal point
#     x = (u * intrinsics["fx"]) + intrinsics["cx"]
#     y = (v * intrinsics["fy"]) + intrinsics["cy"]

#     return np.stack([x, y], axis=1), z > 0


def run_gradio_eval(pred_path: str, gt_path: str, show_gt_pts: bool, show_pred_pts: bool, pred_color_mode: str):
    gt_poses = get_poses(gt_path)
    pred_poses = get_poses(pred_path)

    if not gt_poses or not pred_poses:
        return "Error: Could not load poses from one or both paths.", None, {}

    metrics = calculate_metrics(pred_poses, gt_poses)
    if "error" in metrics:
        return f"Error: {metrics['error']}", None, metrics

    common_names = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))
    p_centers = np.array([pred_poses[n][:3, 3] for n in common_names])
    g_centers = np.array([gt_poses[n][:3, 3] for n in common_names])

    s, R, t = umeyama_alignment(p_centers, g_centers)

    fig = go.Figure()

    if show_gt_pts:
        try:
            gt_pts = get_point_cloud(gt_path)
            if gt_pts is not None and len(gt_pts) > 0:
                fig.add_trace(create_point_cloud_trace(gt_pts, "green", "GT Points"))
        except Exception as e:
            print(f"Could not load GT points: {e}")

    if show_pred_pts:
        try:
            # Determine which file to load based on user selection
            ply_filename = "points_conf.ply" if pred_color_mode == "Confidence" else "points.ply"
            ply_file_path = os.path.join(pred_path, "sparse", ply_filename)

            pred_pts = None
            pred_colors = "red"  # Fallback color

            # Try loading the PLY file directly if it exists
            if os.path.exists(ply_file_path):
                pc = trimesh.load(ply_file_path)
                if isinstance(pc, trimesh.PointCloud):
                    pred_pts = np.array(pc.vertices)
                    if hasattr(pc, "colors") and len(pc.colors) > 0:
                        pred_colors = np.array(pc.colors)

            # Fallback to original get_point_cloud if PLY doesn't exist
            if pred_pts is None:
                pred_pts = get_point_cloud(pred_path)

            if pred_pts is not None and len(pred_pts) > 0:
                # Apply the alignment transform to the prediction point cloud
                # transform: x' = s * R * x + t
                pred_pts_aligned = (s * (R @ pred_pts.T)).T + t.T

                # Update name based on mode
                trace_name = (
                    f"Pred Points ({pred_color_mode})"
                    if isinstance(pred_colors, np.ndarray)
                    else "Pred Points (Aligned)"
                )
                fig.add_trace(create_point_cloud_trace(pred_pts_aligned, pred_colors, trace_name))
        except Exception as e:
            print(f"Could not load Pred points: {e}")

    for name in common_names:
        # Align Pred
        p_c2w = pred_poses[name].copy()
        p_c2w[:3, 3] = s * R @ p_c2w[:3, 3] + t
        p_c2w[:3, :3] = R @ p_c2w[:3, :3]

        g_c2w = gt_poses[name]

        gt_traces = create_frustum_traces(g_c2w, color="green", name=f"GT_{name}")
        pred_traces = create_frustum_traces(p_c2w, color="red", name=f"Pred_{name}")

        for t_trace in gt_traces + pred_traces:
            t_trace.customdata = [name] * len(t_trace.x)
            t_trace.hoverinfo = "name"

        fig.add_traces(gt_traces)
        fig.add_traces(pred_traces)

        # Add Error Vector (RTE)
        fig.add_trace(
            go.Scatter3d(
                x=[g_c2w[0, 3], p_c2w[0, 3]],
                y=[g_c2w[1, 3], p_c2w[1, 3]],
                z=[g_c2w[2, 3], p_c2w[2, 3]],
                mode="lines",
                line=dict(color="yellow", width=2),
                showlegend=False,
            )
        )

    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=0), template="plotly_dark")

    summary_text = (
        f"### Alignment Success\n"
        f"- **Matched Images:** {metrics['num_aligned']}\n"
        f"- **Mean RRE:** {metrics['mean_rre_deg']:.3f}°\n"
        f"- **Mean RTE:** {metrics['mean_rte']:.5f}"
    )

    return summary_text, fig, metrics


def run_gradio_eval_with_names(pred_path, gt_path, show_gt_pts, show_pred_pts, pred_color_mode):
    summary, fig, metrics = run_gradio_eval(pred_path, gt_path, show_gt_pts, show_pred_pts, pred_color_mode)

    # Get names for the dropdown
    gt_poses = get_poses(gt_path)
    pred_poses = get_poses(pred_path)
    common_names = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))

    if common_names:
        return (
            summary,
            fig,
            metrics,
            gr.update(choices=common_names, value=common_names[0]),
            gr.update(maximum=len(common_names) - 1, value=0, visible=True),  # Enable slider
            common_names,
        )
    else:
        return summary, fig, metrics, gr.update(), gr.update(), []


def find_image_file(base_path: str, image_name: str) -> Optional[str]:
    """Helper to find the image file traversing common COLMAP directory structures."""
    possible_roots = [
        base_path,
        os.path.join(base_path, "images"),
        os.path.join(base_path, "..", "images"),
        os.path.join(base_path, "..", "..", "images"),
    ]
    for root in possible_roots:
        if os.path.exists(root):
            cand = os.path.join(root, image_name)
            if os.path.exists(cand):
                return cand
    return None


def render_depth_overlay(img_ref, points_3d, pose, intrinsics):
    """Projects 3D points onto image and draws them with depth coloring."""
    h, w = img_ref.shape[:2]

    # Project points and get depths
    pts_2d, depths = project_points(points_3d, pose, intrinsics)

    # Filter valid points (in front of camera and inside image)
    valid = (depths > 0) & (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)

    pts_valid = pts_2d[valid]
    depths_valid = depths[valid]

    # Sort by depth (farthest first) so closer points overwrite
    sort_idx = np.argsort(depths_valid)[::-1]
    pts_valid = pts_valid[sort_idx]
    depths_valid = depths_valid[sort_idx]

    # Normalize depth for colormap
    if len(depths_valid) > 0:
        d_min, d_max = depths_valid.min(), depths_valid.max()
        norm_depth = (depths_valid - d_min) / (d_max - d_min + 1e-8)
        norm_depth = (norm_depth * 255).astype(np.uint8)

        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        colors = cv2.applyColorMap(norm_depth[:, None], cmap)  # Shape (N, 1, 3)
    else:
        colors = []

    img_viz = img_ref.copy()

    stride = 5
    for pt, color in zip(pts_valid[::stride], colors[::stride]):
        cv2.circle(img_viz, (int(pt[0]), int(pt[1])), 2, color[0].tolist(), -1)

    return img_viz


def update_projection_comparison(camera_name, pred_path, gt_path):
    """Function to update both GT and Pred projection images."""
    if not camera_name:
        return None, None, "Select a camera."

    # 1. Setup Alignment
    gt_poses = get_poses(gt_path)
    pred_poses = get_poses(pred_path)
    common_names = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))

    if camera_name not in common_names:
        return None, None, f"Camera {camera_name} not found in common set."

    p_centers = np.array([pred_poses[n][:3, 3] for n in common_names])
    g_centers = np.array([gt_poses[n][:3, 3] for n in common_names])
    s, R, t = umeyama_alignment(p_centers, g_centers)

    # 2. Load Image (Try to find it in GT path structure)
    img_path = find_image_file(gt_path, camera_name)
    if not img_path:
        # Fallback to pred path
        img_path = find_image_file(pred_path, camera_name)

    if not img_path or not os.path.exists(img_path):
        return None, None, f"Image {camera_name} not found on disk."

    img_ref = cv2.imread(img_path)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

    # 3. Get Intrinsics (We use GT intrinsics for both to compare against the same image)
    gt_intrinsics = get_intrinsics(gt_path, camera_id=1)
    pred_intrinsics = get_intrinsics(pred_path, camera_id=1)
    # Note: If camera_id varies per image, this needs to be dynamic based on cameras.txt

    # 4. Generate GT Projection
    gt_pts = get_point_cloud(gt_path)
    if gt_pts is not None:
        gt_viz = render_depth_overlay(img_ref, gt_pts, gt_poses[camera_name], gt_intrinsics)
    else:
        gt_viz = img_ref  # Return clean image if no points

    # 5. Generate Pred Projection (Aligned)
    pred_pts = get_point_cloud(pred_path)
    if pred_pts is not None:
        # Align Pred Points to GT World
        pred_pts_aligned = (s * (R @ pred_pts.T)).T + t.T
        # Project using GT Pose (to see if geometry matches the real image view)
        pred_viz = render_depth_overlay(img_ref, pred_pts_aligned, gt_poses[camera_name], pred_intrinsics)
    else:
        pred_viz = img_ref

    return gt_viz, pred_viz, f"Visualizing: {camera_name} (Scale: {s:.2f})"


def sync_slider_to_dropdown(idx, names):
    """Updates the dropdown selection based on slider index."""
    if names and 0 <= int(idx) < len(names):
        return names[int(idx)]
    return None


with gr.Blocks(title="SfM Evaluation Suite") as demo:
    gr.Markdown("# 3D Reconstruction Evaluator")
    camera_names_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            pred_input = gr.Textbox(
                label="Prediction Path",
                placeholder="./vggt_outputs/bonsai_2_n100_s42_c2.0",
                value="./vggt_outputs/bonsai_2_n100_s42_c2.0",
            )
            gt_input = gr.Textbox(
                label="Ground Truth Path",
                placeholder="./data/360_v2/bonsai/sparse/0",
                value="./data/360_v2/bonsai/sparse/0",
            )

            with gr.Row():
                chk_gt = gr.Checkbox(label="Show GT Cloud", value=True)
                chk_pred = gr.Checkbox(label="Show Pred Cloud", value=True)

            radio_color = gr.Radio(
                choices=["RGB", "Confidence"], label="Pred Point Color", value="RGB", interactive=True
            )

            btn = gr.Button("Evaluate & Visualize", variant="primary")

            camera_slider = gr.Slider(label="Scroll Cameras", minimum=0, maximum=1, step=1, visible=False)
            camera_dropdown = gr.Dropdown(label="Select Camera to Project", choices=[], allow_custom_value=True)

            output_metrics = gr.Markdown("Results will appear here...")
            output_json = gr.JSON(label="Full Metrics JSON")
            img_info = gr.Markdown("")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="3D Trajectory Comparison")
            with gr.Row():
                with gr.Column():
                    gt_img_display = gr.Image(label="GT Projection")
                with gr.Column():
                    pred_img_display = gr.Image(label="Pred Projection")

    btn.click(
        fn=run_gradio_eval_with_names,
        inputs=[pred_input, gt_input, chk_gt, chk_pred, radio_color],
        outputs=[output_metrics, plot_output, output_json, camera_dropdown, camera_slider, camera_names_state],
    )

    camera_slider.change(
        fn=sync_slider_to_dropdown, inputs=[camera_slider, camera_names_state], outputs=[camera_dropdown]
    )

    camera_dropdown.change(
        fn=update_projection_comparison,
        inputs=[camera_dropdown, pred_input, gt_input],
        outputs=[gt_img_display, pred_img_display, img_info],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
