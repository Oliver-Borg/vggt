import numpy as np
import gradio as gr
import plotly.graph_objects as go
from evaluation import get_poses, umeyama_alignment, calculate_metrics


def create_frustum_traces(c2w: np.ndarray, color: str, name: str, size: float = 0.1) -> list[go.Scatter3d]:
    """Creates Plotly 3D traces for a camera frustum."""
    pts = np.array([[0, 0, 0], [1, 1, 2], [-1, 1, 2], [-1, -1, 2], [1, -1, 2]]) * size
    # Transform to world coordinates
    pts_world = (c2w[:3, :3] @ pts.T + c2w[:3, 3:4]).T

    # Define the 8 lines of the frustum
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

    traces = []
    for line in lines:
        p0, p1 = pts_world[line[0]], pts_world[line[1]]
        traces.append(
            go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False,
                hoverinfo="name",
                name=name,
            )
        )
    return traces


def run_gradio_eval(pred_path: str, gt_path: str):
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

    for name in common_names:
        # Align Pred
        p_c2w = pred_poses[name].copy()
        p_c2w[:3, 3] = s * R @ p_c2w[:3, 3] + t
        p_c2w[:3, :3] = R @ p_c2w[:3, :3]

        g_c2w = gt_poses[name]

        # Add Frustums
        fig.add_traces(create_frustum_traces(g_c2w, color="green", name=f"GT_{name}"))
        fig.add_traces(create_frustum_traces(p_c2w, color="red", name=f"Pred_{name}"))

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


with gr.Blocks(title="SfM Evaluation Suite") as demo:
    gr.Markdown("# 3D Reconstruction Evaluator")

    with gr.Row():
        with gr.Column(scale=1):
            pred_input = gr.Textbox(
                label="Prediction Path",
                placeholder="./vggt_outputs/bonsai_8_n20_s43",
                value="./vggt_outputs/bonsai_8_n20_s43",
            )
            gt_input = gr.Textbox(
                label="Ground Truth Path",
                placeholder="./data/360_v2/bonsai/sparse/0",
                value="./data/360_v2/bonsai/sparse/0",
            )
            btn = gr.Button("Evaluate & Visualize", variant="primary")
            output_metrics = gr.Markdown("Results will appear here...")
            output_json = gr.JSON(label="Full Metrics JSON")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="3D Trajectory Comparison")

    btn.click(fn=run_gradio_eval, inputs=[pred_input, gt_input], outputs=[output_metrics, plot_output, output_json])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
