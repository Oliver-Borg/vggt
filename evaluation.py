import argparse
import datetime
import glob
import json
import os
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple, List, Optional, Any, TypedDict


class EvalMetrics(TypedDict, total=False):
    mean_rre_deg: float
    mean_rte: float
    auc_10: float
    auc_20: float
    auc_30: float
    num_aligned: int
    alignment_scale: float
    error: str


class EvalReport(TypedDict):
    timestamp: str
    pred_path: str
    gt_path: str
    metrics: EvalMetrics
    profiling: Dict[str, Any]


def umeyama_alignment(q: np.ndarray, p: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes optimal similarity transform: y = s * R * x + t.
    https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    # Translation
    n, m = q.shape
    q_mean = q.mean(axis=0)
    p_mean = p.mean(axis=0)
    q_centered = q - q_mean
    p_centered = p - p_mean
    # Computation of the covariance matrix

    H = np.dot(p_centered.T, q_centered) / n
    # First, calculate the SVD of the covariance matrix H,
    U, Sigma, Vt = np.linalg.svd(H)

    # Next, record if the orthogonal matrices contain a reflection,
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1

    # Finally, calculate our optimal rotation matrix R as
    rotation = np.dot(np.dot(U, S), Vt)

    if with_scale:
        var_x = np.var(q_centered, axis=0).sum()
        scale = np.trace(np.dot(np.diag(Sigma), S)) / var_x
    else:
        scale = 1.0

    translation = p_mean - scale * np.dot(rotation, q_mean)
    return float(scale), rotation, translation


def get_poses(path: str) -> Dict[str, np.ndarray]:
    """
    Reads all available COLMAP models in a path and returns the best one.
    Checks for subfolders like 0, 1, 2 if they exist.
    """
    potential_paths: List[str] = [path]
    sparse_path = os.path.join(path, "sparse")

    if os.path.isdir(sparse_path):
        potential_paths.append(sparse_path)
        subdirs = [
            os.path.join(sparse_path, d)
            for d in os.listdir(sparse_path)
            if d.isdigit() and os.path.isdir(os.path.join(sparse_path, d))
        ]
        potential_paths.extend(subdirs)

    best_reconst: Optional[pycolmap.Reconstruction] = None
    max_images: int = -1

    for p in potential_paths:
        if os.path.exists(os.path.join(p, "images.bin")):
            try:
                reconst = pycolmap.Reconstruction(p)
                num_reg = reconst.num_reg_images()
                if num_reg > max_images:
                    max_images = num_reg
                    best_reconst = reconst
            except Exception:
                continue

    if not best_reconst:
        return {}

    poses: Dict[str, np.ndarray] = {}
    for _, image in best_reconst.images.items():
        rigid_stat = image.cam_from_world
        rot_mat = rigid_stat.rotation.matrix()
        tvec = rigid_stat.translation.reshape(3, 1)

        c2w = np.eye(4)
        c2w[:3, :3] = rot_mat.T
        c2w[:3, 3:4] = -np.dot(rot_mat.T, tvec)
        poses[image.name] = c2w

    return poses


def calculate_metrics(pred_poses: Dict[str, np.ndarray], gt_poses: Dict[str, np.ndarray]) -> EvalMetrics:
    common_names = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))
    if len(common_names) < 3:
        return {"error": f"Only {len(common_names)} images matched. Need >= 3 for Umeyama."}

    p_centers = np.array([pred_poses[n][:3, 3] for n in common_names])
    g_centers = np.array([gt_poses[n][:3, 3] for n in common_names])

    s, R_align, t_align = umeyama_alignment(p_centers, g_centers)

    rre_list: List[float] = []
    rte_list: List[float] = []

    for name in common_names:
        p_c2w = pred_poses[name].copy()
        p_c2w[:3, 3] = s * np.dot(R_align, p_c2w[:3, 3]) + t_align
        p_c2w[:3, :3] = np.dot(R_align, p_c2w[:3, :3])

        g_c2w = gt_poses[name]

        # RRE: Geodesic distance
        rel_rot = np.dot(p_c2w[:3, :3].T, g_c2w[:3, :3])
        cos_theta = (np.trace(rel_rot) - 1.0) / 2.0
        rre = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # RTE: Euclidean distance
        rte = np.linalg.norm(p_c2w[:3, 3] - g_c2w[:3, 3])

        rre_list.append(float(rre))
        rte_list.append(float(rte))

    return {
        "mean_rre_deg": round(float(np.mean(rre_list)), 4),
        "mean_rte": round(float(np.mean(rte_list)), 6),
        "auc_10": round(float(np.mean(np.array(rre_list) < 10)), 3),
        "auc_20": round(float(np.mean(np.array(rre_list) < 20)), 3),
        "auc_30": round(float(np.mean(np.array(rre_list) < 30)), 3),
        "num_aligned": len(common_names),
        "alignment_scale": round(s, 6),
    }


def main(pred: str, gt: str) -> None:
    gt_poses = get_poses(gt)
    pred_poses = get_poses(pred)

    if not gt_poses or not pred_poses:
        print(f"Error: Missing or empty reconstruction in pred ({len(pred_poses)}) or gt ({len(gt_poses)})")
        return

    metrics = calculate_metrics(pred_poses, gt_poses)

    stat_json_path = os.path.join(pred, "stat.json")
    profiling = {}
    if os.path.exists(stat_json_path):
        with open(stat_json_path, "r") as f:
            profiling = json.load(f).get("profiling", {})

    report: EvalReport = {
        "timestamp": datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
        "pred_path": pred,
        "gt_path": gt,
        "metrics": metrics,
        "profiling": profiling,
    }

    out_file = os.path.join(pred, "eval_results.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Evaluation for {pred} Complete. Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SfM reconstruction vs GT.")
    parser.add_argument("--pred-glob", required=True, type=str)
    parser.add_argument("--gt", required=True, type=str)
    args = parser.parse_args()

    for pred in sorted(glob.glob(args.pred_glob)):
        main(pred, args.gt)
