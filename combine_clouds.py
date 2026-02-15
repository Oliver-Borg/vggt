import argparse
import pycolmap
import numpy as np
from pathlib import Path
from typing import Tuple


def umeyama_alignment(q: np.ndarray, p: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes optimal similarity transform: p = s * R * q + t.
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


def get_alignment_transform(rec_cam, rec_pts):
    # Match images by name to find common camera centers
    # q: centers from point_source, p: centers from camera_source
    names_cam = {img.name: img.projection_center() for img in rec_cam.images.values()}
    names_pts = {img.name: img.projection_center() for img in rec_pts.images.values()}

    common_names = set(names_cam.keys()) & set(names_pts.keys())

    if len(common_names) < 3:
        raise ValueError(f"Need at least 3 common images for alignment. Found: {len(common_names)}")

    p = np.array([names_cam[name] for name in common_names])
    q = np.array([names_pts[name] for name in common_names])

    return umeyama_alignment(q, p)


def swap_and_align(camera_source_path, point_source_path, output_path):
    rec_cam = pycolmap.Reconstruction(camera_source_path)
    rec_pts = pycolmap.Reconstruction(point_source_path)

    # 1. Find the transform to bring point_source into camera_source space
    # p = s * R * q + t
    s, R, t = get_alignment_transform(rec_cam, rec_pts)
    print(f"Alignment found: Scale={s:.4f}")

    # 2. Create the new reconstruction using camera_source as the base
    # This keeps the cameras and images (poses) from the camera_source
    new_rec = pycolmap.Reconstruction()

    for cam_id, camera in rec_cam.cameras.items():
        new_rec.add_camera(camera)
    for img_id, image in rec_cam.images.items():
        new_rec.add_image(image)

    # 3. Transform and add points from point_source
    # Since we are swapping points, we ignore existing tracks if they don't match
    for pt_id, pt3D in rec_pts.points3D.items():
        # Transform the point: p = s * R * q + t
        xyz_transformed = s * (R @ pt3D.xyz) + t

        # We pass an empty track if you want to avoid ID conflicts/corruption
        # or if the image IDs in the point_source don't match the camera_source
        new_rec.add_point3D(xyz_transformed, pycolmap.Track(), pt3D.color)

    # 4. Export
    output_path.mkdir(parents=True, exist_ok=True)
    new_rec.write(output_path)
    print(f"Reconstruction exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_source", type=str, required=True, help="Path to point cloud with desired cameras.")
    parser.add_argument("--point_source", type=str, required=True, help="Path to point cloud with desired points.")
    parser.add_argument("--output_dir", type=str, default="aligned_swap", help="Output directory.")

    args = parser.parse_args()

    swap_and_align(Path(args.camera_source), Path(args.point_source), Path(args.output_dir))
