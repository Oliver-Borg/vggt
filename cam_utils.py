import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

"""
c2w = pose (3D position and rotation relative to the origin)
w2c = cfw = extrinsics (projection from world coordinates to camera frame)
"""


def build_matrix(translation: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    full_matrix = np.zeros((4, 4))
    full_matrix[:3, :3] = rotation_matrix
    full_matrix[:3, 3] = translation
    full_matrix[3, 3] = 1
    return full_matrix


def decompose_matrix(full_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = full_matrix[:3, 3]
    rotation_matrix = full_matrix[:3, :3]
    return translation, rotation_matrix


def invert_transform(translation: np.ndarray, rotation_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full_matrix = build_matrix(translation, rotation_matrix)
    inverted_matrix = np.linalg.inv(full_matrix)
    return inverted_matrix[:3, 3], inverted_matrix[:3, :3]


def quat_to_mat(quaternion: np.ndarray) -> np.ndarray:
    return Rot.from_quat(quaternion).as_matrix()


def mat_to_quat(rotation_matrix: np.ndarray) -> np.ndarray:
    return Rot.from_matrix(rotation_matrix).as_quat()


def c2w_to_w2c(translation: np.ndarray, rotation_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return invert_transform(translation, rotation_matrix)


def w2c_to_c2w(translation: np.ndarray, rotation_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return invert_transform(translation, rotation_matrix)


def load_poses_from_json(path: Path) -> dict[str, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)

    poses = {}
    for image_name, frame in data.items():
        qvec = np.array(frame["extrinsics"]["qvec"])
        tvec = np.array(frame["extrinsics"]["tvec"])
        rotation_matrix = quat_to_mat(qvec)
        c2w = build_matrix(tvec, rotation_matrix)
        poses[image_name] = c2w
    return poses


def umeyama_alignment_two_points(
    from_points: np.ndarray, to_points: np.ndarray, from_dirs: np.ndarray, to_dirs: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Computes optimal similarity transform: p = s * R * q + t.
    https://en.wikipedia.org/wiki/Kabsch_algorithm

    Modified for exactly two points. Heavily weights point alignment
    to ensure perfect position match, using direction vectors to
    resolve the remaining rotational degree of freedom.
    """
    q = from_points
    p = to_points

    # Translation
    n, m = q.shape
    q_mean = q.mean(axis=0)
    p_mean = p.mean(axis=0)
    q_centered = q - q_mean
    p_centered = p - p_mean
    # Computation of the covariance matrix

    # Heavily weight the point covariance (e.g., 1e6) to perfectly align positions.
    # Add look directions to the covariance matrix to resolve the remaining rotation.
    H = (np.dot(p_centered.T, q_centered) * 1e6 + np.dot(to_dirs.T, from_dirs)) / n
    # First, calculate the SVD of the covariance matrix H,
    U, Sigma, Vt = np.linalg.svd(H)

    # Next, record if the orthogonal matrices contain a reflection,
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1

    # Finally, calculate our optimal rotation matrix R as
    rotation = np.dot(np.dot(U, S), Vt)

    if with_scale:
        # Scale must be computed directly from point variance since Sigma
        # is now artificially inflated by the 1e6 weight.
        var_x = np.var(q_centered, axis=0).sum()
        var_y = np.var(p_centered, axis=0).sum()
        scale = np.sqrt(var_y / var_x) if var_x > 0 else 1.0
    else:
        scale = 1.0

    translation = p_mean - scale * np.dot(rotation, q_mean)
    return float(scale), rotation, translation


def umeyama_alignment(
    from_points: np.ndarray, to_points: np.ndarray, with_scale: bool = True, ignore_outliers: bool = False
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Computes optimal similarity transform: p = s * R * q + t.
    https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    q = from_points
    p = to_points

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

    if not ignore_outliers:
        return float(scale), rotation, translation

    # Outlier rejection
    aligned_q = scale * np.dot(q, rotation.T) + translation
    distances = np.linalg.norm(aligned_q - p, axis=1)
    # Median Absolute Deviation for outlier detection
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    # Keep points within 3 MADs.
    threshold = median_dist + 3 * (mad + 1e-6)
    inliers = distances < threshold
    if not np.all(inliers) and np.sum(inliers) >= 3:
        return umeyama_alignment(q[inliers], p[inliers], with_scale, ignore_outliers=False)

    return float(scale), rotation, translation


def umeyama_alignment_with_orientation(
    from_points: np.ndarray,
    to_points: np.ndarray,
    from_rotations: np.ndarray,
    to_rotations: np.ndarray,
    alpha: float = 1.0,
    with_scale: bool = True,
    ignore_outliers: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute umeyama alignment while respecting orientations.
    We pass points along the camera orientation axis to force the solver to account for these.
    """
    from_x_axis_points = from_points + alpha * from_rotations[:, :, 0]
    from_y_axis_points = from_points + alpha * from_rotations[:, :, 1]
    from_z_axis_points = from_points + alpha * from_rotations[:, :, 2]

    to_x_axis_points = to_points + alpha * to_rotations[:, :, 0]
    to_y_axis_points = to_points + alpha * to_rotations[:, :, 1]
    to_z_axis_points = to_points + alpha * to_rotations[:, :, 2]

    all_from_points = np.concatenate([from_points, from_x_axis_points, from_y_axis_points, from_z_axis_points], axis=0)
    all_to_points = np.concatenate([to_points, to_x_axis_points, to_y_axis_points, to_z_axis_points], axis=0)

    return umeyama_alignment(all_from_points, all_to_points, with_scale, ignore_outliers)


def stochastic_umeyama_alignment(
    from_points: np.ndarray, to_points: np.ndarray, with_scale: bool = True, num_trials: int = 10, num_samples: int = 3
) -> tuple[float, np.ndarray, np.ndarray]:
    best_score = float("inf")
    best_transform = (1.0, np.eye(3), np.zeros(3))

    for _ in range(num_trials):
        indices = np.random.choice(
            from_points.shape[0],
            size=max(min(from_points.shape[0] // num_samples, from_points.shape[0]), 3),
            replace=False,
        )
        s, R, t = umeyama_alignment(from_points[indices], to_points[indices], with_scale=with_scale)

        transformed = s * (R @ from_points.T).T + t
        score = np.mean(np.linalg.norm(transformed - to_points, axis=1))

        if score < best_score:
            best_score = score
            best_transform = (s, R, t)

    return best_transform


def get_metrics(
    gt_poses: dict[str, np.ndarray],
    pred_poses: dict[str, np.ndarray],
    alignment: tuple[float, np.ndarray, np.ndarray] | None = None,
):
    rre_list = []
    rte_list = []
    s, R, t = alignment if alignment is not None else (1.0, np.eye(3), np.zeros(3))

    common_names = set(pred_poses.keys()) & set(gt_poses.keys())

    for name in common_names:
        p_c2w = pred_poses[name]
        g_c2w = gt_poses[name]

        # Align p_c2w
        translation, rotation_matrix = decompose_matrix(p_c2w)
        new_translation = s * (R @ translation) + t
        new_rotation_matrix = R @ rotation_matrix
        p_c2w = build_matrix(new_translation, new_rotation_matrix)

        # RRE: Geodesic distance
        rel_rot = np.dot(p_c2w[:3, :3].T, g_c2w[:3, :3])
        cos_theta = (np.trace(rel_rot) - 1.0) / 2.0
        rre = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # RTE: Euclidean distance
        # TODO Normalize this better
        rte = np.linalg.norm(p_c2w[:3, 3] - g_c2w[:3, 3])
        rre_list.append(float(rre))
        rte_list.append(float(rte))
    return rre_list, rte_list


def verify_look_at_origin(
    c2w_matrix: np.ndarray, is_opencv: bool = False, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> dict:
    """
    Verifies if a c2w matrix is pointing at the world origin [0, 0, 0].
    """
    # 1. Extract camera position
    pos = c2w_matrix[:3, 3]

    # 2. Extract camera forward direction
    if is_opencv:
        # COLMAP/OpenCV looks down the positive Z-axis
        forward_dir = c2w_matrix[:3, 2]
    else:
        # NeRF/OpenGL looks down the negative Z-axis
        forward_dir = -c2w_matrix[:3, 2]

    # 3. Calculate the ideal vector pointing from the camera to the origin
    expected_dir = np.array(origin) - pos

    # 4. Normalize both vectors to compare their directions
    forward_dir_normalized = forward_dir / np.linalg.norm(forward_dir)
    expected_dir_normalized = expected_dir / np.linalg.norm(expected_dir)

    # 5. Calculate the dot product (cosine of the angle between them)

    dot_product = np.dot(forward_dir_normalized, expected_dir_normalized)

    angle_degrees = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    return {
        "position": pos.tolist(),
        "angle_difference_degrees": float(angle_degrees),
        "is_pointing_at_origin": bool(np.isclose(angle_degrees, 0.0, atol=1.0)),
    }
