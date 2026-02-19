import numpy as np
import pycolmap
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


def cfw_to_w2c(cfw: pycolmap.Rigid3d) -> tuple[np.ndarray, np.ndarray]:
    rotation_matrix = quat_to_mat(cfw.rotation.quat)  # np.roll(cfw.rotation.quat, -1)
    translation = cfw.translation.copy()
    return translation, rotation_matrix


def w2c_to_cfw(translation: np.ndarray, rotation_matrix: np.ndarray) -> pycolmap.Rigid3d:
    cfw = pycolmap.Rigid3d()
    cfw.rotation.quat = mat_to_quat(rotation_matrix)
    cfw.translation = translation.copy()
    return cfw


def cfw_to_c2w(cfw: pycolmap.Rigid3d) -> tuple[np.ndarray, np.ndarray]:
    return w2c_to_c2w(*cfw_to_w2c(cfw))


def c2w_to_cfw(translation: np.ndarray, rotation_matrix: np.ndarray) -> pycolmap.Rigid3d:
    return w2c_to_cfw(*c2w_to_w2c(translation, rotation_matrix))


def umeyama_alignment(
    from_points: np.ndarray, to_points: np.ndarray, with_scale: bool = True
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
    return float(scale), rotation, translation


def get_poses(recon: pycolmap.Reconstruction) -> dict[str, np.ndarray]:
    to_return = {}
    for image in recon.images.values():
        to_return[image.name] = build_matrix(*cfw_to_c2w(image.cam_from_world))
    return to_return


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
