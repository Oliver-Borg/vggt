import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as Rot


"""
c2w = pose (3D position and rotation relative to the origin)
w2c = cfw = extrinsics (projection from world coordinates to camera frame)
"""


def invert_transform(translation: np.ndarray, rotation_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full_matrix = np.zeros((4, 4))
    full_matrix[:3, :3] = rotation_matrix
    full_matrix[:3, 3] = translation
    full_matrix[3, 3] = 1

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
    rotation_matrix = quat_to_mat(np.roll(cfw.rotation.quat, -1))
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


def umeyama_alignment(q: np.ndarray, p: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
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
