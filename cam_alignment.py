import numpy as np


def get_alignment_rotation(poses: np.ndarray):
    """Computes a rotation matrix to align the average camera 'up' with +Z."""
    ups = []
    for i in range(len(poses)):
        # Assuming poses is a numpy array of C2W matrices (N, 4, 4) or (N, 3, 4)
        R_c2w = poses[i][:3, :3]

        # Camera 'up' in local space is (0, -1, 0).
        up_world = R_c2w @ np.array([0, -1, 0])
        ups.append(up_world)

    # Average up vector
    mean_up = np.mean(ups, axis=0)
    mean_up /= np.linalg.norm(mean_up)

    # Target vector is positive Z
    target = np.array([0, 0, 1])

    # Calculate rotation between mean_up and target using Rodrigues' rotation formula
    v = np.cross(mean_up, target)
    s = np.linalg.norm(v)
    c = np.dot(mean_up, target)

    if s < 1e-6:
        return np.eye(3)  # Already aligned or opposite

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_align = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
    return R_align
