import unittest
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as Rot
from cam_utils import *
from pycolmap_utils import *


# Import the functions to be tested.
# Assuming they are in a file named 'transforms.py', otherwise adjust import.
class TestTransforms(unittest.TestCase):
    def setUp(self):
        # Create a random rotation matrix and translation vector for testing
        self.rng = np.random.default_rng(42)
        self.rand_rot_vec = self.rng.random(3)
        self.R_orig = Rot.from_rotvec(self.rand_rot_vec).as_matrix()
        self.t_orig = self.rng.random(3) * 10

        # A standard homogeneous matrix constructed from the above
        self.T_orig = np.eye(4)
        self.T_orig[:3, :3] = self.R_orig
        self.T_orig[:3, 3] = self.t_orig

    def test_invert_transform(self):
        """Test if inverting the transform twice returns the original."""
        t_inv, R_inv = invert_transform(self.t_orig, self.R_orig)
        t_rec, R_rec = invert_transform(t_inv, R_inv)

        np.testing.assert_allclose(R_rec, self.R_orig, atol=1e-10)
        np.testing.assert_allclose(t_rec, self.t_orig, atol=1e-10)

    def test_matrix_inverse_property(self):
        """Test if T * T_inv = Identity."""
        t_inv, R_inv = invert_transform(self.t_orig, self.R_orig)

        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv

        identity_check = self.T_orig @ T_inv
        np.testing.assert_allclose(identity_check, np.eye(4), atol=1e-10)

    def test_quat_mat_conversions(self):
        """Test roundtrip conversion between quaternion and rotation matrix."""
        q = mat_to_quat(self.R_orig)
        R_rec = quat_to_mat(q)

        # Check matrix consistency
        np.testing.assert_allclose(R_rec, self.R_orig, atol=1e-10)

    def test_cfw_conversions(self):
        """
        Test the roundtrip:
        (t, R) -> Rigid3d -> (t, R)
        """
        # Create pycolmap object from data
        cfw = w2c_to_cfw(self.t_orig, self.R_orig)

        # Check if output is indeed a Rigid3d object
        self.assertIsInstance(cfw, pycolmap.Rigid3d)

        # Convert back
        t_rec, R_rec = cfw_to_w2c(cfw)

        np.testing.assert_allclose(t_rec, self.t_orig, atol=1e-10)
        np.testing.assert_allclose(R_rec, self.R_orig, atol=1e-10)

    def test_c2w_to_cfw_chain(self):
        """Test the full chain: c2w (input) -> cfw (inverted) -> c2w (re-inverted)."""
        # Convert c2w inputs to a cfw object (which represents w2c)
        cfw = c2w_to_cfw(self.t_orig, self.R_orig)

        # Convert that cfw back to c2w
        t_rec, R_rec = cfw_to_c2w(cfw)

        np.testing.assert_allclose(t_rec, self.t_orig, atol=1e-10)
        np.testing.assert_allclose(R_rec, self.R_orig, atol=1e-10)

    def test_umeyama_alignment_perfect_match(self):
        """Test Umeyama with a known synthetic transform (Scale, Rot, Trans)."""
        n_points = 100
        points_q = self.rng.random((n_points, 3)) * 10

        # Define ground truth transform parameters
        scale_gt = 2.5
        rot_gt = self.R_orig
        trans_gt = np.array([5.0, -3.0, 1.0])

        # Apply transform: p = s * R * q + t
        # Note: standard matrix mult expects column vectors, so we transpose
        points_p = scale_gt * (rot_gt @ points_q.T).T + trans_gt

        # Recover parameters
        scale_est, rot_est, trans_est = umeyama_alignment(points_q, points_p, with_scale=True)

        np.testing.assert_allclose(scale_est, scale_gt, atol=1e-10)
        np.testing.assert_allclose(rot_est, rot_gt, atol=1e-10)
        np.testing.assert_allclose(trans_est, trans_gt, atol=1e-10)

    def test_umeyama_no_scale(self):
        """Test Umeyama when scale estimation is disabled."""
        n_points = 50
        points_q = self.rng.random((n_points, 3))

        # Create transform with scale=1.0 explicitly
        scale_gt = 1.0
        points_p = scale_gt * (self.R_orig @ points_q.T).T + self.t_orig

        # Run with with_scale=False
        scale_est, rot_est, trans_est = umeyama_alignment(points_q, points_p, with_scale=False)

        self.assertEqual(scale_est, 1.0)
        np.testing.assert_allclose(rot_est, self.R_orig, atol=1e-10)
        np.testing.assert_allclose(trans_est, self.t_orig, atol=1e-10)

    def test_umeyama_reflection(self):
        """
        Test edge case where the best alignment might involve a reflection,
        but we enforce a rotation matrix (det=1).
        """
        # Create a set of points
        points_q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)

        # Create p as a reflection of q across X-axis (x -> -x)
        # Umeyama should find a Rotation that approximates this without scaling/shearing illegally
        # if we strictly enforced rotation. However, Kabsch usually handles this by flipping the singular vector
        # to ensure det(R) = 1.

        # Let's rotate it by 90 deg Z to ensure it's a valid rigid body first
        # Rot Z 90: (x,y,z) -> (-y, x, z)
        points_p = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)

        scale, R, t = umeyama_alignment(points_q, points_p, with_scale=False)

        # Determinant should be +1 (valid rotation)
        self.assertAlmostEqual(np.linalg.det(R), 1.0)

        # Transform q with estimated R
        points_p_est = (R @ points_q.T).T + t
        np.testing.assert_allclose(points_p_est, points_p, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
