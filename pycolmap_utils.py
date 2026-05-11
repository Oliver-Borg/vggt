import json
from pathlib import Path

import numpy as np
import pycolmap

from cam_utils import build_matrix, c2w_to_w2c, mat_to_quat, quat_to_mat, w2c_to_c2w

"""
c2w = pose (3D position and rotation relative to the origin)
w2c = cfw = extrinsics (projection from world coordinates to camera frame)
"""


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


def get_poses(recon: pycolmap.Reconstruction) -> dict[str, np.ndarray]:
    to_return = {}
    for image in recon.images.values():
        to_return[image.name] = build_matrix(*cfw_to_c2w(image.cam_from_world))
    return to_return


def load_cameras_json(path: Path) -> pycolmap.Reconstruction:
    """
    Loads camera intrinsics and extrinsics from a JSON file into a pycolmap Reconstruction.
    """
    with open(path, "r") as f:
        data = json.load(f)

    recon = pycolmap.Reconstruction()

    if "frames" in data:
        return recon

    # Track cameras we've already added so we don't duplicate them
    added_cameras = set()

    for frame in data.values():
        cam_id = frame["camera_id"]
        if cam_id not in added_cameras:
            model_str = frame["model"]
            # Handle string formats like "CameraModelId.PINHOLE"
            model_attr = model_str.split(".")[-1]
            model = getattr(pycolmap.CameraModelId, model_attr)

            camera = pycolmap.Camera(
                camera_id=cam_id,
                model=model,
                width=int(frame["width"]),
                height=int(frame["height"]),
                params=frame["params"],
            )
            recon.add_camera(camera)
            added_cameras.add(cam_id)

    # Reconstruct images and map extrinsics
    for i, (image_name, frame) in enumerate(data.items(), start=1):
        image = pycolmap.Image(
            image_id=i,  # 1-indexed is standard for pycolmap
            name=image_name,
            camera_id=frame["camera_id"],
        )

        # Extract the saved C2W extrinsics
        qvec = np.array(frame["extrinsics"]["qvec"])
        tvec = np.array(frame["extrinsics"]["tvec"])

        # Convert quaternion back to rotation matrix
        rotation_matrix = quat_to_mat(qvec)

        # Convert C2W variables back to CFW (cam_from_world) for pycolmap
        cfw = c2w_to_cfw(tvec, rotation_matrix)
        image.cam_from_world = cfw

        recon.add_image(image)

    return recon
