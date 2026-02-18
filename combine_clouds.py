import argparse
import pycolmap
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional
import cv2
import os
from scipy.spatial.transform import Rotation as Rot


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


def get_common_names(rec_cam: pycolmap.Reconstruction, rec_pts: pycolmap.Reconstruction):
    names_cam = {img.name: img.cam_from_world.translation for img in rec_cam.images.values()}
    names_pts = {img.name: img.cam_from_world.translation for img in rec_pts.images.values()}

    common_names = set(names_cam.keys()) & set(names_pts.keys())
    return common_names, names_cam, names_pts


def get_alignment_transform(rec_cam: pycolmap.Reconstruction, rec_pts: pycolmap.Reconstruction):
    # Match images by name to find common camera centers
    # q: centers from point_source, p: centers from camera_source

    common_names, names_cam, names_pts = get_common_names(rec_cam, rec_pts)

    if len(common_names) < 3:
        raise ValueError(f"Need at least 3 common images for alignment. Found: {len(common_names)}")

    p = np.array([names_cam[name] for name in common_names])
    q = np.array([names_pts[name] for name in common_names])

    return umeyama_alignment(q, p)


def load_json_data(path: Path) -> pycolmap.Reconstruction:
    with open(path, "r") as f:
        data = json.load(f)

    cam_angle = data.get("camera_angle_x", 0.0)
    frames = data.get("frames", [])
    frames = sorted(frames, key=lambda x: x["file_path"])

    cameras = {}
    images = {}

    def get_intrinsics(w: int, h: int):

        fl_x = 0.5 * w / np.tan(0.5 * cam_angle)
        fl_y = fl_x
        cx = w / 2.0
        cy = h / 2.0

        return w, h, fl_x, fl_y, cx, cy

    for i, frame in enumerate(frames):
        fname = frame["file_path"] + ".png"
        base_dir = os.path.dirname(path)
        im_path = os.path.join(base_dir, fname)
        im = cv2.imread(im_path)
        frame_intrinsics = get_intrinsics(im.shape[0], im.shape[1])

        name = Path(fname).name

        c2w = np.array(frame["transform_matrix"])
        # c2w[0:3, 1:3] *= -1

        if frame_intrinsics:
            w_i, h_i, fx_i, fy_i, cx_i, cy_i = frame_intrinsics
        else:
            w_i, h_i = 100, 100
            fx_i, fy_i, cx_i, cy_i = 100, 100, 50, 50

        K = np.eye(3)
        K[0, 0] = fx_i
        K[1, 1] = fy_i
        K[0, 2] = cx_i
        K[1, 2] = cy_i

        cam_id = i

        camera = pycolmap.Camera(
            camera_id=cam_id,
            model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
            width=int(w_i),
            height=int(h_i),
            params=[fx_i, cx_i, cy_i],
        )
        cameras[name] = camera
        images[name] = pycolmap.Image(
            image_id=cam_id,
            name=name,
            camera_id=cam_id,
        )

        translation = c2w[:3, 3]
        rotation_matrix = c2w[:3, :3]
        rotation_quat = Rot.from_matrix(rotation_matrix).as_quat()

        images[name].cam_from_world.rotation = pycolmap.Rotation3d(rotation_quat)
        images[name].cam_from_world.translation = translation

    recon = pycolmap.Reconstruction()
    for name, camera in cameras.items():
        recon.add_camera(camera)

    for name, image in images.items():
        recon.add_image(image)

    return recon


def save_cameras_json(
    reconstruction, output_path: Path, alignment: Optional[Tuple] = None, common_names: set[str] | None = None
):
    """
    Exports camera intrinsics and extrinsics to a JSON file.
    If 'alignment' (s, R, t) is provided, the poses are transformed before saving.
    """
    out_data = {}

    # Unpack alignment if provided
    s, R_align, t_align = (1.0, np.eye(3), np.zeros(3)) if alignment is None else alignment

    for img_id, img in reconstruction.images.items():
        cam = reconstruction.cameras[img.camera_id]
        if common_names is not None and img.name not in common_names:
            continue

        # 1. Get current Pose (World-to-Camera)
        cam_from_world = img.cam_from_world
        R_w2c = cam_from_world.rotation.matrix()
        tvec = cam_from_world.translation

        # 2. Convert to Camera Center (Camera-to-World) for easier transformation
        # C = -R^T * t
        center = -R_w2c.T @ tvec

        # 3. Apply Sim(3) Transform to Center: C_new = s * R * C + t
        center_new = s * (R_align @ center) + t_align

        # 4. Apply Rotation Transform to Orientation
        # The camera orientation relative to the world rotates by R_align
        # R_w2c_new = R_w2c_old * R_align^T
        R_w2c_new = R_w2c @ R_align.T

        # 5. Recompute Translation: t_new = -R_w2c_new * C_new
        tvec_new = -R_w2c_new @ center_new

        # 6. Convert Rotation back to Quaternion
        qvec_new = pycolmap.Rotation3d(R_w2c_new).quat

        out_data[img.name] = {
            "camera_id": img.camera_id,
            "model": str(cam.model),
            "width": cam.width,
            "height": cam.height,
            "params": cam.params.tolist(),
            "params_info": cam.params_info,
            "extrinsics": {"qvec": qvec_new.tolist(), "tvec": tvec_new.tolist(), "center": center_new.tolist()},
        }

    sorted_out_data = {k: out_data[k] for k in sorted(out_data.keys())}
    out_data = sorted_out_data

    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=4)
    print(f"Saved camera parameters to {output_path}")


def swap_and_align(camera_source_path: Path, point_source_path: Path, output_path: Path):
    """
    This migrates the cameras from the camera_source_path to the point_source_path in place.
    """
    rec_cam = (
        load_json_data(camera_source_path)
        if camera_source_path.is_file()
        else pycolmap.Reconstruction(camera_source_path)
    )
    rec_pts = pycolmap.Reconstruction(point_source_path)
    orig_rec_pts = pycolmap.Reconstruction(point_source_path)

    # Find the transform to bring camera_source int point_source space
    # p = s * R * q + t
    s, R, t = get_alignment_transform(rec_pts, rec_cam)
    print(f"Alignment found: Scale={s:.4f}")

    img_to_img_map = {image.name: image for image in rec_cam.images.values()}
    img_to_cam_map = {image.name: rec_cam.cameras[image.camera_id] for image in rec_cam.images.values()}
    pts_cam_id_to_img_map = {img.camera_id: img.name for img in rec_pts.images.values()}

    for image_id, image in rec_pts.images.items():
        cam = rec_pts.cameras[image.camera_id]
        cam_id = cam.camera_id
        if cam_id not in pts_cam_id_to_img_map:
            continue

        img_name = pts_cam_id_to_img_map[cam_id]
        if img_name not in img_to_cam_map:
            continue

        src_cam = img_to_cam_map[img_name]
        src_image = img_to_img_map[img_name]

        # Copy the intrinsics from src_cam to cam and the extrinsics from src_image to image

        src_cam.camera_id = cam.camera_id
        cam = src_cam

        Q = Rot.from_matrix(R).as_quat()
        w2c = src_image.cam_from_world
        w2c.translation = s * (R @ w2c.translation) + t
        w2c.rotation.quat *= Q
        image.cam_from_world = w2c

    output_path.mkdir(parents=True, exist_ok=True)
    rec_pts.write(output_path)
    print(f"Reconstruction exported to {output_path}")

    common_names = get_common_names(rec_cam, rec_pts)[0]

    save_cameras_json(rec_cam, output_path / "cameras_aligned_source.json", alignment=(s, R, t), common_names=common_names)

    save_cameras_json(
        orig_rec_pts, output_path / "cameras_reference.json", common_names=common_names
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_source", type=str, required=True, help="Path to point cloud with desired cameras.")
    parser.add_argument("--point_source", type=str, required=True, help="Path to point cloud with desired points.")
    parser.add_argument("--output_dir", type=str, default="aligned_swap", help="Output directory.")

    args = parser.parse_args()

    swap_and_align(Path(args.camera_source), Path(args.point_source), Path(args.output_dir))
