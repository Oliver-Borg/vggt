import argparse
import pycolmap
import numpy as np
import json
from pathlib import Path
import cv2
import os

from cam_utils import cfw_to_c2w, mat_to_quat, umeyama_alignment


def get_common_names(rec_cam: pycolmap.Reconstruction, rec_pts: pycolmap.Reconstruction):
    names_cam = {img.name: cfw_to_c2w(img.cam_from_world)[0] for img in rec_cam.images.values()}
    names_pts = {img.name: cfw_to_c2w(img.cam_from_world)[0] for img in rec_pts.images.values()}

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
        # TODO Check if this is necessary
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
        rotation_quat = mat_to_quat(rotation_matrix)

        images[name].cam_from_world.rotation = pycolmap.Rotation3d(rotation_quat)
        images[name].cam_from_world.translation = translation

    recon = pycolmap.Reconstruction()
    for name, camera in cameras.items():
        recon.add_camera(camera)

    for name, image in images.items():
        recon.add_image(image)

    return recon


def save_cameras_json(reconstruction, output_path: Path, common_names: set[str] | None = None):
    """
    Exports camera intrinsics and extrinsics to a JSON file.
    If 'alignment' (s, R, t) is provided, the poses are transformed before saving.
    """
    out_data = {}

    for img_id, img in reconstruction.images.items():
        cam = reconstruction.cameras[img.camera_id]
        if common_names is not None and img.name not in common_names:
            continue

        translation, rotation_matrix = cfw_to_c2w(img.cam_from_world)
        rot_quat = mat_to_quat(rotation_matrix)

        out_data[img.name] = {
            "camera_id": img.camera_id,
            "model": str(cam.model),
            "width": cam.width,
            "height": cam.height,
            "params": cam.params.tolist(),
            "params_info": cam.params_info,
            "extrinsics": {"qvec": rot_quat.tolist(), "tvec": translation.tolist()},
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
    orig_rec_cam = (
        load_json_data(camera_source_path)
        if camera_source_path.is_file()
        else pycolmap.Reconstruction(camera_source_path)
    )
    rec_pts = pycolmap.Reconstruction(point_source_path)
    orig_rec_pts = pycolmap.Reconstruction(point_source_path)

    # Find the transform to bring point_source into camera_source space
    # p = s * R * q + t
    s, R, t = get_alignment_transform(rec_cam, rec_pts)
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
        rec_pts.cameras[image.camera_id] = src_cam
        image.cam_from_world.translation = src_image.cam_from_world.translation
        image.cam_from_world.rotation.quat = src_image.cam_from_world.rotation.quat

        # Rotate, scale and translate points to match src coord frame
        for p in rec_pts.points3D.values():
            p.xyz = s * (R @ p.xyz) + t


    output_path.mkdir(parents=True, exist_ok=True)
    rec_pts.write(output_path)
    print(f"Reconstruction exported to {output_path}")

    common_names = get_common_names(rec_cam, rec_pts)[0]

    save_cameras_json(rec_cam, output_path / "cameras_aligned_source.json", common_names=common_names)

    save_cameras_json(orig_rec_pts, output_path / "cameras_reference.json", common_names=common_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_source", type=str, required=True, help="Path to point cloud with desired cameras.")
    parser.add_argument("--point_source", type=str, required=True, help="Path to point cloud with desired points.")
    parser.add_argument("--output_dir", type=str, default="aligned_swap", help="Output directory.")

    args = parser.parse_args()

    swap_and_align(Path(args.camera_source), Path(args.point_source), Path(args.output_dir))
