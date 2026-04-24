import argparse
import shutil
import time
import pycolmap
import numpy as np
import json
from pathlib import Path
import cv2
import os

from cam_utils import (
    build_matrix,
    c2w_to_cfw,
    cfw_to_c2w,
    decompose_matrix,
    get_metrics,
    mat_to_quat,
    get_poses,
    umeyama_alignment,
    verify_look_at_origin,
)


def get_common_names(rec_cam: pycolmap.Reconstruction, rec_pts: pycolmap.Reconstruction):
    names_cam = {img.name: cfw_to_c2w(img.cam_from_world)[0] for img in rec_cam.images.values()}
    names_pts = {img.name: cfw_to_c2w(img.cam_from_world)[0] for img in rec_pts.images.values()}

    common_names = set(names_cam.keys()) & set(names_pts.keys())
    return common_names, names_cam, names_pts


def get_alignment_transform(from_recon: pycolmap.Reconstruction, to_recon: pycolmap.Reconstruction):
    # Match images by name to find common camera centers
    # q: centers from point_source, p: centers from camera_source

    common_names, names_from, names_to = get_common_names(from_recon, to_recon)

    if len(common_names) < 3:
        raise ValueError(f"Need at least 3 common images for alignment. Found: {len(common_names)}")

    sorted_names = sorted(list(common_names))
    to_points = np.array([names_to[name] for name in sorted_names])
    from_points = np.array([names_from[name] for name in sorted_names])

    return umeyama_alignment(from_points=from_points, to_points=to_points)


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
        verify_opengl = verify_look_at_origin(c2w, is_opencv=False)
        c2w[0:3, 1:3] *= -1
        verify_colmap = verify_look_at_origin(c2w, is_opencv=True)
        assert verify_colmap["is_pointing_at_origin"] and verify_opengl["is_pointing_at_origin"]

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

        cfw = c2w_to_cfw(translation, rotation_matrix)
        images[name].cam_from_world = cfw

    recon = pycolmap.Reconstruction()
    for name, camera in cameras.items():
        recon.add_camera(camera)

    for name, image in images.items():
        recon.add_image(image)

    return recon


def save_cameras_json(
    reconstruction,
    output_path: Path,
    common_names: set[str] | None = None,
    alignment: tuple[float, np.ndarray, np.ndarray] | None = None,
):
    """
    Exports camera intrinsics and extrinsics to a JSON file.
    If 'alignment' (s, R, t) is provided, the poses are transformed before saving.
    """
    out_data = {}

    s, R, t = alignment if alignment is not None else (1.0, np.eye(3), np.zeros(3))

    for img_id, img in reconstruction.images.items():
        cam = reconstruction.cameras[img.camera_id]
        if common_names is not None and img.name not in common_names:
            continue

        translation, rotation_matrix = cfw_to_c2w(img.cam_from_world)

        translation = s * (R @ translation) + t
        rotation_matrix = R @ rotation_matrix

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


def load_point_cloud(
    point_source_path: Path, return_path: bool = False
) -> pycolmap.Reconstruction | tuple[pycolmap.Reconstruction, Path]:
    try:
        if "cameras.bin" not in os.listdir(point_source_path):
            raise ValueError(f"No cameras.bin found in {point_source_path}")
        pcd = pycolmap.Reconstruction(point_source_path)
        return (pcd, point_source_path) if return_path else pcd
    except Exception as e:
        # We assume a folder is given with 0, 1, 2 etc.
        # In that case we want to find the pcd with the most 3D points
        best_pcd = None
        best_num_points = 0
        best_path = None
        for folder in os.listdir(point_source_path):
            folder_path = Path(os.path.join(point_source_path, folder))
            if os.path.isdir(folder_path):
                pcd = load_point_cloud(folder_path)
                num_points = len(pcd.points3D)
                if num_points > best_num_points:
                    best_pcd = pcd
                    best_num_points = num_points
                    best_path = folder_path

        if best_pcd is not None:
            print("Best point cloud found:", best_path, "with", len(best_pcd.points3D), "points")
            return (best_pcd, best_path) if return_path else best_pcd
        else:
            raise ValueError(f"No point cloud found in {point_source_path}")


def load_cameras(camera_source_path: Path) -> pycolmap.Reconstruction:
    rec = load_json_data(camera_source_path) if camera_source_path.is_file() else load_point_cloud(camera_source_path)
    return rec


def swap_and_align(
    camera_source_path: Path,
    point_source_path: Path,
    output_path: Path,
    use_both_pcds: bool = False,
    align_each_point_set: bool = False,
):
    """
    This migrates the cameras from the camera_source_path to the point_source_path in place.
    It then transforms the points in point_source_path to match the cameras.

    rec_pts will contain cameras and points in the coordinate frame of rec_cam.
    """
    print(f"Loading reconstructions from {camera_source_path} and {point_source_path}")

    rec_cam = load_cameras(camera_source_path)
    orig_rec_cam = load_cameras(camera_source_path)
    rec_pts = load_point_cloud(point_source_path)
    orig_rec_pts = load_point_cloud(point_source_path)

    # Find the transform to bring point_source into camera_source space
    # p = s * R * q + t
    s, R, t = get_alignment_transform(from_recon=rec_pts, to_recon=rec_cam)

    pts_poses = get_poses(rec_pts)
    cam_poses = get_poses(rec_cam)
    keys = sorted(pts_poses.keys() & cam_poses.keys())

    # for c2w in cam_poses.values():
    #     verify = verify_look_at_origin(c2w, is_opencv=True)
    #     assert verify["is_pointing_at_origin"]

    # for i, c2w in enumerate(pts_poses.values()):
    #     translation, rotation_matrix = decompose_matrix(c2w)
    #     translation = s * (R @ translation) + t
    #     rotation_matrix = R @ rotation_matrix
    #     c2w = build_matrix(translation, rotation_matrix)

    #     verify = verify_look_at_origin(c2w, is_opencv=True)
    #     assert verify["is_pointing_at_origin"]

    pts_cam_positions = np.array([s * R @ pts_poses[k][:3, 3] + t for k in keys])
    rec_cam_positions = np.array([cam_poses[k][:3, 3] for k in keys])

    print(f"Alignment found: Scale={s:.4f}")

    img_to_img_map = {image.name: image for image in rec_cam.images.values()}
    img_to_cam_map = {image.name: rec_cam.cameras[image.camera_id] for image in rec_cam.images.values()}
    pts_cam_id_to_img_map = {img.camera_id: img.name for img in rec_pts.images.values()}

    transformed_point_ids = set()

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

        orig_t, orig_R = cfw_to_c2w(image.cam_from_world)
        new_t, new_R = cfw_to_c2w(src_image.cam_from_world)

        # Copy the intrinsics from src_cam to cam and the extrinsics from src_image to image
        src_cam.camera_id = cam.camera_id
        rec_pts.cameras[image.camera_id] = src_cam
        image.cam_from_world.translation = src_image.cam_from_world.translation
        image.cam_from_world.rotation.quat = src_image.cam_from_world.rotation.quat
        # Transform the specific points attached to this camera to preserve local geometry
        if align_each_point_set:
            for p2d in image.points2D:
                if p2d.has_point3D():
                    pid = p2d.point3D_id
                    if pid not in transformed_point_ids and pid in rec_pts.points3D:
                        p3d = rec_pts.points3D[pid]
                        # Convert to local camera frame, then back to world using the new camera pose
                        X_cam = orig_R.T @ (p3d.xyz - orig_t)
                        p3d.xyz = s * (new_R @ X_cam) + new_t
                        transformed_point_ids.add(pid)

    # Rotate, scale and translate remaining points to match src coord frame
    if s > 0.0:
        for pid, p in rec_pts.points3D.items():
            if pid not in transformed_point_ids:
                p.xyz = s * (R @ p.xyz) + t

    # If requested, merge points from the camera source (already in the target coordinate frame)
    if use_both_pcds:
        print("Merging points from both reconstructions...")
        for p in rec_cam.points3D.values():
            # Generate a new unique ID to avoid collisions
            new_id = rec_pts.add_point3D(p.xyz, pycolmap.Track(), p.color)

    output_path.mkdir(parents=True, exist_ok=True)
    rec_pts.write(output_path)
    print(f"Reconstruction exported to {output_path}")

    common_names = get_common_names(rec_cam, rec_pts)[0]

    # all_recons = [rec_cam, rec_pts, orig_rec_cam, orig_rec_pts]
    # names = ["rec_cam", "rec_pts", "orig_rec_cam", "orig_rec_pts"]
    # for i, rec_1 in enumerate(all_recons):
    #     for j, rec_2 in enumerate(all_recons):
    #         print(f"Comparing {names[i]} and {names[j]}")
    #         rre_list, rte_list = get_metrics(get_poses(rec_1), get_poses(rec_2), alignment=(s, R, t))
    #         print(f"Mean RRE: {np.mean(rre_list):.4f}")
    #         print(f"Mean RTE: {np.mean(rte_list):.4f}")

    print("Before alignment")
    rre_list, rte_list = get_metrics(get_poses(orig_rec_cam), get_poses(orig_rec_pts))
    print(f"Mean RRE: {np.mean(rre_list):.4f}")
    print(f"Mean RTE: {np.mean(rte_list):.4f}")

    print("After alignment")
    rre_list, rte_list = get_metrics(get_poses(orig_rec_cam), get_poses(orig_rec_pts), alignment=(s, R, t))
    print(f"Mean RRE: {np.mean(rre_list):.4f}")
    print(f"Mean RTE: {np.mean(rte_list):.4f}")

    print("After swap (these should be 0)")
    rre_list, rte_list = get_metrics(get_poses(rec_pts), get_poses(orig_rec_cam))
    print(f"Mean RRE: {np.mean(rre_list):.4f}")
    print(f"Mean RTE: {np.mean(rte_list):.4f}")

    save_cameras_json(rec_pts, output_path / "copied_cameras.json", common_names=common_names)
    save_cameras_json(orig_rec_cam, output_path / "original_cameras.json", common_names=common_names)
    save_cameras_json(
        orig_rec_pts, output_path / "aligned_cameras.json", common_names=common_names, alignment=(s, R, t)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_source", type=str, required=True, help="Path to point cloud with desired cameras.")
    parser.add_argument("--point_source", type=str, required=True, help="Path to point cloud with desired points.")
    parser.add_argument(
        "--use_both_pcds",
        action="store_true",
        help="If set, points from both reconstructions will be included in the output.",
    )
    parser.add_argument(
        "--align_each_point_set",
        action="store_true",
        help="If set, points associated with each camera will be individually aligned to preserve local geometry. Otherwise, a single global transform is applied to all points.",
    )
    parser.add_argument("--output_dir", type=str, default="aligned_swap", help="Output directory.")

    args = parser.parse_args()

    swap_t1 = time.time()
    swap_and_align(
        Path(args.camera_source) / "sparse",
        Path(args.point_source) / "sparse",
        Path(args.output_dir) / "sparse",
        use_both_pcds=args.use_both_pcds,
        align_each_point_set=args.align_each_point_set,
    )
    swap_t2 = time.time()

    print(f"Swap took {swap_t2 - swap_t1:.2f} seconds")
    copy_t1 = time.time()
    shutil.copytree(Path(args.camera_source) / "images", Path(args.output_dir) / "images", dirs_exist_ok=True)
    shutil.copytree(Path(args.point_source) / "images", Path(args.output_dir) / "images", dirs_exist_ok=True)
    if os.path.exists(Path(args.camera_source) / "depths"):
        shutil.copytree(Path(args.camera_source) / "depths", Path(args.output_dir) / "depths", dirs_exist_ok=True)
    if os.path.exists(Path(args.point_source) / "depths"):
        shutil.copytree(Path(args.point_source) / "depths", Path(args.output_dir) / "depths", dirs_exist_ok=True)
    copy_t2 = time.time()

    print(f"Copy took {copy_t2 - copy_t1:.2f} seconds")

    with open(Path(args.output_dir) / "stat.json", "w") as f:
        json.dump(
            {
                "date": time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime()),
                "input_path": args.camera_source,
                "name": Path(args.output_dir).name,
                "type": "combined",
                "num_images": len(os.listdir(Path(args.output_dir) / "images")),
                "profiling": {
                    "swap_t": swap_t2 - swap_t1,
                    "copy_t": copy_t2 - copy_t1,
                },
            },
            f,
        )
