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
    cfw_to_w2c,
    decompose_matrix,
    get_metrics,
    mat_to_quat,
    get_poses,
    umeyama_alignment,
    verify_look_at_origin,
)
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track


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
    keep_backup_cams: bool = False,
):
    """
    This migrates the cameras from the camera_source_path to the point_source_path.
    It then transforms the points in point_source_path to match the cameras and rebuilds
    the reconstruction using VGGT's batched Numpy matrix tools.
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

    pts_cam_positions = np.array([s * R @ pts_poses[k][:3, 3] + t for k in keys])
    rec_cam_positions = np.array([cam_poses[k][:3, 3] for k in keys])

    print(f"Alignment found: Scale={s:.4f}")

    img_to_img_map = {image.name: image for image in rec_cam.images.values()}
    img_to_cam_map = {image.name: rec_cam.cameras[image.camera_id] for image in rec_cam.images.values()}

    transformed_point_ids = set()

    # Step 1: In-place modify rec_pts to inherit accurate poses and calculate points
    for image_id, image in rec_pts.images.items():
        img_name = image.name

        if img_name in img_to_img_map:
            cam = rec_pts.cameras[image.camera_id]
            src_cam = img_to_cam_map[img_name]
            src_image = img_to_img_map[img_name]

            orig_t, orig_R = cfw_to_c2w(image.cam_from_world)
            new_t, new_R = cfw_to_c2w(src_image.cam_from_world)

            cam.model = src_cam.model
            cam.width = src_cam.width
            cam.height = src_cam.height
            cam.params = src_cam.params

            image.cam_from_world.translation = src_image.cam_from_world.translation
            image.cam_from_world.rotation.quat = src_image.cam_from_world.rotation.quat

            if align_each_point_set:
                for p2d in image.points2D:
                    if p2d.has_point3D():
                        pid = p2d.point3D_id
                        if pid not in transformed_point_ids and pid in rec_pts.points3D:
                            p3d = rec_pts.points3D[pid]
                            X_cam = orig_R.T @ (p3d.xyz - orig_t)
                            p3d.xyz = s * (new_R @ X_cam) + new_t
                            transformed_point_ids.add(pid)

        elif keep_backup_cams:
            orig_t, orig_R = cfw_to_c2w(image.cam_from_world)
            new_t = s * (R @ orig_t) + t
            new_R = R @ orig_R

            aligned_cfw = c2w_to_cfw(new_t, new_R)
            image.cam_from_world.translation = aligned_cfw.translation
            image.cam_from_world.rotation.quat = aligned_cfw.rotation.quat

    # Step 2: Extract aligned data into Numpy arrays for VGGT's array-to-colmap function
    extrinsics_list = []
    intrinsics_list = []
    image_names = []
    image_id_to_fidx = {}

    image_size = np.array([1000, 1000])  # Fallback
    cam_type = "SIMPLE_PINHOLE"
    fidx = 0

    for image_id, image in rec_pts.images.items():
        img_name = image.name
        keep = (img_name in img_to_img_map) or keep_backup_cams
        if not keep:
            continue

        image_id_to_fidx[image_id] = fidx
        image_names.append(img_name)

        # Extrinsics: VGGT expects W2C format
        w2c_t, w2c_R = cfw_to_w2c(image.cam_from_world)
        ext = np.zeros((3, 4))
        ext[:3, :3] = w2c_R
        ext[:3, 3] = w2c_t
        extrinsics_list.append(ext)

        # Intrinsics
        cam = rec_pts.cameras[image.camera_id]
        K = np.eye(3)
        cam_model_str = str(cam.model).split(".")[-1]
        cam_type = cam_model_str
        image_size = np.array([cam.width, cam.height])

        if cam_model_str in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE"]:
            K[0, 0] = cam.params[0]
            K[1, 1] = cam.params[1]
            K[0, 2] = cam.params[2]
            K[1, 2] = cam.params[3]
        else:
            K[0, 0] = cam.params[0]
            K[1, 1] = cam.params[0]
            K[0, 2] = cam.params[1]
            K[1, 2] = cam.params[2]
        intrinsics_list.append(K)
        fidx += 1

    extrinsics = np.stack(extrinsics_list) if extrinsics_list else np.empty((0, 3, 4))
    intrinsics = np.stack(intrinsics_list) if intrinsics_list else np.empty((0, 3, 3))

    points3d = []
    points_rgb = []
    points_xyf = []
    points_errors = []

    for pid, p in rec_pts.points3D.items():
        if s > 0.0 and pid not in transformed_point_ids:
            p.xyz = s * (R @ p.xyz) + t

        # Get first valid observation for this point to link to frames (VGGT ignores full tracks)
        valid_el = None
        for el in p.track.elements:
            if el.image_id in image_id_to_fidx:
                valid_el = el
                break

        if valid_el is not None:
            points3d.append(p.xyz)
            points_rgb.append(p.color)
            points_errors.append(p.error)

            fidx_val = image_id_to_fidx[valid_el.image_id]
            img = rec_pts.images[valid_el.image_id]
            p2d_xy = img.points2D[valid_el.point2D_idx].xy
            points_xyf.append([p2d_xy[0], p2d_xy[1], fidx_val])

    points3d = np.array(points3d) if points3d else np.empty((0, 3))
    points_rgb = np.array(points_rgb) if points_rgb else np.empty((0, 3))
    points_xyf = np.array(points_xyf) if points_xyf else np.empty((0, 3))
    points_errors = np.array(points_errors) if points_errors else np.empty((0,))

    # VGGT natively supports a narrow range of PyCOLMAP camera types
    if cam_type not in ["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        cam_type = "PINHOLE"

    print("Building new reconstruction via VGGT batch_np_matrix_to_pycolmap_wo_track...")
    new_recon = batch_np_matrix_to_pycolmap_wo_track(
        points3d=points3d,
        points_xyf=points_xyf,
        points_rgb=points_rgb,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_size=image_size,
        shared_camera=False,
        camera_type=cam_type,
        points_errors=points_errors,
    )

    # VGGT creates 1:1 camera-to-image mappings where camera_id = fidx + 1
    for orig_image_id, fidx_val in image_id_to_fidx.items():
        new_cam_id = fidx_val + 1

        # Get the perfect camera object we already modified earlier in the script
        orig_cam = rec_pts.cameras[rec_pts.images[orig_image_id].camera_id]

        # Overwrite VGGT's generic camera with our exact model, width, height, and all params
        restored_cam = pycolmap.Camera(
            camera_id=new_cam_id,
            model=orig_cam.model,
            width=orig_cam.width,
            height=orig_cam.height,
            params=orig_cam.params,
        )
        new_recon.cameras[new_cam_id] = restored_cam

    # Restore image names (VGGT overrides them with 'image_{fidx}')
    for f, name in enumerate(image_names):
        if (f + 1) in new_recon.images:
            new_recon.images[f + 1].name = name

    if use_both_pcds:
        print("Merging points from both reconstructions...")
        for p in rec_cam.points3D.values():
            new_recon.add_point3D(p.xyz, pycolmap.Track(), p.color)

    output_path.mkdir(parents=True, exist_ok=True)
    new_recon.write(output_path)
    print(f"Reconstruction exported to {output_path}")

    common_names = get_common_names(rec_cam, new_recon)[0]

    print("Before alignment")
    rre_list, rte_list = get_metrics(get_poses(orig_rec_cam), get_poses(orig_rec_pts))
    print(f"Mean RRE: {np.mean(rre_list):.4f}")
    print(f"Mean RTE: {np.mean(rte_list):.4f}")

    print("After alignment")
    rre_list, rte_list = get_metrics(get_poses(orig_rec_cam), get_poses(orig_rec_pts), alignment=(s, R, t))
    print(f"Mean RRE: {np.mean(rre_list):.4f}")
    print(f"Mean RTE: {np.mean(rte_list):.4f}")

    print("After swap (these should be 0)")
    rre_list, rte_list = get_metrics(get_poses(new_recon), get_poses(orig_rec_cam))
    print(f"Mean RRE: {np.mean(rre_list):.4f}")
    print(f"Mean RTE: {np.mean(rte_list):.4f}")

    save_cameras_json(new_recon, output_path / "copied_cameras.json", common_names=common_names)
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
    parser.add_argument(
        "--keep_backup_cams",
        action="store_true",
        help="If set, cameras missing from the camera_source will be kept and transformed using the global alignment. Otherwise they are removed.",
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
        keep_backup_cams=args.keep_backup_cams,
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
