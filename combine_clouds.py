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
    get_metrics,
    mat_to_quat,
    umeyama_alignment,
    umeyama_alignment_two_points,
    verify_look_at_origin,
)

from pycolmap_utils import (
    c2w_to_cfw,
    cfw_to_c2w,
    cfw_to_w2c,
    get_poses,
    load_cameras_json,
)

from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track, pycolmap_to_batch_np_matrix_full


def get_common_names(rec_cam: pycolmap.Reconstruction, rec_pts: pycolmap.Reconstruction):
    names_cam = {img.name: cfw_to_c2w(img.cam_from_world)[0] for img in rec_cam.images.values()}
    names_pts = {img.name: cfw_to_c2w(img.cam_from_world)[0] for img in rec_pts.images.values()}

    common_names = set(names_cam.keys()) & set(names_pts.keys())
    return common_names, names_cam, names_pts


def get_look_directions(rec: pycolmap.Reconstruction):
    return {
        # The look direction is the positive Z-axis of the camera in world space.
        # This corresponds to the 3rd column of the camera-to-world rotation matrix (R^T).
        img.name: img.cam_from_world.rotation.matrix().T[:, 2]
        for img in rec.images.values()
    }


def get_alignment_transform(from_recon: pycolmap.Reconstruction, to_recon: pycolmap.Reconstruction):
    # Match images by name to find common camera centers
    # q: centers from point_source, p: centers from camera_source

    common_names, names_from, names_to = get_common_names(from_recon, to_recon)

    sorted_names = sorted(list(common_names))
    to_points = np.array([names_to[name] for name in sorted_names])
    from_points = np.array([names_from[name] for name in sorted_names])

    if len(common_names) <= 1:
        return 1.0, np.eye(3), np.zeros(3)

    if len(common_names) == 2:
        from_dirs = np.array([get_look_directions(from_recon)[name] for name in sorted_names])
        to_dirs = np.array([get_look_directions(to_recon)[name] for name in sorted_names])
        return umeyama_alignment_two_points(
            from_points=from_points,
            to_points=to_points,
            from_dirs=from_dirs,
            to_dirs=to_dirs,
        )

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
    except (ValueError, FileNotFoundError):
        # We assume a folder is given with 0, 1, 2 etc.
        # In that case we want to find the pcd with the most 3D points
        best_pcd = None
        best_num_points = 0
        best_path = None
        for folder in os.listdir(point_source_path):
            folder_path = Path(os.path.join(point_source_path, folder))
            if os.path.isdir(folder_path):
                pcd: pycolmap.Reconstruction = load_point_cloud(folder_path)
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


def load_point_clouds(point_source_path: Path) -> list[pycolmap.Reconstruction]:
    try:
        return [
            load_point_cloud(point_source_path / folder)
            for folder in os.listdir(point_source_path)
            if (point_source_path / folder).is_dir() and folder.isnumeric()
        ] or [load_point_cloud(point_source_path)]
    except ValueError:
        return [load_point_cloud(point_source_path)]


def load_cameras(camera_source_path: Path) -> pycolmap.Reconstruction:
    if camera_source_path.is_file():
        rec1 = load_json_data(camera_source_path)
        rec2 = load_cameras_json(camera_source_path)

        return rec1 if rec1.num_images() > rec2.num_images() else rec2
    rec = load_point_cloud(camera_source_path)
    return rec


def _swap_and_align(
    rec_cam: pycolmap.Reconstruction,
    rec_pts: pycolmap.Reconstruction,
    use_both_pcds: bool = False,
    align_each_point_set: bool = False,
    keep_backup_cams: bool = False,
):
    # Find the transform to bring point_source into camera_source space
    # p = s * R * q + t
    s, R, t = get_alignment_transform(from_recon=rec_pts, to_recon=rec_cam)

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

    return new_recon, s, R, t


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

    new_recon, s, R, t = _swap_and_align(
        rec_cam=rec_cam,
        rec_pts=rec_pts,
        use_both_pcds=use_both_pcds,
        align_each_point_set=align_each_point_set,
        keep_backup_cams=keep_backup_cams,
    )

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


def align_to_world_space(rec: pycolmap.Reconstruction, world_rec: pycolmap.Reconstruction):
    s, R, t = get_alignment_transform(from_recon=rec, to_recon=world_rec)
    rec_poses = get_poses(rec)
    world_poses = get_poses(world_rec)

    assert len(rec_poses.keys() & world_poses.keys()) <= len(world_poses)

    # 1. Transform all points into new world space
    for point_id, point in rec.points3D.items():
        point.xyz = s * (R @ point.xyz) + t

    # 2. Transform all camera extrinsics into new world space
    for image_id, image in rec.images.items():
        orig_t, orig_R = cfw_to_c2w(image.cam_from_world)
        new_t = s * (R @ orig_t) + t
        new_R = R @ orig_R

        aligned_cfw = c2w_to_cfw(new_t, new_R)
        image.cam_from_world.translation = aligned_cfw.translation
        image.cam_from_world.rotation.quat = aligned_cfw.rotation.quat

    return rec


def combine_recons(
    partial_recons: list[pycolmap.Reconstruction], glue_recon: pycolmap.Reconstruction
) -> pycolmap.Reconstruction:

    all_recons = [
        align_to_world_space(partial_recon, glue_recon)
        for partial_recon in partial_recons
        if partial_recon.num_points3D() > 0
    ]

    if len(all_recons) == 1:
        return all_recons[0]

    if len(all_recons) == 0:
        raise ValueError("No reconstructions to combine")
        return glue_recon

    all_points3d = []
    all_points_rgb = []
    all_points_xyf = []
    all_points_errors = []

    unique_extrinsics = []
    unique_intrinsics = []
    unique_image_names = []
    img_name_to_global_fidx = {}

    # Extract baseline camera type / size from first recon
    cam_type = "PINHOLE"
    image_size = np.array([1000, 1000])

    if len(all_recons[0].cameras) > 0:
        first_cam = next(iter(all_recons[0].cameras.values()))
        image_size = np.array([first_cam.width, first_cam.height])
        cam_type = str(first_cam.model).split(".")[-1]
        if cam_type not in ["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            cam_type = "PINHOLE"

    for recon in all_recons:
        points3D, points_rgb, points_xyf, points_errors, extrinsics, intrinsics, _, image_names = (
            pycolmap_to_batch_np_matrix_full(recon, camera_type=cam_type)
        )

        if len(points3D) == 0:
            continue

        # Deduplicate images and remap local frame indices to global
        local_to_global_fidx = {}
        for local_fidx, name in enumerate(image_names):
            if name not in img_name_to_global_fidx:
                global_fidx = len(unique_image_names)
                img_name_to_global_fidx[name] = global_fidx
                unique_image_names.append(name)
                unique_extrinsics.append(extrinsics[local_fidx])
                unique_intrinsics.append(intrinsics[local_fidx])
            local_to_global_fidx[local_fidx] = img_name_to_global_fidx[name]

        # Remap points_xyf from local batch index to the shared global index
        if len(points_xyf) > 0:
            for i in range(len(points_xyf)):
                local_fidx = int(points_xyf[i, 2])
                points_xyf[i, 2] = local_to_global_fidx.get(local_fidx, 0)

        all_points3d.append(points3D)
        all_points_rgb.append(points_rgb)
        all_points_xyf.append(points_xyf)
        all_points_errors.append(points_errors)

    # Concatenate all arrays
    comb_points3d = np.concatenate(all_points3d, axis=0) if all_points3d else np.empty((0, 3))
    comb_points_rgb = np.concatenate(all_points_rgb, axis=0) if all_points_rgb else np.empty((0, 3))
    comb_points_xyf = np.concatenate(all_points_xyf, axis=0) if all_points_xyf else np.empty((0, 3))
    comb_points_errors = np.concatenate(all_points_errors, axis=0) if all_points_errors else np.empty((0,))
    comb_extrinsics = np.stack(unique_extrinsics, axis=0) if unique_extrinsics else np.empty((0, 3, 4))
    comb_intrinsics = np.stack(unique_intrinsics, axis=0) if unique_intrinsics else np.empty((0, 3, 3))

    combined_recon = batch_np_matrix_to_pycolmap_wo_track(
        points3d=comb_points3d,
        points_xyf=comb_points_xyf,
        points_rgb=comb_points_rgb,
        extrinsics=comb_extrinsics,
        intrinsics=comb_intrinsics,
        image_size=image_size,
        shared_camera=False,  # TODO Try and address shared camera
        camera_type=cam_type,
        points_errors=comb_points_errors,
    )

    # Restore original image names
    for f, name in enumerate(unique_image_names):
        if (f + 1) in combined_recon.images:
            combined_recon.images[f + 1].name = name

    return combined_recon


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
        help="If set, points associated with each camera will be individually aligned to preserve local geometry. "
        "Otherwise, a single global transform is applied to all points.",
    )
    parser.add_argument(
        "--keep_backup_cams",
        action="store_true",
        help="If set, cameras missing from the camera_source will be kept and transformed using the global alignment. "
        "Otherwise they are removed.",
    )
    parser.add_argument(
        "--glue_parts",
        action="store_true",
        help="If set and camera_source is COLMAP, use VGGT to glue together partial COLMAP reconstructions.",
    )
    parser.add_argument("--output_dir", type=str, default="aligned_swap", help="Output directory.")

    args = parser.parse_args()

    glue_t1 = time.time()

    if args.glue_parts:
        cam_path = Path(args.camera_source)
        partial_recons = load_point_clouds(cam_path / "sparse")
        glue_recon = load_point_cloud(Path(args.point_source) / "sparse")
        combined_recon = combine_recons(partial_recons, glue_recon)
        glued_path = cam_path.parent / f"{cam_path.name}_glued" / "sparse"

        glued_path.mkdir(parents=True, exist_ok=True)
        combined_recon.write(glued_path)
        print(f"Glued reconstruction exported to {glued_path}")
        save_cameras_json(combined_recon, glued_path / "copied_cameras.json")
        cam_path = glued_path
    else:
        cam_path = Path(args.camera_source) / "sparse"

    glue_t2 = time.time()

    print(f"Glue took {glue_t2 - glue_t1:.2f} seconds")

    swap_t1 = time.time()
    swap_and_align(
        cam_path,
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
    if os.path.exists(Path(args.camera_source) / "gt_cameras.json"):
        # TODO Make sure these all save
        shutil.copy2(Path(args.camera_source) / "gt_cameras.json", Path(args.output_dir) / "gt_cameras.json")
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
                    "glue_t": glue_t2 - glue_t1,
                },
            },
            f,
        )
    if os.path.exists(Path(args.output_dir) / "gt_cameras.json"):
        pred_pcd = load_cameras(Path(args.output_dir) / "sparse")
        gt_pcd = load_cameras(Path(args.output_dir) / "gt_cameras.json")
        pred_pcd = align_to_world_space(pred_pcd, gt_pcd)
        save_cameras_json(pred_pcd, Path(args.output_dir) / "aligned_cameras.json")
