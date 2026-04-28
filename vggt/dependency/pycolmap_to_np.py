import numpy as np


def extract_poses_from_reconstruction(
    reconstruction, base_image_path_list, extrinsic, intrinsic, scale, use_backup_cams: bool = False
):
    # Extract optimized extrinsics and intrinsics from the BA reconstruction
    num_frames = len(base_image_path_list)
    optimized_extrinsic = np.zeros((num_frames, 3, 4), dtype=np.float32)
    optimized_intrinsic = np.zeros((num_frames, 3, 3), dtype=np.float32)

    for i in range(num_frames):
        pyimageid = i + 1
        if pyimageid in reconstruction.images:
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]

            # Extract Extrinsic
            if hasattr(pyimage, "cam_from_world"):
                R = pyimage.cam_from_world.rotation.matrix()
                t = pyimage.cam_from_world.translation
            else:
                R = pyimage.qvec2rotmat()
                t = pyimage.tvec

            optimized_extrinsic[i, :3, :3] = R
            optimized_extrinsic[i, :3, 3] = t

            # Extract Intrinsic
            K = np.eye(3, dtype=np.float32)
            params = pycamera.params

            # Compatibility across PyCOLMAP versions
            cam_model_name = pycamera.model.name if hasattr(pycamera, "model") else pycamera.model_name

            if cam_model_name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                K[0, 0] = params[0]  # f
                K[1, 1] = params[0]  # f
                K[0, 2] = params[1]  # cx
                K[1, 2] = params[2]  # cy
            elif cam_model_name == "PINHOLE":
                K[0, 0] = params[0]  # fx
                K[1, 1] = params[1]  # fy
                K[0, 2] = params[2]  # cx
                K[1, 2] = params[3]  # cy
            else:
                print(cam_model_name, "not supported. Using fallback.")
                # Fallback
                K[0, 0] = params[0]
                K[1, 1] = params[1] if len(params) > 1 else params[0]
                K[0, 2] = params[2] if len(params) > 2 else intrinsic[i, 0, 2]
                K[1, 2] = params[3] if len(params) > 3 else intrinsic[i, 1, 2]

            # Rescale intrinsics back to vggt_fixed_resolution
            K[:2, :] /= scale
            optimized_intrinsic[i] = K
        elif use_backup_cams:
            # Fallback if frame was dropped by COLMAP during reconstruction
            optimized_extrinsic[i] = extrinsic[i]
            orig_k = intrinsic[i].copy()
            orig_k[:2, :] /= scale
            optimized_intrinsic[i] = orig_k

    return optimized_extrinsic, optimized_intrinsic
