import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import open3d as o3d

from vggt.utils.geometry import project_world_points_to_cam, unproject_depth_map_to_point_map


def detect_outlier_cameras(
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth_maps: np.ndarray,
    images: np.ndarray,
    depth_conf: np.ndarray | None = None,
    k: int = 5,
    photo_weight: float = 1.0,
    depth_weight: float = 0.1,
    num_outliers: int = 1,
) -> list[int]:
    """
    Detects outlier cameras by reprojecting points from the nearest k cameras,
    performing depth sorting to render depth and color, and computing errors.

    Args:
        extrinsics: Camera extrinsics of shape (B, 3, 4).
        intrinsics: Camera intrinsics of shape (B, 3, 3).
        depth_maps: Depth maps of shape (B, H, W) or (B, H, W, 1).
        images: RGB images of shape (B, H, W, 3) or (B, 3, H, W).
        depth_conf: Optional depth confidence maps of shape (B, H, W).
        k: Number of nearest neighbor cameras to use for reprojection.
        photo_weight: Weight of the photometric L1 loss.
        depth_weight: Weight of the depth L1 loss.
        num_outliers: Number of outlier camera indices to return.

    Returns:
        list[int]: Indices of the top `num_outliers` cameras with the largest errors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = extrinsics.shape[0]
    if depth_maps.ndim == 4:
        depth_maps = depth_maps.squeeze(-1)
    H, W = depth_maps.shape[1], depth_maps.shape[2]

    # Normalize images to [0, 1] if they are uint8
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    # Permute to channel-last (B, H, W, 3) if they are (B, 3, H, W)
    if images.ndim == 4 and images.shape[1] == 3 and images.shape[3] != 3:
        images = images.transpose(0, 2, 3, 1)

    if depth_conf is None:
        depth_conf = np.ones_like(depth_maps)

    # 1. Compute camera centers to find nearest neighbors
    Rs = extrinsics[:, :3, :3]
    ts = extrinsics[:, :3, 3:4]
    translations = -(Rs.transpose(0, 2, 1) @ ts).squeeze(-1)  # (B, 3)

    # 2. Unproject all points to world coordinates beforehand
    world_points_all = unproject_depth_map_to_point_map(depth_maps, extrinsics, intrinsics)

    ext_t = torch.tensor(extrinsics, dtype=torch.float32, device=device)
    int_t = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    images_t = torch.tensor(images, dtype=torch.float32, device=device)
    depth_maps_t = torch.tensor(depth_maps, dtype=torch.float32, device=device)

    camera_errors = []

    for i in range(B):
        # Find nearest k cameras (excluding itself)
        distances = np.linalg.norm(translations - translations[i], axis=1)
        # argsort returns indices of sorted distances. [0] is itself (dist=0).
        nearest_idx = np.argsort(distances)[1 : k + 1]

        neighbor_points = []
        neighbor_colors = []

        # Gather points and colors from neighbors
        for j in nearest_idx:
            # Mask out invalid depth points and low confidence points
            valid_mask = (depth_maps[j] > 1e-8) & (~np.isnan(depth_maps[j])) & (depth_conf[j] > 0.0)
            neighbor_points.append(world_points_all[j][valid_mask])
            neighbor_colors.append(images[j][valid_mask])

        if len(neighbor_points) == 0:
            camera_errors.append(float("inf"))
            continue

        cat_points = np.concatenate(neighbor_points, axis=0)
        cat_colors = np.concatenate(neighbor_colors, axis=0)

        if cat_points.shape[0] == 0:
            camera_errors.append(float("inf"))
            continue

        wp_t = torch.tensor(cat_points, dtype=torch.float32, device=device)
        rgb_t = torch.tensor(cat_colors, dtype=torch.float32, device=device)

        # 3. Project aggregated points into target camera i
        image_points, cam_points = project_world_points_to_cam(wp_t, ext_t[i : i + 1], int_t[i : i + 1])
        image_points = image_points.squeeze(0)  # (N, 2)
        cam_points = cam_points.squeeze(0)  # (3, N)

        u = image_points[:, 0].round().long()
        v = image_points[:, 1].round().long()
        z = cam_points[2, :]

        # Filter out-of-bounds projections
        bounds_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 1e-4)
        u_valid = u[bounds_mask]
        v_valid = v[bounds_mask]
        z_valid = z[bounds_mask]
        rgb_valid = rgb_t[bounds_mask]

        if u_valid.shape[0] == 0:
            camera_errors.append(float("inf"))
            continue

        # 4. Depth Sorting (Painter's Algorithm)
        # Sort descending by z so that closer points are written last and overwrite further ones
        sort_idx = torch.argsort(z_valid, descending=True)
        u_sorted = u_valid[sort_idx]
        v_sorted = v_valid[sort_idx]
        z_sorted = z_valid[sort_idx]
        rgb_sorted = rgb_valid[sort_idx]

        # Initialize Render buffers
        render_depth = torch.zeros((H, W), dtype=torch.float32, device=device)
        render_rgb = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

        # Assign projected points to image plane
        render_depth[v_sorted, u_sorted] = z_sorted
        render_rgb[v_sorted, u_sorted] = rgb_sorted

        # 5. Evaluate against Ground Truth for camera i
        gt_depth = depth_maps_t[i]
        gt_rgb = images_t[i]

        # Compare only where both GT and Render are populated
        eval_mask = (render_depth > 1e-4) & (gt_depth > 1e-8) & (~torch.isnan(gt_depth))

        if eval_mask.sum() == 0:
            camera_errors.append(float("inf"))
            continue

        depth_err = torch.abs(render_depth[eval_mask] - gt_depth[eval_mask]).mean().item()
        photo_err = torch.abs(render_rgb[eval_mask] - gt_rgb[eval_mask]).mean().item()

        total_err = depth_weight * depth_err + photo_weight * photo_err
        camera_errors.append(total_err)

    # 6. Find the indices with the largest errors
    camera_errors_arr = np.array(camera_errors)

    # Handle infinites and NaNs by treating them as the highest possible error (definite outliers)
    camera_errors_arr[np.isnan(camera_errors_arr)] = float("inf")

    outlier_indices = np.argsort(camera_errors_arr)[-num_outliers:].tolist()

    return outlier_indices


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Differentiable conversion from axis-angle to rotation matrix using Rodrigues' formula.
    Args:
        axis_angle: Tensor of shape (B, 3)
    Returns:
        Rotation matrices of shape (B, 3, 3)
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    eps = 1e-6
    # Avoid division by zero
    k = axis_angle / (angles + eps)

    K = torch.zeros(axis_angle.shape[0], 3, 3, device=axis_angle.device)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    I = torch.eye(3, device=axis_angle.device).unsqueeze(0)
    R = I + torch.sin(angles).unsqueeze(-1) * K + (1 - torch.cos(angles)).unsqueeze(-1) * torch.bmm(K, K)
    return R


def optimize_poses_with_registration(
    base_extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth_maps: np.ndarray,
    depth_conf: np.ndarray,
    images: np.ndarray,
    valid_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        base_extrinsics (np.ndarray): Batch of camera extrinsic matricse of shape (S, 3, 4)
        intrinsics (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)
        depth_maps (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        depth_conf (np.ndarray): Batch of depth confidence scores of shape (S, H, W, 1) or (S, H, W)
        images (np.ndarray): Batch of RGB images of shape (S, H, W, 3) or (S, 3, H, W)
        valid_masks (np.ndarray): Batch of valid masks of shape (S, H, W) or (S, H, W, 1)
    Returns optimized poses and depth maps.
    """
    S = base_extrinsics.shape[0]

    # 1. Extract 3D points in the global coordinate frame
    # Assuming this helper returns points of shape (S, H, W, 3)
    world_points_all = unproject_depth_map_to_point_map(depth_maps, base_extrinsics, intrinsics)

    pcds = []
    global_extent = 1.0
    if len(valid_masks.shape) == 4:
        valid_masks = valid_masks.squeeze(-1)

    if images.ndim == 4 and images.shape[1] == 3 and images.shape[3] != 3:
        images = images.transpose(0, 2, 3, 1)

    # 2. Build Open3D PointClouds for each view
    for i in range(S):
        mask = valid_masks[i].astype(bool)

        pts = world_points_all[i][mask]
        colors = images[i][mask]
        confs = depth_conf[i].squeeze()[mask] if depth_conf is not None else None

        if len(pts) == 0:
            pcds.append(o3d.geometry.PointCloud())
            continue

        max_pts = 10000
        if len(pts) > max_pts:
            if confs is not None:
                # Use argpartition to efficiently get the indices of the top max_pts confidences in O(N) time
                idx = np.argpartition(confs, -max_pts)[-max_pts:]
            else:
                idx = np.random.choice(len(pts), max_pts, replace=False)

            pts = pts[idx]
            colors = colors[idx]

        if i == 0:
            global_extent = np.percentile(pts, 99, axis=0).max() - np.percentile(pts, 1, axis=0).min()
        voxel_size = max(global_extent / 100.0, 1e-4)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Normalize RGB to [0, 1] for Open3D
        c_norm = colors.astype(np.float64) / 255.0 if colors.max() > 1.0 else colors
        pcd.colors = o3d.utility.Vector3dVector(c_norm)

        # Downsample and estimate normals for Point-to-Plane ICP
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        pcds.append(pcd_down)

    # 3. Construct the Pose Graph
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    max_corr_dist = max(global_extent / 30.0, 1e-3)

    for i in range(S):
        if i > 0:
            # Initialize nodes at identity since our point clouds are already in world space
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

        for j in range(i + 1, S):
            if len(pcds[i].points) < 100 or len(pcds[j].points) < 100:
                continue

            try:
                # Colored ICP between pairs
                icp_result = o3d.pipelines.registration.registration_colored_icp(
                    pcds[i],
                    pcds[j],
                    max_corr_dist,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
                    ),
                )
            except RuntimeError:
                continue  # Skip this pair and move to the next if no overlap is found

            # If alignment is decent (e.g., >30% overlap), add an edge
            if icp_result.fitness > 0.3:
                info_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    pcds[i], pcds[j], max_corr_dist, icp_result.transformation
                )

                # Flag sequential frames as reliable (odometry), others as uncertain (loop closure)
                is_uncertain = j != i + 1

                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i, j, icp_result.transformation, info_matrix, uncertain=is_uncertain
                    )
                )

    # 4. Global Optimization
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_corr_dist, edge_prune_threshold=0.25, reference_node=0
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    # 5. Apply the optimized deltas back to the original extrinsics
    opt_extrinsics = np.zeros_like(base_extrinsics)
    for i in range(S):
        # delta_T maps the original world space to the globally optimized world space
        delta_T = pose_graph.nodes[i].pose
        delta_T_inv = np.linalg.inv(delta_T)

        # Assumption: base_extrinsics are World-to-Camera (W2C) matrices (standard CV).
        # E_new * X_new = E_old * X_old -> E_new * (delta_T * X_old) = E_old * X_old
        # Therefore, E_new = E_old * delta_T_inv
        # (If your pipeline uses C2W matrices, swap this to E_new = delta_T @ E_old)
        E_old = np.eye(4)
        E_old[:3, :4] = base_extrinsics[i]

        E_new = E_old @ delta_T_inv
        opt_extrinsics[i] = E_new[:3, :4]

    # Return optimized extrinsics alongside the original maps.
    return opt_extrinsics, depth_maps, valid_masks


def optimize_poses(
    base_extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth_maps: np.ndarray,
    depth_conf: np.ndarray,
    images: np.ndarray,
    photo_weight: float = 0.5,
    num_iterations: int = 5000,
    lr: float = 5e-5,
    samples_per_cam: int = 4096,
) -> np.ndarray:
    """
    Optimizes camera extrinsics using an asymmetric depth reprojection loss and optional photometric loss.

    Args:
        base_extrinsics: Initial camera extrinsics of shape (B, 3, 4).
        intrinsics: Camera intrinsics of shape (B, 3, 3).
        depth_maps: Depth maps of shape (B, H, W) or (B, H, W, 1).
        depth_conf: Depth confidence maps of shape (B, H, W).
        images: Optional RGB images of shape (B, H, W, 3) or (B, 3, H, W) for photometric supervision.
        photo_weight: Weight of the photometric L1 loss relative to depth loss.
        num_iterations: Number of optimization steps.
        lr: Learning rate for the poses.
        samples_per_cam: How many pixels to randomly sample per view to prevent OOM.

    Returns:
        np.ndarray: The optimized camera extrinsics (B, 3, 4).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = base_extrinsics.shape[0]
    if depth_maps.ndim == 4:
        depth_maps = depth_maps.squeeze(-1)
    H, W = depth_maps.shape[1], depth_maps.shape[2]

    # Move statics to GPU
    base_ext_t = torch.tensor(base_extrinsics, dtype=torch.float32, device=device)
    intrinsics_t = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    depth_maps_t = torch.tensor(depth_maps, dtype=torch.float32, device=device)
    depth_conf_t = torch.tensor(depth_conf, dtype=torch.float32, device=device)

    if images is not None:
        # Normalize images to [0, 1] if they are uint8
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        images_t = torch.tensor(images, dtype=torch.float32, device=device)

        # If images are channel-first (B, 3, H, W), permute to channel-last (B, H, W, 3)
        if images_t.shape[1] == 3 and images_t.shape[3] != 3:
            images_t = images_t.permute(0, 2, 3, 1)

    # Pre-generate pixel grid once for differentiable unprojection
    v_grid, u_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    u_grid = u_grid.float()
    v_grid = v_grid.float()

    # Learnable parameters: Delta Rotation (Axis-Angle) and Delta Translation
    # We apply these deltas to the initial poses.
    delta_r = torch.zeros((B, 3), dtype=torch.float32, device=device, requires_grad=True)
    delta_t = torch.zeros((B, 3), dtype=torch.float32, device=device, requires_grad=True)

    # Consider decoupling learning rates if rotation/translation converge at different speeds
    optimizer = torch.optim.Adam(
        [{"params": [delta_r], "lr": lr}, {"params": [delta_t], "lr": lr * 5.0}]  # Translation often needs a larger LR
    )

    def get_optimized_extrinsics():
        """Applies learnable deltas to the base extrinsics."""
        R_base = base_ext_t[:, :3, :3]
        t_base = base_ext_t[:, :3, 3]

        # Convert delta axis-angle to delta rotation matrix
        R_delta = axis_angle_to_matrix(delta_r)

        # Apply transformation: new_extrinsic = Delta * Base
        R_new = torch.bmm(R_delta, R_base)
        t_new = torch.bmm(R_delta, t_base.unsqueeze(-1)).squeeze(-1) + delta_t

        # Reconstruct 3x4
        return torch.cat([R_new, t_new.unsqueeze(-1)], dim=-1)

    def diff_unproject(depth, ext, intrinsic, u, v):
        """Fully differentiable unprojection of a batch of specific sampled pixels."""
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        cam_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N, 3)

        R = ext[:3, :3]
        t = ext[:3, 3]

        # World = R^T * (Cam - t). In PyTorch row vector math: (Cam - t) @ R
        world_points = torch.matmul(cam_points - t.unsqueeze(0), R)
        return world_points

    print("Starting Pose Optimization...")

    batch_size = 8  # Optimize over 8 random camera pairs simultaneously to stabilize gradients

    for step in range(num_iterations):
        optimizer.zero_grad()
        current_ext = get_optimized_extrinsics()

        total_loss = 0.0
        valid_pairs_counted = 0

        # 1. Accumulate loss over a mini-batch of camera pairs
        for _ in range(batch_size * 2):  # Allow some failures
            if valid_pairs_counted >= batch_size:
                break

            i = random.randint(0, B - 1)
            j = random.randint(0, B - 1)
            while j == i:
                j = random.randint(0, B - 1)

            # Randomly sample pixels from the source depth map (ignore 0 depth and low confidence)
            valid_mask_i = (depth_maps_t[i] > 1e-8) & (depth_conf_t[i] > 1.0)
            valid_indices = torch.nonzero(valid_mask_i)

            if valid_indices.shape[0] < samples_per_cam:
                continue

            rand_idx = torch.randperm(valid_indices.shape[0])[:samples_per_cam]
            sampled_v = valid_indices[rand_idx, 0]
            sampled_u = valid_indices[rand_idx, 1]
            sampled_depths_i = depth_maps_t[i, sampled_v, sampled_u]

            if images is not None:
                # Extract the RGB colors of the sampled source pixels
                sampled_rgb_i = images_t[i, sampled_v, sampled_u]

            # Unproject to 3D using the CURRENT optimized pose of camera i
            world_points = diff_unproject(
                sampled_depths_i,
                current_ext[i],
                intrinsics_t[i],
                u_grid[sampled_v, sampled_u],
                v_grid[sampled_v, sampled_u],
            )

            # Project into target camera j
            image_points, cam_points = project_world_points_to_cam(
                world_points, current_ext[j].unsqueeze(0), intrinsics_t[j].unsqueeze(0)
            )

            image_points = image_points.squeeze(0)
            cam_points = cam_points.squeeze(0)

            u_proj = image_points[:, 0].round().long()
            v_proj = image_points[:, 1].round().long()
            z_proj = cam_points[2, :]

            # Filter out-of-bounds projections
            bounds_mask = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H) & (z_proj > 0)

            if bounds_mask.sum() < 100:
                continue

            u_proj_valid = u_proj[bounds_mask]
            v_proj_valid = v_proj[bounds_mask]
            z_proj_valid = z_proj[bounds_mask]

            if images is not None:
                # Filter our source RGB values to only those that successfully projected in-bounds
                sampled_rgb_valid = sampled_rgb_i[bounds_mask]

            # Sample the ground truth depth map of camera j
            target_depths = depth_maps_t[j, v_proj_valid, u_proj_valid]
            target_confs = depth_conf_t[j, v_proj_valid, u_proj_valid]

            if images is not None:
                # Sample the target image at the projected coordinates
                target_rgbs = images_t[j, v_proj_valid, u_proj_valid]

            # Ignore target pixels with missing depth
            valid_target_mask = target_depths > 1e-8

            z_proj_valid = z_proj_valid[valid_target_mask]
            target_depths = target_depths[valid_target_mask]
            target_confs = target_confs[valid_target_mask]

            if z_proj_valid.shape[0] == 0:
                continue

            # ASYMMETRIC DEPTH LOSS calculation
            depth_error = z_proj_valid - target_depths
            asymmetric_weight = torch.where(depth_error > 0, 1.0, 0.05)
            depth_loss = (depth_error.abs() * asymmetric_weight * target_confs).mean()

            pair_loss = depth_loss

            # PHOTOMETRIC LOSS calculation
            if images is not None:
                # Filter colors using the valid target depth mask to stay perfectly aligned
                sampled_rgb_valid = sampled_rgb_valid[valid_target_mask]
                target_rgbs = target_rgbs[valid_target_mask]

                # L1 loss between projected source color and actual target color
                photo_loss = F.l1_loss(sampled_rgb_valid, target_rgbs, reduction="none")
                # Weight by depth confidence
                photo_loss = (photo_loss.mean(dim=-1) * target_confs).mean()

                pair_loss = pair_loss + (photo_weight * photo_loss)

            total_loss = total_loss + pair_loss
            valid_pairs_counted += 1

        if valid_pairs_counted == 0:
            continue  # Skip step if no valid pairs were found

        # Average the accumulated loss
        loss = total_loss / valid_pairs_counted

        # Penalize how far the cameras have drifted from their initialization
        anchor_loss_t = delta_t.norm(dim=-1).mean()
        anchor_loss_r = delta_r.norm(dim=-1).mean()

        lambda_t = 0.5  # Weight for translation drift
        lambda_r = 0.1  # Weight for rotation drift (usually needs less strict anchoring)

        anchor_loss = (lambda_t * anchor_loss_t) + (lambda_r * anchor_loss_r)

        # 2. Optional Trajectory Smoothness Regularization
        # Penalizes large sudden jumps in translation between consecutive frames
        translations = current_ext[:, :3, 3]
        velocity = translations[1:] - translations[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        smoothness_loss = acceleration.norm(dim=-1).mean()

        # Add a small weight (e.g., 0.1) to the smoothness prior
        final_loss = loss + anchor_loss + (0.0 * smoothness_loss)

        final_loss.backward()

        # Anchor the first camera to prevent global drift
        delta_r.grad[0] = 0.0
        delta_t.grad[0] = 0.0

        optimizer.step()

        if step % 50 == 0:
            print(
                f"Iteration {step}/{num_iterations} "
                f"| Loss: {loss.item():.4f} "
                f"| Smoothness: {smoothness_loss.item():.4f}"
            )

    print("Optimization Complete.")

    with torch.no_grad():
        final_extrinsics = get_optimized_extrinsics()
        return final_extrinsics.cpu().numpy()
