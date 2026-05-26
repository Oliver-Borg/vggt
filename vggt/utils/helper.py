# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np

import torch
from torch_cluster import fps
import tqdm


def fps_limit_trues(mask: np.ndarray, max_trues: int, points: np.ndarray) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    use GPU-accelerated Farthest Point Sampling (via torch_cluster) to keep exactly max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    # Flatten the spatial dimensions of points to match flatnonzero, keeping the coordinate dim
    valid_points = points.reshape(-1, points.shape[-1])[true_indices]

    valid_points_tensor = torch.from_numpy(valid_points).float()

    if torch.cuda.is_available():
        valid_points_tensor = valid_points_tensor.cuda()

    # fps takes a ratio (0 to 1) instead of an absolute integer K
    sampling_ratio = max_trues / valid_points.shape[0]

    # Returns the indices of the sampled points directly
    sampled_idx_tensor = fps(valid_points_tensor, ratio=sampling_ratio, random_start=True)

    # Squeeze out the exact number requested (in case of floating point rounding)
    fps_idx = sampled_idx_tensor[:max_trues].cpu().numpy()

    # Map the valid_points indices back to the global true_indices
    sampled_indices = true_indices[fps_idx]

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)


def randomly_limit_trues(mask: np.ndarray, max_trues: int, depth_conf: np.ndarray | None = None) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    if depth_conf is not None:
        true_ind_depths = depth_conf.flatten()[true_indices]
        true_ind_depths -= float(true_ind_depths.min())
        true_ind_depths /= float(true_ind_depths.max() + 1e-8)
        # true_ind_depths = true_ind_depths ** 2  # Sharpen the distribution to favor higher confidence points even more

        sampled_indices = np.random.choice(
            true_indices,
            size=max_trues,
            p=true_ind_depths.flatten() / true_ind_depths.sum(),
            replace=False,
        )
    else:
        # randomly pick which True positions to keep
        sampled_indices = np.random.choice(true_indices, size=max_trues, replace=False)  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)


def get_bbox_2d(arr: np.ndarray):
    mask = arr > 0
    if not np.any(mask):
        return (0, 0, 0, 0)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def compute_spatial_frequencies(images: np.ndarray, cutoff_radius: int = 5) -> np.ndarray:
    """
    Applies an FFT high-pass filter to extract high-frequency spatial details.
    """
    # 1. Transform spatial dimensions to frequency domain
    dft = np.fft.fft2(images.astype(float), axes=(1, 2))
    dft_shift = np.fft.fftshift(dft, axes=(1, 2))

    # 2. Create High-Pass Filter mask
    B, H, W, C = images.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    hpf_mask = np.ones((1, H, W, 1))
    mask_area = (y - cy) ** 2 + (x - cx) ** 2 <= cutoff_radius**2
    hpf_mask[0, mask_area, 0] = 0

    # 3. Apply filter and inverse transform
    dft_shift_filtered = dft_shift * hpf_mask
    dft_filtered = np.fft.ifftshift(dft_shift_filtered, axes=(1, 2))
    img_high_freq = np.fft.ifft2(dft_filtered, axes=(1, 2))

    # 4. Return magnitude averaged across channels
    return np.mean(np.abs(img_high_freq), axis=3)


def mask_image_borders(freq_map: np.ndarray, image_masks: np.ndarray) -> np.ndarray:
    """
    Zeroes out the frequency map at the bounding box borders.
    """
    rmin, rmax, cmin, cmax = get_bbox_2d(image_masks[0])

    freq_map[:, rmin, :] = 0
    freq_map[:, rmax, :] = 0
    freq_map[:, :, cmin] = 0
    freq_map[:, :, cmax] = 0

    assert get_bbox_2d(freq_map[0]) != (rmin, rmax, cmin, cmax)
    return freq_map


def calculate_visibility_counts(
    true_indices: np.ndarray, shape_BHW: tuple, depths: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """
    Projects 3D points into all cameras to count how many views share a depth consensus.
    """
    B, H, W = shape_BHW

    if extrinsics.shape[1:] == (3, 4):
        ext_4x4 = np.zeros((B, 4, 4), dtype=float)
        ext_4x4[:, :3, :] = extrinsics
        ext_4x4[:, 3, 3] = 1.0
    else:
        ext_4x4 = extrinsics

    # Unravel back to (Batch, Y, X)
    b_idx, y_idx, x_idx = np.unravel_index(true_indices, (B, H, W))

    # Lift points to 3D Camera Space (raveled to prevent N x N broadcasting)
    z_vals = depths[b_idx, y_idx, x_idx].ravel()
    K_source = intrinsics[b_idx]
    fx = K_source[:, 0, 0].ravel()
    fy = K_source[:, 1, 1].ravel()
    cx_k = K_source[:, 0, 2].ravel()
    cy_k = K_source[:, 1, 2].ravel()

    x_c = (x_idx - cx_k) * z_vals / fx
    y_c = (y_idx - cy_k) * z_vals / fy
    pts_cam = np.stack([x_c, y_c, z_vals, np.ones_like(z_vals)], axis=1)

    # Transform to 3D World Space using the square 4x4 matrix
    c2w = np.linalg.inv(ext_4x4)
    c2w_source = c2w[b_idx]
    pts_world = np.einsum("nij,nj->ni", c2w_source, pts_cam)

    # Base count is 1 (seen by its source camera)
    visibility_counts = np.ones(len(true_indices), dtype=float)

    print("Calculating visibility counts...")
    for i in tqdm.tqdm(range(B)):
        # Project world points into camera i using the 4x4 matrix
        pts_c_i = pts_world @ ext_4x4[i].T
        z_i = pts_c_i[:, 2].ravel()

        valid_z = z_i > 0.05

        u_i = (pts_c_i[:, 0].ravel() * intrinsics[i, 0, 0] / (z_i + 1e-8)) + intrinsics[i, 0, 2]
        v_i = (pts_c_i[:, 1].ravel() * intrinsics[i, 1, 1] / (z_i + 1e-8)) + intrinsics[i, 1, 2]

        valid_u = (u_i >= 0) & (u_i < W - 1)
        valid_v = (v_i >= 0) & (v_i < H - 1)

        # Exclude points that originated from camera i
        valid_proj = valid_z & valid_u & valid_v & (b_idx != i)

        if np.any(valid_proj):
            u_inds = np.round(u_i[valid_proj]).astype(int)
            v_inds = np.round(v_i[valid_proj]).astype(int)

            # ravel to prevent loop-based memory explosion
            sampled_depths = depths[i, v_inds, u_inds].ravel()

            depth_diff = np.abs(z_i[valid_proj] - sampled_depths)
            is_visible = depth_diff < (sampled_depths * 0.05)

            visibility_counts[valid_proj] += is_visible

    return visibility_counts


def sample_by_weights(true_indices: np.ndarray, weights: np.ndarray, max_trues: int, mask_shape: tuple) -> np.ndarray:
    """
    Normalizes weights into probabilities, samples indices, and reconstructs the mask.
    """
    # Normalize weights to use as probabilities
    weights -= float(weights.min())
    weights /= float(weights.max() + 1e-8)
    weights += 1e-5

    sampled_indices = np.random.choice(
        true_indices,
        size=max_trues,
        p=weights / weights.sum(),
        replace=False,
    )

    limited_flat_mask = np.zeros(np.prod(mask_shape), dtype=bool)
    limited_flat_mask[sampled_indices] = True
    return limited_flat_mask.reshape(mask_shape)


def image_freq_limit_trues(
    mask: np.ndarray,
    max_trues: int,
    images: np.ndarray,
    image_masks: np.ndarray,
    depths: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    use_vis_counts: bool = True,
) -> np.ndarray:
    """
    If mask has more than max_trues True values, randomly keep only max_trues of them.
    Probability is weighted heavily by image spatial frequencies, and penalized by frustum overlap.
    """
    true_indices = np.flatnonzero(mask)

    if true_indices.size <= max_trues:
        return mask

    # Format correction for images
    if images.ndim == 4 and images.shape[1] == 3 and images.shape[3] != 3:
        images = images.transpose(0, 2, 3, 1)

    # Format correction for depths to ensure it is strictly (B, H, W)
    if depths.ndim == 4:
        if depths.shape[1] == 1:
            depths = depths[:, 0, :, :]
        elif depths.shape[3] == 1:
            depths = depths[:, :, :, 0]

    # 1. Compute base high-frequency weights
    # TODO Try do this without the filter
    norm_grad_2d = compute_spatial_frequencies(images)
    norm_grad_2d = mask_image_borders(norm_grad_2d, image_masks)

    true_ind_freqs = norm_grad_2d.flatten()[true_indices]

    # 2. Penalize weights based on camera visibility overlap
    shape_BHW = images.shape[:3]
    if use_vis_counts:
        visibility_counts = calculate_visibility_counts(true_indices, shape_BHW, depths, extrinsics, intrinsics)
        true_ind_freqs /= visibility_counts

    # 3. Sample and build the final mask
    return sample_by_weights(true_indices, true_ind_freqs, max_trues, mask.shape)


def uniform_limit_trues(
    mask: np.ndarray,
    max_trues: int,
    points_3d: np.ndarray,
    depth_conf: np.ndarray,
    grid_size: int = 1000,
    min_grid_occupancy: int = 10,
):
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries

    mask = mask & (~np.isnan(points_3d[..., 0]))  # TODO Change to .any(axis=-1)

    true_indices = np.flatnonzero(mask)  # shape = (N_true,)

    # if already within budget, return as-is
    if true_indices.size <= max_trues:
        return mask

    points_3d = points_3d[mask]

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_3d)
    # ret = pcd.voxel_down_sample(0.05)

    flat_points = points_3d.reshape((points_3d.size // 3, 3))
    mins = flat_points.min(axis=0)
    maxes = flat_points.max(axis=0)

    norm_points = (flat_points - mins) / (maxes - mins)

    lower_grid_size = int(max_trues ** (1 / 3))
    upper_grid_size = 2 * grid_size - lower_grid_size

    iters = 0
    best_grid_size = grid_size
    closest_count: int | None = None
    best_min_grid_occupancy = min_grid_occupancy

    # Ideally we will have a value of grid_size that gives us ~100000 occupied voxels
    while iters < 10 and lower_grid_size < upper_grid_size:
        # Cast to uint64 to prevent overflow when grid_size is large
        disc_points = (norm_points * grid_size).round().astype(np.uint64)
        flat_voxel_inds = (
            disc_points[:, 0] * (np.uint64(grid_size) ** 2)
            + disc_points[:, 1] * np.uint64(grid_size)
            + disc_points[:, 2]
        )
        unique_flat_voxel_inds, counts = np.unique(flat_voxel_inds, return_counts=True)
        min_grid_occupancy = int(math.ceil(counts.mean()))
        cur_size = np.count_nonzero(
            counts >= min_grid_occupancy
        )  # Only keep voxels that have >= min_grid_occupancy points
        # cur_size = unique_flat_voxel_inds.size

        if cur_size == max_trues:
            best_grid_size = grid_size
            closest_count = cur_size
            best_min_grid_occupancy = min_grid_occupancy
            break
        elif cur_size < max_trues:
            if closest_count is None or cur_size > closest_count:
                closest_count = cur_size
                best_grid_size = grid_size
                best_min_grid_occupancy = min_grid_occupancy

            lower_grid_size = grid_size
            grid_size = (lower_grid_size + upper_grid_size) // 2
        else:
            upper_grid_size = grid_size
            grid_size = (lower_grid_size + upper_grid_size) // 2
        iters += 1
        print(f"Iter {iters} best count {closest_count} for grid size {best_grid_size}")

    grid_size = best_grid_size
    # Cast to uint64 here as well
    disc_points = (norm_points * grid_size).round().astype(np.uint64)
    flat_voxel_inds = (
        disc_points[:, 0] * (np.uint64(grid_size) ** 2) + disc_points[:, 1] * np.uint64(grid_size) + disc_points[:, 2]
    )
    unique_flat_voxel_inds, counts = np.unique(flat_voxel_inds, return_counts=True)

    # We can then sample the highest confidence point within each

    flat_confs = depth_conf[mask].flatten()
    limited_flat_mask = np.zeros(true_indices.size, dtype=bool)
    inds_sorter = np.lexsort((flat_confs, flat_voxel_inds))
    flat_inds_sorted = flat_voxel_inds[inds_sorter]

    # Correctly identify the last elements of each voxel group
    is_last_of_group = np.append(flat_inds_sorted[:-1] != flat_inds_sorted[1:], True)

    # Filter by the occupancy threshold found during search
    valid_voxels = unique_flat_voxel_inds[counts >= best_min_grid_occupancy]
    if valid_voxels.size == 0:
        valid_voxels = unique_flat_voxel_inds

    last_indices_all = np.where(is_last_of_group)[0]
    mask_valid = np.isin(flat_inds_sorted[last_indices_all], valid_voxels)
    last_indices = last_indices_all[mask_valid]

    if len(last_indices) > max_trues:
        last_indices = last_indices[:max_trues]

    set_mask = np.zeros_like(limited_flat_mask)
    set_mask[last_indices] = True
    sorted_mask = np.zeros_like(limited_flat_mask)
    sorted_mask[inds_sorter] = set_mask

    full_sized_mask = np.zeros(mask.size, dtype=bool)
    full_sized_mask[mask.flatten()] = sorted_mask

    # Assert checks sorted_mask, which contains the final truth flags mapping back correctly
    assert np.count_nonzero(sorted_mask) == closest_count, f"{np.count_nonzero(sorted_mask)} != {closest_count}"

    # restore original shape
    return full_sized_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
            - y_coords (numpy.ndarray): Array of y coordinates for all frames
            - x_coords (numpy.ndarray): Array of x coordinates for all frames
            - f_coords (numpy.ndarray): Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf
