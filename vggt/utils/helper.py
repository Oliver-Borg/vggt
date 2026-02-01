# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


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
        sampled_indices = np.argsort(depth_conf.flatten())[-max_trues:]
    else:
        # randomly pick which True positions to keep
        sampled_indices = np.random.choice(true_indices, size=max_trues, replace=False)  # shape = (max_trues,)

    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)

def uniform_limit_trues(
    mask: np.ndarray,
    max_trues: int,
    points_3d: np.ndarray,
    depth_conf: np.ndarray,
    grid_size: int = 1000,
):
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    # 1D positions of all True entries
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

    # Ideally we will have a value of grid_size that gives us ~100000 occupied voxels
    while iters < 10 and lower_grid_size < upper_grid_size:
        disc_points = (norm_points * grid_size).round().astype(np.uint32)
        flat_inds = disc_points[:, 0] * grid_size ** 2 + disc_points[:, 1] * grid_size + disc_points[:, 2]
        unique_flat_inds: np.ndarray = np.unique(flat_inds)
        cur_size = unique_flat_inds.size
        if cur_size == max_trues:
            best_grid_size = grid_size
            closest_count = cur_size
            break
        elif cur_size < max_trues:
            lower_grid_size = grid_size
            grid_size = (lower_grid_size + upper_grid_size) // 2
            if closest_count is None or cur_size > closest_count:
                closest_count = cur_size
                best_grid_size = grid_size
        else:
            upper_grid_size = grid_size
            grid_size = (lower_grid_size + upper_grid_size) // 2
        iters += 1
        print(f"Iter {iters} best count {closest_count} for grid size {best_grid_size}")
    
    grid_size = best_grid_size
    disc_points = (norm_points * grid_size).round().astype(np.uint32)
    flat_inds = disc_points[:, 0] * grid_size ** 2 + disc_points[:, 1] * grid_size + disc_points[:, 2]
    unique_flat_inds: np.ndarray = np.unique(flat_inds)

    # We can then sample the highest confidence point within each

    flat_confs = depth_conf.flatten()
    # conf_sorter = flat_confs.argsort(order="desc")
    # sorted_confs = flat_confs[conf_sorter]
    # sorted_flat_inds = flat_inds[conf_sorter]
    # build new flat mask: True only at sampled positions
    limited_flat_mask = np.zeros(mask.size, dtype=bool)

    # for flat_ind in unique_flat_inds:
    #     indices = np.where(flat_inds == flat_ind)[0]
    #     confs = flat_confs[indices]
    #     highest_conf = np.argmax(confs)
    #     best_ind = indices[highest_conf]
    #     limited_flat_mask[mask.flatten()][best_ind] = True

    inds_sorter = np.argsort(flat_inds)

    flat_inds_sorted = flat_inds[inds_sorter]
    flat_confs_sorted = flat_confs[inds_sorter]
    orig_inds_sorted = np.arange(flat_inds.size)[inds_sorter]
    best_conf = np.zeros_like(flat_confs_sorted)
    best_conf_ind = np.zeros_like(orig_inds_sorted)

    best_conf[0] = flat_confs_sorted[0]
    best_conf_ind[0] = orig_inds_sorted[0]

    for i in range(1, best_conf.size):
        conf = flat_confs_sorted[i]
        prev_conf = flat_confs_sorted[i - 1]
        ind = orig_inds_sorted[i]
        prev_ind = orig_inds_sorted[i - 1]
        voxel_ind = flat_inds_sorted[i]
        prev_voxel_ind = flat_inds_sorted[i - 1]

        if voxel_ind != prev_voxel_ind:
            best_conf[i] = conf
            best_conf_ind[i] = ind
        elif conf > prev_conf:
            best_conf[i] = conf
            best_conf_ind[i] = ind
        else:
            best_conf[i] = prev_conf
            best_conf_ind[i] = prev_ind

    for i in range(best_conf.size - 2, -1, -1):
        ind = best_conf_ind[i]
        prev_ind = best_conf_ind[i + 1]
        voxel_ind = flat_inds_sorted[i]
        prev_voxel_ind = flat_inds_sorted[i + 1]
        if voxel_ind != prev_voxel_ind:
            # best_conf_ind[i] = ind
            continue
        else:
            best_conf_ind[i] = prev_ind

    set_mask = np.zeros_like(limited_flat_mask)
    set_mask[np.unique(best_conf_ind)] = True
    sorted_mask = np.zeros_like(limited_flat_mask)
    sorted_mask[inds_sorter] = set_mask

    limited_flat_mask[mask.flatten()] = sorted_mask
    
    # TODO Fix the slight mismatch here
    # assert np.count_nonzero(limited_flat_mask) == closest_count

    # restore original shape
    return limited_flat_mask.reshape(mask.shape)


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
