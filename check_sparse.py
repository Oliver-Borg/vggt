import argparse
import pycolmap


def check_sparse_folder(sparse_folder: str) -> bool:
    # Load the reconstruction from the sparse folder
    reconstruction = pycolmap.Reconstruction(sparse_folder)
    print(reconstruction.summary())

    # Check if there are points in the point cloud
    num_points = len(reconstruction.points3D)
    if num_points > 0:
        print(f"Point cloud contains {num_points} points.")
    else:
        print("Point cloud is empty.")
        return False

    # Check if there are cameras in the reconstruction
    num_cameras = len(reconstruction.cameras)
    if num_cameras > 0:
        print(f"Reconstruction contains {num_cameras} cameras.")
    else:
        print("No cameras found in the reconstruction.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Check sparse folder for point cloud and cameras.")
    parser.add_argument(
        "sparse_folder",
        type=str,
        help="Path to the sparse folder output by colmap or vggt.",
    )
    args = parser.parse_args()
    check_sparse_folder(args.sparse_folder)


if __name__ == "__main__":
    main()
