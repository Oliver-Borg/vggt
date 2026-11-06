"""VGGT-Omega inference for the reconstruction pipeline.

Runs in the VGGT-Omega conda env (which has no pycolmap etc.) and writes a raw-prediction cache in the same layout
as the VGGT cache of `demo_colmap.run_vggt` (plus the Omega camera tokens), so that the rest of the pipeline
(point cloud sampling, COLMAP export) runs unchanged in the VGGT env.

Images keep their aspect ratio (Omega's `balanced` preprocessing at `image_resolution`), so there is no square padding:
masks are all ones and `original_coords` = (0, 0, W, H, W_orig, H_orig).
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

OMEGA_ROOT = os.environ.get("VGGT_OMEGA_ROOT", os.path.expanduser("~/work/git/vggt-omega"))
if OMEGA_ROOT not in sys.path:
    sys.path.append(OMEGA_ROOT)

from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

# checkpoint name -> (file, image_resolution); mirrors reconstruct_args.OMEGA_CHECKPOINTS
OMEGA_CHECKPOINTS = {
    "512": ("~/work/checkpoints/vggt-omega/vggt_omega_1b_512.pt", 512),
    "416": ("~/work/checkpoints/vggt-omega/vggt_omega_1b_416_reproduce.pt", 416),
}


def run(scene_dir: str, cache_file: str, checkpoint: str, seed: int = 42):
    torch.manual_seed(seed)
    path, resolution = OMEGA_CHECKPOINTS[checkpoint]
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"VGGT-Omega checkpoint {path} not found (the gated checkpoint has to be downloaded)")

    image_path_list = sorted(glob.glob(os.path.join(scene_dir, "images", "*")))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {scene_dir}/images")
    names = [os.path.basename(p) for p in image_path_list]

    sizes = []
    for p in image_path_list:
        with Image.open(p) as im:
            w, h = im.size
        assert 0.5 <= h / w <= 2.0, f"{p}: aspect ratio {h / w:.2f} would be cropped by the Omega preprocessing"
        sizes.append((w, h))

    device = "cuda"
    model = VGGTOmega().to(device).eval()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    images = load_and_preprocess_images(image_path_list, mode="balanced", image_resolution=resolution)  # [S,3,H,W]
    S, _, H, W = images.shape
    print(f"VGGT-Omega {checkpoint}: {S} images at {W}x{H} (original {sizes[0][0]}x{sizes[0][1]})")

    with torch.inference_mode():
        pred = model(images.to(device))
    pose_enc = pred["pose_enc"].float()  # [1,S,9]
    extrinsic, intrinsic = encoding_to_camera(pose_enc, (H, W))
    depth = pred["depth"].float()[0].cpu().numpy()  # [S,H,W,1]
    conf = pred["depth_conf"].float()[0].cpu().numpy()  # [S,H,W]

    original_coords = torch.tensor([[0, 0, W, H, w, h] for w, h in sizes], dtype=torch.float32)
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    torch.save(
        {
            "images": images.cpu(),
            "original_coords": original_coords,
            "extrinsic": extrinsic[0].cpu().numpy(),
            "intrinsic": intrinsic[0].cpu().numpy(),
            "depth_map": depth,
            "depth_conf": conf,
            "orig_depth_conf": conf.copy(),
            "masks": np.ones((S, H, W, 1), dtype=np.uint8),
            "base_image_path_list": names,
            # Omega-specific: what the camera head reads (final-layer camera + register tokens), the raw pose
            # encoding and the network resolution, for fine-tuning the camera head
            "model": "omega",
            "omega_checkpoint": checkpoint,
            "omega_resolution": resolution,
            "camera_and_register_tokens": pred["camera_and_register_tokens"][0].float().cpu(),  # [S,17,2048]
            "pose_enc": pose_enc[0].cpu(),
            "net_hw": (H, W),
        },
        cache_file,
    )
    print(f"Saved {cache_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--cache_file", required=True)
    ap.add_argument("--checkpoint", default="512", choices=list(OMEGA_CHECKPOINTS))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.scene_dir, a.cache_file, a.checkpoint, a.seed)
