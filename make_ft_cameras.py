"""Builds a copy of a VGGT-Omega reconstruction whose cameras (poses and intrinsics) are replaced by the cameras of a
fine-tuned camera head (`camera_head_gt_smoke.py` output `*_cameras.npz`); the points stay the Omega point cloud.

The npz holds the cameras in the trainer's normalised frame. The similarity trainer frame -> reconstruction frame is
fitted (Umeyama) on the initial predicted cameras, which are the reconstruction's own cameras.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap

from combine_clouds import align_to_world_space, load_cameras, save_cameras_json


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / X.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--step", type=int, default=1000)
    ap.add_argument("--output_dir", required=True)
    a = ap.parse_args()
    recon_dir, out = Path(a.recon_dir), Path(a.output_dir)

    z = np.load(a.npz, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    fidx = z["frame_idx"]
    c2w_tr = {n: z[f"c2w_step{a.step}"][i] for n, i in zip(names, fidx)}
    K_tr = {n: z[f"K_step{a.step}"][i] for n, i in zip(names, fidx)}
    init_tr = {n: z["init_c2w"][i] for n, i in zip(names, fidx)}

    rec = pycolmap.Reconstruction(recon_dir / "sparse")
    by_name = {im.name: im for im in rec.images.values()}
    assert set(by_name) == set(names), "npz frames differ from the reconstruction images"

    def raw_c2w(im):
        w2c = im.cam_from_world.matrix()  # 3x4
        R, t = w2c[:, :3], w2c[:, 3]
        return R.T, -R.T @ t

    src = np.stack([init_tr[n][:3, 3] for n in names])
    dst = np.stack([raw_c2w(by_name[n])[1] for n in names])
    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm(s * src @ R.T + t - dst, axis=1).max()
    rot_resid = max(np.abs(R @ init_tr[n][:3, :3] - raw_c2w(by_name[n])[0]).max() for n in names)
    print(f"trainer -> reconstruction frame: scale {s:.6f}, max centre residual {resid:.2e}, max rotation residual {rot_resid:.2e}")
    assert resid < 1e-3 * np.linalg.norm(dst - dst.mean(0), axis=1).max() and rot_resid < 1e-3

    for n in names:
        im = by_name[n]
        Rc = R @ c2w_tr[n][:3, :3]
        c = s * R @ c2w_tr[n][:3, 3] + t
        Rw = Rc.T
        im.cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(Rw), -Rw @ c)
        K = K_tr[n]
        cam = rec.cameras[im.camera_id]
        cam.params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)

    if out.exists():
        shutil.rmtree(out)
    (out / "sparse").mkdir(parents=True)
    rec.write(out / "sparse")
    shutil.copytree(recon_dir / "images", out / "images")
    for f in ("gt_cameras.json",):
        if (recon_dir / f).exists():
            shutil.copy2(recon_dir / f, out / f)
    stat = json.load(open(recon_dir / "stat.json"))
    stat.update({"type": "omega_ft_cameras", "camera_npz": str(a.npz), "camera_step": a.step,
                 "date": time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime())})
    json.dump(stat, open(out / "stat.json", "w"))

    pred = load_cameras(out / "sparse")
    save_cameras_json(pred, out / "cameras.json")
    if (out / "gt_cameras.json").exists():
        save_cameras_json(align_to_world_space(pred, load_cameras(out / "gt_cameras.json")), out / "aligned_cameras.json")
    print("wrote", out)


if __name__ == "__main__":
    main()
