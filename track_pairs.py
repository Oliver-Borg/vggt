"""Pairwise correspondences from the original VGGT track head, for the 3R-GS epipolar loss.

For every frame i (as the query frame) a fixed grid of ~`n_points` query points is tracked into all other frames j > i
(all pairs), or with `--knn K` into its K nearest frames by Omega camera centre (sparse graph, each pair once; the VGGT
aggregator then only sees the K+1 frames of the window).
Per pair (i, j) the file holds the query points in frame i, the tracked points in frame j (both in pixels of the
original training images, the camera frame of the trainer), the track-head visibility and confidence of both frames, the
pair weight `conf_i * conf_j / max(conf_i * conf_j)` (max over the pair's points) and the mask `weight > mask_threshold`.

Frame order and image sizes come from the VGGT-Omega raw prediction cache of the experiment (`raw_preds.pt`).
Runs in the `vggt` conda env (original VGGT, checkpoint from the torch hub cache like `demo_colmap.load_model`).

Usage: python track_pairs.py --cache_file .../raw_preds.pt --image_dir .../images_2 --out .../track_pairs.pt
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402

_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
RES = 518


def grid_points(x1: float, y1: float, x2: float, y2: float, n_points: int) -> np.ndarray:
    """Cell-centred regular grid with ~n_points points inside the box (x1,y1)-(x2,y2): [N,2] as (x, y)."""
    w, h = x2 - x1, y2 - y1
    ny = max(1, int(round(np.sqrt(n_points * h / w))))
    nx = max(1, int(round(n_points / ny)))
    xs = x1 + (np.arange(nx) + 0.5) * w / nx
    ys = y1 + (np.arange(ny) + 0.5) * h / ny
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def knn_graph(extrinsic: np.ndarray, k: int) -> list:
    """k nearest frames (Euclidean distance between camera centres) of every frame, from w2c extrinsics [S,3,4]."""
    R, t = extrinsic[:, :, :3], extrinsic[:, :, 3]
    centres = -np.einsum("sji,sj->si", R, t)  # -R^T t
    d = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return [list(np.argsort(d[i])[:k]) for i in range(len(d))]


def run(cache_file: str, image_dir: str, out: str, n_points: int, mask_threshold: float, chunk: int = 50, knn: int = 0):
    cache = torch.load(cache_file, map_location="cpu", weights_only=False)
    names = list(cache["base_image_path_list"])
    sizes = np.asarray(cache["original_coords"])[:, -2:]  # (w, h) of the training images
    paths = [os.path.join(image_dir, n) for n in names]
    for p, (w, h) in zip(paths, sizes):
        with Image.open(p) as im:
            assert im.size == (int(w), int(h)), f"{p}: size {im.size} differs from the Omega cache {(w, h)}"
    S = len(names)

    device = "cuda"
    model = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=True)
    state = torch.hub.load_state_dict_from_url(_URL)
    missing, unexpected = model.load_state_dict(
        {k: v for k, v in state.items() if k.startswith(("aggregator.", "track_head."))}, strict=False
    )
    assert not missing, missing
    model = model.to(device).eval()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model.aggregator.to(dtype)  # halves the weight memory; track head runs under autocast like the official demo

    images, _, coords = load_and_preprocess_images_square(paths, RES)  # [S,3,RES,RES], [S,6] in RES space
    images = images.to(device)
    coords = coords.numpy().astype(np.float64)
    box = coords[:, :4]  # x1,y1,x2,y2 of the original image in the RES frame
    wh = coords[:, 4:6]
    assert np.allclose(wh, sizes), (wh[:2], sizes[:2])
    to_px = lambda p, j: (p - box[j, :2]) * (wh[j] / (box[j, 2:] - box[j, :2]))  # RES frame -> image pixels

    grids = [grid_points(*box[i], n_points) for i in range(S)]
    if knn:  # sparse graph: frame i is tracked into its k nearest frames (by Omega camera centre); each pair once
        nbrs = knn_graph(np.asarray(cache["extrinsic"]), knn)
        windows = [[i] + nbrs[i] for i in range(S)]
        edges = [(i, j) for i in range(S) for j in nbrs[i] if not (i in nbrs[j] and j < i)]
    else:  # all pairs: frame i is tracked into all other frames, pairs (i, j) with j > i
        windows = [[i] + [j for j in range(S) if j != i] for i in range(S)]
        edges = list(combinations(range(S), 2))
    track_all, vis_all, conf_all = [], [], []
    for i in range(S):
        order = windows[i]  # the query frame has to be the first frame
        q = torch.from_numpy(grids[i]).float()[None].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            tokens, patch_start = model.aggregator(images[order][None])
        keep = set(model.track_head.feature_extractor.intermediate_layer_idx)  # the other layers are not read by the track head
        tokens = [t if k in keep else None for k, t in enumerate(tokens)]
        torch.cuda.empty_cache()
        outs = []
        for qc in torch.split(q, chunk, dim=1):  # the tracker's memory grows with frames x points
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                tr, vi_, co = model.track_head(tokens, images=images[order][None], patch_start_idx=patch_start, query_points=qc)
            outs.append((tr[-1][0].float().cpu().numpy(), vi_[0].float().cpu().numpy(), co[0].float().cpu().numpy()))
        del tokens
        pos = {f: k for k, f in enumerate(order)}  # frame -> position in this query's window
        track_all.append((pos, np.concatenate([o[0] for o in outs], axis=1)))  # [W,N,2], RES frame
        vis_all.append(np.concatenate([o[1] for o in outs], axis=1))
        conf_all.append(np.concatenate([o[2] for o in outs], axis=1))
        print(f"query frame {i + 1}/{S}", flush=True)

    ei, ej, pi, pj, ci, cj, vi, vj, wgt, msk = [], [], [], [], [], [], [], [], [], []
    for i, j in edges:
        pos, tr = track_all[i]
        ei.append(i)
        ej.append(j)
        pi.append(to_px(grids[i], i))
        pj.append(to_px(tr[pos[j]], j))
        ci.append(conf_all[i][0])
        cj.append(conf_all[i][pos[j]])
        vi.append(vis_all[i][0])
        vj.append(vis_all[i][pos[j]])
        c = conf_all[i][0] * conf_all[i][pos[j]]
        w = c / max(c.max(), 1e-12)  # normalised by the max of the pair
        wgt.append(w)
        msk.append(w > mask_threshold)
    f = lambda a: torch.from_numpy(np.stack(a)).float()
    torch.save(
        {
            "names": names,
            "img_wh": torch.from_numpy(np.asarray(sizes)).float(),
            "ei": torch.tensor(ei),
            "ej": torch.tensor(ej),
            "pts_i": f(pi),
            "pts_j": f(pj),
            "conf_i": f(ci),
            "conf_j": f(cj),
            "vis_i": f(vi),
            "vis_j": f(vj),
            "weight": f(wgt),
            "mask": torch.from_numpy(np.stack(msk)),
            "n_points": n_points,
            "knn": knn,
            "mask_threshold": mask_threshold,
            "res": RES,
        },
        out,
    )
    print(f"saved {out}: {len(ei)} pairs x {len(pi[0])} points, mask fraction {np.mean(msk):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_points", type=int, default=417)
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--knn", type=int, default=0, help="k nearest frames by Omega camera centre; 0 = all pairs")
    a = ap.parse_args()
    run(a.cache_file, a.image_dir, a.out, a.n_points, a.mask_threshold, knn=a.knn)
