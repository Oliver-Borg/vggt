import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
from typing import Any


def load_all_results(scene_name: str) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    """
    Finds and loads all eval_results.json files for a specific scene.
    Returns: { 'colmap': { 20: {...}, 50: {...} }, 'vggt': { ... } }
    """
    data = {"colmap": {}, "vggt": {}}

    for choice in ["colmap", "vggt"]:
        pattern = f"{choice}_outputs/{scene_name}_n*_s*"
        folders = glob.glob(pattern)

        for folder in folders:
            try:
                num_images = int(folder.split("_")[-2].strip("n"))
            except ValueError:
                continue
            try:
                seed = int(folder.split("_")[-1].strip("s"))
            except ValueError:
                continue

            eval_file = os.path.join(folder, "eval_results.json")
            if os.path.exists(eval_file):
                with open(eval_file, "r") as f:
                    res = json.load(f)
                    if "error" not in res["metrics"]:
                        data[choice][(num_images, seed)] = res
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reconstruction metrics vs view count.")
    parser.add_argument("--name", required=True, help="Scene name prefix, e.g., bonsai_8")
    args = parser.parse_args()

    results = load_all_results(args.name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    styles = {
        "colmap": {"color": "#E63946", "marker": "o", "label": "COLMAP (Classic)"},
        "vggt": {"color": "#457B9D", "marker": "s", "label": "VGGT (Transformer)"},
    }

    for choice in ["colmap", "vggt"]:
        sorted_keys = sorted(results[choice].keys())
        sorted_nums = [k[0] for k in sorted_keys]
        seeds = [k[1] for k in sorted_keys]
        if not sorted_keys:
            continue

        rre = [results[choice][(n, s)]["metrics"]["mean_rre_deg"] for n, s in sorted_keys]
        rte = [results[choice][(n, s)]["metrics"]["mean_rte"] for n, s in sorted_keys]

        ax1.plot(sorted_nums, rre, **styles[choice], linewidth=2, markersize=8)
        ax2.plot(sorted_nums, rte, **styles[choice], linewidth=2, markersize=8)

    ax1.set_title(f"Rotation Error ($RRE$) - {args.name}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Mean Error (degrees)", fontsize=12)
    ax1.set_xlabel("Number of Images", fontsize=12)
    ax1.set_xscale("log")
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    ax2.set_title(f"Translation Error ($RTE$) - {args.name}", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Mean Error (normalized units)", fontsize=12)
    ax2.set_xlabel("Number of Images", fontsize=12)
    ax2.set_xscale("log")
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()

    plt.tight_layout()

    out_name = f"plot_{args.name}_comparison.png"
    plt.savefig(out_name, dpi=300)
    print(f"Plot saved to {out_name}")
    plt.show()


if __name__ == "__main__":
    main()
