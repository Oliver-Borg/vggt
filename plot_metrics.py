import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Set


def load_results(choice: str, scene_name: str) -> Dict[int, List[Dict[str, Any]]]:
    """Loads SfM eval_results.json files."""
    data = {}
    pattern = f"{choice}_outputs/{scene_name}_n*_s*"
    for folder in glob.glob(pattern):
        try:
            parts = folder.split("_")
            num_images = int([p for p in parts if p.startswith("n")][-1].strip("n"))
            eval_file = os.path.join(folder, "eval_results.json")
            if os.path.exists(eval_file):
                with open(eval_file, "r") as f:
                    res = json.load(f)
                    if "metrics" in res and "error" not in res["metrics"]:
                        data.setdefault(num_images, []).append(res)
        except (ValueError, IndexError):
            continue
    return data


def load_gsplat_results(choice: str, scene_name: str) -> Dict[int, List[Dict[str, Any]]]:
    """Loads gsplat val_step29999.json files."""
    data = {}
    base_path = os.path.expanduser("~/work/git/gsplat/results")
    pattern = os.path.join(base_path, f"{choice}_outputs", f"{scene_name}_n*_s*", "stats", "val_step29999.json")

    for stat_file in glob.glob(pattern):
        try:
            folder_name = stat_file.split("/")[-3]
            num_images = int(folder_name.split("_n")[-1].split("_s")[0])
            with open(stat_file, "r") as f:
                data.setdefault(num_images, []).append(json.load(f))
        except (ValueError, IndexError):
            continue
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SfM and gsplat metrics on separate subplots.")
    parser.add_argument("--name", required=True, help="Scene name, e.g., bonsai_8")
    args = parser.parse_args()

    # Create 4 subplots: RRE, RTE, PSNR, LPIPS
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    ax_rre, ax_rte, ax_psnr, ax_lpips = axes

    styles = {
        "colmap": {"color": "#E63946", "marker": "o", "label": "COLMAP"},
        "vggt": {"color": "#457B9D", "marker": "s", "label": "VGGT"},
    }

    all_x: Set[int] = set()

    for choice in ["colmap", "vggt"]:
        sfm_data = load_results(choice, args.name)
        gs_data = load_gsplat_results(choice, args.name)

        # Collect all x-values for axis scaling
        union_nums = sorted(list(set(sfm_data.keys()) | set(gs_data.keys())))
        if not union_nums:
            continue
        all_x.update(union_nums)

        # Prepare plotting data
        plot_x_sfm, plot_x_gs = [], []
        m_rre, e_rre = [], [[], []]
        m_rte, e_rte = [], [[], []]
        m_psnr, e_psnr = [], [[], []]
        m_lpips, e_lpips = [], [[], []]

        for n in union_nums:
            if n in sfm_data:
                plot_x_sfm.append(n)
                rre_v = [r["metrics"]["mean_rre_deg"] for r in sfm_data[n]]
                rte_v = [r["metrics"]["mean_rte"] for r in sfm_data[n]]

                for vals, m_list, e_list in [(rre_v, m_rre, e_rre), (rte_v, m_rte, e_rte)]:
                    avg = np.mean(vals)
                    m_list.append(avg)
                    e_list[0].append(avg - np.min(vals))
                    e_list[1].append(np.max(vals) - avg)

            if n in gs_data:
                plot_x_gs.append(n)
                psnr_v = [g["psnr"] for g in gs_data[n]]
                lpips_v = [g["lpips"] for g in gs_data[n]]

                for vals, m_list, e_list in [(psnr_v, m_psnr, e_psnr), (lpips_v, m_lpips, e_lpips)]:
                    avg = np.mean(vals)
                    m_list.append(avg)
                    e_list[0].append(avg - np.min(vals))
                    e_list[1].append(np.max(vals) - avg)

        # Plot Pose Metrics
        if plot_x_sfm:
            ax_rre.errorbar(plot_x_sfm, m_rre, yerr=e_rre, **styles[choice], capsize=4)
            ax_rte.errorbar(plot_x_sfm, m_rte, yerr=e_rte, **styles[choice], capsize=4)

        # Plot Reconstruction Metrics
        if plot_x_gs:
            ax_psnr.errorbar(plot_x_gs, m_psnr, yerr=e_psnr, **styles[choice], capsize=4)
            ax_lpips.errorbar(plot_x_gs, m_lpips, yerr=e_lpips, **styles[choice], capsize=4)

    # Formatting and Cleanup
    all_x_ticks = sorted(list(all_x))
    titles = ["Rotation ($RRE$)", "Translation ($RTE$)", "Quality ($PSNR$)", "Perceptual ($LPIPS$)"]
    y_labels = ["Degrees ↓", "Norm. Units ↓", "dB ↑", "Score ↓"]

    for ax, title, ylabel in zip(axes, titles, y_labels):
        ax.set_xscale("log")
        ax.set_xticks(all_x_ticks)
        ax.set_xticklabels([str(x) for x in all_x_ticks])
        ax.grid(True, which="major", ls="-", alpha=0.15)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Number of Images", fontsize=11)

    # Legend on the first plot only to avoid duplication
    ax_rre.legend(fontsize=10)

    plt.tight_layout()
    out_file = f"full_evaluation_{args.name}.png"
    plt.savefig(out_file, dpi=300)
    print(f"Comprehensive plot saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    main()
