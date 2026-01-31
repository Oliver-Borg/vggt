import argparse
from dataclasses import dataclass
import json
import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Set, Optional


@dataclass
class Param:
    name: str
    pattern: str
    prefix: str
    cast: type[float | str | int]


def extract_params(folder_name: str) -> dict[str, str | float | int | None]:
    regexes = [
        Param(name="num_images", pattern=r"_n(\d+)", prefix="_n", cast=int),
        Param(name="seed", pattern=r"_s(\d+)", prefix="_s", cast=int),
        Param(name="conf_thres_value", pattern=r"_c(\d+\.\d+)", prefix="_c", cast=float),
    ]

    params: dict[str, str | float | int | None] = {}

    for param in regexes:
        name = param.name
        reg = param.pattern
        if (match := re.search(reg, folder_name)) is not None:
            params[name] = param.cast(match.group(1))
        else:
            params[name] = None
    return params


def load_results(choice: str, scene_name: str) -> Dict[int, List[Dict[str, Any]]]:
    """Loads SfM eval_results.json files."""
    data: Dict[int, List[Dict[str, Any]]] = {}
    pattern = f"{choice}_outputs/{scene_name}_n*_s*"
    for folder in glob.glob(pattern):
        try:
            params = extract_params(folder)
            num_images: int = params["num_images"]  # type: ignore
            assert num_images is not None
            eval_file = os.path.join(folder, "eval_results.json")
            if os.path.exists(eval_file):
                with open(eval_file, "r") as f:
                    res: dict[str, float | int | str | dict[str, Any] | None] = json.load(f)
                    res.update(params)
                    if "metrics" in res and "error" not in res["metrics"]:  # type: ignore
                        data.setdefault(num_images, []).append(res)
        except (ValueError, IndexError):
            continue
    return data


def load_gsplat_results(choice: str, scene_name: str) -> Dict[int, List[Dict[str, Any]]]:
    """Loads gsplat val_step6999.json files."""
    data: Dict[int, List[Dict[str, Any]]] = {}
    base_path = os.path.expanduser("~/work/git/gsplat/results")
    pattern = os.path.join(base_path, f"{choice}_outputs", f"{scene_name}_n*_s*", "stats", "val_step6999.json")

    for stat_file in glob.glob(pattern):
        try:
            folder_name = stat_file.split("/")[-3]
            params = extract_params(folder_name)
            num_images: int = params["num_images"]  # type: ignore
            assert num_images is not None
            with open(stat_file, "r") as f:
                stats: dict[str, float | int | str | None] = json.load(f)
                stats["folder_name"] = folder_name
                stats.update(params)
                data.setdefault(num_images, []).append(stats)
        except (ValueError, IndexError):
            continue
    return data


def partition_data(
    data: Dict[int, List[Dict[str, Any]]], split_param: Optional[str]
) -> Dict[Any, Dict[int, List[Dict[str, Any]]]]:
    """Helper to split data dictionary based on a specific parameter value."""
    if not split_param:
        return {None: data}

    partitions: Dict[Any, Dict[int, List[Dict[str, Any]]]] = {}
    for n, items in data.items():
        for item in items:
            val = item.get(split_param)
            if val not in partitions:
                partitions[val] = {}
            if n not in partitions[val]:
                partitions[val][n] = []
            partitions[val][n].append(item)
    return partitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SfM and gsplat metrics on separate subplots.")
    parser.add_argument("--name", required=True, help="Scene name, e.g., bonsai_8")
    parser.add_argument(
        "--split_param", required=False, default=None, help="Parameter to split series by, e.g., conf_thres_value"
    )
    args = parser.parse_args()

    # Create 4 subplots: RRE, RTE, PSNR, LPIPS
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    ax_rre, ax_rte, ax_psnr, ax_lpips = axes

    # Added markersize=4 to make points smaller
    base_styles = {
        "colmap": {"color": "#E63946", "marker": "o", "label": "COLMAP", "markersize": 4},
        "vggt": {"color": "#457B9D", "marker": "s", "label": "VGGT", "markersize": 4},
    }

    # Color map for split series
    cmap = plt.get_cmap("viridis")

    all_x: Set[int] = set()

    for choice in ["colmap", "vggt"]:
        sfm_data = load_results(choice, args.name)
        gs_data = load_gsplat_results(choice, args.name)

        # Partition data if split_param is provided
        sfm_partitions = partition_data(sfm_data, args.split_param)
        gs_partitions = partition_data(gs_data, args.split_param)

        # Get all unique split keys from both sources
        split_keys = sorted(
            list(set(sfm_partitions.keys()) | set(gs_partitions.keys())), key=lambda x: (x is not None, x)
        )

        for idx, split_key in enumerate(split_keys):
            # Select the subset of data for this split key
            curr_sfm = sfm_partitions.get(split_key, {})
            curr_gs = gs_partitions.get(split_key, {})

            # Collect all x-values for axis scaling
            union_nums = sorted(list(set(curr_sfm.keys()) | set(curr_gs.keys())))
            if not union_nums:
                continue
            all_x.update(union_nums)

            # Determine Style
            style = base_styles.get(choice, {}).copy()

            # If we are splitting and the key is valid (not None), modify the style
            if split_key is not None:
                style["label"] = f"{style['label']}-{split_key}"
                if len(split_keys) > 1 or split_key is not None:
                    style["color"] = cmap(idx / max(len(split_keys), 1))

            # Prepare plotting data
            plot_x_sfm, plot_x_gs = [], []
            m_rre, e_rre = [], [[], []]
            m_rte, e_rte = [], [[], []]
            m_psnr, e_psnr = [], [[], []]
            m_lpips, e_lpips = [], [[], []]

            for n in union_nums:
                if n in curr_sfm:
                    plot_x_sfm.append(n)
                    rre_v = [r["metrics"]["mean_rre_deg"] for r in curr_sfm[n]]
                    rte_v = [r["metrics"]["mean_rte"] for r in curr_sfm[n]]

                    for vals, m_list, e_list in [(rre_v, m_rre, e_rre), (rte_v, m_rte, e_rte)]:
                        avg = np.mean(vals)
                        m_list.append(avg)
                        e_list[0].append(avg - np.min(vals))
                        e_list[1].append(np.max(vals) - avg)

                if n in curr_gs:
                    plot_x_gs.append(n)
                    psnr_v = [g["psnr"] for g in curr_gs[n]]
                    lpips_v = [g["lpips"] for g in curr_gs[n]]

                    for vals, m_list, e_list in [(psnr_v, m_psnr, e_psnr), (lpips_v, m_lpips, e_lpips)]:
                        avg = np.mean(vals)
                        m_list.append(avg)
                        e_list[0].append(avg - np.min(vals))
                        e_list[1].append(np.max(vals) - avg)

            # Jitter Strength (multiplicative, +/- 3%)
            JITTER = 0.03

            # Plot Pose Metrics
            if plot_x_sfm:
                # Convert to numpy and apply multiplicative jitter for log scale
                x_vals = np.array(plot_x_sfm, dtype=float)
                noise = np.random.uniform(-JITTER, JITTER, size=x_vals.shape)
                x_vals = x_vals * (1 + noise)

                ax_rre.errorbar(x_vals, m_rre, yerr=e_rre, **style, capsize=4)
                ax_rte.errorbar(x_vals, m_rte, yerr=e_rte, **style, capsize=4)

            # Plot Reconstruction Metrics
            if plot_x_gs:
                x_vals = np.array(plot_x_gs, dtype=float)
                noise = np.random.uniform(-JITTER, JITTER, size=x_vals.shape)
                x_vals = x_vals * (1 + noise)

                ax_psnr.errorbar(x_vals, m_psnr, yerr=e_psnr, **style, capsize=4)
                ax_lpips.errorbar(x_vals, m_lpips, yerr=e_lpips, **style, capsize=4)

    # Formatting and Cleanup
    if not all_x:
        print("No data found to plot.")
        return

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
