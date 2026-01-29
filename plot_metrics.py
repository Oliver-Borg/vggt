import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Any


def load_all_results(scene_name: str) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """
    Finds and loads all eval_results.json files for a specific scene.
    Groups results by num_images regardless of seed.
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

            eval_file = os.path.join(folder, "eval_results.json")
            if os.path.exists(eval_file):
                with open(eval_file, "r") as f:
                    res = json.load(f)
                    if "error" not in res["metrics"]:
                        if num_images not in data[choice]:
                            data[choice][num_images] = []
                        data[choice][num_images].append(res)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregated reconstruction metrics vs view count.")
    parser.add_argument("--name", required=True, help="Scene name prefix, e.g., bonsai_8")
    args = parser.parse_args()

    results = load_all_results(args.name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    styles = {
        "colmap": {"color": "#E63946", "marker": "o", "label": "COLMAP (Classic)"},
        "vggt": {"color": "#457B9D", "marker": "s", "label": "VGGT (Transformer)"},
    }

    for choice in ["colmap", "vggt"]:
        sorted_nums = sorted(results[choice].keys())
        if not sorted_nums:
            continue

        means_rre, errs_rre = [], [[], []]
        means_rte, errs_rte = [], [[], []]

        for n in sorted_nums:
            rre_vals = [res["metrics"]["mean_rre_deg"] for res in results[choice][n]]
            rte_vals = [res["metrics"]["mean_rte"] for res in results[choice][n]]

            m_rre = np.mean(rre_vals)
            means_rre.append(m_rre)
            errs_rre[0].append(m_rre - np.min(rre_vals))
            errs_rre[1].append(np.max(rre_vals) - m_rre)

            m_rte = np.mean(rte_vals)
            means_rte.append(m_rte)
            errs_rte[0].append(m_rte - np.min(rte_vals))
            errs_rte[1].append(np.max(rte_vals) - m_rte)

        ax1.errorbar(
            sorted_nums, means_rre, yerr=errs_rre, **styles[choice], linewidth=2, capsize=4, elinewidth=1.5, alpha=0.8
        )
        ax2.errorbar(
            sorted_nums, means_rte, yerr=errs_rte, **styles[choice], linewidth=2, capsize=4, elinewidth=1.5, alpha=0.8
        )

    for ax, title, ylabel in zip(
        [ax1, ax2],
        ["Rotation Error ($RRE$)", "Translation Error ($RTE$)"],
        ["Mean Error (degrees)", "Mean Error (normalized units)"],
    ):
        ax.set_title(f"{title} - {args.name}", fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Number of Images", fontsize=12)
        ax.set_xscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

    all_x_values = sorted(list(set(list(results["colmap"].keys()) + list(results["vggt"].keys()))))
    for ax in [ax1, ax2]:
        ax.set_xscale("log")
        ax.set_xticks(all_x_values)
        ax.set_xticklabels([str(x) for x in all_x_values])
        ax.minorticks_off()

    plt.tight_layout()
    out_name = f"plot_{args.name}_aggregated.png"
    plt.savefig(out_name, dpi=300)
    print(f"Aggregated plot saved to {out_name}")
    plt.show()


if __name__ == "__main__":
    main()
