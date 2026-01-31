import argparse
from collections import OrderedDict
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class Param:
    name: str
    pattern: str
    cast: type[float | str | int]


def extract_params(folder_name: str) -> dict[str, str | float | int | None]:
    """Extracts parameters from a folder name using regex."""
    regexes = [
        Param(name="num_images", pattern=r"_n(\d+)", cast=int),
        Param(name="seed", pattern=r"_s(\d+)", cast=int),
        Param(name="conf_thres_value", pattern=r"_c(\d+\.\d+)", cast=float),
    ]
    params: dict[str, str | float | int | None] = {}
    for p in regexes:
        if match := re.search(p.pattern, folder_name):
            params[p.name] = p.cast(match.group(1))
        else:
            params[p.name] = None
    return params


def load_metrics_to_df(scene_name: str, methods: List[str]) -> pd.DataFrame:
    """
    Loads both SfM and gsplat metrics for all methods into a single DataFrame.
    """
    records = []

    # (source_name, glob_pattern, json_loader_func)
    sources = [
        ("sfm", "{method}_outputs/{scene}_n*_s*/eval_results.json", _parse_sfm_json),
        (
            "gsplat",
            os.path.expanduser("~/work/git/gsplat/results/{method}_outputs/{scene}_n*_s*/stats/val_step29999.json"),
            _parse_gsplat_json,
        ),
    ]

    for method in methods:
        for source_name, path_template, parser in sources:
            pattern = path_template.format(method=method, scene=scene_name)

            for file_path in glob.glob(pattern):
                try:
                    # Extract folder parameters
                    # We assume folder structure is consistent, getting the folder name relative to the file
                    folder_path = os.path.dirname(file_path)
                    folder_name = os.path.basename(folder_path)
                    # For gsplat, the folder is 3 levels up from the json file in the original code logic
                    if source_name == "gsplat":
                        folder_name = file_path.split("/")[-3]

                    params = extract_params(folder_name)
                    if params["num_images"] is None:
                        continue

                    # Load and parse metrics
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    metrics = parser(data)

                    # Construct record
                    record = {"method": method, "source": source_name, **params, **metrics}
                    records.append(record)

                except (ValueError, IndexError, KeyError, json.JSONDecodeError):
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Merge rows that share the same unique identifiers (method, num_images, seed, etc.)
    # Since SfM and GS metrics come from different files but belong to the same run,
    # we group by identifiers and combine the columns.
    group_cols = ["method", "num_images", "seed", "conf_thres_value"]
    # Filter only columns that exist to avoid key errors
    valid_group_cols = [c for c in group_cols if c in df.columns]

    df_merged = df.groupby(valid_group_cols, dropna=False).first().reset_index()

    return df_merged


def _parse_sfm_json(data: Dict) -> Dict[str, float]:
    """Extracts RRE and RTE from SfM json."""
    if "metrics" in data and "mean_rre_deg" in data["metrics"]:
        return {"rre": data["metrics"]["mean_rre_deg"], "rte": data["metrics"]["mean_rte"]}
    return {}


def _parse_gsplat_json(data: Dict[str, float]) -> Dict[str, float | None]:
    """Extracts PSNR and LPIPS from gsplat json."""
    return {"psnr": data.get("psnr"), "lpips": data.get("lpips")}


def parse_filters(filter_str: str) -> Dict[str, List[Any]]:
    """
    Parses a filter string into a dictionary of lists.
    Format: "key1=val1,val2;key2=val3"
    Tries to infer int/float types.
    """
    filters = {}
    if not filter_str:
        return filters

    for part in filter_str.split(";"):
        if "=" not in part:
            continue
        key, vals = part.split("=", 1)
        parsed_vals = []
        for v in vals.split(","):
            v = v.strip()
            # Attempt type inference for easier filtering against numeric DF columns
            try:
                if "." in v:
                    parsed_vals.append(float(v))
                else:
                    parsed_vals.append(int(v))
            except ValueError:
                parsed_vals.append(v)
        filters[key.strip()] = parsed_vals
    return filters


def plot_metric(
    df: pd.DataFrame,
    x: str,
    y: str,
    series_col: str,
    ax: plt.Axes,
    title: str,
    ylabel: str,
    colors: Dict[str, str],
    markers: Dict[str, str],
    dashes: Dict[str, Any],
    hlines: Dict[str, float] | None = None,
    hranges: Dict[str, Tuple[float, float]] | None = None,
) -> None:
    """
    Generic plotting function using Seaborn.
    Plots line with markers and min/max error bars (PI 100).
    Optionally adds horizontal reference lines (hlines) and shaded regions (hranges).
    """
    if df.empty or y not in df.columns or df[y].isnull().all():
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        return
    df[x] = df[x].fillna(5.0)

    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=series_col,
        style=series_col,
        markers=markers,
        dashes=dashes,
        palette=colors,
        errorbar=("pi", 100),
        err_style="bars",
        ax=ax,
        err_kws={"capsize": 4},
    )

    if hranges:
        for series_name, (min_val, max_val) in hranges.items():
            if pd.notna(min_val) and pd.notna(max_val):
                region_color = colors.get(series_name, "gray")
                ax.axhspan(
                    ymin=min_val,
                    ymax=max_val,
                    color=region_color,
                    alpha=0.1,
                    edgecolor=None,
                    linewidth=0,
                )

    if hlines:
        for series_name, y_val in hlines.items():
            if pd.notna(y_val):
                line_color = colors.get(series_name, "gray")

                ax.axhline(
                    y=y_val,
                    color=line_color,
                    linestyle="-",
                    alpha=0.8,
                    label=series_name,
                    linewidth=1.2,
                )

    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    xlabel = " ".join(x.split("_")).title()
    ax.set_xlabel(xlabel, fontsize=11)
    # ax.set_xscale("log")

    unique_x = sorted(df[x].fillna(5.0).unique().round())
    ax.set_xticks(unique_x)
    ax.set_xticklabels([str(n) for n in unique_x])

    ax.grid(True, which="major", ls="-", alpha=0.15)

    # Handle Legend: Remove individual subplot legends, will add global one later or keep on first
    ax.get_legend().remove()


def main():
    parser = argparse.ArgumentParser(description="Plot SfM and gsplat metrics.")
    parser.add_argument("--name", required=True, help="Scene name, e.g., bonsai_8")
    parser.add_argument("--x_axis", default="num_images", help="X-Axis key")
    parser.add_argument("--split_param", default=None, help="Optional param to split series, e.g., conf_thres_value")
    parser.add_argument("--filter", default=None, help="Filter string, e.g. 'num_images=10,20;conf_thres_value=0.5'")
    args = parser.parse_args()

    df = load_metrics_to_df(args.name, methods=["colmap", "vggt"])

    if df.empty:
        print("No data found.")
        return

    # Apply Filters
    if args.filter:
        filters = parse_filters(args.filter)
        for key, vals in filters.items():
            if key in df.columns:
                print(f"Filtering {key} in {vals}")
                # If filtering by conf_thres_value, keep colmap rows (which don't have this param)
                if key == "conf_thres_value":
                    df = df[df[key].isin(vals) | (df["method"] == "colmap")]
                else:
                    df = df[df[key].isin(vals)]
            else:
                print(f"Warning: Filter key '{key}' not found in dataframe columns.")

        if df.empty:
            print("Dataframe is empty after filtering.")
            return

    if args.split_param and args.split_param in df.columns:
        split_vals = df[args.split_param].fillna("").astype(str)
        df["plot_series"] = df["method"] + "-" + split_vals
        unique_splits = sorted(df[args.split_param].unique())
    else:
        df["plot_series"] = df["method"]
        unique_splits = ["colmap", "vggt"]

    unique_series = sorted(df["plot_series"].unique())

    style_config = {
        "colmap": {"marker": "o", "dashes": ""},
        "vggt": {"marker": "X", "dashes": (2, 2)},
    }

    pal = sns.color_palette("tab10", n_colors=len(unique_splits))
    val_to_color = dict(zip(unique_splits, pal))

    color_map = {}
    marker_map = {}
    dash_map = {}

    for series in unique_series:
        method = "colmap" if "colmap" in series else "vggt"

        if args.split_param:
            val_str = series.replace(f"{method}-", "")
            original_val = next((v for v in unique_splits if str(v) == val_str), None)
            color_map[series] = val_to_color.get(original_val, "#333333")
        else:
            color_map[series] = val_to_color.get(method, "#333333")

        marker_map[series] = style_config[method]["marker"]
        dash_map[series] = style_config[method]["dashes"]

    colmap_means_by_series = {}
    colmap_min_by_series = {}
    colmap_max_by_series = {}

    colmap_df = df[df["method"] == "colmap"]

    if args.x_axis != "num_images" and not colmap_df.empty:
        grouped = colmap_df.groupby("plot_series")

        colmap_means_by_series = grouped.mean(numeric_only=True).to_dict(orient="index")
        colmap_min_by_series = grouped.min(numeric_only=True).to_dict(orient="index")
        colmap_max_by_series = grouped.max(numeric_only=True).to_dict(orient="index")

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    metrics_config = [
        {"y": "rre", "title": "Rotation ($RRE$)", "ylabel": "Degrees ↓"},
        {"y": "rte", "title": "Translation ($RTE$)", "ylabel": "Norm. Units ↓"},
        {"y": "psnr", "title": "Quality ($PSNR$)", "ylabel": "dB ↑"},
        {"y": "lpips", "title": "Perceptual ($LPIPS$)", "ylabel": "Score ↓"},
    ]

    for ax, config in zip(axes, metrics_config):
        plot_df = df
        hlines_dict = OrderedDict()
        hranges_dict = OrderedDict()

        if args.x_axis != "num_images" and not colmap_df.empty:
            plot_df = df[df["method"] != "colmap"]

            for series_name, metric_values in list(
                sorted(list(colmap_means_by_series.items()), key=lambda x: x[1].get(args.split_param))
            ):
                if config["y"] in metric_values:
                    hlines_dict[series_name] = metric_values[config["y"]]

            for series_name in colmap_min_by_series:
                min_val = colmap_min_by_series[series_name].get(config["y"])
                max_val = colmap_max_by_series[series_name].get(config["y"])
                if min_val is not None and max_val is not None:
                    hranges_dict[series_name] = (min_val, max_val)

        plot_metric(
            df=plot_df,
            x=args.x_axis,
            y=config["y"],
            series_col="plot_series",
            ax=ax,
            title=config["title"],
            ylabel=config["ylabel"],
            colors=color_map,
            markers=marker_map,
            dashes=dash_map,
            hlines=hlines_dict,
            hranges=hranges_dict,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles=handles, labels=labels, title="Series", fontsize=10)

    plt.tight_layout()
    out_file = f"full_evaluation {args.name} {args.x_axis} {args.split_param}.png"
    plt.savefig(out_file, dpi=300)
    print(f"Comprehensive plot saved: {out_file}")
    plt.show()


if __name__ == "__main__":
    main()
