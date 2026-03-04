import argparse
from collections import OrderedDict
import glob
import json
import os
from pathlib import Path
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
    default: float | str | int | None = None


regexes = [
    Param(name="num_images", pattern=r"_n(\d+)", cast=int),
    Param(name="seed", pattern=r"_s(\d+)", cast=int),
    Param(name="conf_thres_value", pattern=r"_c(\d+\.\d+)", cast=float),
    Param(name="num_points", pattern=r"_p(\d+)", cast=int),
    Param(name="sampling_mode", pattern=r"_(voxels)|(confidence)|(random)|(ba)", cast=str),
    Param(name="image_mode", pattern=r"_(shuffle)|(distributed)", cast=str, default="shuffle"),
    Param(name="num_cameras", pattern=r"_i(\d+)", cast=int),
    Param(name="gt_eval_data", pattern=r"_(gteval)", cast=str, default="traineval"),
    Param(name="pose_opt", pattern=r"_(poseopt)", cast=str, default="defpose"),
]


def extract_params(folder_name: str) -> dict[str, str | float | int | None]:
    """Extracts parameters from a folder name using regex."""
    params: dict[str, str | float | int | None] = {}
    for p in regexes:
        if match := re.search(p.pattern, folder_name):
            params[p.name] = p.cast([g for g in match.groups() if g is not None][0])
        else:
            params[p.name] = p.default
    return params


def load_metrics_to_df(scene_name: str, methods: list[str], folders: list[str] | None = None) -> pd.DataFrame:
    """
    Loads both SfM and gsplat metrics for all methods into a single DataFrame.
    """
    records = []

    # (source_name, glob_pattern, json_loader_func)
    sources = [
        ("sfm", "~/work/git/vggt/{method}_outputs/{scene}_n*_s*/eval_results.json", _parse_sfm_json),
        (
            "gsplat",
            os.path.expanduser("~/work/git/gsplat/results/{method}_outputs/{scene}_n*_s*/stats/val_step6999.json"),
            _parse_gsplat_json,
        ),
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
                    if folders is not None and len(set(folders) & set(Path(file_path).parts)) == 0:
                        continue

                    # For gsplat, the folder is 3 levels up from the json file in the original code logic
                    if source_name == "gsplat":
                        folder_name = file_path.split("/")[-3]

                    params = extract_params(folder_name)
                    if params["num_images"] is None:
                        continue

                    # Load and parse metrics
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    metrics = parser(data, file_path)

                    # Construct record
                    record = {"method": method, "source": source_name}
                    record.update(params)
                    record.update(metrics)
                    record["file_path"] = file_path
                    records.append(record)

                except (ValueError, IndexError, KeyError, json.JSONDecodeError):
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Merge rows that share the same unique identifiers (method, num_images, seed, etc.)
    # Since SfM and GS metrics come from different files but belong to the same run,
    # we group by identifiers and combine the columns.
    group_cols = ["method"] + [reg.name for reg in regexes]
    # Filter only columns that exist to avoid key errors
    valid_group_cols = [c for c in group_cols if c in df.columns]

    df_merged = df.groupby(valid_group_cols, dropna=False).first().reset_index()

    return df_merged


def _parse_sfm_json(data: Dict, filename: str) -> Dict[str, float]:
    """Extracts RRE and RTE from SfM json."""
    if "metrics" in data and "mean_rre_deg" in data["metrics"]:
        return {"rre": data["metrics"]["mean_rre_deg"], "rte": data["metrics"]["mean_rte"]}
    return {}


def _parse_gsplat_json(data: Dict[str, float], filename: str) -> Dict[str, float | int | None]:
    """Extracts PSNR and LPIPS from gsplat json."""
    parsed_data = {
        "psnr": data.get("psnr"),
        "lpips": data.get("lpips"),
        "val_step": int(filename.split("val_step")[-1].split(".json")[0]),
    }
    if "num_points" in data:
        parsed_data["num_points"] = round(int(data["num_points"]), -3)
    return parsed_data


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
    # df[x] = df[x].fillna(0)

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
    if x in ["num_points"]:
        ax.set_xscale("log")
    else:
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
    plot_graph(args.name, args.x_axis, args.split_param, args.filter)


def plot_graph(name: str, x_axis: str, split_param: str | None = None, filter: str | None = None, folders: list[str] | None = None):
    df = load_metrics_to_df(name, methods=["colmap", "vggt"], folders=folders)

    if df.empty:
        print("No data found.")
        return

    # Apply Filters
    if filter:
        filters = parse_filters(filter)
        for key, vals in filters.items():
            if key in df.columns:
                print(f"Filtering {key} in {vals}")
                # If filtering by conf_thres_value, keep colmap rows (which don't have this param)
                # TODO Maybe just keep Nones?
                if key == "conf_thres_value":
                    df = df[df[key].isin(vals) | (df["method"] == "colmap")]
                else:
                    df = df[df[key].isin(vals)]
            else:
                print(f"Warning: Filter key '{key}' not found in dataframe columns.")

        if df.empty:
            print("Dataframe is empty after filtering.")
            return

    if split_param and split_param in df.columns:
        split_vals = df[split_param].fillna("").astype(str)
        df["plot_series"] = df["method"] + "-" + split_vals
        unique_splits = sorted(df[split_param].unique())
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

        if split_param:
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

    if x_axis == "conf_thres_value" and not colmap_df.empty:
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

        if x_axis == "conf_thres_value" and not colmap_df.empty:
            plot_df = df[df["method"] != "colmap"]

            for series_name, metric_values in list(
                sorted(list(colmap_means_by_series.items()), key=lambda x: x[1].get(split_param))
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
            x=x_axis,
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

    for i in range(4):
        handles, labels = axes[i].get_legend_handles_labels()
        if handles:
            axes[i].legend(handles=handles, labels=labels, title="Series", fontsize=10)

    suffix = f"plots/full_evaluation-{name}-{x_axis}-{split_param}"

    csv_out_file = f"{suffix}.csv"
    os.makedirs(os.path.dirname(csv_out_file), exist_ok=True)
    df.to_csv(csv_out_file, index=False)
    print("Dataframe saved:", Path(csv_out_file))

    plt.tight_layout()
    out_file = f"{suffix}.png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print("Comprehensive plot saved:", Path(out_file))
    plt.show()


if __name__ == "__main__":
    main()
