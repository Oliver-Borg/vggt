import argparse
from collections import OrderedDict
import glob
import json
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class Param:
    name: str
    pattern: str
    cast: type[float | str | int] | Callable[[str], float | str | int]
    default: float | str | int | None = None


regexes = [
    Param(name="num_images", pattern=r"_n(\d+)", cast=int),
    Param(name="seed", pattern=r"_s(\d+)", cast=int),
    Param(name="conf_thres_value", pattern=r"_c(\d+\.\d+)", cast=float),
    Param(name="num_points", pattern=r"_p(\d+)", cast=int),
    Param(name="sampling_mode", pattern=r"_(voxels)|(confidence)|(random)|(ba)", cast=str),
    Param(name="image_mode", pattern=r"_(shuffle)|(distributed)", cast=str, default="shuffle"),
    Param(name="num_cameras", pattern=r"_i(\d+)", cast=int),
    Param(
        name="gt_eval", pattern=r"_(gteval)", cast=lambda x: "GT Eval" if x == "gteval" else "", default="Train Eval"
    ),
    Param(
        name="pose_opt",
        pattern=r"_(poseopt)",
        cast=lambda x: "Pose Opt" if x == "poseopt" else "",
        default="No Pose Opt",
    ),
    Param(
        name="eval_opt",
        pattern=r"_(evalopt)",
        cast=lambda x: "Eval Pose Opt" if x == "evalopt" else "",
        default="Def Eval Poses",
    ),
    Param(
        name="depth_loss",
        pattern=r"_(depth)",
        cast=lambda x: "Depth Loss" if x == "depth" else "",
        default="No Depth Loss",
    ),
    Param(
        name="depth_conf",
        pattern=r"_(conf)",
        cast=lambda x: "Depth Confidence" if x == "conf" else "",
        default="No Depth Confidence",
    ),
    Param(name="choice", pattern=r"(vggt|colmap)_outputs", cast=str),
    Param(name="camera_type", pattern=r"_m(radial)|(pinhole)", cast=str),
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


def apply_presentation_style():
    """High-visibility style with a massive, clear legend."""
    sns.set_theme(style="whitegrid")
    
    sns.set_context("talk", rc={
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 22,
        "legend.title_fontsize": 30,
        "lines.linewidth": 2,
        "lines.markersize": 8,
    })
    
    plt.rcParams.update({
        "font.weight": "normal",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.5",
        "legend.fancybox": True,
        "savefig.dpi": 300
    })


# apply_presentation_style()


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

                    params = extract_params(file_path)
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
    original_x_col: str | None = None,  # No jitter
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
        err_kws={"capsize": 6},
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
                    linewidth=2.0,
                )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    label_col = original_x_col if original_x_col else x
    xlabel = " ".join(label_col.split("_")).title()
    ax.set_xlabel(xlabel)

    if label_col in ["num_points"]:
        ax.set_xscale("log")
    else:
        unique_x = sorted(df[label_col].fillna(5.0).unique().round())
        ax.set_xticks(unique_x)
        ax.set_xticklabels([str(n) for n in unique_x])

    ax.grid(True, which="major", ls="-", alpha=0.15)

    # Handle Legend: Remove individual subplot legends, will add global one later or keep on first
    ax.get_legend().remove()


def plot_pcp(df: pd.DataFrame, out_file: str, color_map: Dict[str, str]):
    """
    Plots a Parallel Coordinate Plot (PCP) for metrics and parameters.
    Uses smooth curves and adds jitter to y-positions to visualize density.
    """
    params = [r.name for r in regexes if r.name in df.columns]
    metrics = ["psnr", "lpips"]

    # Filter metrics present in DF
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return

    num_metrics = len(metrics)
    fig, axes = plt.subplots(
        nrows=num_metrics, ncols=1, figsize=(max(10, len(params) * 1.5), 6 * num_metrics), sharex=False
    )

    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        cols = params + [metric]
        plot_df = df.copy()
        range_map = {}

        for col in cols:
            series = plot_df[col]
            s_numeric = pd.to_numeric(series, errors="coerce")
            is_numeric = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_numeric_dtype(s_numeric)

            non_numeric_params = [p.name for p in regexes if hasattr(p, "cast") and p.cast not in [float, int]]

            if col in non_numeric_params:
                is_numeric = False

            if is_numeric:
                series = s_numeric
                if series.isnull().all():
                    plot_df[col] = 0.5
                    range_map[col] = (0, 0, "num")
                    continue

                mn, mx = series.min(), series.max()
                series = series.fillna(mn)

                if mn == mx:
                    plot_df[col] = 0.5
                else:
                    plot_df[col] = (series - mn) / (mx - mn)
                range_map[col] = (mn, mx, "num")
            else:
                series = series.fillna("N/A").astype(str)
                uniques = sorted(series.unique())
                code_map = {val: i for i, val in enumerate(uniques)}
                codes = series.map(code_map)

                mn, mx = 0, len(uniques) - 1
                if mn == mx:
                    plot_df[col] = 0.5
                else:
                    plot_df[col] = codes / mx
                range_map[col] = (uniques, "cat")

        # Curve smoothness configuration
        num_segments = 20
        t = np.linspace(0, 1, num_segments)
        smooth_t = 3 * t**2 - 2 * t**3

        # Jitter configuration
        jitter_strength = 0.05  # +/- 1% vertical jitter

        # Colormap
        cmap = plt.get_cmap("viridis")

        for idx, row in plot_df.iterrows():
            ys = row[cols].values.astype(float)

            # 1. Determine Color
            metric_norm_val = ys[-1]
            color = cmap(metric_norm_val)

            # 2. Apply Jitter (For visual distinction only)
            # We add random noise to every axis point for this line
            noise = np.random.uniform(-jitter_strength, jitter_strength, size=ys.shape)
            ys_jittered = ys + noise

            # Generate curve coordinates
            curve_xs = []
            curve_ys = []

            for j in range(len(cols) - 1):
                y_curr = ys_jittered[j]
                y_next = ys_jittered[j + 1]

                # Interpolate Y
                interp_y = y_curr * (1 - smooth_t) + y_next * smooth_t
                # Interpolate X
                interp_x = j + t

                curve_xs.append(interp_x)
                curve_ys.append(interp_y)

            full_x = np.concatenate(curve_xs)
            full_y = np.concatenate(curve_ys)

            ax.plot(full_x, full_y, color=color, alpha=0.4, linewidth=1.5)

        # Add Colorbar
        m_min, m_max, _ = range_map[metric]
        norm = plt.Normalize(vmin=m_min, vmax=m_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=30)
        cbar.set_label(metric.upper(), fontweight="bold")

        # Decorate Axes
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha="right")
        ax.set_yticks([])
        # Expand limits slightly to accommodate jitter
        ax.set_ylim(-0.15, 1.15)
        ax.grid(False)

        for i, col in enumerate(cols):
            ax.axvline(i, color="black", linewidth=1.0, alpha=0.2)

            if col in range_map:
                info = range_map[col]
                if info[-1] == "num":
                    mn, mx, _ = info
                    ax.text(i, -0.05, f"{mn:.3g}", ha="center", va="top", fontweight="bold")
                    ax.text(i, 1.05, f"{mx:.3g}", ha="center", va="bottom", fontweight="bold")
                else:
                    uniques, _ = info
                    if len(uniques) <= 10:
                        for idx, u in enumerate(uniques):
                            y_pos = idx / (len(uniques) - 1) if len(uniques) > 1 else 0.5
                            ax.text(
                                i,
                                y_pos,
                                str(u),
                                ha="center",
                                va="center",
                                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                            )
                    else:
                        ax.text(i, -0.05, str(uniques[0]), ha="center", va="top")
                        ax.text(i, 1.05, str(uniques[-1]), ha="center", va="bottom")

        ax.set_title(f"Parallel Coordinate Plot: Parameters vs {metric.upper()}")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print("PCP saved:", out_file)


def main():
    parser = argparse.ArgumentParser(description="Plot SfM and gsplat metrics.")
    parser.add_argument("--name", required=True, help="Scene name, e.g., bonsai_8")
    parser.add_argument("--x_axis", default="num_images", help="X-Axis key")
    parser.add_argument(
        "--split_param", default=None, help="Optional param to split series, e.g., conf_thres_value,sampling_mode"
    )
    parser.add_argument("--filter", default=None, help="Filter string, e.g. 'num_images=10,20;conf_thres_value=0.5'")
    args = parser.parse_args()
    plot_graph(args.name, "default", args.x_axis, args.split_param, args.filter)


def plot_graph(
    name: str, prefix: str, x_axis: str, split_param: str | None = None, filter: str | None = None, folders: list[str] | None = None
):
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

    valid_split_cols = []
    if split_param:
        split_cols = [p.strip() for p in split_param.split(",") if p.strip()]
        valid_split_cols = [c for c in split_cols if c in df.columns and c != "choice"]
        print(f"Invalid split cols: {set(split_cols) - set(valid_split_cols)}")

    if valid_split_cols:
        split_val_series = df[valid_split_cols[0]].fillna("").astype(str)
        for col in valid_split_cols[1:]:
            split_val_series = split_val_series + " | " + df[col].fillna("").astype(str)

        df["plot_series"] = df["method"] + " | " + split_val_series
        unique_splits = sorted(split_val_series.unique())
    else:
        df["plot_series"] = df["method"]
        unique_splits = ["colmap", "vggt"]

    unique_series = sorted(df["plot_series"].unique())

    series_indices = {s: i for i, s in enumerate(unique_series)}
    num_series = len(unique_series)

    centered_indices = {s: i - (num_series - 1) / 2 for s, i in series_indices.items()}

    jitter_col = f"{x_axis}_jitter"

    if x_axis == "num_points":
        jitter_factor = 0.05
        df[jitter_col] = df.apply(
            lambda row: row[x_axis] * (1 + centered_indices.get(row["plot_series"], 0) * jitter_factor), axis=1
        )
    else:
        # Determine appropriate jitter width based on minimum distance between X values
        unique_x = sorted(df[x_axis].dropna().unique())
        if len(unique_x) > 1:
            min_dist = min(np.diff(unique_x))
        else:
            min_dist = 1.0

        jitter_width = min_dist * 0.15

        df[jitter_col] = df.apply(
            lambda row: row[x_axis] + (centered_indices.get(row["plot_series"], 0) * jitter_width), axis=1
        )

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

        if valid_split_cols:
            val_str = series.replace(f"{method} | ", "")
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

    # TODO Make this a parameter for which metrics to use
    metrics_config = [
        # {"y": "rre", "title": "Rotation ($RRE$)", "ylabel": "Degrees ↓"},
        # {"y": "rte", "title": "Translation ($RTE$)", "ylabel": "Norm. Units ↓"},
        {"y": "psnr", "title": "Quality ($PSNR$)", "ylabel": "dB ↑"},
        {"y": "lpips", "title": "Perceptual ($LPIPS$)", "ylabel": "Score ↓"},
    ]

    fig, axes = plt.subplots(1, len(metrics_config), figsize=(8 * len(metrics_config), 4.5))
    if len(metrics_config) == 1:
        axes = [axes]

    for ax, config in zip(axes, metrics_config):
        plot_df = df
        hlines_dict = OrderedDict()
        hranges_dict = OrderedDict()

        if x_axis == "conf_thres_value" and not colmap_df.empty:
            plot_df = df[df["method"] != "colmap"]
            sort_key = valid_split_cols[0] if valid_split_cols else None

            for series_name, metric_values in list(
                sorted(
                    list(colmap_means_by_series.items()),
                    key=lambda x: x[1].get(sort_key) if sort_key and sort_key in x[1] else x[0],
                )
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
            x=jitter_col,
            original_x_col=x_axis,
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

    for i in range(len(axes)):
        handles, labels = axes[i].get_legend_handles_labels()
        processed_labels = [label.replace("colmap", "COLMAP").replace("vggt", "VGGT") for label in labels]
        if handles:
            axes[i].legend(handles=handles, labels=processed_labels, markerscale=1.5)

    suffix = f"plots/{prefix}/full_evaluation-{name}-{x_axis}-{split_param}"

    os.makedirs(os.path.dirname(suffix), exist_ok=True)

    csv_out_file = f"{suffix}.csv"
    os.makedirs(os.path.dirname(csv_out_file), exist_ok=True)
    df.to_csv(csv_out_file, index=False)
    print("Dataframe saved:", Path(csv_out_file))

    plt.tight_layout()
    out_file = f"{suffix}.png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print("Comprehensive plot saved:", Path(out_file))

    pcp_out_file = f"{suffix}_pcp.png"
    plot_pcp(df, pcp_out_file, color_map)

    render_out_base = Path(suffix + "_renders")
    for file_path in df["file_path"].unique():
        p = Path(file_path)
        if "stats" in p.parts and "val_step" in p.name:
            render_src_dir = p.parents[1] / "renders"
            if render_src_dir.exists():
                folder_name = p.parents[1].name
                dest_dir = render_out_base / folder_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for render_file in render_src_dir.glob(f"{p.stem}_*.png"):
                    shutil.copy2(render_file, dest_dir / render_file.name)

    plt.show()


if __name__ == "__main__":
    main()
