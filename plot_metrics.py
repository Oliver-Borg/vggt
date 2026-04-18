import argparse
from collections import OrderedDict
import glob
import json
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Tuple
import numpy as np
import shutil
import warnings

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
import seaborn as sns
import scienceplots

plt.style.use(["science", "grid"])

textwidth = 7.00697 # * 2 / 3
aspect_ratio = 6 / 8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    }
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*Calling float on a single element Series.*")


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
    Param(name="sampling_mode", pattern=r"_(voxels)|(confidence)|(random)|(ba)|(vox3)", cast=str),
    Param(name="image_mode", pattern=r"_(shuffle)|(distributed)|(mfps)|(farthestpose)", cast=str, default=""),
    Param(name="num_cameras", pattern=r"_i(\d+)", cast=int),
    Param(name="gt_eval", pattern=r"_(gteval)", cast=lambda x: "GT Eval" if x == "gteval" else "", default=""),
    Param(
        name="use_gt_extrinsics",
        pattern=r"_(gtext)",
        cast=lambda x: "GT Extrinsics" if x == "gtext" else "",
        default="",
    ),
    Param(
        name="use_gt_intrinsics",
        pattern=r"_(gtint)",
        cast=lambda x: "GT Intrinsics" if x == "gtint" else "",
        default="",
    ),
    Param(
        name="use_gt_points",
        pattern=r"_(gtpcd)",
        cast=lambda x: "GT Points" if x == "gtpcd" else "",
        default="",
    ),
    Param(
        name="pose_opt",
        pattern=r"_(poseopt)",
        cast=lambda x: "Pose Opt" if x == "poseopt" else "",
        default="",
    ),
    Param(
        name="eval_opt",
        pattern=r"_(evalopt)",
        cast=lambda x: "Eval Pose Opt" if x == "evalopt" else "",
        default="",
    ),
    Param(
        name="depth_loss",
        pattern=r"_(depth)",
        cast=lambda x: "Depth Loss" if x == "depth" else "",
        default="No Depth Loss",
    ),
    Param(
        name="error_opa",
        pattern=r"_(erroropa)",
        cast=lambda x: "Err Opa Init" if x == "erroropa" else "",
        default="",
    ),
    Param(
        name="depth_lambda",
        pattern=r"_(dl\d+\.\d+)",
        cast=lambda x: (float(x.replace("dl", ""))) if x.startswith("dl") else 0.0,
        default=0.0,
    ),
    Param(
        name="depth_conf",
        pattern=r"_(conf)",
        cast=lambda x: "Depth Confidence" if x == "conf" else "",
        default="No Depth Confidence",
    ),
    Param(name="choice", pattern=r"(vggt)|(colmap)|(gt)_outputs", cast=str),
    Param(name="camera_type", pattern=r"_m(radial)|(pinhole)", cast=str),
    Param(name="copy_mode", pattern=r"_(crop)|(tiles)|(square)", cast=str, default=None),
    Param(name="val_step", pattern=r"val_step(\d+)", cast=int, default=None),
    Param(name="num_steps", pattern=r"steps(\d+)", cast=int, default=None),
    Param(name="colmap_mode", pattern=r"_(default)|(relaxed)", cast=str, default=None),
    Param(
        name="splatting_strategy",
        pattern=r"_(nomcmc)",
        cast=lambda x: "Default" if x == "nomcmc" else "MCMC",
        default="MCMC",
    ),
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


def load_metrics_to_df(
    scene_name: str,
    methods: list[str],
    folders: list[tuple[str, str]] | None = None,
    val_steps: list[int] = [7000],
) -> pd.DataFrame:
    """
    Loads both SfM and gsplat metrics for all methods into a single DataFrame.
    """
    records: dict[int | tuple[str, str], dict] = {}
    sfm_folders = [pair[0] for pair in folders] if folders is not None else []
    gsplat_folders = [pair[1] for pair in folders] if folders is not None else []

    # (source_name, glob_pattern, json_loader_func)
    sources = [
        [  # TODO Deal with some results having both
            "gsplat",
            os.path.expanduser(
                "~/work/git/gsplat/results/{method}_outputs/{scene}_n*_s*/stats/val_step" + str(i - 1).zfill(4) + ".json"
            ),
            _parse_gsplat_json,
            gsplat_folders,
        ]
        for i in val_steps
    ] + [
        [
            "sfm",
            os.path.expanduser("~/work/git/vggt/{method}_outputs/{scene}_n*_s*/eval_results.json"),
            _parse_sfm_json,
            sfm_folders,
        ],
    ]
    i = 0

    for method in methods:
        for source_name, path_template, parser, source_folders in sources:
            pattern = path_template.format(method=method, scene=scene_name)

            for file_path in glob.glob(pattern):
                try:
                    # Extract folder parameters
                    # We assume folder structure is consistent, getting the folder name relative to the file
                    folder_path = os.path.dirname(file_path)
                    folder_name = os.path.basename(folder_path)

                    folder_overlap: set[str] = set(source_folders) & set(Path(file_path).parts)

                    if folders is not None and len(folder_overlap) == 0:
                        continue

                    source_folder = folder_overlap.pop() if folders is not None else folder_name

                    if folders is not None and source_folder in source_folders:
                        index = source_folders.index(source_folder)
                        key = folders[index]
                    else:
                        key = (str(i), str(i))

                    # For gsplat, the folder is 3 levels up from the json file in the original code logic
                    if source_name == "gsplat":
                        folder_name = file_path.split("/")[-3]
                        key = (key[0], key[1] + "_" + Path(pattern).name)

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
                    if source_name == "gsplat":
                        records[key] = record
                    else:
                        updated = False
                        for sfm_folder, gsplat_folder in folders or []:
                            if sfm_folder == source_folder:
                                for step in val_steps:
                                    key1 = (sfm_folder, gsplat_folder + f"_val_step{step - 1}.json")
                                    if key1 in records:
                                        records[key1].update(metrics)
                                        updated = True

                        # If no corresponding gsplat record exists, save sfm as a standalone record
                        # TODO Get this to work properly without creating orphaned series
                        if not updated:
                            records[(folder_name, "")] = record

                except (ValueError, IndexError, KeyError, json.JSONDecodeError):
                    continue

    records_list = list(records.values())

    for record in records_list:
        if record.get("real_num_points") is not None and record.get("num_points") is None:
            record["num_points"] = record["real_num_points"]

    if not records_list:
        return pd.DataFrame()

    df = pd.DataFrame(records_list)

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
    """Extracts PSNR, LPIPS and SSIM from gsplat json."""
    parsed_data = {
        "psnr": data.get("psnr"),
        "lpips": data.get("lpips"),
        "ssim": data.get("ssim"),
        "num_GS": data.get("num_GS"),
        "val_step": int(filename.split("val_step")[-1].split(".json")[0]),
        "eval_rte": data.get("eval_rte"),
        "eval_rre": data.get("eval_rre"),
    }
    if "num_points" in data:
        parsed_data["real_num_points"] = int(data["num_points"])
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
    y_log_scale: bool = False,
    hatches: Dict[str, str] | None = None,
) -> None:
    """
    Generic plotting function using Seaborn.
    Plots line with markers and min/max error bars (PI 100) or a horizontal bar chart if x is empty.
    Optionally adds horizontal reference lines (hlines) and shaded regions (hranges).
    """

    if df.empty or y not in df.columns or df[y].isnull().all():
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        return
    # df[x] = df[x].fillna(0)

    if not x:
        hue_order = df[series_col].unique()
        sns.barplot(
            data=df,
            x=y,
            y=series_col,
            hue=series_col,
            hue_order=hue_order,
            palette=colors,
            errorbar=("pi", 100),
            capsize=0.1,
            ax=ax,
            orient="h",
            legend=False,
            width=0.5,
        )
        
        if hatches:
            for i, container in enumerate(ax.containers):
                if i < len(hue_order):
                    hatch_pattern = hatches.get(hue_order[i], "")
                    if hatch_pattern:
                        for patch in container:
                            patch.set_hatch(hatch_pattern)

        for container in ax.containers:
            ax.bar_label(container, fontsize=10, fmt="%.3g")

        x_min, x_max = ax.get_xlim()
        padding = (x_max - x_min) * 0.01

        for patch, tick_label in zip(ax.patches, ax.get_yticklabels()):
            x_pos = patch.get_x() + padding
            y_pos = patch.get_y() + patch.get_height() + 0.02

            txt = ax.text(x_pos, y_pos, tick_label.get_text(), ha="left", va="top")
            txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground="white")])

        ax.set_yticks([])  # Hide original y-ticks
        ax.set_ylabel("")  # Remove y-axis label
    else:
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

    if not x:
        ax.set_xlabel(ylabel)
    else:
        ax.set_ylabel(ylabel)
        label_col = original_x_col if original_x_col else x
        xlabel = " ".join(label_col.split("_")).title()
        ax.set_xlabel(xlabel)

        if label_col in ["num_points"]:
            ax.set_xscale("log")
        else:
            unique_x = sorted(df[label_col].fillna(0.0).unique().round())
            ax.set_xticks(unique_x)
            ax.set_xticklabels([str(n) for n in unique_x])

    if y_log_scale and x:
        ax.set_yscale("log")

    ax.grid(True, which="major", ls="-", alpha=0.15)

    # Handle Legend: Remove individual subplot legends, will add global one later or keep on first
    if ax.get_legend():
        ax.get_legend().remove()


def plot_pcp(df: pd.DataFrame, out_file: str, color_map: Dict[str, str], title: str | None = None):
    """
    Plots a Parallel Coordinate Plot (PCP) for metrics and parameters.
    Uses smooth curves and adds jitter to y-positions to visualize density.
    """
    params = [r.name for r in regexes if r.name in df.columns]
    metrics = ["psnr", "lpips", "ssim"]

    # Filter metrics present in DF
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return

    # Average metrics based on identical parameters (ignoring seed)
    # TODO Get this to work properly
    # if "seed" in params:
    #     group_cols = [p for p in params if p != "seed"]
    #     if group_cols:
    #         df = df.groupby(group_cols, dropna=False)[metrics].mean().reset_index()
    #     else:
    #         df = df[metrics].mean().to_frame().T

    # Remove all columns that only have a single value
    df = df.loc[:, df.nunique(dropna=False) > 1]

    # Update params and metrics after filtering
    params = [p for p in params if p in df.columns]
    metrics = [m for m in metrics if m in df.columns]

    if not metrics or not params:
        return

    num_metrics = len(metrics)
    fig, axes = plt.subplots(
        nrows=num_metrics, ncols=1, figsize=(max(10, len(params) * 1.5), 6 * num_metrics), sharex=False
    )

    if num_metrics == 1:
        axes = [axes]

    if title:
        fig.suptitle(f"{title} - Parallel Coordinates", fontsize=16)

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
    plt.savefig(str(Path(out_file).with_suffix(".pdf")))
    print("PCP saved:", out_file, "and PDF")


def main():
    parser = argparse.ArgumentParser(description="Plot SfM and gsplat metrics.")
    parser.add_argument("--name", required=True, help="Scene name, e.g., bonsai_8")
    parser.add_argument("--x_axis", default="num_images", help="X-Axis key")
    parser.add_argument(
        "--split_param", default=None, help="Optional param to split series, e.g., conf_thres_value,sampling_mode"
    )
    parser.add_argument("--filter", default=None, help="Filter string, e.g. 'num_images=10,20;conf_thres_value=0.5'")
    parser.add_argument("--title", default=None, help="Base title for the plots")
    args = parser.parse_args()
    plot_graph(args.name, "default", args.x_axis, args.split_param, args.filter, title=args.title)


def plot_metric_combinations(
    df: pd.DataFrame, out_file: str, color_map: Dict[str, str], marker_map: Dict[str, str], x_axis: str, title: str | None = None
) -> None:
    """
    Plots combinations of metrics (PSNR/LPIPS vs RTE/RRE) as point plots with trend lines.
    """
    x_metrics = [("rre", "Rotation ($RRE$) ↓"), ("rte", "Translation ($RTE$) ↓")]
    y_metrics = [("psnr", "Quality ($PSNR$) ↑"), ("lpips", "Perceptual ($LPIPS$) ↓"), ("ssim", "Perceptual ($SSIM$) ↑")]

    # Ensure metrics exist in the dataframe
    available_cols = df.columns
    valid_x = [m for m in x_metrics if m[0] in available_cols]
    valid_y = [y for y in y_metrics if y[0] in available_cols]

    if not valid_x or not valid_y:
        print("Required metrics for combination plots are not available.")
        return

    num_plots = len(valid_y) * len(valid_x)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    # Standardize axes to 1D array for easy iteration
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    if title:
        fig.suptitle(f"{title} - Metric Combinations", fontsize=16)

    # Add the passed x-axis to the split params for grouping in the combination plot
    plot_df = df.copy()

    if not x_axis or x_axis not in plot_df.columns:
        plot_df["combo_series"] = plot_df["plot_series"]
    else:
        plot_df["combo_series"] = plot_df["plot_series"] + " | " + x_axis + "=" + plot_df[x_axis].astype(str)

    # Generate a color group that ignores the method (colmap/vggt/gt) to sync colors
    def get_color_group(combo_str):
        s = combo_str
        if s.startswith("colmap | "):
            return s[9:]
        if s.startswith("vggt | "):
            return s[7:]
        if s.startswith("gt | "):
            return s[5:]
        if s.startswith("colmap"):
            return s.replace("colmap", "base")
        if s.startswith("vggt"):
            return s.replace("vggt", "base")
        if s.startswith("gt"):
            return s.replace("gt", "base")
        return s

    plot_df["color_group"] = plot_df["combo_series"].apply(get_color_group)

    # Generate distinct colors for every unique parameter combination
    unique_color_groups = sorted(plot_df["color_group"].unique())
    base_pal = sns.color_palette("husl", n_colors=len(unique_color_groups))
    group_to_color = dict(zip(unique_color_groups, base_pal))

    combo_color_map = {}
    combo_marker_map = {}
    for _, row in plot_df.iterrows():
        combo_color_map[row["combo_series"]] = group_to_color[row["color_group"]]
        # Keep the base marker shape to identify the parent series
        combo_marker_map[row["combo_series"]] = marker_map.get(row["plot_series"], "o")

    for i, (y_col, y_label) in enumerate(valid_y):
        for j, (x_col, x_label) in enumerate(valid_x):
            ax_idx = i * len(valid_x) + j
            ax = axes[ax_idx]

            curr_plot_df = plot_df.dropna(subset=[x_col, y_col])

            if curr_plot_df.empty:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
                continue

            # Point plot (scatter)
            sns.scatterplot(
                data=curr_plot_df,
                x=x_col,
                y=y_col,
                hue="combo_series",
                style="combo_series",
                palette=combo_color_map,
                markers=combo_marker_map,
                s=100,
                alpha=0.8,
                ax=ax,
            )

            # Trend lines grouped by the original plot_series
            for series_name in curr_plot_df["plot_series"].unique():
                series_data = curr_plot_df[curr_plot_df["plot_series"] == series_name]
                if len(series_data) > 1:
                    # Sync trend line color with the points
                    trend_color = combo_color_map[series_data.iloc[0]["combo_series"]]
                    sns.regplot(
                        data=series_data,
                        x=x_col,
                        y=y_col,
                        scatter=False,
                        ax=ax,
                        color=trend_color,
                        line_kws={"linestyle": "--", "alpha": 0.5},
                    )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_col.upper()} vs {x_col.upper()}")
            ax.grid(True, which="major", ls="-", alpha=0.15)

            # Keep a legend on the far-right plot
            if ax.get_legend():
                if ax_idx == num_plots - 1:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
                else:
                    ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.savefig(str(Path(out_file).with_suffix(".pdf")), bbox_inches="tight")
    print("Metric combinations plot saved:", Path(out_file), "and PDF")


def plot_graph(
    name: str,
    prefix: str,
    x_axis: str,
    split_param: str | None = None,
    filter: str | None = None,
    folders: list[tuple[str, str]] | None = None,
    create_pcp: bool = True,
    create_combinations: bool = False,
    copy_images: bool = False,
    val_steps: list[int] = [7000],
    title: str | None = None,
    metric_keys: list[str] = ["rre", "rte", "psnr", "lpips", "ssim", "num_GS"],
    dataset_name: str | None = None,
    experiment_name: str | None = None,
    config_dict: dict | None = None,
    apply_jitter: bool = False,
):
    df = load_metrics_to_df(name, methods=["colmap", "vggt", "gt"], folders=folders, val_steps=val_steps)

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

    # split_param = "val_step" if split_param is None else f"{split_param},val_step"

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
        unique_splits = ["colmap", "vggt", "gt"]

    unique_series = sorted(df["plot_series"].unique())

    series_indices = {s: i for i, s in enumerate(unique_series)}
    num_series = len(unique_series)

    centered_indices = {s: i - (num_series - 1) / 2 for s, i in series_indices.items()}

    if not x_axis:
        jitter_col = ""
    else:
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

            jitter_width = min_dist * 0.15 if apply_jitter else 0.0

            max_val = df[x_axis].max()
            min_val = df[x_axis].min()

            df[jitter_col] = df.apply(
                lambda row: row[x_axis] + (centered_indices.get(row["plot_series"], 0) * jitter_width), axis=1
            )

            df[jitter_col] = df[jitter_col].clip(lower=min_val, upper=max_val)
            

    style_config = {
        "colmap": {"marker": "o", "dashes": "", "hatch": ""},
        "vggt": {"marker": "X", "dashes": (2, 2), "hatch": "///"},
        "gt": {"marker": "s", "dashes": (4, 4), "hatch": "\\\\\\"},
    }

    pal = sns.color_palette("tab10", n_colors=len(unique_splits))
    val_to_color = dict(zip(unique_splits, pal))

    color_map = {}
    marker_map = {}
    dash_map = {}
    hatch_map = {}

    for series in unique_series:
        if series.startswith("colmap"):
            method = "colmap"
        elif series.startswith("vggt"):
            method = "vggt"
        elif series.startswith("gt"):
            method = "gt"
        else:
            method = "vggt"

        if valid_split_cols:
            val_str = series.replace(f"{method} | ", "")
            original_val = next((v for v in unique_splits if str(v) == val_str), None)
            color_map[series] = val_to_color.get(original_val, "#333333")
        else:
            color_map[series] = val_to_color.get(method, "#333333")

        marker_map[series] = style_config[method]["marker"]
        dash_map[series] = style_config[method]["dashes"]
        hatch_map[series] = style_config[method]["hatch"]

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
        {"y": "rre", "title": "Rotation ($RRE$)", "ylabel": "Degrees ↓", "ylog": True},
        {"y": "rte", "title": "Translation ($RTE$)", "ylabel": "Norm. Units ↓"},
        {"y": "psnr", "title": "Quality ($PSNR$)", "ylabel": "dB ↑"},
        {"y": "lpips", "title": "Perceptual ($LPIPS$)", "ylabel": "Score ↓"},
        {"y": "ssim", "title": "Perceptual ($SSIM$)", "ylabel": "Score ↑"},
        {"y": "num_GS", "title": "Final Gaussian Count", "ylabel": "Count"},
        {"y": "eval_rre", "title": "Validation Step Rotation ($RRE$)", "ylabel": "Degrees ↓", "ylog": True},
        {"y": "eval_rte", "title": "Validation Step Translation ($RTE$)", "ylabel": "Norm. Units ↓"},
    ]

    metrics = {m["y"]: m for m in metrics_config}

    metrics_config = [metrics[m] for m in metric_keys if m in metrics]

    rows = len(metrics_config) // 3 + (1 if len(metrics_config) % 3 else 0)
    cols = len(metrics_config) // rows + (1 if len(metrics_config) % rows else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows))
    if len(metrics_config) == 1:
        axes = [axes]

    axes = axes.flatten() if hasattr(axes, "flatten") else axes

    if title:
        fig.suptitle(f"{title}", fontsize=16)

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
            y_log_scale=config.get("ylog", False),
            hatches=hatch_map,
        )

    for i in range(len(axes)):
        handles, labels = axes[i].get_legend_handles_labels()
        processed_labels = [
            label.replace("colmap", "COLMAP").replace("vggt", "VGGT").replace("gt", "GT") for label in labels
        ]
        if handles:
            axes[i].legend(handles=handles, labels=processed_labels, markerscale=1.5)

    suffix = f"plots/{prefix}/full_evaluation-{name}-{x_axis}-{split_param}"

    os.makedirs(os.path.dirname(suffix), exist_ok=True)

    csv_out_file = f"{suffix}.csv"
    os.makedirs(os.path.dirname(csv_out_file), exist_ok=True)
    df.to_csv(csv_out_file, index=False)
    print("Dataframe saved:", Path(csv_out_file))

    json_out_file = f"{suffix}.json"
    os.makedirs(os.path.dirname(json_out_file), exist_ok=True)
    df.to_json(json_out_file, orient="records", indent=4)
    print("Dataframe saved:", Path(json_out_file))

    config_out_file = None
    if config_dict:
        config_out_file = f"{suffix}_config.json"
        os.makedirs(os.path.dirname(config_out_file), exist_ok=True)
        with open(config_out_file, "w") as f:
            json.dump(config_dict, f, indent=4)
        print("Config saved:", Path(config_out_file))

    plt.tight_layout()
    out_file = f"{suffix}.png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.savefig(str(Path(out_file).with_suffix(".pdf")))
    print("Comprehensive plot saved:", Path(out_file), "and PDF")

    pcp_out_file = f"{suffix}_pcp.png"
    if create_pcp:
        plot_pcp(df, pcp_out_file, color_map, title=title)

    combo_out_file = f"{suffix}_combos.png"
    if create_combinations:
        plot_metric_combinations(df, combo_out_file, color_map, marker_map, x_axis, title=title)

    render_out_base = Path(suffix + "_renders")
    for file_path in df["file_path"].unique():
        if not copy_images:
            break
        p = Path(file_path)
        if "stats" in p.parts and "val_step" in p.name:
            render_src_dir = p.parents[1] / "renders"
            if render_src_dir.exists():
                folder_name = p.parents[1].name
                dest_dir = render_out_base / folder_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for render_file in render_src_dir.glob(f"{p.stem}_*.png"):
                    shutil.copy2(render_file, dest_dir / render_file.name)

    if dataset_name and experiment_name:
        latest_suffix = f"latest_plots/{experiment_name}_{dataset_name}_latest"
        os.makedirs(os.path.dirname(latest_suffix), exist_ok=True)
        latest_full_png = f"{latest_suffix}_full.png"
        shutil.copy2(out_file, latest_full_png)
        print("Latest copy saved:", Path(latest_full_png))
        latest_full_pdf = f"{latest_suffix}_full.pdf"
        shutil.copy2(str(Path(out_file).with_suffix(".pdf")), latest_full_pdf)
        print("Latest copy saved:", Path(latest_full_pdf))
        latest_full_csv = f"{latest_suffix}_full.csv"
        shutil.copy2(csv_out_file, latest_full_csv)
        print("Latest copy saved:", Path(latest_full_csv))
        latest_full_json = f"{latest_suffix}_full.json"
        shutil.copy2(json_out_file, latest_full_json)
        print("Latest copy saved:", Path(latest_full_json))
        if config_out_file:
            latest_config = f"{latest_suffix}_config.json"
            shutil.copy2(config_out_file, latest_config)
            print("Latest copy saved:", Path(latest_config))
        if create_pcp:
            latest_pcp_png = f"{latest_suffix}_pcp.png"
            if Path(pcp_out_file).exists():
                shutil.copy2(pcp_out_file, latest_pcp_png)
                print("Latest copy saved:", Path(latest_pcp_png))
            latest_pcp_pdf = f"{latest_suffix}_pcp.pdf"
            pcp_pdf_file = str(Path(pcp_out_file).with_suffix(".pdf"))
            if Path(pcp_pdf_file).exists():
                shutil.copy2(pcp_pdf_file, latest_pcp_pdf)
                print("Latest copy saved:", Path(latest_pcp_pdf))
        if create_combinations:
            latest_combo_png = f"{latest_suffix}_comparison.png"
            if Path(combo_out_file).exists():
                shutil.copy2(combo_out_file, latest_combo_png)
                print("Latest copy saved:", Path(latest_combo_png))
            latest_combo_pdf = f"{latest_suffix}_comparison.pdf"
            combo_pdf_file = str(Path(combo_out_file).with_suffix(".pdf"))
            if Path(combo_pdf_file).exists():
                shutil.copy2(combo_pdf_file, latest_combo_pdf)
                print("Latest copy saved:", Path(latest_combo_pdf))

    plt.show()


if __name__ == "__main__":
    main()
