import argparse
from collections import OrderedDict
import glob
import itertools
import json
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Tuple
from line_profiler import profile
import numpy as np
import shutil
import warnings

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import scienceplots

from .plot_cameras import CameraSeries, plot_extrinsics
from .cam_utils import load_poses_from_json
from .plot_point_cloud import plot_point_clouds

plt.style.use(["science", "grid"])

textwidth = 7.00697
aspect_ratio = 6 / 8
scale = 0.5
width = textwidth
height = width * aspect_ratio

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "dejavusans",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "lines.linewidth": 0.75,
        "lines.markersize": 6.0,
        "patch.linewidth": 0.5,
        "axes.linewidth": 0.4,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
    }
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*Calling float on a single element Series.*")


def np_rgba(np_arr: np.ndarray, cmap: str) -> np.ndarray:
    """
    Convert a grayscale image to RGBA using a matplotlib colormap
    Args:
        np_arr (np.ndarray): The grayscale image
        cmap (str): The name of the matplotlib colormap to use
    Returns:
        np.ndarray: The RGBA image
    """

    max_value = np_arr.max()

    normalised = np_arr.astype(np.float32) / max_value
    mapper = plt.get_cmap(cmap)
    rgba = mapper(normalised)
    rgba = rgba * 255
    rgba = rgba.astype(np.uint8)
    # rgba[np_arr == 0] = [21, 59, 106, 255] #153b6a
    return rgba


def np_rgb(np_arr: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """
    Convert a grayscale image to RGB using a matplotlib colormap
    Args:
        np_arr (np.ndarray): The grayscale image
        cmap (str): The name of the matplotlib colormap to use
    Returns:
        np.ndarray: The RGB image
    """
    rgba = np_rgba(np_arr, cmap)
    return rgba[..., :3]


dataset_collections = {
    "outdoor": ["bicycle", "garden", "stump"],
    "indoor": ["bonsai", "kitchen", "counter"],
    "synthetic": [
        "lego",
        "ship",
        "drums",
        # "chair",
        # "ficus",
        # "hotdog",
        # "materials",
        # "mic",
        # "blender_radial",
        # "blender_pinhole",
    ],
    "first": ["bicycle", "bonsai", "lego"],
}


@dataclass
class Param:
    name: str
    pattern: str
    cast: type[float | str | int] | Callable[[str], float | str | int]
    default: float | str | int | None = None
    skip_match: bool = False


regexes = [
    Param(name="num_images", pattern=r"_n(\d+)", cast=int),
    Param(name="seed", pattern=r"_s(\d+)", cast=int),
    Param(name="conf_thres_value", pattern=r"_c(\d+\.\d+)", cast=float),
    Param(name="num_points", pattern=r"_p(\d+)", cast=int),
    Param(
        name="sampling_mode", pattern=r"_(ba)|_(voxels)|_(confidence)|_(random)|_(vox3)|_(fps)|_(imagefps)", cast=str
    ),
    Param(name="near_filtering_strength", pattern=r"_nf(\d+\.\d+|\d+)", cast=str),
    Param(name="near_filtering_quorum", pattern=r"_nq(\d+)", cast=int),
    Param(
        name="reconstruct_pose_opt",
        pattern=r"_(recposeopt)",
        cast=lambda x: "Rec Pose Opt" if x == "recposeopt" else "",
        default="",
    ),
    Param(name="optimisation_iterations", pattern=r"_opti(\d+)", cast=int, default=0),
    Param(name="optimisation_neighbourhood", pattern=r"_optn(\d+)", cast=int, default=10),
    Param(
        name="image_mode",
        pattern=r"_(shuffle)|_(distributed)|_(mfps)|_(farthestpose)|_(nearestpose)",
        cast=str,
        default="",
    ),
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
        name="pose_opt_module",
        pattern=r"_pomod(mcmc)|_pomod(3rgs)|_pomod(sgld)",
        cast=str,
        default="",
    ),
    Param(
        name="eval_opt",
        pattern=r"_(evalopt)",
        cast=lambda x: "Eval Pose Opt" if x == "evalopt" else "",
        default="",
    ),
    Param(
        name="depth_loss_mode",
        pattern=r"_depth(points)|_depth(full)|_depth(closer)",
        cast=str,
        default="",
    ),
    Param(
        name="error_opa",
        pattern=r"_(erroropa)",
        cast=lambda x: "Err Opa Init" if x == "erroropa" else "",
        default="",
    ),
    Param(
        name="depth_lambda",
        pattern=r"_dl(\d+\.\d+)|_dl(\d+)",
        cast=float,
        default=0.0,
    ),
    Param(
        name="depth_conf_mode",
        pattern=r"_conf(standard)|_conf(sigmoid)",
        cast=str,
        default="",
    ),
    Param(name="choice", pattern=r"(vggt)|(colmap)|(gt)_outputs", cast=str),
    Param(name="dataset", pattern=r"(?!)", cast=str, default="", skip_match=True),
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
    Param(name="camera_src", pattern=r"_(colmapcams)|(vggtcams)|(gtcams)", cast=str, default=None),
    Param(name="pcd_src", pattern=r"_(colmappcd)|(vggtpcd)|(gtpcd)|(bothpcd)", cast=str, default=None),
    Param(
        name="align_mode",
        pattern=r"_(amlocal)|_(amglobal)",
        cast=lambda x: "Local Alignment" if x == "amlocal" else "Global Alignment",
        default=None,
    ),
    Param(
        name="align_glue",
        pattern=r"_(glued)",
        cast=lambda _: "Align Glued",
        default=None,
    ),
    Param(
        name="shared_camera",
        pattern=r"_(sharedcam)",
        cast=lambda x: "Shared Cam" if x == "sharedcam" else "",
        default=None,
    ),
    Param(
        name="keep_backup_cams",
        pattern=r"_(fallbackcams)",
        cast=lambda x: "Fallback Cams" if x == "fallbackcams" else "",
        default=None,
    ),
    Param(
        name="random_init",
        pattern=r"_(randinit)",
        cast=lambda x: "Random Init" if x == "randinit" else "",
        default=None,
    ),
    Param(
        name="use_ba",
        pattern=r"_(useba)",
        cast=lambda x: "Use BA" if x == "useba" else "",
        default=None,
    ),
    Param(
        name="feature_extractor",
        pattern=r"_(sift)|_(aliked-sp-sift)|_(aliked-sp)",
        cast=lambda x: x.replace("-", "+"),
        default="",
    ),
    Param(
        name="max_ba_iterations",
        pattern=r"_maxba(\d+)",
        cast=int,
        default=None,
    ),
]

params_dict = {param.name: param for param in regexes}


def convert_to_nullable_ints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts numeric columns that contain only integers and NaNs into pandas'
    nullable 'Int64' type. This prevents integers from becoming floats (e.g., 1.0)
    in JSON dumps and keeps NaNs as null.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if col in params_dict and params_dict[col].cast in [float, str]:
            continue
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            valid_vals = df_clean[col].dropna()
            # If the column has values and all non-NaN values are whole numbers
            if not valid_vals.empty and (valid_vals % 1 == 0).all():
                df_clean[col] = df_clean[col].astype("Int64")
    return df_clean


def extract_params(folder_name: str) -> dict[str, str | float | int | None]:
    """Extracts parameters from a folder name using regex."""
    params: dict[str, str | float | int | None] = {}
    for p in regexes:
        if p.skip_match:
            continue
        if match := re.search(p.pattern, folder_name):
            params[p.name] = p.cast([g for g in match.groups() if g is not None][0])
        else:
            params[p.name] = p.default
    return params


def load_metrics_to_df(
    scene_name: str | list[str],
    methods: list[str],
    folders: list[tuple[str, str]] | None = None,
    val_steps: list[int] = [7000],
) -> pd.DataFrame:
    """
    Loads both SfM and gsplat metrics for all methods into a single DataFrame.
    Accepts a single scene name or a list of scene names.
    """
    # Normalize scene_name to a list
    if isinstance(scene_name, str):
        scene_names = [scene_name]
    else:
        scene_names = scene_name

    # Extract original dataset order to maintain across all plots/tables
    dataset_order = []
    for s_name in scene_names:
        ds_val = "".join([c for c in s_name.replace("_", " ") if not c.isnumeric()]).strip()
        if ds_val not in dataset_order:
            dataset_order.append(ds_val)

    collection_order = []
    for ds_val in dataset_order:
        coll = "other"
        for k, v in dataset_collections.items():
            if ds_val in v:
                coll = k
                break
        if coll not in collection_order:
            collection_order.append(coll)

    records: dict[int | tuple[str, str], dict] = {}
    sfm_folders = [pair[0] for pair in folders] if folders is not None else []
    gsplat_folders = [pair[1] for pair in folders] if folders is not None else []

    # (source_name, glob_pattern, json_loader_func)
    sources = [
        [  # TODO Deal with some results having both
            "gsplat",
            os.path.expanduser(
                "~/work/git/gsplat/results/{method}_outputs/{scene}_n*_s*/stats/val_step"
                + str(i - 1).zfill(4)
                + ".json"
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

    for scene_name in scene_names:
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

                        if "val_step" in metrics:
                            metrics["val_step"] += 1

                        # Construct record
                        record = {
                            "method": method,
                            "source": source_name,
                            "dataset": "".join([c for c in scene_name.replace("_", " ") if not c.isnumeric()]).strip(),
                        }

                        coll = "other"
                        for k, v in dataset_collections.items():
                            if record["dataset"] in v:
                                coll = k
                                break
                        record["dataset_collection"] = coll

                        record.update(params)
                        record.update(metrics)
                        record["file_path"] = file_path
                        record[f"{source_name}_file_path"] = file_path
                        record["folder"] = folder_name
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
                                            records[key1][f"{source_name}_file_path"] = file_path
                                            updated = True

                            # If no corresponding gsplat record exists, save sfm as a standalone record
                            # TODO Get this to work properly without creating orphaned series
                            if not updated:
                                records[(folder_name, "")] = record

                    except (ValueError, IndexError, KeyError, json.JSONDecodeError):
                        continue

    records_list = []
    for k, v in records.items():
        if isinstance(k, tuple):
            records_list.append({**v, "input_folder": k[0], "output_folder": k[1]})
        else:
            records_list.append(v)

    for record in records_list:
        if record.get("real_num_points") is not None and record.get("num_points") is None:
            record["num_points"] = record["real_num_points"]
        if record.get("real_num_points") is not None:
            record["initial_points"] = record["real_num_points"]

    if not records_list:
        return pd.DataFrame()

    df = pd.DataFrame(records_list)
    # Convert 'dataset' to ordered Categorical to implicitly enforce input sorting everywhere
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
    df["dataset_collection"] = pd.Categorical(df["dataset_collection"], categories=collection_order, ordered=True)

    # Merge rows that share the same unique identifiers (method, num_images, seed, etc.)
    # Since SfM and GS metrics come from different files but belong to the same run,
    # we group by identifiers and combine the columns.
    group_cols = ["method"] + [reg.name for reg in regexes]
    # Filter only columns that exist to avoid key errors
    valid_group_cols = [c for c in group_cols if c in df.columns]

    df_merged = df.groupby(valid_group_cols, dropna=False, observed=True).first().reset_index()

    return convert_to_nullable_ints(df_merged)


def _parse_sfm_json(data: Dict, filename: str) -> Dict[str, float | dict[str, float]]:
    """Extracts RRE and RTE from SfM json."""
    if "metrics" in data and "mean_rre_deg" in data["metrics"]:
        metrics = {
            "rre": data["metrics"]["mean_rre_deg"],
            "rte": data["metrics"]["mean_rte"],
            "num_aligned": data["metrics"]["num_aligned"],
            "raw_eval_metrics": {
                "rre": data["metrics"]["all_rre"],
                "rte": data["metrics"]["all_rte"],
            },
        }
        if "mean_depth_l1" in data["metrics"]:
            metrics["pred_depth_l1"] = data["metrics"]["mean_depth_l1"]
            metrics["pred_depth_absrel"] = data["metrics"]["mean_depth_absrel"]
            metrics["pred_depth_rmse"] = data["metrics"]["mean_depth_rmse"]

        if "all_depth_l1" in data["metrics"]:
            metrics["raw_eval_metrics"]["pred_depth_l1"] = data["metrics"]["all_depth_l1"]
            metrics["raw_eval_metrics"]["pred_depth_absrel"] = data["metrics"]["all_depth_absrel"]
            metrics["raw_eval_metrics"]["pred_depth_rmse"] = data["metrics"]["all_depth_rmse"]

        return metrics
    return {}


def get_quality(psnr: float, ssim: float, lpips: float):
    scaled_psnr = (psnr - 14) / (32 - 14)
    scaled_ssim = (ssim - 0.35) / (0.92 - 0.35)
    scaled_lpips = (lpips - 0.06) / (0.60 - 0.06)
    quality = 1 / 3 * (scaled_psnr + scaled_ssim + 1 - scaled_lpips)
    return quality


def get_avge(psnr: float, ssim: float, lpips: float):
    avge = 1 / 3 * (10 ** (-psnr / 10) + (1 - ssim) ** 0.5 + lpips)
    return avge


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
        "raw_metrics": data.get("raw_metrics"),
        "depth_l1": data.get("depth_l1"),
        "depth_abs_rel": data.get("depth_abs_rel"),
    }

    if parsed_data["psnr"] is not None and parsed_data["lpips"] is not None and parsed_data["ssim"] is not None:
        parsed_data["quality"] = get_quality(parsed_data["psnr"], parsed_data["ssim"], parsed_data["lpips"])
        parsed_data["avge"] = get_avge(parsed_data["psnr"], parsed_data["ssim"], parsed_data["lpips"])
        if parsed_data["raw_metrics"] is not None:
            for psnr, lpips, ssim in zip(
                parsed_data["raw_metrics"]["psnr"],
                parsed_data["raw_metrics"]["lpips"],
                parsed_data["raw_metrics"]["ssim"],
            ):
                parsed_data.setdefault("raw_eval_metrics", {"quality": [], "avge": []})
                parsed_data["raw_eval_metrics"]["quality"].append(get_quality(psnr, ssim, lpips))
                parsed_data["raw_eval_metrics"]["avge"].append(get_avge(psnr, ssim, lpips))
    else:
        parsed_data["quality"] = None
        parsed_data["avge"] = None

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
    single_legend: bool = False,
    plot_raw: bool = False,
    hue_order: list[str] | None = None,
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

    _df = df.copy()
    for raw_key in (
        "raw_eval_metrics",
        "raw_metrics",
    ):
        if plot_raw and raw_key in _df.columns:

            def extract_raw(row):
                if isinstance(row.get(raw_key), dict) and y in row.get(raw_key, {}):
                    return row[raw_key][y]
                return [row.get(y)]

            _df[y] = _df.apply(extract_raw, axis=1)
            _df = _df.explode(y).reset_index(drop=True)
            _df[y] = pd.to_numeric(_df[y], errors="coerce")
    df = _df

    if not x:
        if hue_order is None:
            current_hue_order = df[series_col].unique()
        else:
            current_hue_order = [h for h in hue_order if h in df[series_col].values]

        sns.barplot(
            data=df,
            x=y,
            y=series_col,
            hue=series_col,
            hue_order=current_hue_order,
            palette=colors,
            errorbar=("pi", 100),
            capsize=0.05,
            ax=ax,
            orient="h",
            legend=True,
            width=0.5,
        )

        if hatches:
            for i, container in enumerate(ax.containers):
                if i < len(current_hue_order):
                    hatch_pattern = hatches.get(current_hue_order[i], "")
                    if hatch_pattern:
                        for patch in container:
                            patch.set_hatch(hatch_pattern)

            # Explicitly apply hatches to the legend handles so outer logic picks them up
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                hatch_pattern = hatches.get(label, "")
                if hatch_pattern and hasattr(handle, "set_hatch"):
                    handle.set_hatch(hatch_pattern)

        for container in ax.containers:
            ax.bar_label(container, fontsize=10, fmt="%.3g")

        if not single_legend:

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
        plot_df = df
        label_col_check = original_x_col if original_x_col else x

        if label_col_check in ["depth_lambda"]:
            plot_df = df.copy()
            unique_base = sorted(plot_df[label_col_check].fillna(0.0).unique())
            rank_map = {val: i for i, val in enumerate(unique_base)}

            if x != label_col_check:
                jitter = plot_df[x] - plot_df[label_col_check]
                plot_df[x] = plot_df[label_col_check].fillna(0.0).map(rank_map) + jitter
            else:
                plot_df[x] = plot_df[x].fillna(0.0).map(rank_map)

        if hue_order is None:
            current_hue_order = None
        else:
            current_hue_order = [h for h in hue_order if h in plot_df[series_col].values]

        sns.lineplot(
            data=plot_df,
            x=x,
            y=y,
            hue=series_col,
            style=series_col,
            hue_order=current_hue_order,
            style_order=current_hue_order,
            markers=markers,
            dashes=dashes,
            palette=colors,
            errorbar=("pi", 100),
            err_style="band",
            err_kws={"alpha": 0.1},
            ax=ax,
        )

        # Max line
        sns.lineplot(
            data=plot_df,
            x=x,
            y=y,
            hue=series_col,
            style=series_col,
            hue_order=current_hue_order,
            style_order=current_hue_order,
            markers=False,
            dashes=dashes,
            palette=colors,
            estimator="max",
            errorbar=None,
            ax=ax,
            legend=False,
            alpha=0.5,
        )
        # Min line
        sns.lineplot(
            data=plot_df,
            x=x,
            y=y,
            hue=series_col,
            style=series_col,
            hue_order=current_hue_order,
            style_order=current_hue_order,
            markers=False,
            dashes=dashes,
            palette=colors,
            estimator="min",
            errorbar=None,
            ax=ax,
            legend=False,
            alpha=0.5,
        )

    if hranges:
        for series_name, (min_val, max_val) in hranges.items():
            if pd.notna(min_val) and pd.notna(max_val):
                region_color = colors.get(series_name, "gray")
                dash_style = dashes.get(series_name, "-")
                ax.axhspan(
                    ymin=min_val,
                    ymax=max_val,
                    color=region_color,
                    alpha=0.1,
                    edgecolor=None,
                    linewidth=0,
                )
                ax.axhline(y=min_val, color=region_color, linestyle=dash_style, linewidth=1.0, alpha=0.5)
                ax.axhline(y=max_val, color=region_color, linestyle=dash_style, linewidth=1.0, alpha=0.5)

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
        elif label_col in ["depth_lambda"]:
            unique_x = sorted(df[label_col].fillna(0.0).unique())
            ax.set_xticks(range(len(unique_x)))
            ax.set_xticklabels([str(n) for n in unique_x])
        elif label_col not in ["num_images", "val_step"]:
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

        np.random.seed(42)  # For consistent jitter across runs

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
    plt.savefig(out_file, dpi=100)
    plt.savefig(str(Path(out_file).with_suffix(".pdf")), metadata={"CreationDate": None})
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


def save_figure_tex(tex_out_file: str, pdf_path: str, caption: str, label: str, fig_width: float = 1.0):
    """
    Generates a LaTeX figure block and saves it to a specified .tex file.
    """

    latex_figure = (
        "\\begin{figure}[H]\n"
        "    \\centering\n"
        f"    \\includegraphics[width={fig_width}\\linewidth]{{{pdf_path}}}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{figure}\n"
    )

    os.makedirs(os.path.dirname(tex_out_file), exist_ok=True)
    with open(tex_out_file, "w") as f:
        f.write(latex_figure)
    print("LaTeX figure saved:", Path(tex_out_file))


def plot_metric_combinations(
    df: pd.DataFrame,
    out_file: str,
    color_map: Dict[str, str],
    marker_map: Dict[str, str],
    x_axis: str,
    title: str | None = None,
    single_legend: bool = True,
) -> None:
    """
    Plots combinations of metrics (PSNR/LPIPS vs RTE/RRE) as point plots with trend lines.
    """
    x_metrics = [("rre", "RRE ↓"), ("rte", "RTE ↓")]
    y_metrics = [("psnr", "PSNR ↑"), ("lpips", "LPIPS ↓"), ("ssim", "SSIM ↑")]

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

            # Remove individual legends to rely on a single global legend
            if ax.get_legend() and single_legend:
                ax.get_legend().remove()

    if single_legend:
        handles_dict = OrderedDict()
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in handles_dict:
                    handles_dict[l] = h

        if handles_dict:
            handles = list(handles_dict.values())
            ncol = max(1, min(len(handles) // 2, 10))

            remainder = len(handles) % ncol
            if remainder > 0:
                pad_amount = ncol - remainder
                dummy_handle = mlines.Line2D([], [], linestyle="")
                handles.extend([dummy_handle] * pad_amount)
                labels.extend([""] * pad_amount)

            nrow = len(handles) // ncol
            handles_row_first = [handles[r * ncol + c] for c in range(ncol) for r in range(nrow)]
            labels_row_first = [labels[r * ncol + c] for c in range(ncol) for r in range(nrow)]

            fig.legend(
                handles=handles_row_first,
                labels=labels_row_first,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=ncol,
                markerscale=1.0,
            )

    plt.tight_layout()
    plt.savefig(out_file, dpi=100, bbox_inches="tight")
    plt.savefig(str(Path(out_file).with_suffix(".pdf")), bbox_inches="tight", metadata={"CreationDate": None})
    print("Metric combinations plot saved:", Path(out_file), "and PDF")


def add_text_to_image(
    img: Image.Image,
    text: str,
    fontsize: int,
) -> Image.Image:
    font = ImageFont.load_default(size=fontsize)
    # Use a temporary draw object to calculate the text bounding box
    draw_temp = ImageDraw.Draw(img)
    bbox = draw_temp.textbbox((0, 0), text, font=font, spacing=0)
    text_width = bbox[2] - bbox[0]
    text_height = int(bbox[3] - bbox[1])

    padding = 5
    # Create a new image with extra height for the label at the bottom
    new_height = img.height + text_height + 2 * padding
    new_img = Image.new(img.mode, (img.width, new_height), (255, 255, 255))
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    # Position the text in the new bottom area
    # Subtracting bbox offsets ensures the ink starts at our padding boundary
    x_pos, y_pos = padding - bbox[0], img.height + padding - bbox[1]

    # Draw black text directly onto the white padded area
    draw.text((x_pos, y_pos), text, fill=(0, 0, 0), font=font, spacing=0)

    return new_img


def resize_with_padding(im: Image.Image, width: int, height: int):
    w, h = im.size
    h_ratio = height / h
    w_ratio = width / w
    ratio = min(h_ratio, w_ratio)
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    im = im.resize((new_w, new_h))
    new_im = Image.new("RGB", (width, height), (0, 0, 0))
    new_im.paste(im, ((width - new_w) // 2, (height - new_h) // 2))
    return new_im


def _process_single_render(
    row: dict,
    render_src_dir: Path,
    p: Path,
    folder_name: str,
    x_axis: str | None = None,
    show_depth: bool = False,
    show_gt: bool = False,
    render_num: int = 0,
    image_width: int = 1200,
    image_height: int = 800,
):
    images = []
    # Grab only the first image for this validation step
    render_files = sorted(list(render_src_dir.glob(f"{p.stem}_*.jpg"))) + sorted(
        list(render_src_dir.glob(f"{p.stem}_*.png"))
    )

    if not render_files:
        return images

    render = render_files[render_num]
    raw_metrics = row.get("raw_metrics", {})
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}
    depth_factors: list[float] = raw_metrics.get("depth_factor", [])
    depth_l1s: list[float] = raw_metrics.get("depth_l1", [])
    depth_factor = depth_factors[render_num] if depth_factors else None
    depth_l1 = depth_l1s[render_num] if depth_l1s else None

    try:
        img = Image.open(render).convert("RGB")

        w, h = img.size

        # img is composed of 4 parts.
        # gt, predicted, difference and blended all horizontally

        def divide_img(im: Image.Image, splits: int = 4) -> list[Image.Image]:
            w, h = im.size
            width = w // splits
            height = h
            images = []
            for i in range(splits):
                left = i * width
                right = (i + 1) * width
                images.append(im.crop((left, 0, right, height)))
            return images

        if depth_factor is not None:
            # There is an extra column for depth
            if w % 7 == 0 and w % 5 != 0:
                gt_img, pred_img, _, _, depth, _, _ = divide_img(img, splits=7)
            else:
                gt_img, pred_img, _, _, depth = divide_img(img, splits=5)
            depth = np.array(depth).mean(axis=-1)
            # depth *= depth_factor
            # median_depth = np.median(depth[depth > 0]) + 1e-5
            # depth = depth / median_depth / 10
            # depth = np.clip(depth, 0.0, 1.0)
            depth_rgb = np_rgb(depth, "jet")
            depth_rgb[depth == 0, :] = 0
            depth = Image.fromarray(depth_rgb)
        else:
            gt_img, pred_img, _, _ = divide_img(img, splits=4)
            depth = None

        # DEBUGGING
        # w, h = pred_img.size
        # gt_cropped = gt_img.crop((w // 2, 0, w, h))
        # pred_img.paste(gt_cropped, (w // 2, 0))

        if show_gt:
            gt_img = resize_with_padding(gt_img, image_width, image_height)
            gt_img = add_text_to_image(gt_img, "Ground Truth", 48)
            images.append(gt_img)
            return images

        # TODO Decide if we want to do one of these
        # Paste a small version of the diff img in the bottom left corner of the pred_img
        # diff_img = diff_img.resize((diff_img.width // 4, diff_img.height // 4))
        # pred_img.paste(diff_img, (0, h - diff_img.height))

        if depth is not None and show_depth:
            # depth = depth.resize((depth.width // 3, depth.height // 3))
            # pred_img.paste(depth, (0, pred_img.height - depth.height))
            w, h = pred_img.size
            depth = depth.crop((w // 2, 0, w, h))
            pred_img.paste(depth, (w // 2, 0))

        # Crop half the image on the right and paste it
        # w, h = pred_img.size
        # diff_img = diff_img.crop((w // 2, 0, w, h))
        # pred_img.paste(diff_img, (w // 2, 0))

        # Prepare text label
        config_name = (
            str(row.get("plot_series", folder_name))
            .replace("colmap", "COLMAP")
            .replace("vggt", "VGGT")
            .replace("gt", "GT")
            .replace("combined", "Combined")
        )
        if x_axis and pd.notna(row.get(x_axis)):
            config_name = f"{config_name} | {x_axis}={row.get(x_axis)}"

        psnr_values: list[float] = raw_metrics.get("psnr", [])
        lpips_values: list[float] = raw_metrics.get("lpips", [])
        ssim_values: list[float] = raw_metrics.get("ssim", [])

        if psnr_values:
            psnr_val = psnr_values[render_num]
        else:
            psnr_val = row.get("psnr")

        if lpips_values:
            lpips_val = lpips_values[render_num]
        else:
            lpips_val = row.get("lpips")

        if ssim_values:
            ssim_val = ssim_values[render_num]
        else:
            ssim_val = row.get("ssim")

        psnr_str = f"{psnr_val:.2f}" if pd.notna(psnr_val) else "N/A"
        lpips_str = f"{lpips_val:.4f}" if pd.notna(lpips_val) else "N/A"
        ssim_str = f"{ssim_val:.4f}" if pd.notna(ssim_val) else "N/A"

        label_text = f"Config: {config_name}\nPSNR: {psnr_str} | LPIPS: {lpips_str} | SSIM: {ssim_str}"

        pred_img = resize_with_padding(pred_img, image_width, image_height)
        pred_img = add_text_to_image(pred_img, label_text, 48)
        images.append(pred_img)

    except Exception as e:
        print(f"Error processing image {render}: {e}")

    return images


def _stack_images_with_wrap(images: list[Image.Image], max_cols: int = 3):
    rows = [images[i : i + max_cols] for i in range(0, len(images), max_cols)]

    row_widths = [sum(im.size[0] for im in row) for row in rows]
    row_heights = [max(im.size[1] for im in row) for row in rows]

    total_width = max(row_widths)
    total_height = sum(row_heights)

    stacked_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    y_offset = 0
    for row, row_h in zip(rows, row_heights):
        x_offset = 0
        for im in row:
            stacked_img.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
        y_offset += row_h

    return stacked_img


@profile
def create_render_figure(
    df,
    dest_base,
    title: str | None = None,
    dataset_name: str | None = None,
    experiment_name: str | None = None,
    prefix: str | None = None,
    x_axis: str | None = None,
    max_cols: int = 3,
    show_depth: bool = False,
    show_gt: bool = False,
    render_nums: list[int] = [0],
    stack_dataset_renders_horizontally: bool = True,
):
    dest_base = Path(dest_base)
    dest_base.mkdir(parents=True, exist_ok=True)

    dataset_columns = []

    unique_datasets = df["dataset"].unique().tolist()

    cols_per_dataset = max_cols
    if stack_dataset_renders_horizontally and len(unique_datasets) > 1:
        cols_per_dataset = 1
        max_cols = len(unique_datasets)

    image_width = 1200
    image_height = 800 if any({"bicycle", "bonsai"} & set(unique_datasets)) else 1200

    for dataset in unique_datasets:
        dataset_df = df[df["dataset"] == dataset]
        images_to_stack = []
        for i, (idx, row) in enumerate(dataset_df.iterrows()):
            file_path = row.get("file_path", "")
            if not file_path or pd.isna(file_path):
                continue
            p = Path(file_path)
            if "stats" in p.parts and "val_step" in p.name:
                render_src_dir = p.parents[1] / "renders"
                if render_src_dir.exists():
                    folder_name = p.parents[1].name
                    if i == 0 and show_gt:
                        for render_num in render_nums:
                            processed_images = _process_single_render(
                                row,
                                render_src_dir,
                                p,
                                folder_name,
                                x_axis,
                                show_depth,
                                True,
                                render_num,
                                image_width,
                                image_height,
                            )
                            images_to_stack.extend(processed_images)
                    for render_num in render_nums:
                        processed_images = _process_single_render(
                            row,
                            render_src_dir,
                            p,
                            folder_name,
                            x_axis,
                            show_depth,
                            False,
                            render_num,
                            image_width,
                            image_height,
                        )
                        images_to_stack.extend(processed_images)

        if images_to_stack:
            dataset_columns.append(_stack_images_with_wrap(images_to_stack, max_cols=cols_per_dataset))

    if stack_dataset_renders_horizontally:
        stacked_img = _stack_images_with_wrap(dataset_columns, max_cols=max_cols)
    else:
        stacked_img = _stack_images_with_wrap(dataset_columns, max_cols=1)

    out_file = dest_base / "stacked_renders.png"
    stacked_img.save(out_file)
    print(f"Saved stacked renders to {out_file}")

    out_pdf = dest_base / "stacked_renders.pdf"
    stacked_img.save(out_pdf, "PDF", resolution=100.0)
    print(f"Saved stacked renders PDF to {out_pdf}")

    if dataset_name and experiment_name:
        latest_suffix = f"latest_plots/{experiment_name}_{dataset_name}_latest"
        os.makedirs(os.path.dirname(latest_suffix), exist_ok=True)

        latest_render_png = f"{latest_suffix}_renders.png"
        shutil.copy2(out_file, latest_render_png)
        print("Latest copy saved:", Path(latest_render_png))

        latest_render_pdf = f"{latest_suffix}_renders.pdf"
        shutil.copy2(out_pdf, latest_render_pdf)
        print("Latest copy saved:", Path(latest_render_pdf))

        latex_caption = f"{title} ({str(dataset_name).title()})." if title else f"{prefix} - {dataset_name}."
        latex_label = (
            f"fig:renders_{experiment_name}_{dataset_name}"
            if experiment_name
            else f"fig:renders_{prefix}_{dataset_name}"
        )

        tex_out_path = str(Path(latest_render_pdf).with_suffix(".tex"))
        save_figure_tex(
            tex_out_path,
            "Images/04-Results/Renders/" + Path(latest_render_pdf).name,
            caption=latex_caption,
            label=latex_label,
        )
        print("LaTeX figure saved:", Path(tex_out_path))


@profile
def plot_cameras(
    df: pd.DataFrame,
    dest_base: Path,
    x_axis: str | None = None,
    varying_colors: bool = False,
    use_error_colors: bool = False,
    max_cols: int = 3,
):
    dest_base.mkdir(parents=True, exist_ok=True)
    series_list: list[CameraSeries] = []
    df = df.copy()
    cmap = plt.get_cmap("tab10")
    image_names = []
    for i, (idx, row) in enumerate(df.iterrows()):
        file_path = row.get("sfm_file_path", "")
        if not file_path or pd.isna(file_path):
            continue

        p = Path(file_path)
        if p.name == "eval_results.json":
            if i == 0:
                pose_file = p.parent / "gt_cameras.json"
                color = cmap(i) if varying_colors else (0.5, 0.5, 0.5, 1.0)
                color = (*color[:3], 1.0)
                series = CameraSeries(
                    label="Ground Truth",
                    colour=color,
                    poses=load_poses_from_json(pose_file),
                )
                series_list.append(series)

            config_name = (
                str(row.get("plot_series", p.parent.name))
                .replace("colmap", "COLMAP")
                .replace("vggt", "VGGT")
                .replace("gt", "GT")
                .replace("combined", "Combined")
            )
            if x_axis and pd.notna(row.get(x_axis)):
                config_name = f"{config_name} | {x_axis}={row.get(x_axis)}"

            pose_file = p.parent / "aligned_cameras.json"
            color = cmap(i + 1) if varying_colors else (1.0, 0.0, 0.0, 1.0)
            color = (*color[:3], 1.0)
            series = CameraSeries(
                label=config_name,
                colour=color,
                poses=load_poses_from_json(pose_file),
            )
            series_list.append(series)

    if len(series_list) < 3:
        return

    largest_series = series_list[2]
    for series in series_list[2:]:
        image_names.extend(series.poses.keys())
        if len(series.poses) > len(largest_series.poses):
            largest_series = series

    image_names = list(set(image_names))

    plot_extrinsics(
        series_list[0],
        series_list[1:],
        dest_base,
        image_names,
        use_error_colors=use_error_colors,
        max_cols=max_cols,
    )


@profile
def plot_graph(
    name: str | list[str],
    prefix: str,
    x_axis: str,
    split_param: str | None = None,
    filter: str | None = None,
    folders: list[tuple[str, str]] | None = None,
    render_folders: list[str] | None = None,
    camera_folders: list[str] | None = None,
    create_pcp: bool = False,
    create_combinations: bool = False,
    val_steps: list[int] = [7000],
    title: str | None = None,
    metric_keys: list[str] = ["rre", "rte", "psnr", "lpips", "ssim", "num_GS"],
    dataset_name: str | None = None,
    experiment_name: str | None = None,
    config_dict: dict | None = None,
    apply_jitter: bool = False,
    horizontal: bool = True,
    single_legend: bool = True,
    create_table: bool = True,
    print_title: bool = False,
    split_choice: bool = False,
    split_dataset: Literal["none", "individual", "collection"] | bool = "none",
    max_render_cols: int = 3,
    show_depth: bool = False,
    show_gt: bool = False,
    render_nums: list[int] = [0],
    plot_raw: bool = True,
    shared_colors: bool = True,
    make_camera_plot: bool = False,
    make_pcd_plot: bool = False,
    stack_datasets_horizontally: bool = True,
):
    if isinstance(split_dataset, bool):
        split_dataset = "individual" if split_dataset else "none"

    df = load_metrics_to_df(name, methods=["colmap", "vggt", "gt", "combined"], folders=folders, val_steps=val_steps)
    original_name = name
    if isinstance(name, list):
        name = ",".join(name)
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

    sort_cols = valid_split_cols.copy()
    if "choice" in df.columns:
        sort_cols.append("choice")
    elif "method" in df.columns:
        sort_cols.append("method")

    if x_axis:
        sort_cols.append(x_axis)

    if sort_cols:
        df = df.sort_values(by=sort_cols)

    if valid_split_cols:
        split_val_series = df[valid_split_cols[0]].map(lambda x: str(x) if pd.notnull(x) else "")
        for col in valid_split_cols[1:]:
            split_val_series = split_val_series + " | " + df[col].map(lambda x: str(x) if pd.notnull(x) else "")

        df["plot_series"] = df["method"] + " | " + split_val_series
        unique_splits = split_val_series.unique().tolist()
    else:
        df["plot_series"] = df["method"]
        unique_splits = ["colmap", "vggt", "gt", "combined"]

    unique_series = df["plot_series"].unique().tolist()

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
        "combined": {"marker": "D", "dashes": (1, 1), "hatch": "xxx"},
    }

    # Dynamically select tab10 or tab20 based on variation count
    num_splits = len(unique_splits)
    pal_name = "tab10" if num_splits <= 10 else "tab20"

    if shared_colors:
        pal = sns.color_palette(pal_name, n_colors=num_splits)
        val_to_color = dict(zip(unique_splits, pal))
    else:
        # Prevent interleaving when not shared by grouping the assignments by method first
        sorted_series = sorted(unique_series, key=lambda x: (x.split(" | ")[0], x))
        unshared_pal = sns.color_palette("tab10" if len(sorted_series) <= 10 else "tab20", n_colors=len(sorted_series))
        val_to_color = dict(zip(sorted_series, unshared_pal))

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
        elif series.startswith("combined"):
            method = "combined"
        else:
            method = "vggt"

        if shared_colors:
            if valid_split_cols:
                # Safely split purely on the divider to extract the variation string
                parts = series.split(" | ", 1)
                val_str = parts[1] if len(parts) > 1 else parts[0]
                original_val = next((v for v in unique_splits if str(v) == val_str), None)
                color_map[series] = val_to_color.get(original_val, "#333333")
            else:
                color_map[series] = val_to_color.get(method, "#333333")
        else:
            color_map[series] = val_to_color.get(series, "#333333")

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
        {"y": "rre", "title": "RRE ↓", "ylabel": "Degrees", "direction": "↓", "ylog": True},
        {"y": "rte", "title": "RTE ↓", "ylabel": "Norm. Units", "direction": "↓", "ylog": True},
        {"y": "psnr", "title": "PSNR ↑", "ylabel": "dB", "direction": "↑"},
        {"y": "lpips", "title": "LPIPS ↓", "ylabel": "Score", "direction": "↓"},
        {"y": "ssim", "title": "SSIM ↑", "ylabel": "Score", "direction": "↑"},
        {"y": "quality", "title": "Composite Quality ↑", "ylabel": "Score", "direction": "↑"},
        {"y": "avge", "title": "Average Error ↓", "ylabel": "Score", "direction": "↓"},
        {"y": "num_GS", "title": "Final Gaussian Count", "ylabel": "Count"},
        {"y": "eval_rre", "title": "Validation Step RRE ↓", "ylabel": "Degrees", "direction": "↓", "ylog": True},
        {"y": "eval_rte", "title": "Validation Step RTE ↓", "ylabel": "Norm. Units", "direction": "↓", "ylog": True},
        {"y": "num_aligned", "title": "Aligned Cameras ↑", "ylabel": "Count", "direction": "↑"},
        {"y": "real_num_points", "title": "Initial Points", "ylabel": "Count"},
        {"y": "depth_l1", "title": "Splatted Depth L1 ↓", "ylabel": "Loss", "direction": "↓"},
        {"y": "depth_abs_rel", "title": "Splatted Depth Abs Rel ↓", "ylabel": "Score", "direction": "↓"},
        {"y": "pred_depth_l1", "title": "Predicted Depth L1 ↓", "ylabel": "Loss", "direction": "↓"},
        {"y": "pred_depth_absrel", "title": "Predicted Depth Abs Rel ↓", "ylabel": "Score", "direction": "↓"},
        {"y": "pred_depth_rmse", "title": "Predicted Depth RMSE ↓", "ylabel": "Loss", "direction": "↓"},
    ]

    metrics = {m["y"]: m for m in metrics_config}

    metrics_config = [metrics[m] for m in metric_keys if m in metrics]

    datasets = []
    if split_dataset == "individual":
        datasets = list(dict.fromkeys(df["dataset"].dropna())) if "dataset" in df.columns else [None]
    elif split_dataset == "collection":
        datasets = (
            list(dict.fromkeys(df["dataset_collection"].dropna())) if "dataset_collection" in df.columns else [None]
        )

    plot_configs = []

    stack_datasets_vertically = not stack_datasets_horizontally

    if split_choice and split_dataset != "none":
        # Split both choice and dataset
        split_col_choice = "choice" if "choice" in df.columns else "method"
        choices = sorted(df[split_col_choice].dropna().unique().tolist())
        for choice_val in choices:
            iter_product = (
                itertools.product(datasets, metrics_config)
                if stack_datasets_vertically
                else itertools.product(metrics_config, datasets)
            )
            for item in iter_product:
                dataset_val, m_config = item if stack_datasets_vertically else item[::-1]
                cfg = m_config.copy()
                dataset_str = f" - {dataset_val}" if dataset_val else ""
                cfg["title"] = f"{cfg['title']} - {choice_val}{dataset_str}"
                cfg["choice_val"] = choice_val
                cfg["dataset_val"] = dataset_val
                cfg["split_choice"] = True
                cfg["split_dataset"] = split_dataset
                plot_configs.append(cfg)
    elif split_choice:
        split_col = "choice" if "choice" in df.columns else "method"
        choices = sorted(df[split_col].dropna().unique().tolist())
        for choice_val in choices:
            for m_config in metrics_config:
                cfg = m_config.copy()
                cfg["title"] = f"{cfg['title']} - {choice_val}"
                cfg["choice_val"] = choice_val
                cfg["split_col"] = split_col
                plot_configs.append(cfg)
    elif split_dataset != "none":
        iter_product = (
            itertools.product(datasets, metrics_config)
            if stack_datasets_vertically
            else itertools.product(metrics_config, datasets)
        )
        for item in iter_product:
            dataset_val, m_config = item if stack_datasets_vertically else item[::-1]
            cfg = m_config.copy()
            dataset_str = str(dataset_val) if dataset_val else "Unknown"
            cfg["title"] = f"{cfg['title']} - {dataset_str}"
            cfg["dataset_val"] = dataset_val
            cfg["split_dataset"] = split_dataset
            plot_configs.append(cfg)
    else:
        plot_configs = metrics_config.copy()

    if not plot_configs:
        print("No metrics/choices to plot.")
        return

    if split_dataset != "none" and len(datasets) > 0:
        cols = len(datasets) if stack_datasets_horizontally else len(metrics_config)
        rows = len(plot_configs) // cols + (1 if len(plot_configs) % cols else 0)
    else:
        if horizontal:
            rows = len(plot_configs) // 3 + (1 if len(plot_configs) % 3 else 0)
            cols = len(plot_configs) // rows + (1 if len(plot_configs) % rows else 0)
        else:
            cols = len(plot_configs) // 3 + (1 if len(plot_configs) % 3 else 0)
            rows = len(plot_configs) // cols + (1 if len(plot_configs) % cols else 0)

    figwidth = width * scale * cols
    height = figwidth / cols * aspect_ratio * rows

    fig, axes = plt.subplots(rows, cols, figsize=(figwidth, height))
    if len(plot_configs) == 1:
        axes = [axes]

    axes = axes.flatten() if hasattr(axes, "flatten") else axes

    if title and print_title:
        fig.suptitle(f"{title}", fontsize=16)

    for ax, config in zip(axes, plot_configs):
        plot_df = df
        hlines_dict = OrderedDict()
        hranges_dict = OrderedDict()

        if split_choice and "choice_val" in config:
            split_col = config.get("split_col", "choice" if "choice" in df.columns else "method")
            if split_col == "method":
                plot_df = df[df[split_col] == config["choice_val"]]
            else:
                # Keep targeted choice plus the colmap baseline (if its choice is null/NaN)
                plot_df = df[
                    (df[split_col] == config["choice_val"]) | (df[split_col].isna() & (df["method"] == "colmap"))
                ]

        if config.get("split_dataset") == "individual" and "dataset_val" in config:
            plot_df = plot_df[plot_df["dataset"] == config["dataset_val"]]
        elif config.get("split_dataset") == "collection" and "dataset_val" in config:
            plot_df = plot_df[plot_df["dataset_collection"] == config["dataset_val"]]

        if x_axis == "conf_thres_value" and not colmap_df.empty:
            plot_df = plot_df[plot_df["method"] != "colmap"]
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
            single_legend=single_legend,
            plot_raw=plot_raw,
            hue_order=unique_series,
        )

    if single_legend:
        handles_dict = OrderedDict()
        for i in range(len(axes)):
            handles, labels = axes[i].get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in handles_dict:
                    handles_dict[l] = h

        if handles_dict:
            # Reorder handles and labels based on the natural sorted order of unique_series
            ordered_labels = [s for s in unique_series if s in handles_dict]

            # Catch any remaining items (e.g., hline labels not in unique_series)
            ordered_labels.extend([l for l in handles_dict if l not in ordered_labels])
            ordered_handles = [handles_dict[l] for l in ordered_labels]

            processed_labels = [
                l.replace("colmap", "COLMAP")
                .replace("vggt", "VGGT")
                .replace("gt", "GT")
                .replace("combined", "Combined")
                for l in ordered_labels
            ]

            handles = ordered_handles
            labels = processed_labels

            fig_width = fig.get_figwidth()
            max_label_length = max([len(l) for l in labels] + [0])

            estimated_item_width = (max_label_length + 5) * 0.07
            allowed_cols = max(1, int(fig_width / estimated_item_width))
            ncol = min(len(handles), allowed_cols)

            remainder = len(handles) % ncol
            if remainder > 0:
                pad_amount = ncol - remainder
                dummy_handle = mlines.Line2D([], [], linestyle="")
                handles.extend([dummy_handle] * pad_amount)
                labels.extend([""] * pad_amount)

            nrow = len(handles) // ncol
            handles_row_first = [handles[r * ncol + c] for c in range(ncol) for r in range(nrow)]
            labels_row_first = [labels[r * ncol + c] for c in range(ncol) for r in range(nrow)]

            fig.legend(
                handles=handles_row_first,
                labels=labels_row_first,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=ncol,
                markerscale=1.0,
            )
    else:
        for i in range(len(axes)):
            handles, labels = axes[i].get_legend_handles_labels()
            processed_labels = [
                label.replace("colmap", "COLMAP")
                .replace("vggt", "VGGT")
                .replace("gt", "GT")
                .replace("combined", "Combined")
                for label in labels
            ]
            if handles:
                axes[i].legend(handles=handles, labels=processed_labels, markerscale=1.0)

    suffix = f"plots/{prefix}/full_evaluation-{name}-{x_axis}-{split_param}"

    os.makedirs(os.path.dirname(suffix), exist_ok=True)
    serial_df = df.copy()

    if "raw_metrics" in serial_df.columns:
        serial_df = serial_df.drop("raw_metrics", axis=1)
    if "raw_eval_metrics" in serial_df.columns:
        serial_df = serial_df.drop("raw_eval_metrics", axis=1)

    csv_out_file = f"{suffix}.csv"
    os.makedirs(os.path.dirname(csv_out_file), exist_ok=True)
    serial_df.to_csv(csv_out_file, index=False)
    print("Dataframe saved:", Path(csv_out_file))

    json_out_file = f"{suffix}.json"
    os.makedirs(os.path.dirname(json_out_file), exist_ok=True)
    serial_df.to_json(json_out_file, orient="records", indent=4)
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
    plt.savefig(out_file, dpi=100, bbox_inches="tight")
    plt.savefig(str(Path(out_file).with_suffix(".pdf")), bbox_inches="tight", metadata={"CreationDate": None})
    print("Comprehensive plot saved:", Path(out_file), "and PDF")

    pcp_out_file = f"{suffix}_pcp.png"
    if create_pcp:
        plot_pcp(df, pcp_out_file, color_map, title=title if print_title else None)

    combo_out_file = f"{suffix}_combos.png"
    if create_combinations:
        plot_metric_combinations(
            df,
            combo_out_file,
            color_map,
            marker_map,
            x_axis,
            title=title if print_title else None,
            single_legend=single_legend,
        )

    if render_folders is not None:
        render_out_base = Path(suffix + "_renders")
        # Use render_folders to filter df based on "folder" column
        render_df = df[df["folder"].isin(render_folders)]

        if not render_df.empty:
            render_df = render_df[render_df["val_step"] == max(render_df["val_step"])]

        if not render_df.empty:
            create_render_figure(
                render_df,
                render_out_base,
                title=title,
                dataset_name=dataset_name,
                experiment_name=experiment_name,
                prefix=prefix,
                x_axis=x_axis,
                max_cols=max_render_cols,
                show_depth=show_depth,
                show_gt=show_gt,
                render_nums=render_nums,
                stack_dataset_renders_horizontally=stack_datasets_horizontally,
            )
    if camera_folders is not None and make_camera_plot:
        camera_df = df[df["input_folder"].isin(camera_folders)]
        if not camera_df.empty:
            plot_cameras(
                camera_df, Path(suffix + "_cameras"), x_axis=x_axis, use_error_colors=True, max_cols=max_render_cols
            )

    if camera_folders is not None and make_pcd_plot:
        pcd_df = df[df["input_folder"].isin(camera_folders)]
        if not pcd_df.empty:
            plot_point_clouds(pcd_df, Path(suffix + "_pcd"), x_axis=x_axis, max_cols=max_render_cols * 2)

    if dataset_name and experiment_name:
        latest_suffix = f"latest_plots/{experiment_name}_{dataset_name}_latest"
        os.makedirs(os.path.dirname(latest_suffix), exist_ok=True)
        latest_full_png = f"{latest_suffix}_full.png"
        shutil.copy2(out_file, latest_full_png)
        print("Latest copy saved:", Path(latest_full_png))
        latest_full_pdf = f"{latest_suffix}_full.pdf"
        shutil.copy2(str(Path(out_file).with_suffix(".pdf")), latest_full_pdf)
        print("Latest copy saved:", Path(latest_full_pdf))

        # Cameras saved to Path(suffix + "_cameras") / "camera_alignment.png"
        # and Path(suffix + "_cameras") / "camera_alignment.pdf"

        latest_camera_png = f"{latest_suffix}_camera_alignment.png"
        camera_png_file = Path(suffix + "_cameras") / "camera_alignment.png"
        if camera_png_file.exists():
            shutil.copy2(camera_png_file, latest_camera_png)
            print("Latest copy saved:", Path(latest_camera_png))
        latest_camera_pdf = f"{latest_suffix}_camera_alignment.pdf"
        camera_pdf_file = Path(suffix + "_cameras") / "camera_alignment.pdf"
        if camera_pdf_file.exists():
            shutil.copy2(camera_pdf_file, latest_camera_pdf)
            print("Latest copy saved:", Path(latest_camera_pdf))
            tex_out_path = str(Path(latest_camera_pdf).with_suffix(".tex"))
            latex_caption = f"{title} ({str(dataset_name).title()})." if title else f"{prefix} - {dataset_name}."
            latex_label = (
                f"fig:cameras_{experiment_name}_{dataset_name}"
                if experiment_name
                else f"fig:cameras_{prefix}_{dataset_name}"
            )
            save_figure_tex(
                tex_out_path,
                "Images/04-Results/Renders/" + Path(latest_camera_pdf).name,
                caption=latex_caption,
                label=latex_label,
            )

        if make_pcd_plot:
            latest_pcd_png = f"{latest_suffix}_point_clouds.png"
            pcd_png_file = Path(suffix + "_pcd") / "point_clouds.png"
            if pcd_png_file.exists():
                shutil.copy2(pcd_png_file, latest_pcd_png)
                print("Latest copy saved:", Path(latest_pcd_png))
            latest_pcd_pdf = f"{latest_suffix}_point_clouds.pdf"
            pcd_pdf_file = Path(suffix + "_pcd") / "point_clouds.pdf"
            if pcd_pdf_file.exists():
                shutil.copy2(pcd_pdf_file, latest_pcd_pdf)
                print("Latest copy saved:", Path(latest_pcd_pdf))
                tex_out_path = str(Path(latest_pcd_pdf).with_suffix(".tex"))
                latex_caption = f"{title} ({str(dataset_name).title()})." if title else f"{prefix} - {dataset_name}."
                latex_label = (
                    f"fig:pointclouds_{experiment_name}_{dataset_name}"
                    if experiment_name
                    else f"fig:pointclouds_{prefix}_{dataset_name}"
                )
                save_figure_tex(
                    tex_out_path,
                    "Images/04-Results/PointClouds/" + Path(latest_pcd_pdf).name,
                    caption=latex_caption,
                    label=latex_label,
                )

        latex_caption = f"{title} ({str(dataset_name).title()})." if title else f"{prefix} - {dataset_name}."
        latex_label = f"fig:{experiment_name}_{dataset_name}" if experiment_name else f"fig:{prefix}_{dataset_name}"

        save_figure_tex(
            str(Path(latest_full_pdf).with_suffix(".tex")),
            "Images/04-Results/" + Path(latest_full_pdf).name,
            caption=latex_caption,
            label=latex_label,
            fig_width=min(1.0, cols / 3),
        )

        print("LaTeX figure saved:", Path(str(Path(latest_full_pdf).with_suffix(".tex"))))

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

    if create_table:
        plot_table(
            name=original_name,
            prefix=prefix,
            x_axis=x_axis,
            split_param=split_param,
            filter=filter,
            folders=folders,
            val_steps=val_steps,
            title=title,
            metric_keys=metric_keys,
            metrics=metrics,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            split_choice=split_choice,
            split_dataset=split_dataset,
        )


def plot_table(
    name: str | list[str],
    prefix: str,
    x_axis: str,
    split_param: str | None = None,
    filter: str | None = None,
    folders: list[tuple[str, str]] | None = None,
    val_steps: list[int] = [7000],
    title: str | None = None,
    metric_keys: list[str] = ["rre", "rte", "psnr", "lpips", "ssim", "num_GS"],
    metrics: dict[str, dict[str, str]] = {},
    dataset_name: str | None = None,
    experiment_name: str | None = None,
    split_choice: bool = False,
    split_dataset: Literal["none", "individual", "collection"] | bool = "none",
):
    if isinstance(split_dataset, bool):
        split_dataset = "individual" if split_dataset else "none"

    dynamic_rounding = False
    df = load_metrics_to_df(name, methods=["colmap", "vggt", "gt", "combined"], folders=folders, val_steps=val_steps)

    if df.empty:
        print("No data to plot")
        return

    # Apply filters
    if filter:
        filters = parse_filters(filter)
        for col, vals in filters.items():
            if col in df.columns:
                df = df[df[col].isin(vals)]

    # Select relevant columns: choice, x_axis, split params, and metrics
    columns_to_include = ["choice"]

    if split_dataset == "individual" and "dataset" in df.columns:
        columns_to_include.append("dataset")
    elif split_dataset == "collection" and "dataset_collection" in df.columns:
        columns_to_include.append("dataset_collection")

    if x_axis:
        columns_to_include.append(x_axis)

    if split_param:
        split_cols = [p.strip() for p in split_param.split(",") if p.strip()]
        columns_to_include.extend(split_cols)
    else:
        split_cols = []

    columns_to_include.extend(metric_keys)

    # Remove duplicates
    columns_to_include = list(dict.fromkeys(columns_to_include))
    # Keep only existing columns
    columns_to_include = [col for col in columns_to_include if col in df.columns]

    non_metric_cols = [col for col in columns_to_include if col not in metric_keys or col in split_cols]

    # Take the mean of all metrics based on non_metric_cols
    df_for_agg = df[columns_to_include]
    df_table = df_for_agg.groupby(non_metric_cols, dropna=False, observed=True).mean().reset_index()

    # Format method names for presentation (feature from plot_graph)
    if "choice" in df_table.columns:
        df_table["choice"] = df_table["choice"].replace(
            {"colmap": "COLMAP", "vggt": "VGGT", "gt": "GT", "combined": "Combined"}
        )

    # Sort the dataframe
    sort_cols = []
    if split_dataset == "individual" and "dataset" in df_table.columns:
        sort_cols.append("dataset")
    elif split_dataset == "collection" and "dataset_collection" in df_table.columns:
        sort_cols.append("dataset_collection")

    if split_choice and "choice" in df_table.columns:
        sort_cols.append("choice")
    elif "choice" in df_table.columns and "choice" not in sort_cols:
        sort_cols.append("choice")

    if x_axis and x_axis in df_table.columns:
        sort_cols.append(x_axis)
    sort_cols.extend([c for c in split_cols if c in df_table.columns])
    sort_cols.extend(["method", "seed"])
    sort_cols = [col for col in sort_cols if col in df_table.columns]
    sort_cols = list(dict.fromkeys(sort_cols))  # Remove any duplicates preserving order

    if sort_cols:
        df_table = df_table.sort_values(by=sort_cols)

    # After grouping and averaging, some integer columns might become float.
    # Let's try to convert them back if they are whole numbers.
    for col in df_table.columns:
        if pd.api.types.is_float_dtype(df_table[col]):
            # Check if all non-NaN values are whole numbers
            if (df_table[col].dropna() % 1 == 0).all():
                # Using Int64 to support NaNs
                df_table[col] = df_table[col].astype("Int64")

    rename_map = {}
    if "choice" in df_table.columns:
        rename_map["choice"] = "Choice"

    if "dataset" in df_table.columns:
        rename_map["dataset"] = "Dataset"

    if "dataset_collection" in df_table.columns:
        rename_map["dataset_collection"] = "Dataset Collection"

    if x_axis and x_axis in df_table.columns:
        rename_map[x_axis] = x_axis.replace("_", " ").title()

    if split_param:
        for c in split_cols:
            if c in df_table.columns:
                rename_map[c] = c.replace("_", " ").title()

    for m in metric_keys:
        if m in df_table.columns and m in metrics:
            metric_title = metrics[m]["title"]
            if m == "eval_rre":
                metric_title = metric_title.replace("Validation Step", "Val")
            if m == "eval_rte":
                metric_title = metric_title.replace("Validation Step", "Val")
            rename_map[m] = metric_title

    df_table_renamed = df_table.rename(columns=rename_map)
    inv_rename_map = {v: k for k, v in rename_map.items()}

    # Dynamic variables for LaTeX table
    latex_caption = f"{title} ({str(dataset_name).title()})" if title else f"{prefix} - {dataset_name}"
    latex_label = f"tab:{experiment_name}_{dataset_name}" if experiment_name else f"tab:{prefix}_{dataset_name}"

    # Determine split keys for formatting groups
    split_keys = []
    if split_dataset == "individual" and "Dataset" in df_table_renamed.columns:
        split_keys.append("Dataset")
    elif split_dataset == "individual" and "dataset" in df_table_renamed.columns:
        split_keys.append("dataset")
    elif split_dataset == "collection" and "Dataset Collection" in df_table_renamed.columns:
        split_keys.append("Dataset Collection")
    elif split_dataset == "collection" and "dataset_collection" in df_table_renamed.columns:
        split_keys.append("dataset_collection")

    if split_choice and "Choice" in df_table_renamed.columns:
        split_keys.append("Choice")
    elif split_choice and "choice" in df_table_renamed.columns:
        split_keys.append("choice")

    if split_keys:
        configs = df_table_renamed[split_keys].drop_duplicates().to_dict("records")
    else:
        configs = [{}]

    # Generate column format string from full df
    column_format = ""
    for col_name in df_table_renamed.columns:
        if pd.api.types.is_numeric_dtype(df_table_renamed[col_name]):
            column_format += "r"
        else:
            column_format += "l"

    latex_bodies = []
    for config in configs:
        mask = pd.Series(True, index=df_table_renamed.index)
        for k, v in config.items():
            mask &= df_table_renamed[k] == v
        sub_df = df_table_renamed[mask].copy()

        # Format sub_df
        for col_name in sub_df.columns:
            col = sub_df[col_name]
            original_metric_key = inv_rename_map.get(col_name)
            is_metric = original_metric_key and original_metric_key in metrics

            if pd.api.types.is_numeric_dtype(df_table_renamed[col_name]):
                is_integer = pd.api.types.is_integer_dtype(df_table_renamed[col_name])

                best_val, second_best_val = None, None
                if is_metric and pd.api.types.is_numeric_dtype(col):
                    metric_info = metrics.get(original_metric_key, {})
                    direction = metric_info.get("direction")
                    if direction:
                        sorted_vals = col.dropna().sort_values(ascending=(direction == "↓")).round(3)
                        unique_sorted_vals = sorted_vals.unique()
                        if len(unique_sorted_vals) > 0:
                            best_val = unique_sorted_vals[0]
                        if len(unique_sorted_vals) > 1:
                            second_best_val = unique_sorted_vals[1]

                formatted_col = []
                for val in col:
                    if pd.isna(val):
                        formatted_col.append("")
                        continue
                    s = f"{val:d}" if is_integer else (f"{val:.3g}" if dynamic_rounding else f"{val:.3f}")
                    if best_val is not None and np.isclose(round(val, 3), best_val):
                        formatted_col.append(f"\\textbf{{{s}}}")
                    elif second_best_val is not None and np.isclose(round(val, 3), second_best_val):
                        formatted_col.append(f"\\underline{{{s}}}")
                    else:
                        formatted_col.append(s)
                sub_df[col_name] = formatted_col
            else:
                sub_df[col_name] = col.apply(lambda x: str(x) if pd.notna(x) else "")

        # Get body latex
        sub_latex = sub_df.to_latex(index=False, header=False, escape=False, na_rep="")
        body_lines = []
        for line in sub_latex.splitlines():
            if line.strip().endswith(r"\\"):
                body_lines.append(line)
        latex_bodies.append("\n".join(body_lines))

    # Build full table latex using a dummy df to fetch the header and footer robustly
    dummy_df = df_table_renamed.head(1).copy()
    dummy_marker = "DUMMY_ROW_MARKER_12345"
    dummy_df.iloc[0, 0] = dummy_marker
    dummy_latex = dummy_df.to_latex(
        index=False,
        column_format=column_format,
        escape=False,
        caption=latex_caption,
        label=latex_label,
        position="H",
        na_rep="",
    )

    lines = dummy_latex.splitlines()
    header_lines = []
    footer_lines = []
    found_dummy = False

    for line in lines:
        if dummy_marker in line:
            found_dummy = True
            continue
        if not found_dummy:
            header_lines.append(line)
        else:
            footer_lines.append(line)

    header_part = "\n".join(header_lines)
    footer_part = "\n".join(footer_lines)

    rule_cmd = r"\midrule"
    if r"\midrule" not in header_part and r"\hline" in header_part:
        rule_cmd = r"\hline"

    joined_bodies = f"\n{rule_cmd}\n".join(latex_bodies)
    latex_table = f"{header_part}\n{joined_bodies}\n{footer_part}"

    latex_table = latex_table.replace("\\begin{tabular}", "\\small\n\\begin{tabular}")

    # Save to file
    suffix = f"plots/{experiment_name}_{dataset_name}_{prefix}"
    os.makedirs(os.path.dirname(suffix), exist_ok=True)

    tex_out_file = f"{suffix}.tex"
    os.makedirs(os.path.dirname(tex_out_file), exist_ok=True)
    with open(tex_out_file, "w") as f:
        f.write(latex_table)
    print("LaTeX table saved:", Path(tex_out_file))

    # Also save CSV for reference
    csv_out_file = f"{suffix}.csv"
    df_table.to_csv(csv_out_file, index=False)
    print("CSV table saved:", Path(csv_out_file))

    if dataset_name and experiment_name:
        latest_suffix = f"latest_plots/{experiment_name}_{dataset_name}_latest"
        os.makedirs(os.path.dirname(latest_suffix), exist_ok=True)
        latest_tex = f"{latest_suffix}_table.tex"
        shutil.copy2(tex_out_file, latest_tex)
        print("Latest LaTeX table saved:", Path(latest_tex))
        latest_csv = f"{latest_suffix}_table.csv"
        shutil.copy2(csv_out_file, latest_csv)
        print("Latest CSV table saved:", Path(latest_csv))


if __name__ == "__main__":
    main()
