import os

from line_profiler import profile
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from PIL import Image

from .plot_utils import add_text_to_image, np_rgb, resize_with_padding, save_figure_tex, stack_images_with_wrap


def get_bbox_2d(arr):
    mask = ~np.isnan(arr)
    if not np.any(mask):
        return (0, 0, 0, 0)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def _process_single_depth_conf(
    row: dict,
    input_folder: Path,
    x_axis: str | None = None,
    render_num: int = 0,
    image_width: int = 1200,
    image_height: int = 800,
    dataset_name: str = "default",
):
    from PIL import ImageOps

    images_out = []
    images_dir = input_folder / "images"
    depths_dir = input_folder / "depths"

    if not images_dir.exists() or not depths_dir.exists():
        return images_out

    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    img_files = sorted([p for p in images_dir.iterdir() if p.suffix in valid_exts])

    if not img_files or render_num >= len(img_files):
        return images_out

    img_path = img_files[render_num]
    img_name = img_path.name

    depth_path = depths_dir / f"depth_{img_name}.npy"
    conf_path = depths_dir / f"raw_conf_{img_name}.npy"

    # try:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    rmin, rmax, cmin, cmax = 0, h - 1, 0, w - 1

    unscaled_max_depth = 0.0
    unscaled_max_conf = 0.0

    if depth_path.exists():
        depth_arr = np.load(depth_path)
        depth_h, depth_w = depth_arr.shape[:2]
        h_scale = h / depth_h
        w_scale = w / depth_w
        if h_scale != w_scale:
            rmin, rmax, cmin, cmax = get_bbox_2d(depth_arr)
            depth_arr = depth_arr[rmin : rmax + 1, cmin : cmax + 1]
        valid_depths = depth_arr[depth_arr > 0]
        if valid_depths.size > 0:
            min_depth = valid_depths.min()
            unscaled_max_depth = valid_depths.max()
            depth_arr = depth_arr - min_depth
            depth_arr = np.maximum(depth_arr, 0.0)
            median_depth = np.median(depth_arr[depth_arr > 0]) + 1e-5
            depth_arr = depth_arr / median_depth

        vmaxes = {
            "lego": 2.5,
            "drums": 3.0,
            "ship": 2.5,
            "bonsai": 1.5,
            "counter": 5.0,
            "kitchen": 6.0,
            "bicycle": 7.0,
            "garden": 4.0,
            "stump": 20.0,
            "default": 1.0,
        }
        vmax = vmaxes.get(dataset_name, 1.0)
        depth_rgb = np_rgb(depth_arr, "viridis", vmax=vmax)
        zero_depths = depth_arr == 0.0
        depth_rgb[zero_depths, :] = 0
        depth_img = Image.fromarray(depth_rgb).resize((w, h))
    else:
        depth_img = Image.new("RGB", (w, h), (0, 0, 0))

    if conf_path.exists():
        conf_arr = np.load(conf_path)
        conf_arr = conf_arr[rmin : rmax + 1, cmin : cmax + 1]
        unscaled_max_conf = conf_arr.max()
        conf_arr = np.maximum(conf_arr - 1.0, 0.0)
        conf_rgb = np_rgb(conf_arr, "plasma", vmax=max(np.max(conf_arr), 1.0))
        zero_confs = conf_arr <= 0.0
        conf_rgb[zero_confs, :] = 0
        conf_img = Image.fromarray(conf_rgb).resize((w, h))
    else:
        conf_img = Image.new("RGB", (w, h), (0, 0, 0))

    # Alpha blend the heatmaps over the original grayscale image for visual context
    img_gray = img.convert("L").convert("RGB")
    depth_overlay = depth_img  # Image.blend(img_gray, depth_img, 0.6)
    conf_overlay = conf_img  # Image.blend(img_gray, conf_img, 0.6)

    combined_w = w * 3
    combined_img = Image.new("RGB", (combined_w, h))
    combined_img.paste(img, (0, 0))
    combined_img.paste(depth_overlay, (w, 0))
    combined_img.paste(conf_overlay, (w * 2, 0))

    pred_img = resize_with_padding(combined_img, image_width, image_height)

    config_name = (
        str(row.get("plot_series", input_folder.name))
        .replace("colmap", "COLMAP")
        .replace("vggt", "VGGT")
        .replace("gt", "GT")
        .replace("combined", "Combined")
    )
    if x_axis and pd.notna(row.get(x_axis)):
        config_name = f"{config_name} | {x_axis}={row.get(x_axis)}"

    label_text = (
        f"{dataset_name}: {config_name}\nOriginal | "
        f"Depth (max={unscaled_max_depth:.2f}) | Confidence (max={unscaled_max_conf:.2f})"
    )

    final_img = add_text_to_image(pred_img, label_text, 32)
    final_img = ImageOps.expand(final_img, border=4, fill="black")
    final_img = ImageOps.expand(final_img, border=20, fill="white")

    images_out.append(final_img)

    # except Exception as e:
    #     print(f"Error processing depth/conf for {img_name}: {e}")

    return images_out


@profile
def create_depth_conf_figure(
    df,
    dest_base,
    title: str | None = None,
    dataset_name: str | None = None,
    experiment_name: str | None = None,
    prefix: str | None = None,
    x_axis: str | None = None,
    max_cols: int = 3,
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
    image_height = 400 if any({"bicycle", "bonsai"} & set(unique_datasets)) else 600

    for dataset in unique_datasets:
        dataset_df = df[df["dataset"] == dataset]
        images_to_stack = []
        for i, (idx, row) in enumerate(dataset_df.iterrows()):
            file_path = row.get("sfm_file_path", "")
            if not file_path or pd.isna(file_path):
                continue
            input_folder = Path(file_path).parent

            for render_num in render_nums:
                processed_images = _process_single_depth_conf(
                    row=row,
                    input_folder=input_folder,
                    x_axis=x_axis,
                    render_num=render_num,
                    image_width=image_width,
                    image_height=image_height,
                    dataset_name=dataset,
                )
                images_to_stack.extend(processed_images)

        if images_to_stack:
            dataset_columns.append(stack_images_with_wrap(images_to_stack, max_cols=cols_per_dataset))

    if not dataset_columns:
        return

    if stack_dataset_renders_horizontally:
        stacked_img = stack_images_with_wrap(dataset_columns, max_cols=max_cols)
    else:
        stacked_img = stack_images_with_wrap(dataset_columns, max_cols=1)

    out_file = dest_base / "stacked_depth_conf.jpg"
    jpg_scaling_factor = 1
    stacked_img.resize((stacked_img.width // jpg_scaling_factor, stacked_img.height // jpg_scaling_factor)).save(
        out_file, quality=95, optimize=True
    )
    print(f"Saved stacked depth/conf to {out_file}")

    out_pdf = dest_base / "stacked_depth_conf.pdf"
    stacked_img.save(out_pdf, "PDF", resolution=100.0, quality=100, optimize=True)
    print(f"Saved stacked depth/conf PDF to {out_pdf}")

    if dataset_name and experiment_name:
        latest_suffix = f"latest_plots/{experiment_name}_{dataset_name}_latest"
        os.makedirs(os.path.dirname(latest_suffix), exist_ok=True)

        latest_dc_jpg = f"{latest_suffix}_depth_conf.jpg"
        shutil.copy2(out_file, latest_dc_jpg)
        print("Latest copy saved:", Path(latest_dc_jpg))

        latest_dc_pdf = f"{latest_suffix}_depth_conf.pdf"
        shutil.copy2(out_pdf, latest_dc_pdf)
        print("Latest copy saved:", Path(latest_dc_pdf))

        latex_caption = (
            f"{title} Depth and Confidence ({str(dataset_name).title()})."
            if title
            else f"{prefix} - {dataset_name} Depth and Confidence."
        )
        latex_label = (
            f"fig:depthconf_{experiment_name}_{dataset_name}"
            if experiment_name
            else f"fig:depthconf_{prefix}_{dataset_name}"
        )

        tex_out_path = str(Path(latest_dc_pdf).with_suffix(".tex"))
        save_figure_tex(
            tex_out_path,
            "Images/04-Results/Depths/" + Path(latest_dc_pdf).name,
            caption=latex_caption,
            label=latex_label,
            lowres_path="Images/04-Results/Depths/" + Path(latest_dc_jpg).name,
        )
        print("LaTeX figure saved:", Path(tex_out_path))
