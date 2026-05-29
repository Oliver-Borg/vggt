import os
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt


def np_rgba(np_arr: np.ndarray, cmap: str, vmax: float | None = None) -> np.ndarray:
    """
    Convert a grayscale image to RGBA using a matplotlib colormap
    Args:
        np_arr (np.ndarray): The grayscale image
        cmap (str): The name of the matplotlib colormap to use
    Returns:
        np.ndarray: The RGBA image
    """

    max_value = np_arr.max() if vmax is None else vmax
    if max_value == 0:
        max_value = 1e-5

    normalised = np_arr.astype(np.float32) / max_value
    normalised = np.clip(normalised, 0.0, 1.0)
    mapper = plt.get_cmap(cmap)
    rgba = mapper(normalised)
    rgba = rgba * 255
    rgba = rgba.astype(np.uint8)
    # rgba[np_arr == 0] = [21, 59, 106, 255] #153b6a
    return rgba


def np_rgb(np_arr: np.ndarray, cmap: str = "viridis", vmax: float | None = None) -> np.ndarray:
    """
    Convert a grayscale image to RGB using a matplotlib colormap
    Args:
        np_arr (np.ndarray): The grayscale image
        cmap (str): The name of the matplotlib colormap to use
    Returns:
        np.ndarray: The RGB image
    """
    rgba = np_rgba(np_arr, cmap, vmax=vmax)
    return rgba[..., :3]


def add_text_to_image(
    img: Image.Image,
    text: str,
    fontsize: int,
) -> Image.Image:
    font = ImageFont.load_default(size=fontsize)
    # Use a temporary draw object to calculate the text bounding box
    draw_temp = ImageDraw.Draw(img)
    bbox = draw_temp.textbbox((0, 0), text, font=font, spacing=0)
    # text_width = bbox[2] - bbox[0]
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


def stack_images_with_wrap(images: list[Image.Image], max_cols: int = 3):
    rows = [images[i: i + max_cols] for i in range(0, len(images), max_cols)]

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


def save_figure_tex(
    tex_out_file: str,
    highres_path: str,
    caption: str,
    label: str,
    fig_width: float = 1.0,
    lowres_path: str | None = None,
):
    """
    Generates a LaTeX figure block and saves it to a specified .tex file.
    """

    latex_figure = (
        (
            "\\begin{figure}[H]\n"
            "    \\centering\n"
            "    \\ifhighres\n"
            f"        \\includegraphics[width={fig_width}\\linewidth]{{{highres_path}}}\n"
            "    \\else\n"
            f"        \\includegraphics[width={fig_width}\\linewidth]{{{lowres_path}}}\n"
            "    \\fi\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{{label}}}\n"
            "\\end{figure}\n"
        )
        if lowres_path is not None
        else (
            "\\begin{figure}[H]\n"
            "    \\centering\n"
            f"    \\includegraphics[width={fig_width}\\linewidth]{{{highres_path}}}\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{{label}}}\n"
            "\\end{figure}\n"
        )
    )

    os.makedirs(os.path.dirname(tex_out_file), exist_ok=True)
    with open(tex_out_file, "w") as f:
        f.write(latex_figure)
    print("LaTeX figure saved:", Path(tex_out_file))
