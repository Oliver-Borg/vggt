import numpy as np
import matplotlib.pyplot as plt


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
