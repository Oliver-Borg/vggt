from time import time

import torch
import numpy as np
from transformers import pipeline
from PIL import Image

torch.backends.cudnn.enabled = False
# 1. Set up the device and load the SAM 2 pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"

# The mask-generation pipeline automatically handles grid generation to extract all masks
generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=device)

# 2. Load and prepare your image
# image_path = "data/360_v2/bicycle/images_4/_DSC8679.JPG"
image_path = "data/360_v2/bonsai/images_2/DSCF5565.JPG"
image = Image.open(image_path).convert("RGB")

# 3. Run inference
# SAM 2 does not use text prompts for automatic mask generation.
# Instead, the pipeline automatically grids the image to segment everything.
# 'points_per_batch' controls parallelization (lower it if you run out of GPU memory).
outputs = generator(
    image,
    points_per_batch=64,
    pred_iou_thresh=0.75,  # Default is typically higher; lower accepts less confident objects
    stability_score_thresh=0.80,  # Lower accepts masks with less strictly defined borders
)

# 5. Extract the masks
masks = outputs["masks"]
scores = outputs["scores"]

print(f"Successfully extracted {len(masks)} masks!")

# 6. Generate random colors and overlay masks onto the original image
# Convert the original PIL image to an RGBA numpy array
image_rgba = np.array(image.convert("RGBA"))
overlay = np.zeros_like(image_rgba)

alpha = 0.5  # Transparency of the masks (0.0 is invisible, 1.0 is completely solid)

# The pipeline returns a list of tensors/arrays, so we handle them individually
for mask in masks:
    # Convert individual mask to numpy if it's a tensor
    mask_np = mask.cpu().numpy() if hasattr(mask, "cpu") else np.array(mask)

    # Squeeze extra dimensions if present (e.g., [1, H, W] -> [H, W])
    if mask_np.ndim == 3:
        mask_np = np.squeeze(mask_np, axis=0)

    # Generate a random bright RGB color
    color = np.random.randint(0, 255, size=3, dtype=np.uint8)

    # Find valid mask pixels
    valid_pixels = mask_np > 0

    # Apply the color and alpha transparency to the overlay layer
    overlay[valid_pixels, :3] = color
    overlay[valid_pixels, 3] = int(255 * alpha)

# Composite the overlay onto the original image
overlay_img = Image.fromarray(overlay, "RGBA")
final_result = Image.alpha_composite(image.convert("RGBA"), overlay_img)

# Save the output
output_path = "sam2_overlay_result_bonsai.jpg"
final_result.convert("RGB").save(output_path)
print(f"Overlay saved successfully to {output_path}")
