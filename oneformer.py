import torch
import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image

# Bypass the cuDNN mismatch error to avoid a lengthy environment reinstallation
torch.backends.cudnn.enabled = False

# 1. Set up the device and load the OneFormer model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Using the ADE20k Swin-Tiny model for general purpose universal segmentation
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny").to(device)

# 2. Load and prepare your image
# image_path = "data/360_v2/bicycle/images_4/_DSC8679.JPG"
image_path = "data/360_v2/bonsai/images_2/DSCF5565.JPG"
image = Image.open(image_path).convert("RGB")

# 3. Run inference
# OneFormer uses task tokens. We use "panoptic" to segment both distinct objects ("things")
# and background regions ("stuff") so no pixels are left unsegmented.
inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# 5. Extract the masks
# Post-process to get the panoptic segmentation map sized to the original image
panoptic_segmentation = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[image.size[::-1]]  # Requires [height, width] format
)[0]

# OneFormer returns a single 2D tensor where each unique pixel value corresponds to a different segmented region
segmentation_map = panoptic_segmentation["segmentation"].cpu().numpy()
unique_segments = np.unique(segmentation_map)

print(f"Successfully extracted {len(unique_segments)} distinct regions!")

# 6. Generate random colors and overlay masks onto the original image
# Convert the original PIL image to an RGBA numpy array
image_rgba = np.array(image.convert("RGBA"))
overlay = np.zeros_like(image_rgba)

alpha = 0.5  # Transparency of the masks (0.0 is invisible, 1.0 is completely solid)

# Iterate through every unique segment ID found in the image
for seg_id in unique_segments:
    # Generate a random bright RGB color
    color = np.random.randint(0, 255, size=3, dtype=np.uint8)

    # Find valid mask pixels for this specific segment
    valid_pixels = segmentation_map == seg_id

    # Apply the color and alpha transparency to the overlay layer
    overlay[valid_pixels, :3] = color
    overlay[valid_pixels, 3] = int(255 * alpha)

# Composite the overlay onto the original image
overlay_img = Image.fromarray(overlay, "RGBA")
final_result = Image.alpha_composite(image.convert("RGBA"), overlay_img)

# Save the output
output_path = "oneformer_overlay_result.jpg"
final_result.convert("RGB").save(output_path)
print(f"Overlay saved successfully to {output_path}")
