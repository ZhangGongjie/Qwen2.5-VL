import json
import random
import io
import ast
import os
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET

import torch
from transformers import AutoProcessor
from qwenvl.model.qwen2_5_3dvl import Qwen2_5_3DVL_ForConditionalGeneration
from qwenvl.data.utils_3dvl import get_coord3d, resize_coord3d_resize, coord3d_to_flat_patches
from qwenvl.data.data_qwen import smart_resize


model_path = "./Qwen2.5-VL-3B-Instruct-SFT3D-coco3d-scannet2d-stage2"

model = Qwen2_5_3DVL_ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)


# --- Example Usage ---

# Define paths for your example data (replace with your actual paths)
image_path = "data/coco_3d/val2017/000000030213.jpg"
depth_path = "data/coco_3d/depth/000000030213_remove_edges.png"
cam_params_path = "data/coco_3d/camera_parameters/000000030213.json" # or .txt for ScanNet format

# Check if example files exist (optional, but recommended)
if not all(os.path.exists(p) for p in [image_path, depth_path, cam_params_path]):
    print("Warning: Example data paths do not exist. Please update them in inference.py")
    # You might want to exit or provide default dummy data here
    # For now, we'll proceed assuming the paths will be corrected by the user.

# 1. Load Image
try:
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}. Exiting.")
    exit()

# 2. Compute initial 3D coordinates
#    Note: get_coord3d needs the original image for dimensions, but reads depth/cam params itself.
print("Computing initial 3D coordinates...")
try:
    coord3d_original = get_coord3d(image, depth_path, cam_params_path) # Shape: (orig_H, orig_W, 3)
except FileNotFoundError:
    print(f"Error: Depth or camera parameters file not found. Check paths: {depth_path}, {cam_params_path}. Exiting.")
    exit()
except Exception as e:
    print(f"Error computing 3D coordinates: {e}. Exiting.")
    exit()

# 3. Prepare Text Prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            # --- Replace with your desired prompt ---
            {"type": "text", "text": "Where is the wodden bucket on the floor? Please provide 3D Coord."},
            # -----------------------------------------
        ],
    },
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Process Image and Text with Processor
print("Processing image and text...")
# Note: The processor handles resizing the image according to its internal logic (using smart_resize implicitly)
#       It also prepares input_ids, attention_mask, pixel_values, and image_grid_thw
inputs = processor(text=[text], images=[image], return_tensors="pt")

# 5. Determine Resized Image Dimensions and Resize Coordinate Map
pixel_values = inputs['pixel_values'] # Shape: (1, 3, H, W) - already resized by processor
resized_height, resized_width = pixel_values.shape[2], pixel_values.shape[3]

print(f"Resizing coordinate map to {resized_height}x{resized_width}...")
coord3d_resized = resize_coord3d_resize(coord3d_original, (resized_height, resized_width), mode="bilinear") # Shape: (H, W, 3)

# 6. Flatten Coordinate Map into Patches
print("Flattening coordinate map into patches...")
# Get patch/merge sizes from the processor's vision config or data_args used during training
# Assuming defaults used in Qwen2.5VL if not available directly
patch_size = getattr(processor, 'patch_size', 14) # Default Qwen 2.5 VL patch size
merge_size = getattr(processor, 'merge_size', 2)   # Default Qwen 2.5 VL merge size
temporal_patch_size = getattr(processor, 'temporal_patch_size', 2) # Default 1 for images

coord3d_flat_patches = coord3d_to_flat_patches(
    coord3d_resized,
    patch_size=patch_size,
    merge_size=merge_size,
    temporal_patch_size=temporal_patch_size
) # Shape: (num_patches, channels * temporal_patch_size * patch_size^2)
coord3d = torch.tensor(coord3d_flat_patches) # Convert to tensor

# Ensure grid_thw matches the structure expected by coord3d_to_flat_patches
# The processor returns grid_thw which should align if parameters match
assert coord3d.shape[0] == inputs['image_grid_thw'][0].prod().item() // (merge_size**2), \
    f"Mismatch between flattened coord3d patches ({coord3d.shape[0]}) and expected patches based on image_grid_thw ({inputs['image_grid_thw'][0].prod().item() // (merge_size**2)})"


# 7. Move inputs to device
print("Moving inputs to device...")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
coord3d = coord3d.to(model.device, dtype=model.dtype) # Ensure correct dtype and device

# 8. Generate Response
print("Generating response...")
with torch.no_grad():
    generate_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pixel_values=inputs['pixel_values'],
        image_grid_thw=inputs['image_grid_thw'],
        coord3d=coord3d, # Pass the processed 3D coordinates
        max_new_tokens=1024,
        do_sample=False,
        # Add other generation parameters as needed (temperature, top_k, etc.)
    )

# 9. Decode and Print Output
print("Decoding output...")
decoded_output = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

# --- Optional: Clean up the output ---
# Find the assistant's response start
assistant_prompt = "<|im_start|>assistant\n"
response_start_index = decoded_output.find(assistant_prompt)
if response_start_index != -1:
    response_text = decoded_output[response_start_index + len(assistant_prompt):]
else:
    response_text = decoded_output # Fallback if pattern not found

# Remove end token if present
end_token = "<|im_end|>"
if response_text.endswith(end_token):
    response_text = response_text[:-len(end_token)].strip()

# --- Print the final response ---
print("\n--- Model Response ---")
print(response_text)
print("--------------------\n")

