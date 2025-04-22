import json
import random
import io
import ast
import os
from PIL import Image, ImageDraw, ImageFont
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET

import torch
from transformers import AutoProcessor
from qwenvl.model.qwen2_5_3dvl import Qwen2_5_3DVL_ForConditionalGeneration
from qwenvl.data.utils_3dvl import get_coord3d, resize_coord3d_resize, coord3d_to_flat_patches
from qwenvl.data.data_qwen import smart_resize


# Define constants (matching data_qwen.py)
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>" # Placeholder in user message content
IMAGE_PAD_TOKEN = "<|image_pad|>" # Actual token placeholder for the model
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"


model_path = "./Qwen2.5-VL-3B-Instruct-SFT3D-coco3d-scannet2d-stage2"

model = Qwen2_5_3DVL_ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)


# Set the model's image token
image_token = getattr(processor, 'image_token', "<image>")
print(f"Using image token: {image_token}")

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
    print(f"Original image dimensions: {orig_width}x{orig_height}")
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}. Exiting.")
    exit()

# 2. Prepare Text Prompt
user_prompt = "Where is the wodden bucket on the floor? Please provide 3D coordinate."
print(f"User prompt: {user_prompt}")

# Create a simple prompt with image token
# The processor will automatically handle the image token replacement
# For Qwen models we use this format
prompt = f"<|im_start|>user\n{image_token}{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
print(f"Formatted prompt with image token: {prompt}")

# 3. Process Text and Image Together
print("Processing text and image together...")
inputs = processor(text=[prompt], images=[image], return_tensors="pt")
image_grid_thw = inputs['image_grid_thw']  # Shape: (num_images, 3)
print(f"Processed image_grid_thw: {image_grid_thw}")

# Get patch/merge sizes (can take from processor or hardcode based on training)
patch_size = 14  # Default Qwen 2.5 VL patch size
merge_size = 2   # Default Qwen 2.5 VL merge size
temporal_patch_size = 2  # Default for Qwen2.5VL 3D

# Calculate the number of merged patches
num_merged_patches = image_grid_thw[0].prod().item() // (merge_size**2)
print(f"Calculated number of merged patches: {num_merged_patches}")

# 4. Compute initial 3D coordinates
print("Computing initial 3D coordinates...")
try:
    coord3d_original = get_coord3d(image, depth_path, cam_params_path) # Shape: (orig_H, orig_W, 3)
except FileNotFoundError:
    print(f"Error: Depth or camera parameters file not found. Check paths: {depth_path}, {cam_params_path}. Exiting.")
    exit()
except Exception as e:
    print(f"Error computing 3D coordinates: {e}. Exiting.")
    exit()

# 5. Resize and Flatten Coordinate Map
factor = patch_size * merge_size
min_pixels, max_pixels = 784, 50176

resized_height, resized_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)


print(f"Resizing coordinate map to {resized_height}x{resized_width}...")
coord3d_resized = resize_coord3d_resize(coord3d_original, (resized_height, resized_width), mode="bilinear") # Shape: (H, W, 3)

    coord3d_resized,
    patch_size=patch_size,
    merge_size=merge_size,
    temporal_patch_size=temporal_patch_size
) # Shape: (num_patches, channels * temporal_patch_size * patch_size^2)
coord3d = torch.tensor(coord3d_flat_patches) # Convert to tensor
coord3d = coord3d.to(inputs["pixel_values"].device)

# Verify patch count consistency
print(f"Flattened coord3d shape: {coord3d.shape}")
print(f"Expected patches based on grid_thw: {num_merged_patches}")
try:
    assert coord3d.shape[0] == num_merged_patches, \
        f"Mismatch between flattened coord3d patches ({coord3d.shape[0]}) and calculated merged patches ({num_merged_patches})"
    print("Patch count verification successful!")
except AssertionError as e:
    print(f"Warning: {e}")
    print("Continuing anyway, but results may be incorrect.")

# 6. Move inputs to device
print("Moving inputs to device...")
inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
coord3d = coord3d.to(model.device, dtype=model.dtype) # Ensure correct dtype and device

# 7. Generate Response
print("Generating response...")
with torch.no_grad():
model_inputs = {
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask'],
    'pixel_values': inputs['pixel_values'],
    'image_grid_thw': inputs['image_grid_thw'],
    'coord3d': coord3d
}

print("Input shapes:")
for k, v in model_inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

with torch.no_grad():
    generate_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id
    )

# 8. Decode and Print Output
print("Decoding output...")
decoded_output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
response_text = decoded_output.strip()

# --- Print the final response ---
print("\n--- Model Response ---")
print(response_text)
print("--------------------\n")

