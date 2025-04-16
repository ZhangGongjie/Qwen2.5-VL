import json
import random
import io
import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET

import torch
from transformers import AutoProcessor
from qwenvl.model.qwen2_5_3dvl import Qwen2_5_3DVL_ForConditionalGeneration


model_path = "./Qwen2.5-VL-3B-Instruct-SFT3D-coco3d-scannet2d-stage2"

model = Qwen2_5_3DVL_ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

