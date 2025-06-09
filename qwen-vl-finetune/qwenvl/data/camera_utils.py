import json
import torch
import numpy as np
from typing import Dict, Tuple, Optional

def load_camera_parameters(camera_params_path: str) -> Optional[Dict]:
    """Load camera parameters from a JSON file.
    
    Args:
        camera_params_path: Path to the camera parameters JSON file
        
    Returns:
        Dictionary containing camera parameters (fx, fy, cx, cy) or None if file not found
    """
    try:
        with open(camera_params_path, 'r') as f:
            params = json.load(f)
            
        # Extract intrinsics matrix
        intrinsics = np.array(params['intrinsics'])
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None

def adjust_camera_parameters(
    params: Dict,
    original_size: Tuple[int, int],
    new_size: Tuple[int, int]
) -> Dict:
    """Adjust camera parameters when image is resized.
    
    Args:
        params: Dictionary containing original camera parameters
        original_size: Original image size (height, width)
        new_size: New image size (height, width)
        
    Returns:
        Dictionary containing adjusted camera parameters
    """
    if params is None:
        return None
        
    h_orig, w_orig = original_size
    h_new, w_new = new_size
    
    # Calculate scaling factors
    scale_h = h_new / h_orig
    scale_w = w_new / w_orig
    
    # Adjust focal lengths and principal points
    adjusted_params = {
        'fx': params['fx'] * scale_w,
        'fy': params['fy'] * scale_h,
        'cx': params['cx'] * scale_w,
        'cy': params['cy'] * scale_h
    }
    
    return adjusted_params

def generate_camera_aware_position_encoding_grid(
    height: int,
    width: int,
    camera_params: Dict,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate a grid of position embeddings based on camera parameters.
    
    Args:
        height: Image height
        width: Image width
        camera_params: Dictionary containing camera parameters (fx, fy, cx, cy)
        device: Device to place the tensor on
        
    Returns:
        Tensor of shape (height, width, 2) containing camera-aware position encodings
    """
    if camera_params is None:
        # If no camera parameters, return a default grid (all rays pointing forward)
        return torch.zeros((height, width, 3), device=device)
    
    # Create coordinate grids
    v, u = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    
    # Convert to float
    u = u.float()
    v = v.float()
    
    # Calculate ray directions
    x = (u - camera_params['cx']) / camera_params['fx']
    y = (v - camera_params['cy']) / camera_params['fy']
    
    # Stack
    rays = torch.stack([x, y], dim=-1)
    
    return rays 