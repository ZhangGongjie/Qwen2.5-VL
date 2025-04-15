import math
import ujson as json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def get_coord3d(image, depth_path, cam_params_path):
    """
    Reads a color (RGB) image, a corresponding depth image, 
    and camera parameters in order to compute 3D coordinates 
    (X, Y, Z) for each pixel.

    Args:
        image (PIL.Image): The RGB image (used only to get width/height).
        depth_path (str): The path to the depth image.
        cam_params_path (str): The path to the camera parameters file.

    Returns:
        torch.Tensor: A tensor of shape (H, W, 3) containing
                      the 3D coordinates for each pixel.
    """
    width, height = image.size

    # 1) Try to load the depth image. If failed, fill with NaNs.
    try:
        depth_image = Image.open(depth_path)
        depth_image = np.array(depth_image, dtype=np.float32)
        # The assumption here is that the depth is stored in millimeters,
        # hence dividing by 1000 to get meters.
        depth_image = depth_image / 1000.0
        depth_image[depth_image == 0] = np.nan
    except Exception as exn:
        print(f"Warning: Failed to open depth image {depth_path}. Exception: {exn}")
        depth_image = np.full((height, width), np.nan, dtype=np.float32)

    # 2) Try to load camera parameters (intrinsics, rotation, translation).
    #    If loading fails, generate default parameters.
    try:
        with open(cam_params_path, "r") as f:
            if cam_params_path.endswith(".json"):
                cam_params = json.load(f)

            elif cam_params_path.endswith(".txt"):  # scannet_2d format
                # 这部分代码用于将 scannet_2d 的相机参数转换成和 .json 中相同的格式
                extrinsics_4x4 = np.loadtxt(cam_params_path)  # 从txt读取外参(4x4)
                intrinsics_4x4 = np.loadtxt("/mnt/gongjie_NAS2/Datasets/scannet_2d/intrinsics.txt")  # (4x4)内参

                # 提取旋转和平移 (假设txt文件中存的是从相机到世界的变换矩阵)
                R = extrinsics_4x4[:3, :3]   # 取出 3x3 旋转
                t = extrinsics_4x4[:3, 3]    # 取出 3x1 平移向量

                # 提取内参 3x3, 并从中获取 fx, fy, cx, cy
                K = intrinsics_4x4[:3, :3]  # 取出 3x3 内参矩阵
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]

                cam_params = {
                    "intrinsics": [[fx, 0, cx],
                                   [0, fy, cy],
                                   [0, 0,   1]],
                    "rotation": R.tolist(),
                    "translation": t.tolist()
                }

            else:
                raise ValueError(f"Unsupported camera parameters file format: {cam_params_path}")

    except Exception as exn:
        print(f"Warning: Failed to open camera parameters {cam_params_path}. Exception: {exn}")
        fx = fy = max(width, height)
        cx = width / 2.0
        cy = height / 2.0
        cam_params = {
            "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0]
        }

    # Extract intrinsics (focal lengths fx, fy, and principal point cx, cy)
    K = np.array(cam_params["intrinsics"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 3) Create a meshgrid for pixel coordinates (u, v).
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 4) Convert depth + pixel coordinates to 3D coordinates in the camera frame.
    #    X = (u - cx) * depth / fx
    #    Y = (v - cy) * depth / fy
    #    Z = depth
    x = (u - cx) * depth_image / fx
    y = (v - cy) * depth_image / fy
    z = depth_image
    coord3d = np.stack((x, y, z), axis=2)  # shape: (H, W, 3)

    # 5) Convert from camera coordinate system to world coordinate system if needed.
    R = np.array(cam_params["rotation"])
    T = np.array(cam_params["translation"])
    coord3d_flat = coord3d.reshape(-1, 3).T  # shape: (3, H*W)
    coord3d_world = R @ coord3d_flat + T[:, np.newaxis]
    coord3d = coord3d_world.T.reshape(height, width, 3)

    # 6) Multiply the Z channel by -1 (a common practice when coordinate systems differ).
    if cam_params_path.endswith(".txt"):  # scannet_2d format
        pass  # scannet format does not need to convert
    else:
        coord3d[:, :, -1] *= -1

    # 7) Return as a PyTorch tensor.
    return torch.tensor(coord3d, dtype=torch.float32)


def resize_coord3d_resize(coord3d, new_size, mode='bicubic'):
    """
    Resizes a 3D coordinate map to the new spatial size using interpolation.

    Args:
        coord3d (torch.Tensor): A tensor of shape (H, W, 3) containing 3D coords.
        new_size (tuple): (new_height, new_width).
        mode (str): Interpolation mode (e.g., 'nearest', 'bilinear', 'bicubic').

    Returns:
        torch.Tensor: A tensor of shape (new_height, new_width, 3) 
                      with interpolated 3D coordinates.
    """
    # Permute from (H, W, 3) to (1, 3, H, W) for PyTorch interpolate
    coord3d_perm = coord3d.permute(2, 0, 1).unsqueeze(0)
    # Perform interpolation
    coord3d_interp = F.interpolate(coord3d_perm, size=new_size, mode=mode, align_corners=True)
    # Bring back to shape (H, W, 3)
    coord3d_resized = coord3d_interp.squeeze(0).permute(1, 2, 0)
    return coord3d_resized


def get_coord3d_info(image_path, depth_path, cam_params_path, image, interp_mode='bicubic'):
    """
    Combines the steps of reading an image, creating a 3D coordinate map,
    and then resizing that map to match the size of the given 'image'.

    Args:
        image_path (str): Path to the RGB image.
        depth_path (str): Path to the depth image.
        cam_params_path (str): Path to camera parameters file.
        image (PIL.Image): The RGB image (used for final size).
        interp_mode (str): Interpolation mode for resizing.

    Returns:
        torch.Tensor: The resized 3D coordinates of shape (H, W, 3).
    """
    new_width, new_height = image.size
    # 1) Compute 3D coords at original resolution
    coord3d = get_coord3d(Image.open(image_path), depth_path, cam_params_path)
    # 2) Resize them to match the new image size
    coord3d_resized = resize_coord3d_resize(coord3d, new_size=(new_height, new_width), mode=interp_mode)
    return coord3d_resized


def coord3d_to_flat_patches(coord3d_resized, patch_size, merge_size, temporal_patch_size=1, is_return_grid_size=False):
    """
    Splits the resized 3D coordinate map into flat patches suitable for 
    tasks like chunked processing or attention-based models.

    Args:
        coord3d_resized (torch.Tensor): (H, W, 3) 3D coordinates.
        patch_size (int): The height/width of each patch (spatial).
        merge_size (int): Additional grouping of patches along spatial dimension.
        temporal_patch_size (int): In case of temporal dimension (e.g. frames), 
                                   number of frames per patch.
        is_return_grid_size (bool): Whether to return the grid size 
                                    (T, grid_h, grid_w) along with patches.

    Returns:
        np.ndarray or (np.ndarray, tuple): Flattened patches of shape 
        (total_patches, channels * temporal_patch_size * patch_size^2).
        If is_return_grid_size=True, returns (patches, (T, grid_h, grid_w)).
    """
    # Extract shape and move to NumPy (B=1, because we just have one image/tensor).
    resized_height, resized_width, _ = coord3d_resized.shape
    patches = np.expand_dims(coord3d_resized.numpy(), axis=0)  # shape: (1, H, W, 3)
    patches = patches.transpose(0, 3, 1, 2)                    # shape: (1, 3, H, W)

    # Here T=1 for a single image. If temporal_patch_size > 1, we might replicate last frame.
    T = patches.shape[0]
    remainder = T % temporal_patch_size
    if remainder != 0:
        # If we need to pad, replicate the last frame to fill up to temporal_patch_size
        repeats = np.repeat(patches[-1][np.newaxis], temporal_patch_size - remainder, axis=0)
        patches = np.concatenate([patches, repeats], axis=0)

    # Recompute T after potential padding
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    # Rearrange data into patches:
    # The structure below effectively does a block-splitting along H, W, and T.
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # Reorder axes so we have a consistent layout
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    # Flatten the patches along spatial/temporal axes
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    grid_size = (grid_t, grid_h, grid_w)

    if is_return_grid_size:
        return flatten_patches, grid_size
    else:
        return flatten_patches


def generate_3D_positional_encoding(coordinates, channels, grid_dims, scale=500.0):
    """
    Generates a sinusoidal positional encoding in 3D (X, Y, Z) space,
    typically used for attention or transformer models that need 
    explicit position information.

    Args:
        coordinates (torch.Tensor): Flattened 3D coordinates of shape (N, vec_len).
        channels (int): Total number of channels for the positional encoding.
                        Must be divisible by 6 (since we do sin/cos for X, Y, Z).
        grid_dims (torch.Tensor): A shape descriptor of how data is split 
                                  (e.g., T, width, height) for each chunk.
        scale (float): Maximum frequency of the sinusoid.

    Returns:
        (torch.Tensor, torch.Tensor):
            - Concatenated positional encodings of shape (sum_of_patches, patch_area, channels)
            - A corresponding mask indicating NaN locations, (sum_of_patches, patch_area)
    """
    device = coordinates.device
    dtype = coordinates.dtype

    # The next few variables define an expected shape in coordinates
    pt = 2
    pc = 3
    ps = 14
    vec_len = pt * pc * ps * ps  # The expected length for each flattened coordinate vector

    # Check if we have the correct input shape
    if coordinates.shape[1] != vec_len:
        raise ValueError(f"Expected coordinate vector length {vec_len}, got {coordinates.shape[1]}.")

    # We do sin/cos expansions for each dimension (x, y, z). 
    # 'channels' must be divisible by 6 => 2 expansions (sin/cos) * 3 dims (x, y, z).
    L = channels // 6
    if L * 6 != channels:
        raise ValueError("num_channels must be divisible by 6.")

    # Generate frequency values from 1 to `scale`
    freq = torch.linspace(1.0, scale, steps=L, device=device, dtype=torch.float32).view(L, 1, 1, 1)

    out_encodings = []
    out_nan_masks = []

    counts = []
    sizes = []

    # Each entry in 'grid_dims' presumably corresponds to (T, W, H) or (time, width, height).
    # We compute how many total coordinates we have for that chunk.
    for i in range(grid_dims.shape[0]):
        t_dim, w_dim, h_dim = grid_dims[i].tolist()
        cnt = int(t_dim * w_dim * h_dim * ps * ps)
        counts.append(cnt)
        sizes.append((int(h_dim * ps), int(w_dim * ps)))

    # Reshape from (N, pt*pc*ps*ps) to something more manageable
    coordinates = coordinates.view(-1, pt * pc)

    start = 0
    for cnt, size in zip(counts, sizes):
        end = start + cnt

        # (cnt, pt*pc) => then expand into (cnt*pt, pc) => shape (cnt*pt, pc)
        grid_coords = coordinates[start:end]
        grid_coords = grid_coords.view(cnt * pt, pc)

        # Then interpret as shape: (cnt*pt, pc, H, W) for some dimension H,W = size
        grid_coords = grid_coords.view(-1, pc, *size)

        # Create a mask for NaN values
        mask = torch.isnan(grid_coords)
        nan_mask = mask.any(dim=1)

        # Replace NaNs with zeros for the sin/cos calculations
        grid_coords = grid_coords.masked_fill(mask, 0)

        # Scale coordinates for sinusoidal embeddings
        # The final shape will be: (batch, L, pc, H, W) for sin/cos expansions
        coord_scaled = grid_coords.unsqueeze(1).float() * freq * 2 * math.pi

        # Compute sin/cos
        sin_vals = torch.sin(coord_scaled)
        cos_vals = torch.cos(coord_scaled)

        # Combine them along a new dimension
        sincos = torch.stack((sin_vals, cos_vals), dim=2)  # shape: (batch, L, 2, pc, H, W)
        # Reorder so that pc dimension is last or suitably placed for flatten
        sincos = sincos.permute(0, 3, 2, 1, 4, 5)  # shape: (batch, pc, 2, L, H, W)

        # Now pack as (pt, channels, H, W). 
        # 'channels' is 2 expansions (sin/cos) * L frequencies * 3 coords = 6L
        pe = sincos.reshape(pt, channels, *size).to(dtype=dtype)

        # Expand the mask and zero out positions in the encoding where input is NaN
        nan_mask_exp = nan_mask.unsqueeze(1).expand_as(pe)
        pe = pe.masked_fill(nan_mask_exp, 0)

        # Flatten each patch
        pe = pe.view(-1, pt * ps * ps, channels)
        nan_mask = nan_mask.to(dtype=dtype).view(-1, pt * ps * ps)

        out_encodings.append(pe)
        out_nan_masks.append(nan_mask)

        start = end

    # Concatenate across all chunks
    return torch.cat(out_encodings, dim=0), torch.cat(out_nan_masks, dim=0)
