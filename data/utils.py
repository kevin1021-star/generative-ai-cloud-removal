import torch
import torch.nn.functional as F

def upsample_tensor(tensor, target_size=(128, 128), mode='bilinear'):
    """
    Upsamples a tensor [B, C, H_in, W_in] to [B, C, H_out, W_out].
    Supports 3D tensors by adding and removing a batch dimension.
    """
    has_batch = len(tensor.shape) == 4
    if not has_batch:
        tensor = tensor.unsqueeze(0) # add batch dim
        
    upsampled = F.interpolate(tensor, size=target_size, mode=mode, align_corners=False)
    
    if not has_batch:
        upsampled = upsampled.squeeze(0) # remove batch dim
        
    return upsampled

def calculate_ndvi(red, nir, eps=1e-8):
    """
    Calculates the Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)
    """
    return (nir - red) / (nir + red + eps)

def calculate_albedo_from_s2(s2_tensor):
    """
    Estimates broadband shortwave albedo from Sentinel-2 bands using Liang's formula:
    Inputs: s2_tensor of shape [..., 6, H, W] containing [Blue, Green, Red, NIR, SWIR1, SWIR2]
    Formula:
    alpha = 0.356*Blue + 0.130*Red + 0.373*NIR + 0.085*SWIR1 + 0.072*SWIR2 - 0.0018
    """
    # Extract bands
    blue   = s2_tensor[..., 0, :, :]
    red    = s2_tensor[..., 2, :, :]
    nir    = s2_tensor[..., 3, :, :]
    swir1  = s2_tensor[..., 4, :, :]
    swir2  = s2_tensor[..., 5, :, :]
    
    albedo = 0.356 * blue + 0.130 * red + 0.373 * nir + 0.085 * swir1 + 0.072 * swir2 - 0.0018
    
    # Clip to physically realistic albedo bounds [0.0, 1.0]
    return torch.clamp(albedo, 0.0, 1.0)
