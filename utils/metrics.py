import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_func

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two PyTorch tensors.
    Args:
        img1, img2: Tensors of shape [B, C, H, W]
    """
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    # avoid division by zero
    mse = torch.clamp(mse, min=1e-8)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return torch.mean(psnr).item()

def calculate_ssim(img1, img2):
    """
    Computes Structural Similarity Index (SSIM) between two PyTorch tensors.
    Args:
        img1, img2: Tensors of shape [B, C, H, W] in [0, 1] range.
    """
    # Convert to numpy and transpose to HWC
    im1_np = img1.detach().cpu().numpy()
    im2_np = img2.detach().cpu().numpy()
    
    ssim_vals = []
    B = im1_np.shape[0]
    for b in range(B):
        # Transpose to [H, W, C]
        im1 = np.transpose(im1_np[b], (1, 2, 0))
        im2 = np.transpose(im2_np[b], (1, 2, 0))
        
        # Check channel count to set channel_axis
        s_val = ssim_func(im1, im2, data_range=1.0, channel_axis=-1)
        ssim_vals.append(s_val)
        
    return float(np.mean(ssim_vals))

def calculate_sam(img1, img2, eps=1e-8):
    """
    Computes Spectral Angle Mapper (SAM) between two PyTorch tensors.
    SAM measures the spectral similarity in radians.
    Args:
        img1, img2: Tensors of shape [B, C, H, W]
    """
    # Reshape to [B, C, H*W] -> transpose to [B, H*W, C]
    B, C, H, W = img1.shape
    v1 = img1.view(B, C, -1).permute(0, 2, 1) # [B, H*W, C]
    v2 = img2.view(B, C, -1).permute(0, 2, 1) # [B, H*W, C]
    
    # Dot product along spectral dimension (C)
    dot_product = torch.sum(v1 * v2, dim=-1) # [B, H*W]
    
    # Norms
    norm1 = torch.norm(v1, p=2, dim=-1) # [B, H*W]
    norm2 = torch.norm(v2, p=2, dim=-1) # [B, H*W]
    
    # Cosine similarity
    cos_theta = dot_product / (norm1 * norm2 + eps)
    # Clamp to avoid nan in arccos due to numerical precision
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    
    # Spectral Angle in radians
    sam_rad = torch.acos(cos_theta)
    
    return torch.mean(sam_rad).item()
