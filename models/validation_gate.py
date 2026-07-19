import torch
import torch.nn as nn
import torch.nn.functional as F
from data.utils import calculate_ndvi, calculate_albedo_from_s2, upsample_tensor

class PhysicsValidationGate(nn.Module):
    """
    Physics Validation Gate.
    Implements:
      1. Differentiable in-loop physics loss functions (NDVI, Albedo, and SAR gradients).
      2. Post-hoc validation flag checker for reconstructed scenes.
    """
    def __init__(self, ndvi_min=-0.1, ndvi_max=0.9, albedo_min=0.0, albedo_max=0.6):
        super().__init__()
        self.ndvi_min = ndvi_min
        self.ndvi_max = ndvi_max
        self.albedo_min = albedo_min
        self.albedo_max = albedo_max

    def get_gradients(self, img):
        """Computes Sobel gradients to find structural edges of an image."""
        # img: [B, C, H, W]
        B, C, H, W = img.shape
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        
        # Apply to each channel and stack
        grad_x = F.conv2d(img.view(B*C, 1, H, W), sobel_x, padding=1).view(B, C, H, W)
        grad_y = F.conv2d(img.view(B*C, 1, H, W), sobel_y, padding=1).view(B, C, H, W)
        
        return grad_x, grad_y

    def compute_physics_loss(self, recon_x0, gt_x0, s2_lr, sar_hr_fused):
        """
        Differentiable In-Loop Physics Loss.
        Computes NDVI penalty, Albedo penalty, and SAR-optical gradient alignment.
        """
        B, C, H, W = recon_x0.shape
        
        # 1. NDVI Loss
        # LISS-IV Green=0, Red=1, NIR=2
        recon_ndvi = calculate_ndvi(recon_x0[:, 1, :, :], recon_x0[:, 2, :, :])
        gt_ndvi = calculate_ndvi(gt_x0[:, 1, :, :], gt_x0[:, 2, :, :])
        
        # Differentiable L1 loss between reconstructed and ground truth NDVI
        ndvi_loss = F.l1_loss(recon_ndvi, gt_ndvi)
        
        # Penalize if NDVI is out of bounds
        out_of_bounds_ndvi = torch.clamp(recon_ndvi - self.ndvi_max, min=0.0) + torch.clamp(self.ndvi_min - recon_ndvi, min=0.0)
        ndvi_penalty = out_of_bounds_ndvi.mean()
        
        # 2. Broadband Albedo Loss
        # Fusing LISS-IV bands with Sentinel-2 SWIR bands
        # Sentinel-2 Blue=0, Red=2, NIR=3, SWIR1=4, SWIR2=5
        # Upsample Sentinel-2 SWIR to LISS-IV resolution
        s2_hr = upsample_tensor(s2_lr, target_size=(H, W), mode='bilinear')
        
        # Reconstructed S2-style composite
        # We replace B2 (Blue) with upsampled S2 Blue, B3 (Green) with recon Green, 
        # B4 (Red) with recon Red, B8 (NIR) with recon NIR, B11/12 with S2 SWIR.
        recon_s2_comp = torch.zeros(B, 6, H, W, device=recon_x0.device)
        recon_s2_comp[:, 0, :, :] = s2_hr[:, 0, :, :] # Blue
        recon_s2_comp[:, 1, :, :] = recon_x0[:, 0, :, :] # Green (LISS-IV equivalent)
        recon_s2_comp[:, 2, :, :] = recon_x0[:, 1, :, :] # Red
        recon_s2_comp[:, 3, :, :] = recon_x0[:, 2, :, :] # NIR
        recon_s2_comp[:, 4, :, :] = s2_hr[:, 4, :, :] # SWIR1
        recon_s2_comp[:, 5, :, :] = s2_hr[:, 5, :, :] # SWIR2
        
        # Calculate Albedo
        recon_albedo = calculate_albedo_from_s2(recon_s2_comp)
        
        # Ground Truth S2 composite
        gt_s2_comp = recon_s2_comp.clone()
        gt_s2_comp[:, 1:4, :, :] = gt_x0 # Inject GT optical
        gt_albedo = calculate_albedo_from_s2(gt_s2_comp)
        
        albedo_loss = F.l1_loss(recon_albedo, gt_albedo)
        
        # Penalize if Albedo is out of bounds
        out_of_bounds_albedo = torch.clamp(recon_albedo - self.albedo_max, min=0.0) + torch.clamp(self.albedo_min - recon_albedo, min=0.0)
        albedo_penalty = out_of_bounds_albedo.mean()
        
        # 3. Structural Gradient Alignment (SAR vs. Optical)
        # We want the structural boundaries in our reconstructed image to align with the SAR guides
        recon_gx, recon_gy = self.get_gradients(recon_x0)
        sar_gx, sar_gy = self.get_gradients(sar_hr_fused)
        
        # Minimize difference in gradient direction and magnitude
        gradient_loss = F.mse_loss(recon_gx, sar_gx) + F.mse_loss(recon_gy, sar_gy)
        
        # Total combined differentiable physics loss
        total_physics_loss = ndvi_loss + 0.1 * ndvi_penalty + albedo_loss + 0.1 * albedo_penalty + 0.5 * gradient_loss
        
        return total_physics_loss

    @torch.no_grad()
    def post_hoc_validate(self, recon_x0, s2_lr):
        """
        Runs post-hoc validation checks on a completed reconstruction.
        Returns:
            passed (bool): True if all checks pass, False if anomalies detected.
            metrics (dict): Dict of validation metrics and bounds flags.
        """
        H, W = recon_x0.shape[2:]
        recon_ndvi = calculate_ndvi(recon_x0[:, 1, :, :], recon_x0[:, 2, :, :])
        
        s2_hr = upsample_tensor(s2_lr, target_size=(H, W), mode='bilinear')
        recon_s2_comp = torch.zeros(recon_x0.shape[0], 6, H, W, device=recon_x0.device)
        recon_s2_comp[:, 0, :, :] = s2_hr[:, 0, :, :]
        recon_s2_comp[:, 1, :, :] = recon_x0[:, 0, :, :]
        recon_s2_comp[:, 2, :, :] = recon_x0[:, 1, :, :]
        recon_s2_comp[:, 3, :, :] = recon_x0[:, 2, :, :]
        recon_s2_comp[:, 4, :, :] = s2_hr[:, 4, :, :]
        recon_s2_comp[:, 5, :, :] = s2_hr[:, 5, :, :]
        recon_albedo = calculate_albedo_from_s2(recon_s2_comp)
        
        # Compute anomaly rates
        ndvi_anomaly_mask = (recon_ndvi < self.ndvi_min) | (recon_ndvi > self.ndvi_max)
        albedo_anomaly_mask = (recon_albedo < self.albedo_min) | (recon_albedo > self.albedo_max)
        
        ndvi_anomaly_rate = float(ndvi_anomaly_mask.float().mean().cpu().numpy())
        albedo_anomaly_rate = float(albedo_anomaly_mask.float().mean().cpu().numpy())
        
        # Strict pass threshold: less than 1% anomalous pixels
        passed = (ndvi_anomaly_rate < 0.01) and (albedo_anomaly_rate < 0.01)
        
        metrics = {
            'ndvi_mean': float(recon_ndvi.mean().cpu().numpy()),
            'ndvi_anomaly_rate': ndvi_anomaly_rate,
            'albedo_mean': float(recon_albedo.mean().cpu().numpy()),
            'albedo_anomaly_rate': albedo_anomaly_rate,
            'passed': passed
        }
        
        return passed, metrics

    def guided_correction_step(self, x_0_pred, condition, step_size=0.01):
        """
        Dynamically adjusts the x_0 prediction inside the diffusion sampling loop
        by taking a small step down the gradient of the physics validation check.
        """
        # Enable gradients temporarily for correction step (even inside @torch.no_grad)
        with torch.enable_grad():
            x_0_pred_detached = x_0_pred.detach().clone().requires_grad_(True)
            
            # Unpack condition components to reconstruct S2 and SAR references
            # condition is [B, 13, 128, 128]
            # clear_pixels: 0:3, TPT_pred: 3:6, memory_bank_mean: 6:9, sar_fused: 9:12, mask: 12:13
            sar_fused = condition[:, 9:12, :, :].detach()
            
            # Setup simplified gradient alignment loss as correction objective
            recon_gx, recon_gy = self.get_gradients(x_0_pred_detached)
            sar_gx, sar_gy = self.get_gradients(sar_fused)
            struct_loss = F.mse_loss(recon_gx, sar_gx) + F.mse_loss(recon_gy, sar_gy)
            
            # NDVI boundary penalty
            recon_ndvi = calculate_ndvi(x_0_pred_detached[:, 1, :, :], x_0_pred_detached[:, 2, :, :])
            out_of_bounds = torch.clamp(recon_ndvi - self.ndvi_max, min=0.0) + torch.clamp(self.ndvi_min - recon_ndvi, min=0.0)
            ndvi_penalty = out_of_bounds.mean()
            
            total_loss = struct_loss + ndvi_penalty
            
            # Compute gradient
            total_loss.backward()
            
            # Step in direction of negative gradient
            corrected_x0 = x_0_pred_detached - step_size * x_0_pred_detached.grad
        
        return torch.clamp(corrected_x0.detach(), 0.0, 1.0)

