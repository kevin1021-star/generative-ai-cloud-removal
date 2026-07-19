import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    """Embeds scalar time steps into a high-dimensional vector space."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim // 4, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        # Sine-cosine sinusoidal positional encoding
        half_dim = self.dim // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # MLP
        emb = F.silu(self.linear1(emb))
        emb = self.linear2(emb)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU()
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # Add time step embedding
        time_emb = self.time_mlp(t_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        return h + self.shortcut(x)

class ConditionalDenoisingUnet(nn.Module):
    """
    Conditional Denoising U-Net for Diffusion.
    Inputs:
      - x_t: Noisy image [B, 3, 128, 128]
      - t: Time steps [B]
      - condition: Concatenated priors [B, 13, 128, 128]
        (clear_pixels [3], TPT_pred [3], memory_bank_mean [3], sar_fused [3], cloud_shadow_mask [1])
    Total input channels = 3 + 13 = 16 channels
    """
    def __init__(self, in_channels=16, out_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # Encoder contracting path
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        self.res1 = ResBlock(64, 64, time_emb_dim)
        self.down1 = nn.MaxPool2d(2) # 64x64
        
        self.res2 = ResBlock(64, 128, time_emb_dim)
        self.down2 = nn.MaxPool2d(2) # 32x32
        
        self.res3 = ResBlock(128, 256, time_emb_dim)
        self.down3 = nn.MaxPool2d(2) # 16x16
        
        # Bottleneck
        self.mid_res1 = ResBlock(256, 256, time_emb_dim)
        self.mid_res2 = ResBlock(256, 256, time_emb_dim)
        
        # Decoder expanding path
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest') # 32x32
        self.res4 = ResBlock(256 + 128, 128, time_emb_dim)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest') # 64x64
        self.res5 = ResBlock(128 + 64, 64, time_emb_dim)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest') # 128x128
        self.res6 = ResBlock(64 + 64, 64, time_emb_dim)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x_t, t, condition):
        # Concatenate noisy image with condition maps
        x = torch.cat([x_t, condition], dim=1) # [B, 16, 128, 128]
        
        # Embed time
        t_emb = self.time_embed(t)
        
        # Encode
        h0 = self.init_conv(x)
        h1 = self.res1(h0, t_emb)
        h2 = self.down1(h1)
        
        h3 = self.res2(h2, t_emb)
        h4 = self.down2(h3)
        
        h5 = self.res3(h4, t_emb)
        h6 = self.down3(h5)
        
        # Bottleneck
        h6 = self.mid_res1(h6, t_emb)
        h6 = self.mid_res2(h6, t_emb)
        
        # Decode with skip connections
        h = self.up1(h6) # [B, 256, 32, 32]
        h = torch.cat([h, h4], dim=1) # h4 is [B, 128, 32, 32], matches spatial dimension
        h = self.res4(h, t_emb)
        
        h = self.up2(h) # [B, 128, 64, 64]
        h = torch.cat([h, h2], dim=1) # h2 is [B, 64, 64, 64], matches spatial dimension
        h = self.res5(h, t_emb)
        
        h = self.up3(h) # [B, 64, 128, 128]
        h = torch.cat([h, h0], dim=1) # h0 is [B, 64, 128, 128], matches spatial dimension
        h = self.res6(h, t_emb)
        
        noise_pred = self.out_conv(h)
        return noise_pred


class PhysicsGuidedDiffusionDecoder(nn.Module):
    """
    DDPM / DDIM wrapper with fast DPM-style 10-step sampling and in-loop physics loss capabilities.
    """
    def __init__(self, unet_model, num_steps=10, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.unet = unet_model
        self.num_steps = num_steps
        
        # Set up DDPM schedule parameters
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register parameters as constant tensors (non-trainable parameters)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x_0, t, noise):
        """Adds noise to the original ground truth image (forward diffusion process)."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_x_start_from_noise(self, x_t, t, noise_pred):
        """Estimates the clean target image x_0 from the predicted noise at step t (used for in-loop physics checks)."""
        sqrt_recip_alphas_cumprod_t = (1.0 / self.sqrt_alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1.0)).view(-1, 1, 1, 1)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise_pred

    @torch.no_grad()
    def sample_ddim(self, condition, steps=10, physics_gate=None):
        """
        Fast DDIM Denoising Sampler (10 steps).
        Supports taking multiple stochastic runs to generate an uncertainty map.
        Args:
            condition: Concatenated conditioning maps [B, 13, 128, 128]
            steps: Number of sampling steps (defaults to 10)
            physics_gate: Optional physics checker module
        Returns:
            x_0: Reconstructed optical image [B, 3, 128, 128]
            uncertainty: Spatial variance map [B, 1, 128, 128]
        """
        B = condition.shape[0]
        device = condition.device
        
        # We run the sampling loop 3 times stochastic to compute pixel variance (Uncertainty Map)
        num_runs = 3
        runs_x0 = []
        
        # Step intervals for fast sampling (e.g. 10 steps)
        time_steps = torch.arange(0, self.num_steps).long().to(device)
        
        for run_idx in range(num_runs):
            # Start from random isotropic Gaussian noise
            x_t = torch.randn(B, 3, 128, 128, device=device)
            
            # Loop backwards from steps-1 to 0
            for i in reversed(range(steps)):
                t = torch.full((B,), time_steps[i], device=device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self.unet(x_t, t, condition)
                
                # Predict x_0 at current step
                x_0_pred = self.predict_x_start_from_noise(x_t, t, noise_pred)
                x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)
                
                # Dynamic physics-guided adjustment if a physics validation gate is attached
                if physics_gate is not None:
                    # Apply physics gradient optimization step inside the diffusion loop
                    x_0_pred = physics_gate.guided_correction_step(x_0_pred, condition)
                
                # Compute DDIM step
                if i > 0:
                    alpha_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
                    sigma_t = 0.0 # DDIM parameter (eta=0 makes it deterministic given noise x_T)
                    
                    # Compute x_{t-1}
                    dir_xt = torch.sqrt(1.0 - alpha_prev - sigma_t**2) * noise_pred
                    x_t = torch.sqrt(alpha_prev) * x_0_pred + dir_xt
                else:
                    x_t = x_0_pred
                    
            runs_x0.append(x_t)
            
        # Calculate final reconstructed mean and variance (uncertainty map)
        runs_x0_stack = torch.stack(runs_x0, dim=0) # [num_runs, B, 3, 128, 128]
        reconstructed_mean = torch.mean(runs_x0_stack, dim=0) # [B, 3, 128, 128]
        
        # Uncertainty is the variance of generated values across runs, averaged over channels
        uncertainty = torch.var(runs_x0_stack, dim=0).mean(dim=1, keepdim=True) # [B, 1, 128, 128]
        # Normalize uncertainty to [0, 1] range for visualization
        uncertainty = torch.clamp(uncertainty * 100.0, 0.0, 1.0)
        
        return reconstructed_mean, uncertainty
