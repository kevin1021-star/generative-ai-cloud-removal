import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    Computes cross-attention where:
      - Query (Q) is derived from high-resolution optical features (LISS-IV edge/structure templates)
      - Key (K) and Value (V) are derived from upsampled low-resolution SAR structural features
    """
    def __init__(self, query_dim, key_dim, value_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.q_conv = nn.Conv2d(query_dim, out_dim, kernel_size=1)
        self.k_conv = nn.Conv2d(key_dim, out_dim, kernel_size=1)
        self.v_conv = nn.Conv2d(value_dim, out_dim, kernel_size=1)
        
        self.out_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def forward(self, q_input, kv_input):
        """
        Args:
            q_input: [B, query_dim, H, W] (High-res optical template)
            kv_input: [B, key_dim, H, W] (Upsampled low-res SAR)
        Returns:
            fused: [B, out_dim, H, W]
        """
        B, C, H, W = q_input.shape
        
        # Project inputs
        Q = self.q_conv(q_input) # [B, out_dim, H, W]
        K = self.k_conv(kv_input) # [B, out_dim, H, W]
        V = self.v_conv(kv_input) # [B, out_dim, H, W]
        
        # Transposed Channel Attention (Restormer style) to prevent OOM
        # Flatten spatial coordinates to [B, num_heads, head_dim, H*W]
        Q = Q.view(B, self.num_heads, self.head_dim, H * W)
        K = K.view(B, self.num_heads, self.head_dim, H * W)
        V = V.view(B, self.num_heads, self.head_dim, H * W)
        
        # Compute channel attention map [B, num_heads, head_dim, head_dim]
        # (reduces spatial dimension from 16384 to 16, saving ~17GB of RAM per head)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / ((H * W) ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Multiply attention with values: [B, num_heads, head_dim, H*W]
        out = torch.matmul(attn_probs, V)
        
        # Reshape back to image grid: [B, out_dim, H, W]
        out = out.view(B, -1, H, W)
        
        return self.out_conv(out)

class SARCoherenceFusion(nn.Module):
    """
    Sentinel-1 SAR Coherence and Structure Fusion Network.
    Upsamples low-res 10m SAR (2 bands) to 5.8m using high-res LISS-IV spatial edges
    via cross-attention.
    """
    def __init__(self, sar_channels=2, optical_channels=3, fusion_dim=64):
        super().__init__()
        
        # High-res Optical structure encoder (to extract sharp edges)
        self.optical_encoder = nn.Sequential(
            nn.Conv2d(optical_channels, fusion_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim // 2),
            nn.ReLU(True),
            nn.Conv2d(fusion_dim // 2, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(True)
        )
        
        # Low-res SAR encoder
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, fusion_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim // 2),
            nn.ReLU(True),
            nn.Conv2d(fusion_dim // 2, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(True)
        )
        
        # Cross-Attention Block
        self.cross_attention = CrossAttentionBlock(
            query_dim=fusion_dim,
            key_dim=fusion_dim,
            value_dim=fusion_dim,
            out_dim=fusion_dim
        )
        
        # Refinement network to smooth fusion borders
        self.refiner = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(True),
            nn.Conv2d(fusion_dim, optical_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # output is in optical channel range [0, 1]
        )

    def forward(self, sar_lr, optical_hr):
        """
        Args:
            sar_lr: Low-resolution SAR tensor [B, 2, 64, 64]
            optical_hr: High-resolution optical template [B, 3, 128, 128]
        Returns:
            fused_hr: Fused high-resolution SAR-optical structure [B, 3, 128, 128]
        """
        # 1. Bilinear upsample SAR to match HR dimensions [B, 2, 128, 128]
        sar_up = F.interpolate(sar_lr, size=optical_hr.shape[2:], mode='bilinear', align_corners=False)
        
        # 2. Extract features
        opt_feats = self.optical_encoder(optical_hr) # [B, fusion_dim, 128, 128] (Query source)
        sar_feats = self.sar_encoder(sar_up)         # [B, fusion_dim, 128, 128] (Key/Value source)
        
        # 3. Fuse features using cross-attention
        fused_feats = self.cross_attention(q_input=opt_feats, kv_input=sar_feats) # [B, fusion_dim, 128, 128]
        
        # 4. Refine back to optical bands
        fused_hr = self.refiner(fused_feats) # [B, 3, 128, 128]
        
        return fused_hr
