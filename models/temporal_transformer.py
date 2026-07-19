import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPhenologyTransformer(nn.Module):
    """
    Temporal Phenology Transformer (TPT).
    Takes a historical sequence of cloud-free optical patches [B, T, C, H, W]
    and predicts the expected optical patch [B, C, H, W] at time t.
    """
    def __init__(self, in_channels=3, embed_dim=64, num_layers=2, num_heads=4, patch_size=128):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Spatial Encoder: downscale spatial dimensions
        # 128x128 -> 64x64 -> 32x32 -> 16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 4, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(True),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True)
        )
        
        # Positional Encoding for time steps
        self.time_pos_embed = nn.Parameter(torch.zeros(1, 10, embed_dim)) # max 10 steps
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Spatial Decoder: upscale back to original resolution
        # 16x16 -> 32x32 -> 64x64 -> 128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(embed_dim // 4, in_channels, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Sigmoid() # pixel intensities are [0, 1]
        )

    def forward(self, history):
        """
        Args:
            history (torch.Tensor): [B, T, C, H, W] where T is sequence length
        Returns:
            predicted_t (torch.Tensor): [B, C, H, W]
        """
        B, T, C, H, W = history.shape
        
        # 1. Encode spatial dimension of all steps: [B * T, C, H, W]
        history_flat = history.view(B * T, C, H, W)
        feat_flat = self.encoder(history_flat) # [B * T, embed_dim, h_lat, w_lat]
        
        h_lat, w_lat = feat_flat.shape[2], feat_flat.shape[3]
        
        # 2. Reshape for Temporal Transformer: [B, h_lat, w_lat, T, embed_dim]
        feat = feat_flat.view(B, T, self.embed_dim, h_lat, w_lat)
        feat = feat.permute(0, 3, 4, 1, 2) # [B, h_lat, w_lat, T, embed_dim]
        
        # Flatten spatial dimensions to run transformer in parallel for each pixel position
        # [B * h_lat * w_lat, T, embed_dim]
        feat_seq = feat.reshape(B * h_lat * w_lat, T, self.embed_dim)
        
        # Add temporal positional embedding
        feat_seq = feat_seq + self.time_pos_embed[:, :T, :]
        
        # 3. Process time sequence with Transformer Encoder
        out_seq = self.transformer(feat_seq) # [B * h_lat * w_lat, T, embed_dim]
        
        # Take the last time step as the prediction target
        out_last = out_seq[:, -1, :] # [B * h_lat * w_lat, embed_dim]
        
        # 4. Decode spatial features back to original resolution
        out_feat = out_last.view(B, h_lat, w_lat, self.embed_dim).permute(0, 3, 1, 2) # [B, embed_dim, h_lat, w_lat]
        predicted_t = self.decoder(out_feat) # [B, C, H, W]
        
        return predicted_t
