import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import LISS4CloudRemovalDataset
from data.utils import upsample_tensor
from models.cloud_detector import MultimodalCloudDetector
from models.memory_bank import SpectralMemoryBank
from models.temporal_transformer import TemporalPhenologyTransformer
from models.sar_fusion import SARCoherenceFusion
from models.diffusion_decoder import ConditionalDenoisingUnet, PhysicsGuidedDiffusionDecoder
from models.validation_gate import PhysicsValidationGate

def train_pipeline(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output and checkpoint dirs
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    # Initialize Dataset and DataLoader
    data_dir = "C:/Users/AS/.gemini/antigravity/scratch/liss4_cloud_removal/data/synthetic"
    dataset = LISS4CloudRemovalDataset(data_dir=data_dir)
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.RandomSubsetSplit(dataset, [train_size, val_size]) if hasattr(torch.utils.data, 'RandomSubsetSplit') else torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 1. Initialize Memory Bank with training data
    print("Initializing Spectral Memory Bank...")
    memory_bank = SpectralMemoryBank()
    for sample in train_loader:
        gt = sample['gt'] # [B, 3, 128, 128]
        mask = torch.ones_like(sample['cloud_mask']) # simulate clear updating
        for b in range(gt.shape[0]):
            # mock coords (0.5, 0.5) and season 5
            memory_bank.update_from_patch(0.5, 0.5, 5, gt[b], mask[b])
            
    # 2. Instantiate Models
    print("Instantiating models...")
    cloud_detector = MultimodalCloudDetector().to(device)
    tpt = TemporalPhenologyTransformer().to(device)
    sar_fusion = SARCoherenceFusion().to(device)
    
    unet = ConditionalDenoisingUnet().to(device)
    diffusion = PhysicsGuidedDiffusionDecoder(unet, num_steps=config['models']['diffusion']['num_steps']).to(device)
    
    physics_gate = PhysicsValidationGate(
        ndvi_min=config['physics']['ndvi']['min_val'],
        ndvi_max=config['physics']['ndvi']['max_val'],
        albedo_min=config['physics']['albedo']['min_val'],
        albedo_max=config['physics']['albedo']['max_val']
    ).to(device)
    
    # Optimizers
    opt_detector = optim.Adam(cloud_detector.parameters(), lr=config['models']['cloud_detector']['lr'])
    opt_tpt = optim.Adam(tpt.parameters(), lr=config['models']['tpt']['lr'])
    opt_sar = optim.Adam(sar_fusion.parameters(), lr=config['models']['sar_fusion']['lr'])
    opt_diff = optim.Adam(diffusion.parameters(), lr=config['models']['diffusion']['lr'])
    
    # Standard loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Train loops (For demo purposes we train for 3 epochs)
    epochs = 3
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        cloud_detector.train()
        tpt.train()
        sar_fusion.train()
        diffusion.unet.train()
        
        epoch_det_loss = 0.0
        epoch_tpt_loss = 0.0
        epoch_sar_loss = 0.0
        epoch_diff_loss = 0.0
        epoch_physics_loss = 0.0
        
        for batch in train_loader:
            # Move items to device
            gt = batch['gt'].to(device)
            cloudy = batch['cloudy'].to(device)
            cloud_mask = batch['cloud_mask'].to(device)
            shadow_mask = batch['shadow_mask'].to(device)
            sar = batch['sar'].to(device) # [B, 2, 64, 64]
            s2 = batch['s2'].to(device) # [B, 6, 64, 64]
            history = batch['history'].to(device) # [B, 5, 3, 128, 128]
            
            # --- TASK 1: Train Cloud Detector ---
            opt_detector.zero_grad()
            # Upsample S2 blue (channel 0) and SAR for detector input
            s2_blue_up = upsample_tensor(s2[:, 0:1, :, :], target_size=(128, 128))
            sar_up = upsample_tensor(sar, target_size=(128, 128))
            
            pred_cloud_logits, pred_shadow_logits = cloud_detector(cloudy, s2_blue_up, sar_up)
            
            loss_cloud = bce_loss(pred_cloud_logits, cloud_mask)
            loss_shadow = bce_loss(pred_shadow_logits, shadow_mask)
            loss_det = loss_cloud + loss_shadow
            
            loss_det.backward()
            opt_detector.step()
            epoch_det_loss += loss_det.item()
            
            # --- TASK 2: Train Temporal Phenology Transformer ---
            opt_tpt.zero_grad()
            pred_tpt = tpt(history)
            loss_tpt = l1_loss(pred_tpt, gt)
            
            loss_tpt.backward()
            opt_tpt.step()
            epoch_tpt_loss += loss_tpt.item()
            
            # --- TASK 3: Train SAR Coherence Fusion ---
            opt_sar.zero_grad()
            # Use TPT prediction as the optical baseline to fuse with SAR
            pred_sar_fused = sar_fusion(sar, pred_tpt.detach())
            loss_sar = l1_loss(pred_sar_fused, gt)
            
            loss_sar.backward()
            opt_sar.step()
            epoch_sar_loss += loss_sar.item()
            
            # --- TASK 4: Train Physics-Guided Diffusion ---
            opt_diff.zero_grad()
            
            # Generate condition maps
            # Unmask (clear) parts of input LISS-IV
            combined_mask = torch.clamp(cloud_mask + shadow_mask, 0.0, 1.0)
            clear_pixels = cloudy * (1.0 - combined_mask)
            
            # Query memory bank mean maps (mock query)
            B = gt.shape[0]
            mb_mean_list = []
            for b in range(B):
                mb_mean, _ = memory_bank.query_patch_prior(0.5, 0.5, 5, (3, 128, 128))
                mb_mean_list.append(mb_mean)
            mb_mean_tensor = torch.stack(mb_mean_list, dim=0).to(device)
            
            # Concatenate conditions
            # 3 (clear_pixels) + 3 (pred_tpt) + 3 (mb_mean) + 3 (pred_sar_fused) + 1 (combined_mask) = 13 channels
            condition = torch.cat([
                clear_pixels,
                pred_tpt.detach(),
                mb_mean_tensor,
                pred_sar_fused.detach(),
                combined_mask
            ], dim=1)
            
            # Forward diffusion: sample step t and add noise
            t_steps = torch.randint(0, diffusion.num_steps, (B,), device=device).long()
            noise = torch.randn_like(gt)
            x_t = diffusion.q_sample(x_0=gt, t=t_steps, noise=noise)
            
            # Predict noise
            noise_pred = diffusion.unet(x_t, t_steps, condition)
            loss_diff_recon = l1_loss(noise_pred, noise)
            
            # Differentiable In-loop Physics Loss
            # Estimate x_0 from predicted noise
            x_0_pred = diffusion.predict_x_start_from_noise(x_t, t_steps, noise_pred)
            x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)
            
            loss_phys = physics_gate.compute_physics_loss(
                recon_x0=x_0_pred,
                gt_x0=gt,
                s2_lr=s2,
                sar_hr_fused=pred_sar_fused.detach()
            )
            
            # Combined Loss
            total_diff_loss = loss_diff_recon + config['models']['diffusion']['physics_guidance_scale'] * loss_phys
            
            total_diff_loss.backward()
            opt_diff.step()
            
            epoch_diff_loss += loss_diff_recon.item()
            epoch_physics_loss += loss_phys.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Det Loss: {epoch_det_loss/len(train_loader):.4f} | "
              f"TPT Loss: {epoch_tpt_loss/len(train_loader):.4f} | "
              f"SAR Loss: {epoch_sar_loss/len(train_loader):.4f} | "
              f"Diff Loss: {epoch_diff_loss/len(train_loader):.4f} | "
              f"Phys Loss: {epoch_physics_loss/len(train_loader):.4f}")

    # Save checkpoints
    torch.save(cloud_detector.state_dict(), os.path.join(checkpoint_dir, "cloud_detector.pt"))
    torch.save(tpt.state_dict(), os.path.join(checkpoint_dir, "tpt.pt"))
    torch.save(sar_fusion.state_dict(), os.path.join(checkpoint_dir, "sar_fusion.pt"))
    torch.save(diffusion.state_dict(), os.path.join(checkpoint_dir, "diffusion.pt"))
    
    print("Training orchestration complete! Checkpoints saved.")

if __name__ == "__main__":
    config_file = "C:/Users/AS/.gemini/antigravity/scratch/liss4_cloud_removal/config/config.yaml"
    train_pipeline(config_file)
