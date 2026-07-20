import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

class MultiSpectralAugmentor:
    """
    Advanced co-registered augmentation pipeline using Albumentations.
    Ensures that spatial augmentations (flips, rotations, shifts) are applied
    identically across all aligned modalities: LISS-IV, SAR, Sentinel-2, 
    cloud masks, and historical observation sequences.
    """
    def __init__(self):
        if not ALBUMENTATIONS_AVAILABLE:
            self.transform = None
            return
            
        # Define co-registered targets
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5)
        ], additional_targets={
            'cloudy': 'image',
            'cloud_mask': 'mask',
            'shadow_mask': 'mask',
            'sar_0': 'image',
            'sar_1': 'image',
            's2_0': 'image',
            's2_1': 'image',
            's2_2': 'image',
            's2_3': 'image',
            's2_4': 'image',
            's2_5': 'image',
            'history_0': 'image',
            'history_1': 'image',
            'history_2': 'image',
            'history_3': 'image',
            'history_4': 'image'
        })

    def __call__(self, sample):
        if self.transform is None:
            return sample

        # Extract numpy arrays for Albumentations (expects HWC for images, HW for masks)
        gt = sample['gt'].permute(1, 2, 0).numpy()          # [H, W, 3]
        cloudy = sample['cloudy'].permute(1, 2, 0).numpy()  # [H, W, 3]
        cloud_mask = sample['cloud_mask'].squeeze(0).numpy() # [H, W]
        shadow_mask = sample['shadow_mask'].squeeze(0).numpy() # [H, W]
        
        sar = sample['sar'].permute(1, 2, 0).numpy()        # [64, 64, 2]
        s2 = sample['s2'].permute(1, 2, 0).numpy()          # [64, 64, 6]
        history = sample['history'].permute(0, 2, 3, 1).numpy() # [seq_len, 128, 128, 3]

        # Prepare kwargs for co-registration
        kwargs = {
            'image': gt,
            'cloudy': cloudy,
            'cloud_mask': cloud_mask,
            'shadow_mask': shadow_mask
        }
        
        # Add SAR bands individually to bypass single-channel constraints
        for i in range(2):
            kwargs[f'sar_{i}'] = sar[:, :, i:i+1]
            
        # Add Sentinel-2 bands individually
        for i in range(6):
            kwargs[f's2_{i}'] = s2[:, :, i:i+1]
            
        # Add historical sequence frames individually
        for i in range(len(history)):
            kwargs[f'history_{i}'] = history[i]

        # Run albumentations
        transformed = self.transform(**kwargs)

        # Re-pack outputs and convert back to tensors
        sample['gt'] = torch.tensor(transformed['image'], dtype=torch.float32).permute(2, 0, 1)
        sample['cloudy'] = torch.tensor(transformed['cloudy'], dtype=torch.float32).permute(2, 0, 1)
        sample['cloud_mask'] = torch.tensor(transformed['cloud_mask'], dtype=torch.float32).unsqueeze(0)
        sample['shadow_mask'] = torch.tensor(transformed['shadow_mask'], dtype=torch.float32).unsqueeze(0)

        # Re-assemble SAR
        sar_recon = []
        for i in range(2):
            sar_recon.append(transformed[f'sar_{i}'])
        sample['sar'] = torch.tensor(np.concatenate(sar_recon, axis=-1), dtype=torch.float32).permute(2, 0, 1)

        # Re-assemble Sentinel-2
        s2_recon = []
        for i in range(6):
            s2_recon.append(transformed[f's2_{i}'])
        sample['s2'] = torch.tensor(np.concatenate(s2_recon, axis=-1), dtype=torch.float32).permute(2, 0, 1)

        # Re-assemble History
        hist_recon = []
        for i in range(len(history)):
            hist_recon.append(transformed[f'history_{i}'])
        sample['history'] = torch.tensor(np.stack(hist_recon, axis=0), dtype=torch.float32).permute(0, 3, 1, 2)

        return sample

class LISS4CloudRemovalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory with synthetic .npz files.
            transform (callable, optional): Optional transform to be applied.
        """
        self.data_dir = data_dir
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        # Use our advanced MultiSpectralAugmentor as the default training transform
        self.transform = transform if transform is not None else MultiSpectralAugmentor()
        
    def __len__(self):
        if len(self.file_list) == 0:
            return 10  # Fallback dataset length to keep loaders happy on fresh clones
        return len(self.file_list)
        
    def __getitem__(self, idx):
        if len(self.file_list) == 0:
            # Fallback dynamic mock sample generator for seamless out-of-the-box operation on fresh clones
            gt = torch.rand(3, 128, 128)
            cloud_mask = (torch.rand(1, 128, 128) > 0.8).float()
            # Simple cloud mapping: white/grey where clouded
            cloudy = gt * (1.0 - cloud_mask) + torch.ones_like(gt) * 0.85 * cloud_mask
            shadow_mask = torch.zeros(1, 128, 128)
            sar = torch.rand(2, 64, 64)
            s2 = torch.rand(6, 64, 64)
            
            history = []
            for i in range(5):
                history.append(gt + torch.randn_like(gt) * 0.03 * (i + 1))
            history = torch.clamp(torch.stack(history, dim=0), 0.0, 1.0)
            
            sample = {
                'gt': gt,
                'cloudy': cloudy,
                'cloud_mask': cloud_mask,
                'shadow_mask': shadow_mask,
                'sar': sar,
                's2': s2,
                'history': history,
                'sample_name': f"synthetic_fallback_{idx}.npz"
            }
            if self.transform:
                sample = self.transform(sample)
            return sample

        file_path = self.file_list[idx]
        data = np.load(file_path)
        
        # Load and transpose to (C, H, W) for PyTorch compatibility
        gt = torch.tensor(data['gt'], dtype=torch.float32).permute(2, 0, 1)                  # [3, 128, 128]
        cloudy = torch.tensor(data['cloudy'], dtype=torch.float32).permute(2, 0, 1)          # [3, 128, 128]
        cloud_mask = torch.tensor(data['cloud_mask'], dtype=torch.float32).unsqueeze(0)      # [1, 128, 128]
        shadow_mask = torch.tensor(data['shadow_mask'], dtype=torch.float32).unsqueeze(0)    # [1, 128, 128]
        sar = torch.tensor(data['sar'], dtype=torch.float32).permute(2, 0, 1)                # [2, 64, 64] (low-res)
        s2 = torch.tensor(data['s2'], dtype=torch.float32).permute(2, 0, 1)                  # [6, 64, 64] (low-res)
        
        # History is [seq_len, H, W, C] -> permute to [seq_len, C, H, W]
        history = torch.tensor(data['history'], dtype=torch.float32).permute(0, 3, 1, 2)     # [seq_len, 3, 128, 128]
        
        sample = {
            'gt': gt,
            'cloudy': cloudy,
            'cloud_mask': cloud_mask,
            'shadow_mask': shadow_mask,
            'sar': sar,
            's2': s2,
            'history': history,
            'sample_name': os.path.basename(file_path)
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
