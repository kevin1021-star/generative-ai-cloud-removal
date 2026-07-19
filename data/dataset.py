import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class LISS4CloudRemovalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory with synthetic .npz files.
            transform (callable, optional): Optional transform to be applied.
        """
        self.data_dir = data_dir
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
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
