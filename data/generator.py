import os
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

class SyntheticSatelliteDataGenerator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.size_hr = self.config['data']['image_size']
        self.size_lr = self.size_hr // 2  # Simulate resolution mismatch (128 vs 64)
        self.seq_len = self.config['data']['sequence_length']
        self.num_samples = self.config['data']['synthetic_samples']
        
        self.output_dir = os.path.join(
            os.path.dirname(config_path), 
            "../data/synthetic"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_base_scene(self):
        """Generates a synthetic high-resolution ground truth scene with roads, fields, rivers, and cities."""
        scene = np.zeros((self.size_hr, self.size_hr, 3), dtype=np.float32) # Green, Red, NIR
        
        # 1. Background fields (vegetation/soil)
        # Create grid of fields
        for i in range(0, self.size_hr, 32):
            for j in range(0, self.size_hr, 32):
                # NDVI determine the ratio of Red vs NIR
                veg_type = np.random.rand()
                if veg_type > 0.4: # Vegetated field
                    green = 0.15 + np.random.rand() * 0.05
                    red = 0.08 + np.random.rand() * 0.03
                    nir = 0.45 + np.random.rand() * 0.15  # High NIR
                else: # Soil/Barren field
                    green = 0.12 + np.random.rand() * 0.04
                    red = 0.18 + np.random.rand() * 0.04  # High Red
                    nir = 0.20 + np.random.rand() * 0.04
                scene[i:i+32, j:j+32, 0] = green
                scene[i:i+32, j:j+32, 1] = red
                scene[i:i+32, j:j+32, 2] = nir
                
        # 2. Curving River
        # Define a path
        x = np.arange(self.size_hr)
        y = (self.size_hr // 2) + np.sin(x / 10.0) * 15.0
        for xi in range(self.size_hr):
            yi = int(y[xi])
            if 0 <= yi < self.size_hr:
                # River is dark in NIR, slightly green
                scene[xi, max(0, yi-2):min(self.size_hr, yi+3), 0] = 0.08 # Green
                scene[xi, max(0, yi-2):min(self.size_hr, yi+3), 1] = 0.04 # Red
                scene[xi, max(0, yi-2):min(self.size_hr, yi+3), 2] = 0.02 # NIR (water absorbs NIR)

        # 3. Roads (Straight lines)
        road_y = self.size_hr // 3
        scene[max(0, road_y-1):min(self.size_hr, road_y+2), :, 0] = 0.15 # Grayish
        scene[max(0, road_y-1):min(self.size_hr, road_y+2), :, 1] = 0.15
        scene[max(0, road_y-1):min(self.size_hr, road_y+2), :, 2] = 0.15

        # 4. Urban centers (Highly reflective concrete squares)
        # Create some buildings
        for _ in range(5):
            bx = np.random.randint(10, self.size_hr - 20)
            by = np.random.randint(10, self.size_hr - 20)
            # High reflectance concrete
            scene[bx:bx+8, by:by+8, 0] = 0.35
            scene[bx:bx+8, by:by+8, 1] = 0.38
            scene[bx:bx+8, by:by+8, 2] = 0.39
            
        return scene

    def generate_clouds(self, base_scene):
        """Generates random cloud and shadow masks and applies them to the base scene."""
        cloud_mask = np.zeros((self.size_hr, self.size_hr), dtype=np.float32)
        shadow_mask = np.zeros((self.size_hr, self.size_hr), dtype=np.float32)
        
        # Draw 1-2 cloud blobs
        for _ in range(np.random.randint(1, 3)):
            cx = np.random.randint(20, self.size_hr - 20)
            cy = np.random.randint(20, self.size_hr - 20)
            radius = np.random.randint(15, 30)
            
            # Simple radial cloud
            y, x = np.ogrid[-cx:self.size_hr-cx, -cy:self.size_hr-cy]
            mask = x*x + y*y <= radius*radius
            cloud_mask[mask] = 1.0
            
            # Shadow is shifted (e.g., northeast shift)
            sx = cx + 8
            sy = cy - 8
            y_s, x_s = np.ogrid[-sx:self.size_hr-sx, -sy:self.size_hr-sy]
            s_mask = x_s*x_s + y_s*y_s <= radius*radius
            shadow_mask[s_mask] = 1.0
            
        # Smooth boundaries using Gaussian filter
        cloud_mask = gaussian_filter(cloud_mask, sigma=2.0)
        cloud_mask = (cloud_mask > 0.3).astype(np.float32)
        
        shadow_mask = gaussian_filter(shadow_mask, sigma=2.0)
        shadow_mask = (shadow_mask > 0.3).astype(np.float32)
        # Avoid cloud-shadow overlap
        shadow_mask = np.clip(shadow_mask - cloud_mask, 0, 1)
        
        # Apply clouds and shadows to base scene
        cloudy_scene = base_scene.copy()
        # Clouds are bright white/gray
        for b in range(3):
            cloudy_scene[:, :, b] = (1.0 - cloud_mask) * cloudy_scene[:, :, b] + cloud_mask * (0.8 + 0.1 * np.random.rand())
        
        # Shadows are dark
        for b in range(3):
            cloudy_scene[:, :, b] = (1.0 - shadow_mask) * cloudy_scene[:, :, b] + shadow_mask * (cloudy_scene[:, :, b] * 0.3)
            
        return cloudy_scene, cloud_mask, shadow_mask

    def generate_sar(self, base_scene):
        """Generates low-res (10m) synthetic Sentinel-1 SAR image (VV, VH) from high-res scene."""
        # SAR backscatter is sensitive to roughness and structure.
        # High returns on urban areas, moderate on vegetation, low on water (specular reflection).
        # We downsample base scene to lr resolution first, then generate SAR.
        lr_scene = np.zeros((self.size_lr, self.size_lr, 3), dtype=np.float32)
        for b in range(3):
            # Block-average downsample (simulate sensor resolution)
            lr_scene[:, :, b] = base_scene[::2, ::2, b]
            
        # VV: strong backscatter from buildings (double bounce) and roads (edges)
        vv = np.zeros((self.size_lr, self.size_lr), dtype=np.float32)
        # VH: cross-polarization, strong from volume scattering (vegetation)
        vh = np.zeros((self.size_lr, self.size_lr), dtype=np.float32)
        
        for i in range(self.size_lr):
            for j in range(self.size_lr):
                g, r, nir = lr_scene[i, j, 0], lr_scene[i, j, 1], lr_scene[i, j, 2]
                ndvi = (nir - r) / (nir + r + 1e-6)
                
                # Check features based on reflectance
                if nir < 0.05: # Water
                    vv[i, j] = 0.05 + 0.02 * np.random.randn()
                    vh[i, j] = 0.01 + 0.005 * np.random.randn()
                elif g > 0.3 and r > 0.3 and nir > 0.3: # Urban
                    vv[i, j] = 0.85 + 0.10 * np.random.randn()
                    vh[i, j] = 0.65 + 0.10 * np.random.randn()
                elif ndvi > 0.2: # Vegetation
                    vv[i, j] = 0.40 + 0.08 * np.random.randn()
                    vh[i, j] = 0.35 + 0.08 * np.random.randn()
                else: # Soil/Road
                    vv[i, j] = 0.25 + 0.05 * np.random.randn()
                    vh[i, j] = 0.12 + 0.03 * np.random.randn()
                    
        # Clip SAR values between 0 and 1
        vv = np.clip(vv, 0.0, 1.0)
        vh = np.clip(vh, 0.0, 1.0)
        
        return np.stack([vv, vh], axis=-1)

    def generate_s2(self, base_scene):
        """Generates low-res (10m) Sentinel-2 optical bands (Blue, Green, Red, NIR, SWIR1, SWIR2)."""
        lr_scene = np.zeros((self.size_lr, self.size_lr, 3), dtype=np.float32)
        for b in range(3):
            lr_scene[:, :, b] = base_scene[::2, ::2, b]
            
        s2 = np.zeros((self.size_lr, self.size_lr, 6), dtype=np.float32)
        
        # Populate Sentinel-2 bands
        # LISS-IV Green, Red, NIR map to S2 Green (B3), Red (B4), NIR (B8)
        s2[:, :, 1] = lr_scene[:, :, 0] # Green
        s2[:, :, 2] = lr_scene[:, :, 1] # Red
        s2[:, :, 3] = lr_scene[:, :, 2] # NIR
        
        # Extrapolate Blue (B2): highly correlated with Green but slightly darker
        s2[:, :, 0] = np.clip(s2[:, :, 1] * 0.85 + 0.02 * np.random.randn(*s2[:, :, 0].shape), 0.0, 1.0)
        
        # Extrapolate SWIR1 (B11) and SWIR2 (B12): sensitive to soil moisture and clay.
        # Soil reflects SWIR heavily, vegetation absorbs SWIR compared to NIR, water absorbs completely.
        for i in range(self.size_lr):
            for j in range(self.size_lr):
                g, r, nir = lr_scene[i, j, 0], lr_scene[i, j, 1], lr_scene[i, j, 2]
                ndvi = (nir - r) / (nir + r + 1e-6)
                if nir < 0.05: # Water
                    s2[i, j, 4] = 0.01 + 0.005 * np.random.randn() # SWIR1
                    s2[i, j, 5] = 0.005 + 0.002 * np.random.randn() # SWIR2
                elif ndvi > 0.2: # Vegetation
                    s2[i, j, 4] = 0.15 + 0.03 * np.random.randn()
                    s2[i, j, 5] = 0.08 + 0.02 * np.random.randn()
                else: # Soil/Urban
                    s2[i, j, 4] = 0.40 + 0.05 * np.random.randn()
                    s2[i, j, 5] = 0.35 + 0.05 * np.random.randn()
                    
        s2 = np.clip(s2, 0.0, 1.0)
        return s2

    def generate_history(self, base_scene):
        """Generates historical sequence showing crop phenology (seasonal growth)."""
        history = []
        for step in range(self.seq_len):
            # Alter vegetation level based on chronological step to simulate growth
            growth_factor = 0.6 + 0.4 * np.sin(step / self.seq_len * np.pi)
            hist_scene = base_scene.copy()
            for i in range(self.size_hr):
                for j in range(self.size_hr):
                    g, r, nir = base_scene[i, j, 0], base_scene[i, j, 1], base_scene[i, j, 2]
                    ndvi = (nir - r) / (nir + r + 1e-6)
                    if ndvi > 0.2: # Vegetation changes seasonally
                        hist_scene[i, j, 2] = r + (nir - r) * growth_factor # adjust NIR
                        hist_scene[i, j, 0] = g * (0.8 + 0.2 * growth_factor) # adjust Green
            # Add a bit of registration noise and noise
            noise = 0.01 * np.random.randn(*hist_scene.shape)
            hist_scene = np.clip(hist_scene + noise, 0.0, 1.0)
            history.append(hist_scene)
            
        return np.stack(history, axis=0) # [seq_len, size_hr, size_hr, 3]

    def run(self):
        """Runs the generation of self.num_samples datasets."""
        print(f"Generating {self.num_samples} synthetic satellite datasets to {self.output_dir}...")
        for idx in range(self.num_samples):
            # Create a clean high-res ground truth scene
            gt_scene = self.generate_base_scene()
            
            # Generate clouds & shadows
            cloudy, cloud_mask, shadow_mask = self.generate_clouds(gt_scene)
            
            # Generate SAR
            sar = self.generate_sar(gt_scene)
            
            # Generate Sentinel-2
            s2 = self.generate_s2(gt_scene)
            
            # Generate timeline history
            history = self.generate_history(gt_scene)
            
            # Save to compressed numpy archive
            filepath = os.path.join(self.output_dir, f"sample_{idx:04d}.npz")
            np.savez_compressed(
                filepath,
                gt=gt_scene,
                cloudy=cloudy,
                cloud_mask=cloud_mask,
                shadow_mask=shadow_mask,
                sar=sar,
                s2=s2,
                history=history
            )
            
        print("Data generation complete!")

if __name__ == "__main__":
    import sys
    config_file = "C:/Users/AS/.gemini/antigravity/scratch/liss4_cloud_removal/config/config.yaml"
    generator = SyntheticSatelliteDataGenerator(config_file)
    generator.run()
