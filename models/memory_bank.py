import torch
import numpy as np

class SpectralMemoryBank:
    """
    Spectral Memory Bank for storing and retrieving historical cloud-free spectra.
    Uses coordinate grid location and seasonal phase as keys.
    """
    def __init__(self, grid_size=(10, 10), num_seasons=12, num_bands=3):
        """
        Args:
            grid_size (tuple): Spatial grid division for coordinates (latitude, longitude).
            num_seasons (int): Number of seasonal bins (e.g. 12 months).
            num_bands (int): Number of optical spectral bands (Green, Red, NIR).
        """
        self.grid_size = grid_size
        self.num_seasons = num_seasons
        self.num_bands = num_bands
        
        # We store running sums and counts to calculate online mean and variance
        self.sums = np.zeros((*grid_size, num_seasons, num_bands), dtype=np.float32)
        self.sq_sums = np.zeros((*grid_size, num_seasons, num_bands), dtype=np.float32)
        self.counts = np.zeros((*grid_size, num_seasons), dtype=np.float32) + 1e-5 # avoid div by zero
        
        # Default global averages for uninitialized grid queries (typical values)
        # Green: 0.15, Red: 0.15, NIR: 0.35 (vegetation dominant)
        self.global_mean = np.array([0.15, 0.15, 0.35], dtype=np.float32)
        self.global_std = np.array([0.05, 0.05, 0.10], dtype=np.float32)

    def _get_grid_indices(self, lat, lon):
        """Maps continuous lat/lon coordinates to grid bins."""
        # Simple mapping assuming lat/lon normalized between 0 and 1
        lat_idx = int(np.clip(lat * self.grid_size[0], 0, self.grid_size[0] - 1))
        lon_idx = int(np.clip(lon * self.grid_size[1], 0, self.grid_size[1] - 1))
        return lat_idx, lon_idx

    def update_from_patch(self, lat, lon, season_idx, patch, mask):
        """
        Updates the memory bank statistics using clean pixels of a patch.
        Args:
            lat (float), lon (float): Geolocation center of patch.
            season_idx (int): Current season index (0 to num_seasons-1).
            patch (torch.Tensor/np.ndarray): Clean ground truth optical patch [3, H, W].
            mask (torch.Tensor/np.ndarray): Binary mask [1, H, W] indicating clear (1) or cloudy/shadow (0) pixels.
        """
        if isinstance(patch, torch.Tensor):
            patch = patch.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
            
        lat_idx, lon_idx = self._get_grid_indices(lat, lon)
        
        # Filter clean pixels
        # mask is [1, H, W], patch is [3, H, W]
        clear_pixels = patch * mask # [3, H, W]
        
        # Accumulate statistics for this grid location and season
        for b in range(self.num_bands):
            valid_pixels = clear_pixels[b][mask[0] > 0.5]
            if len(valid_pixels) > 0:
                self.sums[lat_idx, lon_idx, season_idx, b] += np.sum(valid_pixels)
                self.sq_sums[lat_idx, lon_idx, season_idx, b] += np.sum(valid_pixels ** 2)
                
        # Increment counts by number of clear pixels added
        num_clear = np.sum(mask)
        self.counts[lat_idx, lon_idx, season_idx] += num_clear

    def query_pixel_stats(self, lat, lon, season_idx):
        """
        Retrieves historical mean and standard deviation for a location and season.
        Returns:
            mean (np.ndarray): [3]
            std (np.ndarray): [3]
        """
        lat_idx, lon_idx = self._get_grid_indices(lat, lon)
        count = self.counts[lat_idx, lon_idx, season_idx]
        
        if count < 10.0: # Insufficient historical samples, return global defaults
            return self.global_mean, self.global_std
            
        mean = self.sums[lat_idx, lon_idx, season_idx] / count
        var = (self.sq_sums[lat_idx, lon_idx, season_idx] / count) - (mean ** 2)
        std = np.sqrt(np.clip(var, 1e-6, 1.0))
        
        return mean, std

    def query_patch_prior(self, lat, lon, season_idx, patch_shape):
        """
        Queries and returns a prior mean and variance map for a patch.
        Args:
            patch_shape (tuple): (C, H, W)
        Returns:
            mean_map (torch.Tensor): [C, H, W]
            std_map (torch.Tensor): [C, H, W]
        """
        mean, std = self.query_pixel_stats(lat, lon, season_idx)
        
        C, H, W = patch_shape
        mean_map = torch.from_numpy(mean).view(C, 1, 1).repeat(1, H, W)
        std_map = torch.from_numpy(std).view(C, 1, 1).repeat(1, H, W)
        
        return mean_map, std_map
