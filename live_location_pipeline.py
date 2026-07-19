import math
import io
import urllib.request
import urllib.error
from PIL import Image
import numpy as np
import torch
import os

def deg2num(lat_deg, lon_deg, zoom):
    """Convert Lat/Lon to XYZ tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def get_esri_satellite_tile(x, y, z):
    """Fetches a single tile from Esri World Imagery (open for non-commercial/demo use)."""
    # ESRI uses standard XYZ, but their URL format is {z}/{y}/{x}
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 LISS-IV-Hackathon-Bot'})
    try:
        response = urllib.request.urlopen(req, timeout=5)
        return Image.open(io.BytesIO(response.read())).convert('RGB')
    except Exception as e:
        print(f"Failed to fetch tile {z}/{y}/{x}: {e}")
        # Return a blank green image as fallback
        return Image.new('RGB', (256, 256), color=(34, 139, 34))

def fetch_simulated_liss4_image(lat, lon):
    """
    Fetches high-res satellite imagery around the target coordinate and
    downsamples it to perfectly simulate LISS-IV (5.8m/pixel) resolution.
    
    A 128x128 LISS-IV image covers ~742x742 meters on the ground.
    At zoom level 16, Esri tiles are roughly 2.4m/pixel at the equator.
    We will fetch a 3x3 grid at zoom 16, crop the center 742m equivalent,
    and downsample to 128x128.
    """
    zoom = 16
    cx, cy = deg2num(lat, lon, zoom)
    
    # Fetch 3x3 grid
    full_image = Image.new('RGB', (256 * 3, 256 * 3))
    for i in range(-1, 2):
        for j in range(-1, 2):
            tile = get_esri_satellite_tile(cx + i, cy + j, zoom)
            full_image.paste(tile, ((i + 1) * 256, (j + 1) * 256))
            
    # The center of the 3x3 grid is the center of the middle tile (approximate)
    # We want a crop that represents ~742x742 meters.
    # At zoom 16 near latitude 20 (India), resolution is ~2.2 m/px.
    # So 742m is about 337 pixels. Let's crop a 337x337 box from the center.
    
    center_x, center_y = 256 * 1.5, 256 * 1.5
    crop_size = 337 / 2.0
    box = (
        int(center_x - crop_size),
        int(center_y - crop_size),
        int(center_x + crop_size),
        int(center_y + crop_size)
    )
    
    cropped_high_res = full_image.crop(box)
    
    # Simulating LISS-IV Spatial Resolution (5.8m)
    # Downsample to 128x128 to match LISS-IV exact spec
    liss4_simulated = cropped_high_res.resize((128, 128), Image.LANCZOS)
    
    return liss4_simulated, cropped_high_res

def prepare_live_sample(lat, lon, add_clouds=True):
    """
    Fetches the live image, converts it to LISS-IV tensor format,
    adds synthetic clouds for the pipeline to remove, and prepares auxiliaries.
    """
    from real_data_pipeline import add_synthetic_clouds, create_synthetic_auxiliaries
    
    # 1. Fetch & Simulate LISS-IV Spatial Resolution
    liss4_img, high_res_img = fetch_simulated_liss4_image(lat, lon)
    
    # 2. Simulate LISS-IV Spectral Bands (Green, Red, NIR)
    arr = np.array(liss4_img, dtype=np.float32) / 255.0  # [H, W, 3] in RGB
    
    green = arr[:, :, 1]
    red = arr[:, :, 0]
    # LISS-IV NIR approximation from RGB (vegetation reflects heavily in NIR)
    # We approximate NIR using Green (vegetation) and scaling down Blue/Red
    nir_proxy = arr[:, :, 1] * 0.6 + arr[:, :, 0] * 0.2 + arr[:, :, 2] * 0.2
    nir_proxy = np.clip(nir_proxy * 1.2, 0.0, 1.0) # Boost to simulate high vegetation reflectance
    
    liss4_tensor = torch.tensor(np.stack([green, red, nir_proxy], axis=0), dtype=torch.float32)
    
    if add_clouds:
        # Image is clean -> add synthetic clouds for testing
        gt = liss4_tensor.clone()
        cloudy, cloud_mask = add_synthetic_clouds(liss4_tensor, cloud_coverage=0.4)
    else:
        cloudy = liss4_tensor.clone()
        gt = liss4_tensor.clone()
        brightness = liss4_tensor.mean(dim=0)
        cloud_mask = (brightness > 0.7).float().unsqueeze(0)
        
    # Generate Hackathon Auxiliaries (Simulated Sentinel-1 SAR & Sentinel-2 Optical)
    sar, s2, history = create_synthetic_auxiliaries(liss4_tensor)
    
    sample = {
        'gt': gt,
        'cloudy': cloudy,
        'cloud_mask': cloud_mask,
        'sar': sar,
        's2': s2,
        'history': history,
        'high_res_ref': high_res_img # for showing the judges the high-res view
    }
    
    return sample
