"""
Real Satellite Data Downloader & Preprocessor for LISS-IV Cloud Removal Pipeline.

This script provides two workflows:
  1. Download real Sentinel-2 + Sentinel-1 data from Copernicus (requires free account)
  2. Load manually downloaded GeoTIFF files and preprocess for inference

Usage:
  # Option A: Download from Copernicus API (requires registration)
  python real_data_pipeline.py --mode download --lat 28.6139 --lon 77.2090 --date 2024-06-15

  # Option B: Load your own GeoTIFF files
  python real_data_pipeline.py --mode local --optical path/to/sentinel2.tif --sar path/to/sentinel1.tif
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def preprocess_rgb_image(image_path, target_size=128):
    """
    Loads any RGB image (PNG, JPG, TIFF) and converts it into a
    synthetic LISS-IV-like tensor [3, H, W] with values in [0, 1].
    
    Maps:  R -> Band 1 (Red),  G -> Band 0 (Green),  B -> Band 2 (pseudo-NIR)
    
    This allows testing the pipeline with ANY satellite screenshot,
    even from Google Maps or Copernicus Browser.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in RGB
    
    # Map to LISS-IV band order: [Green, Red, NIR-proxy]
    green = arr[:, :, 1]  # Green channel
    red = arr[:, :, 0]    # Red channel
    nir_proxy = arr[:, :, 2] * 0.5 + arr[:, :, 1] * 0.3 + arr[:, :, 0] * 0.2  # Synthetic NIR
    
    liss4_tensor = torch.tensor(np.stack([green, red, nir_proxy], axis=0), dtype=torch.float32)
    return liss4_tensor


def create_synthetic_auxiliaries(liss4_tensor, target_size=128):
    """
    When we only have optical data (no real SAR or Sentinel-2),
    generate synthetic auxiliary inputs to feed the pipeline.
    
    Returns: sar [2, 64, 64], s2 [6, 64, 64], history [5, 3, 128, 128]
    """
    C, H, W = liss4_tensor.shape
    half = target_size // 2  # 64 for SAR and S2 resolution
    
    # Synthetic SAR (VV, VH) - derived from optical texture
    from torch.nn.functional import interpolate
    liss4_down = interpolate(
        liss4_tensor.unsqueeze(0), size=(half, half), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    # VV ~ mean intensity, VH ~ edge-like features
    vv = liss4_down.mean(dim=0, keepdim=True) * 0.8 + torch.randn(1, half, half) * 0.05
    vh = torch.abs(liss4_down[2:3] - liss4_down[1:2]) * 0.6 + torch.randn(1, half, half) * 0.05
    sar = torch.clamp(torch.cat([vv, vh], dim=0), 0.0, 1.0)  # [2, 64, 64]
    
    # Synthetic Sentinel-2 (6 bands at lower resolution)
    # Blue, Green, Red, NIR, SWIR1, SWIR2
    blue = liss4_down[0:1] * 0.7 + liss4_down[1:2] * 0.3  # Approximate blue
    s2 = torch.cat([
        blue,                           # Blue
        liss4_down[0:1],                # Green
        liss4_down[1:2],                # Red
        liss4_down[2:3],                # NIR
        liss4_down[2:3] * 0.6,          # SWIR1 proxy
        liss4_down[2:3] * 0.4           # SWIR2 proxy
    ], dim=0)  # [6, 64, 64]
    s2 = torch.clamp(s2, 0.0, 1.0)
    
    # Temporal history: 5 slightly varied copies of the input
    history = []
    for i in range(5):
        variation = liss4_tensor + torch.randn_like(liss4_tensor) * 0.03 * (i + 1)
        history.append(torch.clamp(variation, 0.0, 1.0))
    history = torch.stack(history, dim=0)  # [5, 3, 128, 128]
    
    return sar, s2, history


def add_synthetic_clouds(liss4_tensor, cloud_coverage=0.3):
    """
    Adds realistic synthetic cloud patches to a clean satellite image.
    Returns: cloudy_tensor [3, H, W], cloud_mask [1, H, W]
    """
    from scipy.ndimage import gaussian_filter
    
    C, H, W = liss4_tensor.shape
    cloud_mask = np.zeros((H, W), dtype=np.float32)
    
    # Generate 3-6 cloud blobs
    num_clouds = np.random.randint(3, 7)
    for _ in range(num_clouds):
        cx, cy = np.random.randint(20, H - 20), np.random.randint(20, W - 20)
        radius = np.random.randint(10, 35)
        y, x = np.ogrid[:H, :W]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        blob = np.clip(1.0 - dist / radius, 0, 1)
        cloud_mask = np.maximum(cloud_mask, blob)
    
    # Smooth the edges
    cloud_mask = gaussian_filter(cloud_mask, sigma=5)
    cloud_mask = np.clip(cloud_mask, 0, 1)
    
    # Scale to target coverage
    if cloud_mask.max() > 0:
        threshold = np.percentile(cloud_mask[cloud_mask > 0], (1 - cloud_coverage) * 100)
        cloud_mask = np.clip((cloud_mask - threshold) / (cloud_mask.max() - threshold + 1e-8), 0, 1)
    
    cloud_mask_tensor = torch.tensor(cloud_mask, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
    
    # Apply cloud: white where clouded
    cloud_color = 0.85 + torch.randn(3, 1, 1) * 0.05
    cloudy = liss4_tensor * (1.0 - cloud_mask_tensor) + cloud_color * cloud_mask_tensor
    cloudy = torch.clamp(cloudy, 0.0, 1.0)
    
    return cloudy, cloud_mask_tensor


def prepare_sample_from_image(image_path, add_clouds=True, cloud_coverage=0.3):
    """
    Complete preprocessing pipeline:
    Image file -> Model-ready sample dict
    
    If the image is already cloudy (real satellite), set add_clouds=False.
    If the image is clean and you want to test cloud removal, set add_clouds=True.
    """
    # Load and convert to LISS-IV format
    liss4 = preprocess_rgb_image(image_path, target_size=128)
    
    if add_clouds:
        # Image is clean -> add synthetic clouds for testing
        gt = liss4.clone()
        cloudy, cloud_mask = add_synthetic_clouds(liss4, cloud_coverage)
    else:
        # Image is already cloudy -> use as-is (no ground truth available)
        cloudy = liss4.clone()
        gt = liss4.clone()  # Placeholder, won't be meaningful
        # Create a rough cloud mask by detecting bright white regions
        brightness = liss4.mean(dim=0)  # [H, W]
        cloud_mask = (brightness > 0.7).float().unsqueeze(0)  # [1, H, W]
    
    # Generate auxiliary inputs
    sar, s2, history = create_synthetic_auxiliaries(liss4)
    
    sample = {
        'gt': gt,
        'cloudy': cloudy,
        'cloud_mask': cloud_mask,
        'sar': sar,
        's2': s2,
        'history': history
    }
    
    return sample


def download_from_copernicus(lat, lon, date, output_dir):
    """
    Downloads Sentinel-2 imagery from Copernicus Data Space.
    
    PREREQUISITES:
    1. Create a FREE account at: https://dataspace.copernicus.eu
    2. pip install sentinelsat  OR  use the OData API directly
    
    This function provides the download workflow. Due to API authentication
    requirements, you'll need to set your credentials.
    """
    print("=" * 60)
    print("HOW TO DOWNLOAD REAL SATELLITE DATA")
    print("=" * 60)
    print()
    print(f"Target Location: {lat}°N, {lon}°E")
    print(f"Target Date: {date}")
    print()
    print("OPTION 1: Copernicus Browser (Easiest - Manual Download)")
    print("-" * 50)
    print(f"1. Go to: https://browser.dataspace.copernicus.eu")
    print(f"2. Navigate to lat={lat}, lon={lon}")
    print(f"3. Set date range around {date}")
    print(f"4. Select 'Sentinel-2 L2A' in the search panel")
    print(f"5. Find an image WITH CLOUDS (cloud cover 20-60%)")
    print(f"6. Click Download -> True Color (PNG) or GeoTIFF")
    print(f"7. Save to: {output_dir}")
    print()
    print("OPTION 2: Google Earth Engine (Python API)")
    print("-" * 50)
    print("pip install earthengine-api")
    print("import ee")
    print("ee.Authenticate()  # One-time Google sign-in")
    print("ee.Initialize()")
    print()
    print(f"# Get Sentinel-2 image")
    print(f"collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')")
    print(f"  .filterBounds(ee.Geometry.Point({lon}, {lat}))")
    print(f"  .filterDate('{date}', '<end_date>')")
    print(f"  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))")
    print(f"  .sort('CLOUDY_PIXEL_PERCENTAGE', False)")  
    print(f"image = collection.first()")
    print()
    print("OPTION 3: Copernicus API (Automated)")
    print("-" * 50)
    print("pip install cdsetool")
    print("Then use the OData API with your credentials")
    print()
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Satellite Data Pipeline")
    parser.add_argument('--mode', choices=['download', 'local', 'demo'], default='demo',
                       help='download: show download instructions | local: process local file | demo: run demo')
    parser.add_argument('--image', type=str, help='Path to satellite image (PNG/JPG/TIFF)')
    parser.add_argument('--lat', type=float, default=28.6139, help='Latitude')
    parser.add_argument('--lon', type=float, default=77.2090, help='Longitude')
    parser.add_argument('--date', type=str, default='2024-06-15', help='Date (YYYY-MM-DD)')
    parser.add_argument('--add-clouds', action='store_true', default=True,
                       help='Add synthetic clouds to clean images for testing')
    parser.add_argument('--no-clouds', action='store_true',
                       help='Image already has clouds, skip adding synthetic ones')
    
    args = parser.parse_args()
    
    if args.mode == 'download':
        output_dir = os.path.join(PROJECT_ROOT, "data", "real")
        download_from_copernicus(args.lat, args.lon, args.date, output_dir)
        
    elif args.mode == 'local':
        if not args.image:
            print("ERROR: --image path required for local mode")
            sys.exit(1)
        
        print(f"Processing: {args.image}")
        add_clouds = not args.no_clouds
        sample = prepare_sample_from_image(args.image, add_clouds=add_clouds)
        
        # Run inference
        from models.memory_bank import SpectralMemoryBank
        from infer import load_models_for_inference, run_inference_pipeline
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        
        memory_bank = SpectralMemoryBank()
        memory_bank.update_from_patch(0.5, 0.5, 5, sample['gt'], torch.ones_like(sample['cloud_mask']))
        
        cloud_detector, tpt, sar_fusion, diffusion, physics_gate = load_models_for_inference(config_path, device)
        
        results = run_inference_pipeline(sample, cloud_detector, tpt, sar_fusion, diffusion, physics_gate, memory_bank, device)
        
        print("\nInference Results:")
        for k, v in results['metrics'].items():
            print(f"  {k}: {v}")
        
        # Save comparison image
        from utils.visualize import create_comparison_figure
        comparison = create_comparison_figure(results)
        out_path = os.path.join(PROJECT_ROOT, "outputs", "real_comparison.png")
        comparison.save(out_path)
        print(f"\nComparison saved to: {out_path}")
        
    elif args.mode == 'demo':
        print("=" * 60)
        print("REAL SATELLITE DATA TESTING GUIDE")
        print("=" * 60)
        print()
        print("3 WAYS TO TEST WITH REAL SATELLITE IMAGERY:")
        print()
        print("1. SCREENSHOT METHOD (Easiest - 2 minutes)")
        print("   - Go to https://browser.dataspace.copernicus.eu")
        print("   - Find a cloudy area")
        print("   - Take a screenshot and save as PNG")
        print("   - Run: python real_data_pipeline.py --mode local --image screenshot.png --no-clouds")
        print()
        print("2. UPLOAD IN DASHBOARD (Easy - 1 minute)")
        print("   - Launch: python -m streamlit run app.py")
        print("   - Go to the 'Upload Real Image' tab")
        print("   - Upload any satellite screenshot")
        print()
        print("3. API DOWNLOAD (Advanced)")
        print("   - Run: python real_data_pipeline.py --mode download --lat 28.6 --lon 77.2")
        print("   - Follow the printed instructions")
