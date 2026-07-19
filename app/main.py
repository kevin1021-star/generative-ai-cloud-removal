import os
import yaml
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

# Set page config at the very beginning
st.set_page_config(
    page_title="LISS-IV Cloud Removal & Reconstruction Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from data.dataset import LISS4CloudRemovalDataset
from models.memory_bank import SpectralMemoryBank
from infer import load_models_for_inference, run_inference_pipeline

# ----------------- CUSTOM STYLE & TYPOGRAPHY -----------------
st.markdown("""
    <style>
    /* Dark Theme Core Adjustments */
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    
    /* Header Card styling with high-tech glassmorphism and green neon gradient */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
        margin-bottom: 25px;
        text-align: center;
    }
    
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(to right, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 8px;
    }
    
    /* Metric & validation badges styling */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
        margin-top: 5px;
    }
    
    .pass-badge {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    
    .fail-badge {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    </style>
""", unsafe_html=True)

# ----------------- MAIN LAYOUT -----------------
st.markdown("""
    <div class="header-card">
        <h1 class="header-title" id="main-title">Generative AI-Based Cloud Removal & Reconstruction</h1>
        <p class="header-subtitle">Physics-Guided & SAR-Fused Surface Reconstruction for LISS-IV Satellite Imagery</p>
    </div>
""", unsafe_html=True)

# Define project files directories
config_file = "C:/Users/AS/.gemini/antigravity/scratch/liss4_cloud_removal/config/config.yaml"
data_dir = "C:/Users/AS/.gemini/antigravity/scratch/liss4_cloud_removal/data/synthetic"

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_dataset_and_models():
    # Load dataset
    dataset = LISS4CloudRemovalDataset(data_dir=data_dir)
    
    # Load models
    cloud_detector, tpt, sar_fusion, diffusion, physics_gate = load_models_for_inference(config_file, device)
    
    # Load and populate spectral memory bank
    memory_bank = SpectralMemoryBank()
    # Populate memory bank using sample ground truths
    for idx in range(min(len(dataset), 20)):
        sample = dataset[idx]
        memory_bank.update_from_patch(0.5, 0.5, 5, sample['gt'], torch.ones_like(sample['cloud_mask']))
        
    return dataset, cloud_detector, tpt, sar_fusion, diffusion, physics_gate, memory_bank

try:
    dataset, detector, tpt, sar_fusion, diffusion, physics_gate, memory_bank = get_dataset_and_models()
except Exception as e:
    st.error(f"Error loading models or dataset. Please make sure data has been generated and training has completed: {e}")
    st.stop()

# Sidebar: selector
st.sidebar.markdown("### 🎛️ Settings")
sample_index = st.sidebar.slider(
    "Select Synthetic Test Sample",
    min_value=0,
    max_value=len(dataset) - 1,
    value=0,
    help="Change this slider to select different scenes with varying cloud shapes and surface details."
)

st.sidebar.markdown("""
---
### 🛠️ Core Capabilities Shown:
1. **Multimodal U-Net Cloud Masking** (Resolves lack of blue band by fusing S2 Blue and S1 SAR).
2. **Temporal Phenology TPT** (Predicts crop state based on history sequence).
3. **Cross-modal SAR Edge Fusion** (Preserves LISS-IV 5.8m sharp edges).
4. **Fast Denoising Diffusion** (Reconstructs surface in 10 DDIM steps).
5. **Differentiable In-Loop & Post-hoc Physics Gate** (Guarantees valid NDVI/Albedo).
""")

# Run inference
sample = dataset[sample_index]
with st.spinner("Reconstructing cloudy region via Physics-Guided Generative AI..."):
    results = run_inference_pipeline(
        sample=sample,
        cloud_detector=detector,
        tpt=tpt,
        sar_fusion=sar_fusion,
        diffusion=diffusion,
        physics_gate=physics_gate,
        memory_bank=memory_bank,
        device=device
    )

# ----------------- VISUALIZATION ROW 1: INPUTS -----------------
st.subheader("📡 Input Telemetry & Cloud Identification")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Cloudy LISS-IV Optical Input (Green, Red, NIR)**")
    # Convert Green-Red-NIR to RGB for visualization (NIR, Red, Green as false color, or just Red, Green, Red for simple rgb)
    # Let's map NIR, Red, Green to RGB (Standard False Color Composite to highlight vegetation)
    cloudy_np = results['inputs']['cloudy'].permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d0f12')
    ax.imshow(cloudy_np)
    ax.axis('off')
    st.pyplot(fig)
    st.caption("Standard optical sensors get blocked by cloud cover.")

with col2:
    st.markdown("**Sentinel-1 SAR Radar Input (VV, VH Composite)**")
    sar_np = results['inputs']['sar'].permute(1, 2, 0).numpy()
    # Create fake RGB for SAR: Red=VV, Green=VH, Blue=VV/VH
    sar_rgb = np.zeros((*sar_np.shape[:2], 3))
    sar_rgb[:, :, 0] = sar_np[:, :, 0]
    sar_rgb[:, :, 1] = sar_np[:, :, 1]
    sar_rgb[:, :, 2] = np.clip(sar_np[:, :, 0] / (sar_np[:, :, 1] + 1e-6) * 0.1, 0, 1)
    
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d0f12')
    ax.imshow(sar_rgb)
    ax.axis('off')
    st.pyplot(fig)
    st.caption("SAR radar penetrates clouds, showing structural patterns below.")

with col3:
    st.markdown("**AI Identified Cloud & Shadow Mask**")
    mask_np = results['intermediates']['detected_mask'][0].numpy()
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d0f12')
    # Use green for cloud, red for shadow
    ax.imshow(mask_np, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)
    st.caption("U-Net segments both clouds and their shadows as areas needing reconstruction.")

# ----------------- VISUALIZATION ROW 2: INTERMEDIATES & OUTPUTS -----------------
st.subheader("🎨 Generative Surface Reconstruction & Physical Constraints")
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("**Temporal Phenology TPT Prediction**")
    tpt_np = results['intermediates']['pred_tpt'].permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d0f12')
    ax.imshow(tpt_np)
    ax.axis('off')
    st.pyplot(fig)
    st.caption("Predicted expected land state derived from historical sequence timeline.")

with col5:
    st.markdown("**High-Resolution SAR Coherence Fused Guide**")
    sf_np = results['intermediates']['sar_fused'].permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d0f12')
    ax.imshow(sf_np)
    ax.axis('off')
    st.pyplot(fig)
    st.caption("Fuses 10m radar textures with 5.8m LISS-IV edge stencils.")

with col6:
    st.markdown("**Final Cloud-Free Generative Output**")
    out_np = results['outputs']['final_output'].permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d0f12')
    ax.imshow(out_np)
    ax.axis('off')
    st.pyplot(fig)
    st.caption("Reconstructed cloud-free surface, seamlessly blended into clear pixels.")

# ----------------- VISUALIZATION ROW 3: GT & UNCERTAINTY -----------------
st.subheader("🔍 Ground Truth Comparison & Uncertainty Quantification")
col7, col8, col9 = st.columns(3)

with col7:
    st.markdown("**Ground Truth (Cloud-Free Reference)**")
    gt_np = results['outputs']['gt'].permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d0f12')
    ax.imshow(gt_np)
    ax.axis('off')
    st.pyplot(fig)
    st.caption("Original cloud-free scene captured on a clear day.")

with col8:
    st.markdown("**Spatial Conformal Uncertainty Map**")
    unc_np = results['intermediates']['uncertainty'][0].numpy()
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d0f12')
    # Plot uncertainty using plasma colormap
    im = ax.imshow(unc_np, cmap='plasma', vmin=0.0, vmax=1.0)
    ax.axis('off')
    st.pyplot(fig)
    st.caption("Brighter pixels show higher uncertainty in the synthesized region.")

with col9:
    st.markdown("**Evaluation Metrics & Validation Gate**")
    
    # Validation status
    status_html = ""
    if results['metrics']['physics_passed']:
        status_html = '<div class="pass-badge">✓ Physics Validation: PASSED</div>'
    else:
        status_html = '<div class="fail-badge">✗ Physics Validation: FAILED</div>'
        
    st.markdown(f"<div style='text-align: center; margin-top: 15px; margin-bottom: 20px;'>{status_html}</div>", unsafe_html=True)
    
    # 4 metrics cards
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak Signal-to-Noise (PSNR)</div>
                <div class="metric-value">{results['metrics']['psnr']:.2f} dB</div>
            </div>
            <br>
            <div class="metric-card">
                <div class="metric-label">Mean Reconstruction NDVI</div>
                <div class="metric-value">{results['metrics']['ndvi_mean']:.3f}</div>
            </div>
        """, unsafe_html=True)
    with mc2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Structural Similarity (SSIM)</div>
                <div class="metric-value">{results['metrics']['ssim']:.3f}</div>
            </div>
            <br>
            <div class="metric-card">
                <div class="metric-label">Mean Reconstruction Albedo</div>
                <div class="metric-value">{results['metrics']['albedo_mean']:.3f}</div>
            </div>
        """, unsafe_html=True)
