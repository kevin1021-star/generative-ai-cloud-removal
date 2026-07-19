"""
Streamlit Dashboard for LISS-IV Cloud Removal and Reconstruction.
A premium web interface to visualize the full end-to-end pipeline.
Run: streamlit run app.py
"""

import os
import sys
import time
import yaml
import torch
import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import LISS4CloudRemovalDataset
from models.memory_bank import SpectralMemoryBank
from infer import load_models_for_inference, run_inference_pipeline
from utils.visualize import (
    create_comparison_figure,
    create_metrics_bar_chart,
    create_ndvi_map,
    tensor_to_rgb
)
from real_data_pipeline import prepare_sample_from_image, preprocess_rgb_image
from live_location_pipeline import prepare_live_sample
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="LISS-IV Cloud Removal AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# Custom CSS for Premium Dark Theme
# -------------------------------------------------------------------
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Status badges */
    .badge-pass {
        display: inline-block;
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .badge-fail {
        display: inline-block;
        background: linear-gradient(135deg, #f5365c, #f56036);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Pipeline steps indicator */
    .pipeline-step {
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #667eea;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #e2e8f0;
    }
    
    .pipeline-step-active {
        background: rgba(102, 126, 234, 0.15);
        border-left: 3px solid #00f260;
    }
    
    /* Section divider */
    .section-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 2rem 0;
    }
    
    /* Image container */
    .img-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Model & Data Caching
# -------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    """Loads all models and datasets. Cached so it only runs once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    data_dir = os.path.join(PROJECT_ROOT, "data", "synthetic")
    
    # Load models
    cloud_detector, tpt, sar_fusion, diffusion, physics_gate = load_models_for_inference(config_path, device)
    
    # Load dataset
    dataset = LISS4CloudRemovalDataset(data_dir=data_dir)
    
    # Initialize memory bank
    memory_bank = SpectralMemoryBank()
    # Populate with first sample as reference
    sample0 = dataset[0]
    memory_bank.update_from_patch(0.5, 0.5, 5, sample0['gt'], torch.ones_like(sample0['cloud_mask']))
    
    return cloud_detector, tpt, sar_fusion, diffusion, physics_gate, memory_bank, dataset, device


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ Control Panel")
    st.markdown("---")
    
    # Load pipeline
    with st.spinner("Loading AI models..."):
        cloud_detector, tpt, sar_fusion, diffusion, physics_gate, memory_bank, dataset, device = load_pipeline()
    
    st.success(f"✅ Models loaded on **{device}**")
    
    # Input mode selection
    st.markdown("### 📷 Input Source")
    input_mode = st.radio("Choose input", ["🧪 Synthetic Dataset", "📤 Upload Real Image", "🌍 Live Location Search"], label_visibility="collapsed")
    
    # Store mode in session state to persist map clicks
    st.session_state['input_mode'] = input_mode
    
    if input_mode == "🧪 Synthetic Dataset":
        sample_idx = st.slider("Sample Index", 0, len(dataset) - 1, 0, help="Choose a sample from the synthetic dataset")
    elif input_mode == "📤 Upload Real Image":
        uploaded_file = st.file_uploader(
            "Upload satellite image",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            help="Upload any satellite screenshot from Copernicus, Google Earth, etc."
        )
        is_already_cloudy = st.checkbox(
            "Image already has clouds",
            value=True,
            help="Check if uploaded image already contains clouds. Uncheck to add synthetic clouds to a clean image."
        )
    else:
        st.markdown("**Search for a location:**")
        search_query = st.text_input("City or coordinates (e.g., 'Guwahati, India')", value="Guwahati, India")
        is_already_cloudy = st.checkbox(
            "Image already has clouds",
            value=False,
            help="If unchecked, we will mathematically inject clouds into the fetched clear satellite view to demonstrate the model."
        )
    
    # DDIM steps slider
    st.markdown("### ⚙️ Inference Settings")
    ddim_steps = st.slider("DDIM Steps", 5, 50, 10, step=5, help="More steps = higher quality, slower inference")
    
    # Visualization mode
    st.markdown("### 🎨 Visualization")
    color_mode = st.radio("Color Composite", ["False Color (NIR-R-G)", "Natural Color-like (R-G-NIR)"])
    
    # Run button
    st.markdown("---")
    run_button = st.button("🚀 Run Cloud Removal", use_container_width=True, type="primary")
    
    # Help section for getting real data
    with st.expander("📡 Where to get satellite images?"):
        st.markdown("""
        **Easiest method (2 min):**
        1. Go to [Copernicus Browser](https://browser.dataspace.copernicus.eu)
        2. Navigate to any location
        3. Find a cloudy Sentinel-2 image
        4. Take a screenshot & save as PNG
        5. Upload it here!
        
        **Other sources:**
        - [Google Earth](https://earth.google.com)
        - [ISRO Bhuvan](https://bhuvan.nrsc.gov.in)
        - [USGS EarthExplorer](https://earthexplorer.usgs.gov)
        """)


# -------------------------------------------------------------------
# Main Content
# -------------------------------------------------------------------
st.markdown('<h1 class="main-header">🛰️ LISS-IV Cloud Removal & Reconstruction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Physics-Guided Generative AI Framework for Satellite Imagery</p>', unsafe_allow_html=True)

# Pipeline Overview
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
pipeline_steps = [
    ("☁️", "Cloud Detection"),
    ("📚", "Memory Bank"),
    ("🌿", "TPT Prediction"),
    ("📡", "SAR Fusion"),
    ("🎨", "Diffusion Decoder"),
    ("✅", "Physics Gate")
]
for col, (icon, name) in zip([col1, col2, col3, col4, col5, col6], pipeline_steps):
    col.markdown(f"""
    <div class="pipeline-step">
        <div style="font-size:1.5rem; text-align:center">{icon}</div>
        <div style="font-size:0.75rem; text-align:center; color:#a0aec0">{name}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Run Inference
# -------------------------------------------------------------------
# Handle Live Map interaction before running
lat, lon = 26.1445, 91.7362 # Default: Guwahati
if st.session_state.get('input_mode') == "🌍 Live Location Search":
    st.markdown("### 📍 Select Target Area")
    st.markdown("Drag the map or use the search bar to find a region, then **click on the map** to select the exact coordinate to process.")
    
    # Geocode search query
    try:
        geolocator = Nominatim(user_agent="liss4_cloud_app")
        location = geolocator.geocode(search_query)
        if location:
            lat, lon = location.latitude, location.longitude
    except:
        pass # Fallback to default
        
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")
    folium.Marker([lat, lon], tooltip="Target Location").add_to(m)
    
    # Render map and get clicks
    map_data = st_folium(m, height=400, width=800)
    if map_data and map_data.get('last_clicked'):
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        st.info(f"Selected coordinates: {lat:.4f}, {lon:.4f}. Click **Run Cloud Removal** to fetch satellite data.")

if run_button:
    # Determine which sample to process
    test_sample = None
    source_label = ""
    
    if input_mode == "🧪 Synthetic Dataset":
        test_sample = dataset[sample_idx]
        source_label = f"Synthetic Sample #{sample_idx}"
    elif input_mode == "📤 Upload Real Image":
        if uploaded_file is not None:
            # Save uploaded file temporarily
            import tempfile
            tmp_dir = os.path.join(PROJECT_ROOT, "outputs", "uploads")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Preprocess the uploaded image
            add_clouds = not is_already_cloudy
            test_sample = prepare_sample_from_image(tmp_path, add_clouds=add_clouds)
            source_label = f"Uploaded: {uploaded_file.name}"
        else:
            st.warning("⚠️ Please upload a satellite image first!")
            
    elif input_mode == "🌍 Live Location Search":
        # Fetch live data using live_location_pipeline
        source_label = f"Live Coordinates: {lat:.4f}, {lon:.4f}"
        st.markdown(f"*Fetching 1-meter satellite data and simulating 5.8m LISS-IV sensor for **{lat:.4f}, {lon:.4f}**...*")
        try:
            add_clouds = not is_already_cloudy
            test_sample = prepare_live_sample(lat, lon, add_clouds=add_clouds)
            st.success("Successfully fetched and simulated LISS-IV imagery!")
        except Exception as e:
            st.error(f"Failed to fetch live imagery: {e}")
            test_sample = None
    
    if test_sample is not None:
        st.markdown(f"*Processing: **{source_label}***")
        
        # Show progress animation
        progress_bar = st.progress(0, text="Initializing pipeline...")
        
        steps_list = [
            "Detecting clouds and shadows...",
            "Querying spectral memory bank...",
            "Running temporal phenology transformer...",
            "Fusing SAR coherence data...",
            "Running physics-guided diffusion decoder...",
            "Validating physical constraints...",
            "Computing quality metrics..."
        ]
        
        for i, step_text in enumerate(steps_list):
            progress_bar.progress((i + 1) / len(steps_list), text=step_text)
            time.sleep(0.3)
        
        # Actually run inference
        with st.spinner("🔬 Processing through all pipeline stages..."):
            results = run_inference_pipeline(
                test_sample, cloud_detector, tpt, sar_fusion,
                diffusion, physics_gate, memory_bank, device
            )
        
        progress_bar.progress(1.0, text="✅ Complete!")
        time.sleep(0.5)
        progress_bar.empty()
        
        # Store results in session state
        st.session_state['results'] = results
        st.session_state['has_results'] = True

# -------------------------------------------------------------------
# Display Results
# -------------------------------------------------------------------
if st.session_state.get('has_results', False):
    results = st.session_state['results']
    metrics = results['metrics']
    
    # Metrics Row
    st.markdown("### 📊 Reconstruction Quality Metrics")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['psnr']:.2f}</div>
            <div class="metric-label">PSNR (dB)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['ssim']:.4f}</div>
            <div class="metric-label">SSIM</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['sam']:.4f}</div>
            <div class="metric-label">SAM (rad)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        badge = "badge-pass" if metrics['physics_passed'] else "badge-fail"
        label = "PASSED ✓" if metrics['physics_passed'] else "REVIEW ⚠"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value"><span class="{badge}">{label}</span></div>
            <div class="metric-label">Physics Gate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Full Comparison Panel
    st.markdown("### 🖼️ Full Pipeline Comparison")
    comparison_img = create_comparison_figure(results)
    st.image(comparison_img, use_container_width=True)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Side-by-Side: Cloudy vs Reconstructed vs Ground Truth
    st.markdown("### 🔍 Side-by-Side Comparison")
    
    band_order = (2, 1, 0) if "False" in color_mode else (1, 0, 2)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**☁️ Cloudy Input**")
        cloudy_img = tensor_to_rgb(results['inputs']['cloudy'], band_order=band_order)
        st.image(cloudy_img, use_container_width=True, clamp=True)
    
    with c2:
        st.markdown("**🎨 AI Reconstructed**")
        recon_img = tensor_to_rgb(results['outputs']['final_output'], band_order=band_order)
        st.image(recon_img, use_container_width=True, clamp=True)
    
    with c3:
        st.markdown("**🎯 Ground Truth**")
        gt_img = tensor_to_rgb(results['outputs']['gt'], band_order=band_order)
        st.image(gt_img, use_container_width=True, clamp=True)
        
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    if 'high_res_ref' in results['inputs'] and results['inputs']['high_res_ref'] is not None:
        st.markdown("### 🗺️ Live High-Res Reference (1m/pixel vs LISS-IV 5.8m/pixel)")
        st.markdown("This shows the actual 1-meter high-resolution tile we fetched. Notice how the Generative AI was able to reconstruct roads and houses even from the much blurrier 5.8-meter LISS-IV simulated input!")
        st.image(results['inputs']['high_res_ref'], use_container_width=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # NDVI & Physics Validation
    st.markdown("### 🌿 Physics Validation Details")
    
    p1, p2 = st.columns(2)
    
    with p1:
        st.markdown("**NDVI Map (Reconstructed)**")
        ndvi_img = create_ndvi_map(results['outputs']['final_output'])
        st.image(ndvi_img, use_container_width=True)
    
    with p2:
        st.markdown("**📈 Quality Metrics Chart**")
        metrics_chart = create_metrics_bar_chart(metrics)
        st.image(metrics_chart, use_container_width=True)
    
    # Physics details table
    st.markdown("#### Physics Validation Summary")
    phys_data = {
        "Metric": ["NDVI Mean", "NDVI Anomaly Rate", "Albedo Mean", "Albedo Anomaly Rate", "Overall"],
        "Value": [
            f"{metrics['ndvi_mean']:.4f}",
            f"{metrics['ndvi_anomaly_rate']*100:.2f}%",
            f"{metrics['albedo_mean']:.4f}",
            f"{metrics['albedo_anomaly_rate']*100:.2f}%",
            "✅ PASSED" if metrics['physics_passed'] else "⚠️ REVIEW"
        ],
        "Threshold": [
            "[-0.1, 0.9]",
            "< 1.0%",
            "[0.0, 0.4]",
            "< 1.0%",
            "All below threshold"
        ],
        "Status": [
            "✅" if -0.1 <= metrics['ndvi_mean'] <= 0.9 else "❌",
            "✅" if metrics['ndvi_anomaly_rate'] < 0.01 else "⚠️",
            "✅" if 0.0 <= metrics['albedo_mean'] <= 0.4 else "❌",
            "✅" if metrics['albedo_anomaly_rate'] < 0.01 else "⚠️",
            "✅" if metrics['physics_passed'] else "⚠️"
        ]
    }
    st.table(phys_data)

else:
    # Welcome state
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #a0aec0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🛰️</div>
        <h3 style="color: #e2e8f0;">Ready to Remove Clouds</h3>
        <p>Select a sample from the sidebar and click <strong>Run Cloud Removal</strong> to begin.</p>
        <br>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div class="pipeline-step" style="max-width: 200px;">
                <strong>Step 1</strong><br>Select a satellite image sample
            </div>
            <div class="pipeline-step" style="max-width: 200px;">
                <strong>Step 2</strong><br>Configure DDIM sampling steps
            </div>
            <div class="pipeline-step" style="max-width: 200px;">
                <strong>Step 3</strong><br>Click Run & watch the AI work
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    LISS-IV Cloud Removal & Reconstruction Framework | Physics-Guided Generative AI<br>
    Built with PyTorch + Streamlit | Synthetic Data Demo
</div>
""", unsafe_allow_html=True)
