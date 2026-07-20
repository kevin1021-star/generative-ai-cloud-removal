# 🛰️ PG-SMDNet: Physics-Guided State Reconstruction and Cloud Removal for LISS-IV Satellite Imagery

Alter Ego's state-of-the-art, hackathon-winning framework for automated real-time cloud removal, surface reconstruction, and downstream analytics (NDVI vegetation tracking, NDWI water body delineation, and Sentinel-1 SAR structure maps) in ISRO LISS-IV multi-spectral imagery.

---

## 🚀 Ultra-Premium Presentation Stack (React + FastAPI) — *Primary Presentation*

Our primary user interface is an ultra-premium, full-stack React dashboard featuring interactive Leaflet maps, live OpenStreetMap Nominatim geocoding, real-time Open-Meteo & SRTM DEM elevation queries, and side-by-side Generative AI model comparisons.

### 1. Start the FastAPI AI Backend
The PyTorch models are loaded in memory via an asynchronous FastAPI REST server.
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python api.py
```
*Backend runs on: `http://localhost:8000`*

### 2. Start the React Frontend
Our presentation page featuring Framer Motion micro-animations, interactive maps, and a 3D zero-gravity floating astronaut with cursor-responsive tracking.
```bash
# Navigate to the frontend directory
cd frontend

# Install Node modules (if not already done)
npm install

# Start the Vite development server
npm run dev
```
*Frontend runs on: `http://localhost:5173`*

---

## 📊 Fallback Sandbox Interface (Streamlit) — *Alternative Presentation*

If you wish to view the legacy single-page Streamlit mockup (which is what opens by default when running `app.py`), you can run the following fallback command from the project root:

```bash
python -m streamlit run app.py
```
*Streamlit dashboard runs on: `http://localhost:8501`*

---

## 🎨 Key Features & Architecture
- **Multi-Modal Diffusion Model**: Blends Sentinel-1 SAR coherence, Sentinel-2 SWIR/optical imagery, and 5-frame LISS-IV temporal reference sequences.
- **Physics Validation Gate**: Implements in-loop differentiable loss functions matching physical constants (Albedo, NDVI indexes, and SAR spatial gradients).
- **Interactive Nominatim Geocoding**: Type in any address (like "Guwahati" or "New Delhi"), geocode it, and retrieve live contextual climate/elevation context.
- **Dynamic Leaflet Layering**: Drop pins on actual satellite locations to run inference in real-time.
- **Comparative AI Tab**: Directly compares our **PG-SMDNet** model outputs and metrics against **Traditional GANs (pix2pix)** and **TPT Transformers** side-by-side.
