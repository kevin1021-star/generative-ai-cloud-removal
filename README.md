# 🛰️ PG-SMDNet: Physics-Guided Cloud Removal & State Reconstruction

Alter Ego's state-of-the-art framework for automated real-time cloud removal, surface reconstruction, and multi-spectral analytics (NDVI vegetation tracking, NDWI water body delineation, and Sentinel-1 SAR structure maps) in ISRO LISS-IV satellite imagery.

Built for the **Bharatiya Antariksh Hackathon 2026** (ISRO / Bhoonidhi).

---

## 🌐 Live Deployments
* **Interactive Frontend (Vercel):** [https://generative-ai-cloud-removal-4jua.vercel.app](https://generative-ai-cloud-removal-4jua.vercel.app)
* **API Backend (Render):** [https://generative-ai-cloud-removal.onrender.com](https://generative-ai-cloud-removal.onrender.com)

---

## 🎨 Core Architecture
Our framework bypasses traditional pixel inpainting (GANs) which generate unscientific artifacts. Instead, it learns atmospheric optics and physical constraints to reconstruct authentic ground-truth spectra:

* **Sentinel-1 SAR Coherence Fusion:** Fuses microwave radar backscatter (VV/VH) which passes directly through cloud moisture to capture physical structures under the clouds.
* **Temporal Phenology Transformer (TPT):** Learns historical seasonal crop cycles from previous cloud-free images to predict current vegetation states.
* **Physics Validation Gate:** An in-loop validation module that enforces physical constants (Albedo bounds & NDVI vegetation indexes) to prevent the AI from generating unscientific data.

---

## 📊 Interactive Features
* **Leaflet Geocoding Portal:** Enter any location (e.g. `Guwahati, India`) to fetch live Open-Meteo climate records and SRTM DEM elevation data.
* **Band-Composite Toggles:** Switch between **Simulated Natural Color (RGB)** and **Standard False Color Composite (FCC: NIR-Red-Green)** dynamically.
* **Spectral Reflectance Curves:** Click any pixel to plot its reflectance signature (Green, Red, NIR bands) against reference crop curves.
* **Downstream Analytics:** Watch the app automatically calculate crop health distributions and map obscured urban infrastructure.

---

## 🚀 Local Quick Start (1-Click)

The repository comes with one-click startup scripts that boot both servers and open your default browser automatically:

### For Windows:
Double-click the **`run_project.bat`** file in the root folder, or run in PowerShell:
```powershell
.\run_project.bat
```

### For macOS / Linux:
Run in Terminal:
```bash
chmod +x run_project.sh
./run_project.sh
```

---

*Developed by Team Alter Ego | Python 3.10 · PyTorch · FastAPI · React.js · Leaflet*
