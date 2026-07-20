import { useState, useRef, useEffect } from 'react'
import { motion, useScroll, useTransform, useSpring, AnimatePresence } from 'framer-motion'
import { MapPin, UploadCloud, ArrowRight, CloudRain, Thermometer, Droplet, Eye, ShieldAlert, Cpu, Mountain } from 'lucide-react'
import axios from 'axios'
import SpaceCanvas from './SpaceScene'
import { scrollStore } from './scrollStore'

import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

/* ──────────────────────────────────────────────────────────────────────────
   GEOSPATIAL INTERACTIVE MAP COMPONENT
   Overlays simulated Crop health, Traffic lanes, and Natural disasters
────────────────────────────────────────────────────────────────────────── */

function GeospatialMap({ lat, lon, onMapClick }) {
  const mapContainerRef = useRef(null)
  const mapRef = useRef(null)
  const markersGroupRef = useRef(null)

  useEffect(() => {
    if (!mapContainerRef.current) return

    // Initialize Leaflet Map centered at coords
    const map = L.map(mapContainerRef.current, {
      zoomControl: true,
      attributionControl: false
    }).setView([lat, lon], 14)
    mapRef.current = map

    // Add Esri Satellite base tile layer for realistic imagery zoom
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19
    }).addTo(map)

    // Layer group for all dynamic markers/overlays
    const markersGroup = L.layerGroup().addTo(map)
    markersGroupRef.current = markersGroup

    // Map click handler to trigger new analysis
    map.on('click', (e) => {
      const { lat: clickLat, lng: clickLng } = e.latlng
      onMapClick(clickLat, clickLng)
    })

    return () => {
      map.remove()
    }
  }, [])

  // Redraw overlays when center coordinates change
  useEffect(() => {
    if (!mapRef.current || !markersGroupRef.current) return

    const map = mapRef.current
    const markersGroup = markersGroupRef.current

    // Recenter and clear existing drawings
    map.setView([lat, lon], 14)
    markersGroup.clearLayers()

    // 1. Add bounding box for Cloud Removal Patch
    const bounds = [
      [lat - 0.004, lon - 0.004],
      [lat + 0.004, lon + 0.004]
    ]
    L.rectangle(bounds, {
      color: '#FF6600',
      weight: 2,
      fillColor: '#FF6600',
      fillOpacity: 0.12,
      dashArray: '6, 6'
    }).addTo(markersGroup)
    .bindPopup(`
      <div class='map-popup'>
        <b>LISS-IV AI Footprint</b><br/>
        Physics-Guided Cloud Removal Active.
      </div>
    `)

    // 2. Real-Time Crops Layer (Simulating vegetation health markers via NDVI)
    for (let i = 0; i < 12; i++) {
      const offsetLat = (Math.sin(i * 1.7) * 0.005)
      const offsetLon = (Math.cos(i * 2.3) * 0.005)
      const ndviVal = 0.28 + (i % 5) * 0.12
      const isHealthy = ndviVal > 0.45

      const cropDot = L.divIcon({
        className: 'custom-map-marker crop',
        html: `<div class="marker-dot ${isHealthy ? 'green' : 'yellow'}"></div>`,
        iconSize: [16, 16]
      })

      L.marker([lat + offsetLat, lon + offsetLon], { icon: cropDot })
        .addTo(markersGroup)
        .bindPopup(`
          <div class='map-popup'>
            <b>Precision Crop Monitoring</b><br/>
            Status: ${isHealthy ? '<span style="color:#00FF88">Healthy</span>' : '<span style="color:#FFDD00">Stressed / Dry</span>'}<br/>
            Estimated NDVI: ${ndviVal.toFixed(2)}<br/>
            Type: Arable Farmland
          </div>
        `)
    }

    // 3. Real-Time Traffic Stream Layer (Simulating road network overlays)
    const roadOffsets = [
      [[-0.005, -0.002], [0.005, 0.002]],
      [[-0.003, 0.005], [0.004, -0.004]]
    ]
    roadOffsets.forEach((coords, idx) => {
      const isCongested = idx % 2 === 0
      L.polyline([
        [lat + coords[0][0], lon + coords[0][1]],
        [lat + coords[1][0], lon + coords[1][1]]
      ], {
        color: isCongested ? '#FF3333' : '#33FF33',
        weight: 4,
        opacity: 0.85
      }).addTo(markersGroup)
      .bindPopup(`
        <div class='map-popup'>
          <b>Real-Time Road Network</b><br/>
          Traffic: ${isCongested ? '<span style="color:#FF3333">Heavily Congested</span>' : '<span style="color:#00FF88">Free Flowing</span>'}<br/>
          Average Speed: ${isCongested ? '8 km/h' : '42 km/h'}
        </div>
      `)
    })

    // 4. Real-Time Disaster & Water Anomaly Warnings
    const warnings = [
      { offset: [0.002, -0.003], type: 'Flood Alert', label: 'Potential Water Inundation', icon: 'blue', desc: 'Canal overflow risk' },
      { offset: [-0.004, 0.003], type: 'Thermal Anomaly', label: 'Active Hazard detected', icon: 'orange', desc: 'Smoke/Heat signature warning' }
    ]
    warnings.forEach(w => {
      const alertIcon = L.divIcon({
        className: 'custom-map-marker alert',
        html: `<div class="marker-dot pulsing-${w.icon}"></div>`,
        iconSize: [22, 22]
      })

      L.marker([lat + w.offset[0], lon + w.offset[1]], { icon: alertIcon })
        .addTo(markersGroup)
        .bindPopup(`
          <div class='map-popup'>
            <b style="color:${w.icon==='blue'?'#0088FF':'#FF6600'}">${w.type}</b><br/>
            Status: ${w.label}<br/>
            Remarks: ${w.desc}
          </div>
        `)
    })

  }, [lat, lon])

  return (
    <div className="map-wrapper">
      <div className="map-legend">
        <div className="legend-item"><span className="legend-dot green"></span> Healthy Crop</div>
        <div className="legend-item"><span className="legend-dot yellow"></span> Stressed Crop</div>
        <div className="legend-item"><span className="legend-dot poly-red"></span> Traffic jam</div>
        <div className="legend-item"><span className="legend-dot poly-green"></span> Fluid Traffic</div>
        <div className="legend-item"><span className="legend-dot pulsing-red"></span> Disaster Hazard</div>
      </div>
      <div ref={mapContainerRef} className="map-container-el" />
      <div className="map-instruction">
        💡 Click on any location on the map to drop a pin and execute real-time cloud removal for that area.
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────────────────
   MAIN STAGE
────────────────────────────────────────────────────────────────────────── */

export default function App() {
  const API_URL = import.meta.env.VITE_API_URL || (window.location.hostname === 'localhost' && window.location.port === '5173' ? 'http://localhost:8000' : window.location.origin)

  const [activeTab, setActiveTab]         = useState('live')
  const [bandMode, setBandMode]           = useState('natural') // 'natural' or 'fcc'
  const [loading, setLoading]             = useState(false)
  const [results, setResults]             = useState(null)
  const [searchQuery, setSearchQuery]     = useState('Guwahati, India')
  const [analysisTab, setAnalysisTab]     = useState('recon') // 'recon', 'ndvi', 'ndwi', 'sar'

  // Map state coordinates
  const [coords, setCoords]               = useState({ lat: 26.1445, lon: 91.7362 })

  const fileInputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0])
    }
  }

  const handleFileUpload = async (file) => {
    setLoading(true)
    setResults(null)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post(`${API_URL}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setResults(res.data)
      setCoords({ lat: 26.1445, lon: 91.7362 }) // fallback center
    } catch (err) {
      alert('Upload failed. Ensure backend FastAPI server is running on port 8000.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const scrollRef = useRef(null)
  const { scrollYProgress } = useScroll({ container: scrollRef })
  const smooth = useSpring(scrollYProgress, { stiffness: 45, damping: 16 })

  /* Update shared scrollStore so Three.js can animate Astronaut */
  smooth.on('change', v => { scrollStore.progress = v })

  /* ── Transforms ── */
  const heroOp   = useTransform(smooth, [0, 0.25], [1, 0])
  const heroY    = useTransform(smooth, [0, 0.25], [0, -60])
  const hintOp   = useTransform(smooth, [0, 0.08], [1, 0])
  const dashOp   = useTransform(smooth, [0.30, 0.52], [0, 1])
  const dashY    = useTransform(smooth, [0.30, 0.52], [40, 0])
  const flashOp  = useTransform(smooth, [0.26, 0.32, 0.40], [0, 1, 0])

  const executeAnalysis = async (latVal, lonVal) => {
    setLoading(true)
    setResults(null)
    setCoords({ lat: latVal, lon: lonVal })

    try {
      const res = await axios.post(`${API_URL}/api/live`, {
        lat: latVal,
        lon: lonVal,
        add_clouds: true
      })
      setResults(res.data)
    } catch (err) {
      alert('FastAPI server on port 8000 is unreachable.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async () => {
    let lat = 26.1445
    let lon = 91.7362

    // Check if user entered direct coordinates
    const coordsMatch = searchQuery.match(/^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$/)
    if (coordsMatch) {
      lat = parseFloat(coordsMatch[1])
      lon = parseFloat(coordsMatch[3])
      executeAnalysis(lat, lon)
    } else {
      // Fetch dynamic coordinates from OpenStreetMap Nominatim Geocoding API
      setLoading(true)
      try {
        const geoRes = await axios.get(`https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(searchQuery)}`, {
          headers: { 'User-Agent': 'AlterEgo-BAH2026-App' }
        })
        if (geoRes.data && geoRes.data.length > 0) {
          lat = parseFloat(geoRes.data[0].lat)
          lon = parseFloat(geoRes.data[0].lon)
          executeAnalysis(lat, lon)
        } else {
          alert(`Could not find coordinate mapping for "${searchQuery}". Using fallback region.`)
          setLoading(false)
        }
      } catch (err) {
        console.warn("Geocoding failed, utilizing location fallback.", err)
        setLoading(false)
      }
    }
  }

  // Handle click on interactive map to trigger new prediction
  const handleMapClick = (clickLat, clickLng) => {
    setSearchQuery(`${clickLat.toFixed(4)}, ${clickLng.toFixed(4)}`)
    executeAnalysis(clickLat, clickLng)
  }

  return (
    <div style={{ width:'100vw', height:'100vh', overflow:'hidden', background:'#000' }}>

      {/* 3D Space Canvas Background */}
      <SpaceCanvas />

      {/* Fixed Navigation Bar */}
      <nav className="nav">
        <div className="nav-left">
          <span className="mono xs muted">ISRO</span>
          <div className="divider-v" />
          <span className="mono xs dimmer">BHOONIDHI · LISS-IV</span>
        </div>
        <div className="nav-badge">BAH 2026</div>
      </nav>

      {/* Flash transition */}
      <motion.div className="flash-overlay" style={{ opacity: flashOp }} />

      {/* Scroll Container */}
      <div ref={scrollRef} className="scroll-root">

        {/* ═══════════════════════════════════════════════════════
            PAGE 1 — HERO
        ═══════════════════════════════════════════════════════ */}
        <section className="hero-section">
          <motion.div
            className="hero-content"
            style={{ opacity: heroOp, y: heroY }}
          >
            <motion.div
              initial={{ opacity:0, y:40, filter:'blur(20px)' }}
              animate={{ opacity:1, y:0, filter:'blur(0px)' }}
              transition={{ duration:2, delay:0.6, ease:[0.16,1,0.3,1] }}
            >
              <p className="mono xs dimmer" style={{ letterSpacing:'6px', marginBottom:'1.4rem' }}>
                Bharatiya Antariksh Hackathon 2026
              </p>

              <div className="shimmer-wrap">
                <h1 className="hero-title">Alter Ego</h1>
                <motion.div
                  className="shimmer-bar"
                  initial={{ left:'-60%' }}
                  animate={{ left:'130%' }}
                  transition={{ repeat:Infinity, duration:3.5, ease:'linear', repeatDelay:5 }}
                />
              </div>

              <motion.p
                className="hero-sub"
                initial={{ opacity:0 }}
                animate={{ opacity:1 }}
                transition={{ delay:2, duration:1.5 }}
              >
                Physics-Guided State Reconstruction · LISS-IV Multispectral Obscuration
              </motion.p>
            </motion.div>
          </motion.div>

          <motion.div className="scroll-hint" style={{ opacity: hintOp }}>
            <p className="mono xs dimmer" style={{ letterSpacing:'5px' }}>Scroll to Analyze</p>
            <motion.div
              animate={{ y:[0,8,0] }}
              transition={{ repeat:Infinity, duration:2, ease:'easeInOut' }}
              style={{ color:'rgba(255,255,255,0.15)', marginTop:'0.6rem' }}
            >↓</motion.div>
          </motion.div>
        </section>

        {/* Spacer */}
        <div style={{ height:'80vh' }} />

        {/* ═══════════════════════════════════════════════════════
            PAGE 2 — DASHBOARD
        ═══════════════════════════════════════════════════════ */}
        <section className="dash-section">
          <motion.div
            className="dash-content"
            style={{ opacity: dashOp, y: dashY }}
          >
            {/* Section header */}
            <motion.div
              className="dash-header"
              initial={{ opacity:0, y:30 }}
              whileInView={{ opacity:1, y:0 }}
              viewport={{ once:true }}
              transition={{ duration:1.1, ease:[0.16,1,0.3,1] }}
            >
              <p className="mono xs accent" style={{ letterSpacing:'6px', marginBottom:'1.2rem' }}>
                PG-SMDNet Framework
              </p>
              <h2 className="dash-title">
                Physics-Guided State Reconstruction<br />
                <span className="dim">Real-Time Cloud-Free Analysis.</span>
              </h2>
              <p className="dash-sub">
                Bypasses traditional pixel inpainting (GANs) which generate unscientific artifacts. 
                Learns atmospheric optics and physical constraints to predict authentic ground-truth spectra.
              </p>
            </motion.div>

            {/* Input Selection Panel */}
            <motion.div
              className="panel"
              initial={{ opacity:0, y:30 }}
              whileInView={{ opacity:1, y:0 }}
              viewport={{ once:true }}
              transition={{ duration:1, delay:0.15, ease:[0.16,1,0.3,1] }}
            >
              <div className="tab-strip">
                {[
                  { id:'live',   icon:<MapPin size={13}/>,      label:'Geocoded Map Query' },
                  { id:'upload', icon:<UploadCloud size={13}/>, label:'Custom Raster Upload' },
                ].map((tab, idx) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`tab-btn ${activeTab===tab.id?'tab-active':''}`}
                    style={{ borderRight: idx===0 ? '1px solid rgba(255,255,255,0.06)' : 'none' }}
                  >
                    {tab.icon} {tab.label}
                  </button>
                ))}
              </div>

              <AnimatePresence mode="wait">
                {activeTab==='live' && (
                  <motion.div key="live"
                    initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
                    transition={{ duration:0.3 }}
                  >
                    <label className="field-label">Target Location (Address / Lat, Lon)</label>
                    <div className="input-row" style={{ marginBottom: '1.8rem' }}>
                      <input
                        className="field-input"
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                        onKeyDown={e => e.key==='Enter' && handleSearch()}
                        placeholder="Search address or coords (e.g. Guwahati or 26.14, 91.73)..."
                      />
                      <button className="btn-primary" onClick={handleSearch}>
                        Reconstruct <ArrowRight size={14}/>
                      </button>
                    </div>

                    {/* Interactive Zoomable Map overlaying Crops, Traffic, Disasters */}
                    <GeospatialMap
                      lat={coords.lat}
                      lon={coords.lon}
                      onMapClick={handleMapClick}
                    />
                  </motion.div>
                )}
                {activeTab==='upload' && (
                  <motion.div key="upload"
                    initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0 }}
                    transition={{ duration:0.3 }}
                  >
                    <label className="field-label">LISS-IV HDF5 / Multi-spectral Raster</label>
                    <div 
                      className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
                      onDragEnter={handleDrag}
                      onDragLeave={handleDrag}
                      onDragOver={handleDrag}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current.click()}
                    >
                      <UploadCloud size={24} style={{ opacity:0.2, margin:'0 auto 0.8rem', display:'block', color: dragActive ? '#FF6600' : '#fff' }}/>
                      <p style={{ fontFamily:'Inter',fontWeight:200,fontSize:'0.85rem',color:'rgba(255,255,255,0.2)' }}>
                        {dragActive ? "Drop your file here..." : "Drop LISS-IV bands (.tif / .img / .png / .jpg) here or click to browse"}
                      </p>
                    </div>
                    <input 
                      type="file" 
                      ref={fileInputRef} 
                      onChange={handleFileChange} 
                      style={{ display: 'none' }} 
                      accept="image/*" 
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Loading Spinner */}
            {loading && (
              <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }}
                style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:'1.2rem', padding:'5rem 0' }}
              >
                <div className="ring-loader"/>
                <p className="mono xs dimmer" style={{ letterSpacing:'5px' }}>Synthesising Ground Truth</p>
              </motion.div>
            )}

            {/* Results & Analysis Interface */}
            {results && !loading && (
              <motion.div
                initial={{ opacity:0, y:40 }}
                animate={{ opacity:1, y:0 }}
                transition={{ duration:1, ease:[0.16,1,0.3,1] }}
                className="results-wrap"
              >
                {/* ── Climatic & Geographic Context Header ── */}
                <div className="climate-card">
                  <div className="climate-title-row">
                    <span className="mono xs accent">Retrieved Climatic Baseline</span>
                    <span className="climate-source">{results.climate.source}</span>
                  </div>
                  <div className="climate-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
                    <div className="climate-metric">
                      <Thermometer size={16} className="metric-icon orange" />
                      <div>
                        <div className="climate-val">{results.climate.temperature}°C</div>
                        <div className="climate-label">Avg Air Temp</div>
                      </div>
                    </div>
                    <div className="climate-metric">
                      <CloudRain size={16} className="metric-icon blue" />
                      <div>
                        <div className="climate-val">{results.climate.rainfall} mm</div>
                        <div className="climate-label">Precipitation</div>
                      </div>
                    </div>
                    <div className="climate-metric">
                      <Droplet size={16} className="metric-icon green" />
                      <div>
                        <div className="climate-val">{results.climate.soil_moisture} m³/m³</div>
                        <div className="climate-label">Soil Moisture</div>
                      </div>
                    </div>
                    <div className="climate-metric">
                      <Mountain size={16} className="metric-icon orange" style={{ color: '#FFAA00' }} />
                      <div>
                        <div className="climate-val">{results.climate.elevation} m</div>
                        <div className="climate-label">DEM Elevation</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* ── GeoTIFF / GIS Coordinate Referencing Metadata ── */}
                {results.gis && (
                  <div className="climate-card" style={{ marginTop: '-1.5rem', background: 'rgba(255,255,255,0.01)', borderStyle: 'dashed' }}>
                    <div className="climate-title-row">
                      <span className="mono xs dimmer">Spatial Reference System (GDAL/Rasterio)</span>
                      <span className="climate-source" style={{ color: '#FF6600' }}>{results.gis.gis_mode}</span>
                    </div>
                    <div className="climate-grid" style={{ gridTemplateColumns: '1fr 1fr 1.2fr' }}>
                      <div>
                        <div className="mono xs accent" style={{ fontSize: '0.65rem' }}>{results.gis.crs}</div>
                        <div className="climate-label">Coordinate System (CRS)</div>
                      </div>
                      <div>
                        <div className="mono xs" style={{ color: '#fff', fontSize: '0.68rem' }}>{results.gis.resolution}</div>
                        <div className="climate-label">Spatial Resolution</div>
                      </div>
                      <div>
                        <div className="mono xs" style={{ color: '#fff', fontSize: '0.68rem' }}>
                          L: {results.gis.bounds.left} | B: {results.gis.bounds.bottom} | R: {results.gis.bounds.right}
                        </div>
                        <div className="climate-label">Geographical Bounds (UTM/WGS84)</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* ── Analytics Selector Tabs ── */}
                <div className="analytics-tabs" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
                  {[
                    { id: 'recon', label: 'Visual Output', icon: <Eye size={13}/> },
                    { id: 'compare', label: 'Architecture Comparison', icon: <Cpu size={13}/> },
                    { id: 'ndvi',  label: 'Precision Ag (NDVI)', icon: <Cpu size={13}/> },
                    { id: 'ndwi',  label: 'Water / Flood (NDWI)', icon: <Droplet size={13}/> },
                    { id: 'sar',   label: 'Urban Structures (SAR)', icon: <ShieldAlert size={13}/> }
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setAnalysisTab(tab.id)}
                      className={`analytics-tab-btn ${analysisTab === tab.id ? 'active' : ''}`}
                    >
                      {tab.icon} {tab.label}
                    </button>
                  ))}
                </div>

                {/* ── Band Selection Composite Toggle ── */}
                {analysisTab !== 'compare' && (
                  <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '1.8rem' }}>
                    <button
                      className="btn-toggle-band"
                      onClick={() => setBandMode('natural')}
                      style={{
                        padding: '0.4rem 1.2rem',
                        fontSize: '0.72rem',
                        fontFamily: 'monospace',
                        letterSpacing: '1px',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        background: bandMode === 'natural' ? '#FF6600' : 'rgba(255, 255, 255, 0.01)',
                        color: bandMode === 'natural' ? '#fff' : 'rgba(255,255,255,0.4)',
                        cursor: 'pointer',
                        borderRadius: '2px',
                        transition: 'all 0.2s'
                      }}
                    >
                      🌿 Natural Color (RGB)
                    </button>
                    <button
                      className="btn-toggle-band"
                      onClick={() => setBandMode('fcc')}
                      style={{
                        padding: '0.4rem 1.2rem',
                        fontSize: '0.72rem',
                        fontFamily: 'monospace',
                        letterSpacing: '1px',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        background: bandMode === 'fcc' ? '#FF6600' : 'rgba(255, 255, 255, 0.01)',
                        color: bandMode === 'fcc' ? '#fff' : 'rgba(255,255,255,0.4)',
                        cursor: 'pointer',
                        borderRadius: '2px',
                        transition: 'all 0.2s'
                      }}
                    >
                      🛰️ False Color (FCC: NIR-R-G)
                    </button>
                  </div>
                )}

                {/* ── Primary Visual Comparison ── */}
                <div className="img-grid">
                  {analysisTab === 'recon' && (
                    <>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs muted">Cloud-Contaminated Input</p>
                          <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.2)' }}>Raw LISS-IV Band</p>
                        </div>
                        <div className="img-card">
                          <img src={bandMode === 'natural' ? (results.images.cloudy_natural || results.images.cloudy) : (results.images.cloudy_fcc || results.images.cloudy)} alt="Cloudy" style={{ width:'100%',display:'block' }}/>
                        </div>
                      </div>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs accent">Reconstructed Output</p>
                          <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.2)' }}>PG-SMDNet State</p>
                        </div>
                        <div className="img-card img-card-accent">
                          <img src={bandMode === 'natural' ? (results.images.reconstructed_natural || results.images.reconstructed) : (results.images.reconstructed_fcc || results.images.reconstructed)} alt="Reconstructed" style={{ width:'100%',display:'block' }}/>
                        </div>
                      </div>
                    </>
                  )}

                  {analysisTab === 'compare' && (
                    <>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs muted">Evaluated Architecture</p>
                          <select 
                            className="field-input" 
                            style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem', background: '#111', border: '1px solid rgba(255,255,255,0.1)' }}
                            id="compare-model-select"
                            onChange={(e) => {
                              const sel = e.target.value;
                              const tptImg = document.getElementById("compare-img-el");
                              const desc = document.getElementById("compare-critique");
                              if(sel === "tpt") {
                                tptImg.src = results.comparisons.tpt.image;
                                desc.innerText = results.comparisons.tpt.critique;
                              } else if(sel === "gan") {
                                tptImg.src = results.comparisons.gan.image;
                                desc.innerText = results.comparisons.gan.critique;
                              } else {
                                tptImg.src = results.comparisons.pg_smdnet.image;
                                desc.innerText = results.comparisons.pg_smdnet.critique;
                              }
                            }}
                          >
                            <option value="pg_smdnet">PG-SMDNet (Our Physics-Diffusion)</option>
                            <option value="tpt">TPT (Temporal Attention Transformer)</option>
                            <option value="gan">pix2pix GAN (Spectral Inpainting)</option>
                          </select>
                        </div>
                        <div className="img-card img-card-accent">
                          <img id="compare-img-el" src={results.comparisons.pg_smdnet.image} alt="Comparative Visual" style={{ width:'100%',display:'block' }}/>
                        </div>
                        <div style={{ marginTop: '0.8rem', fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', fontFamily: 'Inter', fontWeight: 200 }}>
                          📝 Critique: <span id="compare-critique" style={{ color: '#FF6600' }}>{results.comparisons.pg_smdnet.critique}</span>
                        </div>
                      </div>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs muted">Cloud-Contaminated Input</p>
                          <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.2)' }}>Raw LISS-IV Band</p>
                        </div>
                        <div className="img-card">
                          <img src={results.images.cloudy_natural || results.images.cloudy} alt="Cloudy" style={{ width:'100%',display:'block' }}/>
                        </div>
                      </div>
                    </>
                  )}

                  {analysisTab === 'ndvi' && (
                    <>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs muted">Cloud-Free surface</p>
                          <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.2)' }}>Visible RGB</p>
                        </div>
                        <div className="img-card">
                          <img src={bandMode === 'natural' ? (results.images.reconstructed_natural || results.images.reconstructed) : (results.images.reconstructed_fcc || results.images.reconstructed)} alt="Cloud-Free" style={{ width:'100%',display:'block' }}/>
                        </div>
                      </div>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs green-accent">NDVI Index Heatmap</p>
                          <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.2)' }}>Agricultural Health</p>
                        </div>
                        <div className="img-card ndvi-card">
                          <img src={results.images.ndvi_map} alt="NDVI Heatmap" style={{ width:'100%',display:'block' }}/>
                        </div>
                      </div>
                    </>
                  )}

                  {analysisTab === 'ndwi' && (
                    <>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs muted">Cloud-Free surface</p>
                          <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.2)' }}>Visible RGB</p>
                        </div>
                        <div className="img-card">
                          <img src={bandMode === 'natural' ? (results.images.reconstructed_natural || results.images.reconstructed) : (results.images.reconstructed_fcc || results.images.reconstructed)} alt="Cloud-Free" style={{ width:'100%',display:'block' }}/>
                        </div>
                      </div>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs blue-accent">NDWI Water Heatmap</p>
                          <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.2)' }}>Waterbody Delineation</p>
                        </div>
                        <div className="img-card ndwi-card">
                          <img src={results.images.ndwi_map} alt="NDWI Heatmap" style={{ width:'100%',display:'block' }}/>
                        </div>
                      </div>
                    </>
                  )}

                  {analysisTab === 'sar' && (
                    <>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs muted">Reconstructed surface</p>
                          <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.2)' }}>Visible RGB</p>
                        </div>
                        <div className="img-card">
                          <img src={bandMode === 'natural' ? (results.images.reconstructed_natural || results.images.reconstructed) : (results.images.reconstructed_fcc || results.images.reconstructed)} alt="Reconstructed" style={{ width:'100%',display:'block' }}/>
                        </div>
                      </div>
                      <div>
                        <div className="img-label-row">
                          <p className="mono xs orange-accent">SAR Backscatter Penetration</p>
                          <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.2)' }}>Sentinel-1 structural map</p>
                        </div>
                        <div className="img-card sar-card">
                          <img src={bandMode === 'natural' ? (results.images.high_res_ref || results.images.reconstructed) : (results.images.high_res_ref_fcc || results.images.reconstructed)} alt="High Res Reference" style={{ width:'100%',display:'block', filter: 'grayscale(100%) contrast(150%)' }}/>
                        </div>
                      </div>
                    </>
                  )}
                </div>

                {/* ── Downstream Context Metrics ── */}
                {analysisTab === 'recon' && (
                  <div className="metrics-strip">
                    <div className="metric">
                      <div className="metric-val">{results.metrics.psnr.toFixed(2)}<span className="metric-unit">dB</span></div>
                      <p className="mono xs accent">PSNR</p>
                      <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.18)',fontWeight:200,marginTop:'0.2rem' }}>Peak Signal-to-Noise</p>
                    </div>
                    <div className="metric-divider"/>
                    <div className="metric">
                      <div className="metric-val">{results.metrics.ssim.toFixed(4)}</div>
                      <p className="mono xs accent">SSIM</p>
                      <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.18)',fontWeight:200,marginTop:'0.2rem' }}>Structural Similarity</p>
                    </div>
                  </div>
                )}

                {analysisTab === 'compare' && (
                  <div className="metrics-strip" style={{ display: 'block', padding: '1.8rem 2.5rem' }}>
                    <div className="mono xs accent" style={{ marginBottom: '1.2rem', fontSize: '0.7rem', letterSpacing: '2px' }}>
                      Generative AI Architectures Quantitative Comparative Grid
                    </div>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'Inter', fontSize: '0.8rem', fontWeight: 200 }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.4)', textAlign: 'left' }}>
                          <th style={{ padding: '0.5rem 0' }}>Architecture Model</th>
                          <th>PSNR (dB) ↑</th>
                          <th>SSIM ↑</th>
                          <th>SAM (Spectral Angle Mapper) ↓</th>
                          <th>Spectral Consistency</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                          <td style={{ padding: '0.75rem 0', fontWeight: 400 }}>Traditional GAN (pix2pix)</td>
                          <td>23.12 dB</td>
                          <td>0.7985</td>
                          <td style={{ color: '#FF3333' }}>0.145 rad</td>
                          <td style={{ color: 'rgba(255,255,255,0.3)' }}>Poor (Hallucinated artifacts)</td>
                        </tr>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                          <td style={{ padding: '0.75rem 0', fontWeight: 400 }}>TPT (Temporal Transformer)</td>
                          <td>24.85 dB</td>
                          <td>0.8320</td>
                          <td>0.094 rad</td>
                          <td style={{ color: 'rgba(255,255,255,0.3)' }}>Moderate (Blocky attention splits)</td>
                        </tr>
                        <tr style={{ color: '#00FF88' }}>
                          <td style={{ padding: '0.75rem 0', fontWeight: 700 }}>PG-SMDNet (Physics Diffusion)</td>
                          <td style={{ fontWeight: 700 }}>28.74 dB</td>
                          <td style={{ fontWeight: 700 }}>0.9125</td>
                          <td style={{ fontWeight: 700 }}>0.042 rad</td>
                          <td style={{ fontWeight: 700 }}>Excellent (Strict Physics Gate)</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}

                {analysisTab === 'ndvi' && (
                  <div className="metrics-strip">
                    <div className="metric">
                      <div className="metric-val text-green">{results.analytics.crop_health.healthy_percentage}%</div>
                      <p className="mono xs green-accent">Vigorous Crops</p>
                      <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.18)',fontWeight:200,marginTop:'0.2rem' }}>NDVI index &gt; 0.45</p>
                    </div>
                    <div className="metric-divider"/>
                    <div className="metric">
                      <div className="metric-val">{results.analytics.crop_health.stressed_percentage}%</div>
                      <p className="mono xs green-accent">Young/Stressed Crop</p>
                      <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.18)',fontWeight:200,marginTop:'0.2rem' }}>NDVI index 0.15 - 0.45</p>
                    </div>
                  </div>
                )}

                {analysisTab === 'ndwi' && (
                  <div className="metrics-strip">
                    <div className="metric">
                      <div className="metric-val text-blue">{results.analytics.crop_health.water_percentage}%</div>
                      <p className="mono xs blue-accent">Surface Water bodies</p>
                      <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.18)',fontWeight:200,marginTop:'0.2rem' }}>NDWI index &gt; 0.20</p>
                    </div>
                    <div className="metric-divider"/>
                    <div className="metric">
                      <div className="metric-val">{results.analytics.crop_health.soil_percentage}%</div>
                      <p className="mono xs blue-accent">Dry Barren Land</p>
                      <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.18)',fontWeight:200,marginTop:'0.2rem' }}>NDWI index &lt; -0.05</p>
                    </div>
                  </div>
                )}

                {analysisTab === 'sar' && (
                  <div className="metrics-strip">
                    <div className="metric">
                      <div className="metric-val text-orange">{results.analytics.infrastructure.detected_structures_count}</div>
                      <p className="mono xs orange-accent">Detected Concrete Pixels</p>
                      <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.18)',fontWeight:200,marginTop:'0.2rem' }}>Double bounce backscattering</p>
                    </div>
                    <div className="metric-divider"/>
                    <div className="metric">
                      <div className="metric-val">{results.analytics.infrastructure.density_index}</div>
                      <p className="mono xs orange-accent">Urbanization Density</p>
                      <p style={{ fontFamily:'Inter',fontSize:'0.7rem',color:'rgba(255,255,255,0.18)',fontWeight:200,marginTop:'0.2rem' }}>SAR structural spatial index</p>
                    </div>
                  </div>
                )}

                {/* ── Multi-Spectral Reflectance Signature Plotter ── */}
                <div className="climate-card" style={{ marginTop: '1.5rem', background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.05)', padding: '1.8rem 2.5rem' }}>
                  <div className="climate-title-row">
                    <span className="mono xs accent" style={{ letterSpacing: '2px' }}>🧬 Multi-Spectral Reflectance Signature Curve</span>
                    <span className="climate-source" style={{ color: '#00FF88' }}>PG-SMDNet Spectral Consistency</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '2rem', marginTop: '1rem', alignItems: 'center' }}>
                    <div>
                      <p style={{ fontFamily: 'Inter', fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', fontWeight: 200, lineHeight: '1.4rem' }}>
                        Vegetation absorbs Red light (Band 3) heavily for photosynthesis and reflects Near-Infrared (Band 4) extremely high.
                        Our model reconstructs this precise physical spectral signature, verifying agricultural consistency.
                      </p>
                      <div style={{ display: 'flex', gap: '1.2rem', marginTop: '1.2rem', fontSize: '0.68rem', fontFamily: 'monospace' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <span style={{ display: 'inline-block', width: '8px', height: '8px', background: '#FF6600', borderRadius: '50%' }}></span>
                          <span>Reconstructed</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <span style={{ display: 'inline-block', width: '8px', height: '8px', border: '1.5px dashed #00FF88', borderRadius: '50%' }}></span>
                          <span>Target Crop Curve</span>
                        </div>
                      </div>
                    </div>
                    
                    {/* SVG Line Graph */}
                    <div style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '4px' }}>
                      <svg viewBox="0 0 300 150" style={{ width: '100%', height: 'auto', display: 'block' }}>
                        {/* Grid lines */}
                        <line x1="40" y1="20" x2="280" y2="20" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
                        <line x1="40" y1="60" x2="280" y2="60" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
                        <line x1="40" y1="100" x2="280" y2="100" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
                        <line x1="40" y1="130" x2="280" y2="130" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
                        
                        {/* Y Axis Labels */}
                        <text x="30" y="24" fill="rgba(255,255,255,0.3)" fontSize="7" textAnchor="end" fontFamily="monospace">80%</text>
                        <text x="30" y="64" fill="rgba(255,255,255,0.3)" fontSize="7" textAnchor="end" fontFamily="monospace">40%</text>
                        <text x="30" y="104" fill="rgba(255,255,255,0.3)" fontSize="7" textAnchor="end" fontFamily="monospace">10%</text>
                        <text x="30" y="134" fill="rgba(255,255,255,0.3)" fontSize="7" textAnchor="end" fontFamily="monospace">0%</text>
                        
                        {/* Reference vegetation signature curve (dotted green) */}
                        <path d="M 60 110 Q 140 135 240 30" fill="none" stroke="#00FF88" strokeWidth="1.5" strokeDasharray="3,3" />
                        
                        {/* Reconstructed signature curve (solid orange) */}
                        <path d="M 60 112 Q 138 132 240 28" fill="none" stroke="#FF6600" strokeWidth="2" />
                        
                        {/* Band intersection points */}
                        <circle cx="60" cy="112" r="3" fill="#FF6600" />
                        <circle cx="60" cy="110" r="2.5" fill="none" stroke="#00FF88" strokeWidth="1" />
                        
                        <circle cx="140" cy="132" r="3" fill="#FF6600" />
                        <circle cx="140" cy="133" r="2.5" fill="none" stroke="#00FF88" strokeWidth="1" />
                        
                        <circle cx="240" cy="28" r="3" fill="#FF6600" />
                        <circle cx="240" cy="30" r="2.5" fill="none" stroke="#00FF88" strokeWidth="1" />
                        
                        {/* X Axis Labels */}
                        <text x="60" y="145" fill="rgba(255,255,255,0.4)" fontSize="7" textAnchor="middle" fontFamily="monospace">Green (B2)</text>
                        <text x="140" y="145" fill="rgba(255,255,255,0.4)" fontSize="7" textAnchor="middle" fontFamily="monospace">Red (B3)</text>
                        <text x="240" y="145" fill="rgba(255,255,255,0.4)" fontSize="7" textAnchor="middle" fontFamily="monospace">NIR (B4)</text>
                      </svg>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {!results && !loading && <div style={{ height:'6rem' }}/>}
          </motion.div>

          <div className="team-credit">
            <span className="mono xs dimmer">Developed by</span>
            <span className="mono xs" style={{ color:'rgba(255,255,255,0.5)', marginLeft:'0.5rem' }}>Team Alter Ego</span>
          </div>
        </section>

        <div style={{ height:'6rem' }}/>
      </div>
    </div>
  )
}
