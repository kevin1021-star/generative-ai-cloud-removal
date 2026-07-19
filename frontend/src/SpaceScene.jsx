import { useRef, useMemo, Suspense } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Stars, useGLTF, Environment, Decal, useTexture } from '@react-three/drei'
import * as THREE from 'three'
import { scrollStore } from './scrollStore'

/* ──────────────────────────────────────────────────────────────────────────
   3D GLTF ASTRONAUT MODEL WITH CUSTOM DECALS & MOUSE HOVER TRACKING
────────────────────────────────────────────────────────────────────────── */

function Astronaut3D() {
  const ref = useRef()
  
  // Track mouse offset smoothly using a mutable ref
  const mouseOffset = useRef({ x: 0, y: 0 })

  // Load GLTF model
  const { nodes, materials } = useGLTF('/Astronaut.glb')
  
  // Get pointer position
  const { pointer } = useThree()

  // Load flag and logo textures for decals
  const [flagTex, isroTex] = useTexture([
    '/flag_india.png',
    '/isro_logo.png'
  ])

  // Optimize material for deep space reflections
  const material = useMemo(() => {
    const mat = materials['Astronaut_mat'].clone()
    mat.roughness = 0.32
    mat.metalness = 0.85
    return mat
  }, [materials])

  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.elapsedTime
    const p = scrollStore.progress // 0 → 1

    /* ── 1. Sinusoidal Free-Floating Motion ── */
    const floatY = Math.sin(t * 0.4) * 0.30
    const floatX = Math.cos(t * 0.25) * 0.18
    const floatZ = Math.sin(t * 0.18) * 0.12

    /* ── 2. Mouse Pointer Hover Inertia (smooth lerping) ── */
    const targetMouseX = pointer.x * 0.8
    const targetMouseY = pointer.y * 0.6

    mouseOffset.current.x = THREE.MathUtils.lerp(mouseOffset.current.x, targetMouseX, 0.05)
    mouseOffset.current.y = THREE.MathUtils.lerp(mouseOffset.current.y, targetMouseY, 0.05)

    /* ── 3. Combined Positions (Float + Scroll + Hover) ── */
    // Starting Y is -2.0 so the head sits fully inside the viewport
    const scrollX = 2.4 - p * 3.4
    const scrollY = -2.0 + p * 1.5
    const scrollZ = -1.8 - p * 3.0

    ref.current.position.set(
      floatX + scrollX + mouseOffset.current.x,
      floatY + scrollY + mouseOffset.current.y,
      floatZ + scrollZ
    )

    /* ── 4. Rotation / Tilt response to Hover ── */
    // Apply default rotation offsets in combination with the hover tilt
    ref.current.rotation.x = Math.sin(t * 0.08) * 0.12 - mouseOffset.current.y * 0.35
    ref.current.rotation.y = -0.6 + t * 0.05 + mouseOffset.current.x * 0.45 // added default -0.6 rotation
    ref.current.rotation.z = Math.cos(t * 0.06) * 0.08 - mouseOffset.current.x * 0.25
  })

  // nodes['node-0'] is the main mesh in Google's Astronaut.glb
  const meshNode = nodes['node-0']

  return (
    /* Moved scale and default base rotation to the parent group.
       The child mesh is left at scale [1, 1, 1] so Decal projections 
       remain mathematically stable and do not detach or float separately! */
    <group ref={ref} scale={[2.6, 2.6, 2.6]}>
      <mesh
        geometry={meshNode.geometry}
        material={material}
        castShadow
        receiveShadow
      >
        {/* ── Indian Flag Decal on the Right Sleeve / Shoulder ── */}
        <Decal
          /* Position in local unscaled mesh coordinates */
          position={[0.12, 1.4, 0.04]}
          /* Rotation facing the right sleeve */
          rotation={[0, Math.PI / 2, 0]}
          scale={[0.07, 0.05, 0.07]}
          map={flagTex}
        />

        {/* ── ISRO Logo Decal on the Center Chest ── */}
        <Decal
          /* Position in local unscaled mesh coordinates */
          position={[0.0, 1.41, 0.15]}
          rotation={[0, 0, 0]}
          scale={[0.12, 0.05, 0.09]}
          map={isroTex}
        />
      </mesh>
    </group>
  )
}

function StarField() {
  const ref = useRef()
  useFrame(({ clock }) => {
    if (ref.current) ref.current.rotation.y = clock.elapsedTime * 0.005
  })
  return (
    <group ref={ref}>
      <Stars radius={130} depth={50} count={7000} factor={4.5} saturation={0} fade speed={0.4} />
    </group>
  )
}

/* Orange-amber cosmic nebula backdrop */
function NebulaBackground() {
  const tex = useMemo(() => {
    const size = 128
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = size
    const ctx = canvas.getContext('2d')
    const grd = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2)
    grd.addColorStop(0, 'rgba(155, 60, 15, 0.20)')
    grd.addColorStop(0.5, 'rgba(75, 20, 5, 0.06)')
    grd.addColorStop(1, 'rgba(0, 0, 0, 0)')
    ctx.fillStyle = grd
    ctx.fillRect(0, 0, size, size)
    return new THREE.CanvasTexture(canvas)
  }, [])

  return (
    <mesh position={[4, 0, -10]}>
      <planeGeometry args={[16, 12]} />
      <meshBasicMaterial map={tex} transparent opacity={0.4} depthWrite={false} blending={THREE.AdditiveBlending} />
    </mesh>
  )
}

/* ──────────────────────────────────────────────────────────────────────────
   MAIN STAGE
────────────────────────────────────────────────────────────────────────── */

export default function SpaceCanvas() {
  return (
    <Canvas
      style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0 }}
      camera={{ position: [0, 0, 8], fov: 45 }}
      gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.15 }}
      shadows
    >
      <ambientLight intensity={0.25} color="#0b0e14" />
      <directionalLight position={[-6, 4, 8]} intensity={3.5} color="#f4eedc" castShadow />
      <directionalLight position={[6, -2, -4]} intensity={0.6} color="#4488cc" />
      
      <Environment preset="night" />

      <StarField />
      <NebulaBackground />
      
      <Suspense fallback={null}>
        <Astronaut3D />
      </Suspense>
    </Canvas>
  )
}

useGLTF.preload('/Astronaut.glb')
