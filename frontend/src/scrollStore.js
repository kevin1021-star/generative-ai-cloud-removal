// scrollStore.js — module-level ref shared between App and Three.js scene
// No React state = no re-renders, just raw fast reads inside useFrame
export const scrollStore = { progress: 0 }
