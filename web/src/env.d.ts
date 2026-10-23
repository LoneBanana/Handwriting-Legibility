// Augments Vite's ImportMetaEnv (declared by vite/client) with this app's variables.
interface ImportMetaEnv {
  readonly VITE_API_BASE?: string
  readonly VITE_MEDIAPIPE_WASM_BASE?: string
  readonly VITE_FACE_LANDMARKER_MODEL?: string
}
