// Copies the MediaPipe vision WASM runtime from node_modules into public/ so the
// face-expression panel is self-hosted (no CDN at runtime). FilesetResolver picks
// the SIMD build when the browser supports it and the no-SIMD build otherwise, so
// only those two pairs are needed.
import { copyFile, mkdir, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const source = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm')
const target = join(root, 'public', 'mediapipe', 'wasm')
const files = [
  'vision_wasm_internal.js',
  'vision_wasm_internal.wasm',
  'vision_wasm_nosimd_internal.js',
  'vision_wasm_nosimd_internal.wasm',
]

async function isFresh(src, dest) {
  try {
    const [a, b] = await Promise.all([stat(src), stat(dest)])
    return a.size === b.size && b.mtimeMs >= a.mtimeMs
  } catch {
    return false
  }
}

await mkdir(target, { recursive: true })
let copied = 0
for (const name of files) {
  const src = join(source, name)
  const dest = join(target, name)
  if (await isFresh(src, dest)) continue
  await copyFile(src, dest)
  copied += 1
}
if (copied > 0) console.log(`mediapipe: copied ${copied} wasm file(s) to public/mediapipe/wasm`)
