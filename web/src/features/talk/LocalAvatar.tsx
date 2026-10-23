import { useEffect, useState } from 'react'
import type { AvatarController } from './avatarController'
import { AvatarFace } from './AvatarFace'
import { maxDifference, type Blendshapes } from './blendshapes'
import type { FaceStore } from './faceStore'

/**
 * Runs the built-in avatar's animation loop: each frame the controller
 * produces blendshapes, the face is redrawn when they change, and the
 * expression panel's store gets the exact values (ground truth).
 */
export function LocalAvatar({ controller, store }: { controller: AvatarController; store: FaceStore }) {
  const [shapes, setShapes] = useState<Blendshapes>(() => controller.current)

  useEffect(() => {
    let frame = 0
    let shown = controller.current
    const tick = (now: number) => {
      const next = controller.step(now)
      store.publish(next, now)
      if (maxDifference(next, shown) > 0.002) {
        shown = next
        setShapes(next)
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [controller, store])

  return <AvatarFace shapes={shapes} />
}
