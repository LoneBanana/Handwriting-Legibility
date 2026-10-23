import { describe, expect, it, vi } from 'vitest'
import { blendshapesFrom } from './blendshapes'
import { emotionTargets } from './expressions'
import { createFaceStore } from './faceStore'

describe('createFaceStore', () => {
  it('notifies at a gentle rate and keeps the snapshot stable in between', () => {
    const store = createFaceStore({ notifyEveryMs: 125 })
    const listener = vi.fn()
    const unsubscribe = store.subscribe(listener)
    const shapes = emotionTargets('content', 1)
    let previous = store.getSnapshot()
    let changes = 0
    for (let t = 0; t <= 1000; t += 16) {
      store.publish(shapes, t)
      const snapshot = store.getSnapshot()
      if (snapshot !== previous) changes += 1
      previous = snapshot
    }
    expect(listener.mock.calls.length).toBe(changes)
    expect(changes).toBeGreaterThanOrEqual(7)
    expect(changes).toBeLessThanOrEqual(9)
    unsubscribe()
    store.publish(shapes, 2000)
    expect(listener.mock.calls.length).toBe(changes)
  })

  it('analyses what it is given', () => {
    const store = createFaceStore()
    for (let t = 0; t <= 1500; t += 16) store.publish(emotionTargets('surprised', 1), t)
    const snapshot = store.getSnapshot()
    expect(snapshot.hasData).toBe(true)
    expect(snapshot.faceFound).toBe(true)
    expect(snapshot.estimate.family).toBe('surprised')
    expect(snapshot.regions.outerBrows).toBeGreaterThan(0.5)
  })

  it('reports when no face is visible and relaxes the bars', () => {
    const store = createFaceStore({ notifyEveryMs: 0 })
    store.publish(emotionTargets('excited', 1), 0)
    for (let t = 100; t <= 1500; t += 100) store.publish(null, t)
    const snapshot = store.getSnapshot()
    expect(snapshot.faceFound).toBe(false)
    expect(snapshot.reading.au.AU12).toBeLessThan(0.05)
  })

  it('is not swayed by a talking jaw', () => {
    const store = createFaceStore({ notifyEveryMs: 0 })
    for (let t = 0; t <= 3000; t += 16) {
      const jaw = 0.3 + 0.3 * Math.sin((2 * Math.PI * 4 * t) / 1000) // syllables at about 4 per second
      store.publish(blendshapesFrom({ jawOpen: jaw }), t)
    }
    expect(store.getSnapshot().estimate.family).toBe('neutral')
  })

  it('resets to an empty state', () => {
    const store = createFaceStore()
    store.publish(emotionTargets('angry', 1), 0)
    store.reset()
    expect(store.getSnapshot().hasData).toBe(false)
    expect(store.getSnapshot().reading.au.AU4).toBe(0)
  })
})
