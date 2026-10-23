import { useEffect, useRef, useState } from 'react'

type Props = {
  limitMs: number
  /** Changes whenever a new problem starts, restarting the timer. */
  resetKey: string
  reducedMotion: boolean
}

/**
 * A gentle per-problem timer. It never cuts the learner off: when the time is
 * used up the bar turns striped and the learner can still answer.
 */
export function TimerBar({ limitMs, resetKey, reducedMotion }: Props) {
  const [fraction, setFraction] = useState(1)
  const [over, setOver] = useState(false)
  const startedAt = useRef(0)

  // Each problem mounts a fresh TimerBar (its parent is keyed by problem), so the state
  // starts full; the effect only has to run the clock.
  useEffect(() => {
    startedAt.current = performance.now()
    let frame = 0
    let interval = 0
    const tick = () => {
      const elapsed = performance.now() - startedAt.current
      const remaining = Math.max(0, 1 - elapsed / limitMs)
      setFraction(remaining)
      if (remaining <= 0) {
        setOver(true)
        return false
      }
      return true
    }
    if (reducedMotion) {
      interval = window.setInterval(() => {
        if (!tick()) window.clearInterval(interval)
      }, 1000)
    } else {
      const loop = () => {
        if (tick()) frame = requestAnimationFrame(loop)
      }
      frame = requestAnimationFrame(loop)
    }
    return () => {
      cancelAnimationFrame(frame)
      window.clearInterval(interval)
    }
  }, [limitMs, resetKey, reducedMotion])

  return (
    <div className="timer" data-over={over}>
      <div
        className="timer__track"
        role="timer"
        aria-label={`Time for this problem: ${Math.round(limitMs / 1000)} seconds`}
      >
        <div className="timer__fill" style={{ transform: `scaleX(${over ? 1 : fraction})` }} />
      </div>
      <p className="small muted" aria-live="polite" style={{ margin: 'var(--space-1) 0 0' }}>
        {over ? 'Time’s up for a speed point — take your time and finish.' : ''}
      </p>
    </div>
  )
}
