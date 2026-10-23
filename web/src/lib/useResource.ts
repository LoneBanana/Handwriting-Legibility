import { useCallback, useEffect, useState } from 'react'

type Resource<T> = {
  data: T | null
  error: unknown
  loading: boolean
  reload: () => void
}

/** Loads data once (and again on `reload`), ignoring results that arrive after unmount. */
export function useResource<T>(load: () => Promise<T>): Resource<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)
    load()
      .then((value) => {
        if (active) setData(value)
      })
      .catch((err: unknown) => {
        if (active) setError(err)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
    // `load` is expected to be a stable module-level function or memoised callback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt])

  const reload = useCallback(() => setAttempt((n) => n + 1), [])
  return { data, error, loading, reload }
}
