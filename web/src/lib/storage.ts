// localStorage can be missing or throw (private windows, blocked site data,
// sandboxed iframes). Every access goes through these helpers so the app keeps
// working without it.

export function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

export function writeStorage(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key)
    else window.localStorage.setItem(key, value)
  } catch {
    // Storage unavailable: the value simply isn't remembered.
  }
}
