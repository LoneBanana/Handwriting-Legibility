// A small set of stroke icons. Decorative by default (aria-hidden); pass `label`
// when the icon is the only content of a control.

const PATHS = {
  speaker: 'M4 9v6h4l5 4V5L8 9H4zm12.5 3a4.5 4.5 0 0 0-2.5-4v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z',
  stop: 'M6 6h12v12H6z',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  cross: 'M6 6l12 12M18 6L6 18',
  upload: 'M12 16V4m0 0L7 9m5-5l5 5M5 20h14',
  camera: 'M4 8h3l2-3h6l2 3h3v11H4zM12 17a4 4 0 1 0 0-8 4 4 0 0 0 0 8z',
  mic: 'M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zm-7 9a7 7 0 0 0 14 0M12 19v3',
  send: 'M4 12l16-8-6 16-2.5-6.5L4 12z',
  refresh: 'M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6',
  sparkle: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z',
  copy: 'M9 9h10v11H9zM5 15V4h10',
  star: 'M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z',
  bolt: 'M13 2L4 14h7l-1 8 9-12h-7z',
  play: 'M7 4l13 8-13 8z',
  arrowRight: 'M5 12h14m-6-6l6 6-6 6',
  info: 'M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20zm0-11v6m0-9.5v.5',
  pause: 'M7 5h4v14H7zm6 0h4v14h-4z',
} as const

export type IconName = keyof typeof PATHS

const FILLED: ReadonlySet<IconName> = new Set<IconName>(['speaker', 'stop', 'play', 'pause', 'star', 'bolt'])

export function Icon({ name, label, size = 20 }: { name: IconName; label?: string; size?: number }) {
  const filled = FILLED.has(name)
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? 'currentColor' : 'none'}
      stroke={filled ? 'none' : 'currentColor'}
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      focusable="false"
    >
      <path d={PATHS[name]} />
    </svg>
  )
}
