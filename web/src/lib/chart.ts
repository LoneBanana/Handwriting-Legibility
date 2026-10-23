export type ChartPoint = { label: string; y: number }

type Options = {
  width: number
  height: number
  padding: { top: number; right: number; bottom: number; left: number }
}

/** Picks a "nice" step (1, 2, 5 × 10^n) so roughly `count` ticks span the range. */
export function niceStep(range: number, count = 4): number {
  if (!(range > 0)) return 1
  const raw = range / count
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const normalized = raw / magnitude
  const nice = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10
  return nice * magnitude
}

export function chartGeometry(points: ChartPoint[], { width, height, padding }: Options) {
  const plot = {
    left: padding.left,
    right: width - padding.right,
    top: padding.top,
    bottom: height - padding.bottom,
  }
  const values = points.map((p) => p.y)
  let min = values.length ? Math.min(...values) : 0
  let max = values.length ? Math.max(...values) : 1
  const nonNegative = values.every((v) => v >= 0)
  if (min === max) {
    min -= 1
    max += 1
  }
  const step = niceStep(max - min)
  min = Math.floor(min / step) * step
  if (nonNegative) min = Math.max(0, min) // levels and counts never go below zero
  max = Math.ceil(max / step) * step

  const xFor = (index: number) =>
    points.length <= 1 ? (plot.left + plot.right) / 2 : plot.left + (index / (points.length - 1)) * (plot.right - plot.left)
  const yFor = (value: number) => plot.bottom - ((value - min) / (max - min)) * (plot.bottom - plot.top)

  const coords = points.map((p, i) => ({ x: xFor(i), y: yFor(p.y) }))
  const linePath = coords.map((c, i) => `${i === 0 ? 'M' : 'L'}${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' ')
  const areaPath = coords.length
    ? `${linePath} L${coords[coords.length - 1].x.toFixed(1)},${plot.bottom} L${coords[0].x.toFixed(1)},${plot.bottom} Z`
    : ''
  const yTicks: { value: number; y: number }[] = []
  for (let v = min; v <= max + step / 2; v += step) {
    const value = Number(v.toFixed(6))
    yTicks.push({ value, y: yFor(value) })
  }
  return { plot, points: coords, linePath, areaPath, yTicks }
}
