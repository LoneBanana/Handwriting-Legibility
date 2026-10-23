import { FACE_REGIONS, REGION_LABELS, type FaceRegion } from './faceAnalysis'

type RegionDrawing = {
  /** Outline drawn in a quiet colour, then again in the accent colour as it activates. */
  shapes: { d: string; stroke?: boolean }[]
  label: string[]
  labelAt: { x: number; y: number; anchor: 'start' | 'end' }
  /** Where the leader line touches the region. */
  target: { x: number; y: number }
}

// A schematic face in a 340 × 250 viewBox: face centred on x = 170, labels on both sides.
const REGIONS: Record<FaceRegion, RegionDrawing> = {
  innerBrows: {
    shapes: [{ d: 'M148,80 L162,77 M178,77 L192,80', stroke: true }],
    label: ['Inner', 'brows'],
    labelAt: { x: 96, y: 58, anchor: 'end' },
    target: { x: 152, y: 79 },
  },
  outerBrows: {
    shapes: [{ d: 'M126,86 Q134,79 148,80 M192,80 Q206,79 214,86', stroke: true }],
    label: ['Outer', 'brows'],
    labelAt: { x: 244, y: 58, anchor: 'start' },
    target: { x: 208, y: 82 },
  },
  eyes: {
    shapes: [{ d: 'M128,104 Q144,93 160,104 Q144,113 128,104 Z M180,104 Q196,93 212,104 Q196,113 180,104 Z' }],
    label: ['Eyes'],
    labelAt: { x: 96, y: 108, anchor: 'end' },
    target: { x: 128, y: 104 },
  },
  cheeks: {
    shapes: [{ d: 'M117,138 a15,10 0 1,0 30,0 a15,10 0 1,0 -30,0 Z M193,138 a15,10 0 1,0 30,0 a15,10 0 1,0 -30,0 Z' }],
    label: ['Cheeks'],
    labelAt: { x: 96, y: 146, anchor: 'end' },
    target: { x: 118, y: 140 },
  },
  nose: {
    shapes: [{ d: 'M170,100 L162,132 Q170,139 178,132 Z' }],
    label: ['Nose'],
    labelAt: { x: 244, y: 114, anchor: 'start' },
    target: { x: 176, y: 124 },
  },
  upperLip: {
    shapes: [{ d: 'M154,146 Q170,141 186,146 L186,153 Q170,150 154,153 Z' }],
    label: ['Upper', 'lip'],
    labelAt: { x: 244, y: 142, anchor: 'start' },
    target: { x: 186, y: 149 },
  },
  mouthCorners: {
    shapes: [{ d: 'M142,163 a6,6 0 1,0 12,0 a6,6 0 1,0 -12,0 Z M186,163 a6,6 0 1,0 12,0 a6,6 0 1,0 -12,0 Z' }],
    label: ['Mouth', 'corners'],
    labelAt: { x: 96, y: 182, anchor: 'end' },
    target: { x: 143, y: 166 },
  },
  lips: {
    shapes: [{ d: 'M154,163 Q170,154 186,163 Q170,172 154,163 Z' }],
    label: ['Lips'],
    labelAt: { x: 244, y: 176, anchor: 'start' },
    target: { x: 184, y: 166 },
  },
  jaw: {
    shapes: [{ d: 'M142,190 Q170,214 198,190 Q170,203 142,190 Z' }],
    label: ['Jaw'],
    labelAt: { x: 244, y: 212, anchor: 'start' },
    target: { x: 194, y: 196 },
  },
}

const ACTIVE = 0.25

function describe(regions: Record<FaceRegion, number>): string {
  const active = FACE_REGIONS.filter((region) => regions[region] >= 0.2).sort((a, b) => regions[b] - regions[a])
  if (active.length === 0) return 'Face diagram: the face is relaxed.'
  return `Face diagram. Moving most: ${active
    .slice(0, 4)
    .map((region) => REGION_LABELS[region].toLowerCase())
    .join(', ')}.`
}

/** A labelled face whose regions light up as their action units activate. */
export function FaceDiagram({ regions, dimmed = false }: { regions: Record<FaceRegion, number>; dimmed?: boolean }) {
  return (
    <svg className="face-diagram" viewBox="0 0 340 250" role="img" aria-label={describe(regions)} data-dimmed={dimmed}>
      <path
        d="M170,34 C210,34 232,64 232,110 C232,160 206,206 170,214 C134,206 108,160 108,110 C108,64 130,34 170,34 Z"
        className="face-diagram__outline"
      />
      {FACE_REGIONS.map((region) => {
        const drawing = REGIONS[region]
        const value = Math.min(1, Math.max(0, regions[region]))
        const active = value >= ACTIVE
        const lineEnd = drawing.labelAt.anchor === 'end' ? drawing.labelAt.x + 6 : drawing.labelAt.x - 6
        return (
          <g key={region} className="face-diagram__region" data-active={active}>
            {drawing.shapes.map((shape, i) => (
              <g key={i}>
                <path d={shape.d} className={shape.stroke ? 'face-diagram__base-stroke' : 'face-diagram__base'} />
                <path
                  d={shape.d}
                  className={shape.stroke ? 'face-diagram__glow-stroke' : 'face-diagram__glow'}
                  style={{ opacity: 0.1 + 0.9 * value }}
                />
              </g>
            ))}
            <line
              x1={lineEnd}
              y1={drawing.labelAt.y - 4}
              x2={drawing.target.x}
              y2={drawing.target.y}
              className="face-diagram__leader"
            />
            <text x={drawing.labelAt.x} y={drawing.labelAt.y - (drawing.label.length - 1) * 7} textAnchor={drawing.labelAt.anchor}>
              {drawing.label.map((line, i) => (
                <tspan key={line} x={drawing.labelAt.x} dy={i === 0 ? 0 : 15}>
                  {line}
                </tspan>
              ))}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
