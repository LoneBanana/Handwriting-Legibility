import { useId } from 'react'
import type { Blendshapes } from './blendshapes'
import { FACE_VIEWBOX, faceGeometry } from './avatarGeometry'

const EYE_WHITE = '#fbfaf7'
const TOOTH = '#fffdf8'

/**
 * The built-in avatar, drawn from 52 blendshape values. Colours come from the
 * --skin/--feature/--lip/... tokens so it follows the light, dark, cream and
 * high-contrast themes.
 */
export function AvatarFace({ shapes, label = 'The talking avatar' }: { shapes: Blendshapes; label?: string }) {
  // useId() contains characters that some browsers reject inside url(#…) references.
  const id = `avatar-${useId().replace(/[^\w-]/g, '')}`
  const face = faceGeometry(shapes)
  const { mouth } = face
  const mouthClip = `${id}-mouth`

  return (
    <svg className="avatar-face" viewBox={FACE_VIEWBOX} role="img" aria-label={label} focusable="false">
      {/* shoulders and neck */}
      <path d="M26,240 C32,214 62,203 100,203 C138,203 168,214 174,240 Z" fill="var(--accent)" />
      <rect x="83" y="180" width="34" height="34" rx="10" fill="var(--skin-shade)" />
      {/* ears */}
      {[36, 164].map((cx) => (
        <g key={cx}>
          <ellipse cx={cx} cy="116" rx="8.5" ry="14" fill="var(--skin)" />
          <path
            d={cx < 100 ? 'M38,108 q-5,8 0,16' : 'M162,108 q5,8 0,16'}
            fill="none"
            stroke="var(--feature)"
            strokeOpacity="0.3"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </g>
      ))}
      <path d={face.head} fill="var(--skin)" stroke="var(--feature)" strokeOpacity="0.25" strokeWidth="1.2" />
      {/* hair */}
      <path
        d="M33,116 C27,50 64,19 100,19 C136,19 173,50 167,116 C163,86 152,66 132,58 C120,64 106,62 96,55 C82,62 60,64 47,74 C40,86 35,100 33,116 Z"
        fill="var(--feature)"
      />

      {/* forehead lines (inner/outer brows raised) and frown lines (brows lowered) */}
      <g fill="none" stroke="var(--feature)" strokeLinecap="round" strokeWidth="1.3">
        <path d="M82,70 Q100,66 118,70 M86,76 Q100,73 114,76" opacity={face.foreheadLines} />
        <path d="M96.5,90 l-1,7 M103.5,90 l1,7" opacity={face.furrow} />
      </g>

      {/* cheeks */}
      {face.cheeks.map((cheek) => (
        <ellipse key={cheek.side} cx={cheek.cx} cy={cheek.cy} rx="13" ry="8" fill="var(--cheek)" opacity={cheek.opacity} />
      ))}

      {/* eyes */}
      {face.eyes.map((eye) => {
        const clip = `${id}-eye-${eye.side}`
        return (
          <g key={eye.side}>
            <clipPath id={clip}>
              <path d={eye.opening} />
            </clipPath>
            <path d={eye.opening} fill={EYE_WHITE} />
            <g clipPath={`url(#${clip})`}>
              <circle cx={eye.iris.x} cy={eye.iris.y} r="6.6" fill="var(--iris)" />
              <circle cx={eye.iris.x} cy={eye.iris.y} r="3" fill="#161210" />
              <circle cx={eye.iris.x + 2} cy={eye.iris.y - 2.2} r="1.3" fill="#ffffff" />
            </g>
            <path d={eye.upperLid} fill="none" stroke="var(--feature)" strokeWidth="2.4" strokeLinecap="round" />
            <path d={eye.lowerLid} fill="none" stroke="var(--feature)" strokeOpacity="0.45" strokeWidth="1.1" strokeLinecap="round" />
            <path
              d={eye.crinklePath}
              fill="none"
              stroke="var(--feature)"
              strokeWidth="1.2"
              strokeLinecap="round"
              opacity={eye.crinkle * 0.8}
            />
          </g>
        )
      })}

      {/* eyebrows */}
      {face.brows.map((brow) => (
        <path key={brow.side} d={brow.path} fill="none" stroke="var(--feature)" strokeWidth="5.5" strokeLinecap="round" />
      ))}

      {/* nose */}
      <g fill="none" stroke="var(--feature)" strokeLinecap="round">
        <path d="M98,112 Q95,127 91.5,135" strokeOpacity="0.3" strokeWidth="1.8" />
        <path d={face.nose.bottom} strokeOpacity="0.5" strokeWidth="2" />
        <path d="M93,117 Q100,114 107,117 M94,122 Q100,119.5 106,122" strokeWidth="1.2" opacity={face.nose.wrinkles * 0.9} />
      </g>
      {face.nose.nostrils.map((nostril) => (
        <ellipse key={nostril.cx} cx={nostril.cx} cy={nostril.cy} rx={nostril.rx} ry="1.5" fill="var(--feature)" opacity="0.45" />
      ))}

      {/* mouth */}
      <clipPath id={mouthClip}>
        <path d={mouth.outline} />
      </clipPath>
      <path d={mouth.outline} fill="var(--mouth-inside)" />
      <g clipPath={`url(#${mouthClip})`}>
        {mouth.tongue && (
          <ellipse cx={mouth.tongue.cx} cy={mouth.tongue.cy} rx={mouth.tongue.rx} ry={mouth.tongue.ry} fill="var(--lip)" opacity="0.75" />
        )}
        {mouth.teeth && (
          <rect x={mouth.teeth.x} y={mouth.teeth.y} width={mouth.teeth.width} height={mouth.teeth.height} rx="1.5" fill={TOOTH} />
        )}
      </g>
      <path d={mouth.outline} fill="none" stroke="var(--lip)" strokeWidth={mouth.lipWidth} strokeLinejoin="round" />
      {mouth.opening < 2 && (
        <path d={mouth.seam} fill="none" stroke="var(--mouth-inside)" strokeWidth="1.3" strokeLinecap="round" />
      )}
      {mouth.dimples.map((dimple) => (
        <path
          key={dimple.side}
          d={dimple.path}
          fill="none"
          stroke="var(--feature)"
          strokeWidth="1.3"
          strokeLinecap="round"
          opacity={dimple.opacity}
        />
      ))}
    </svg>
  )
}
