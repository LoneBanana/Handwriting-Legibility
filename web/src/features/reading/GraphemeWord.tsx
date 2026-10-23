import { graphemeSegments } from './graphemes'

type Props = {
  word: string
  graphemes: string[]
  focus: [number, number] | null
  small?: boolean
}

/** A word coloured by grapheme with the practised pattern highlighted. */
export function GraphemeWord({ word, graphemes, focus, small = false }: Props) {
  const segments = graphemeSegments(word, graphemes, focus)
  const focusText = segments
    .filter((s) => s.inFocus)
    .map((s) => s.text)
    .join('')
  return (
    <span className={small ? 'word word--small' : 'word'}>
      <span className="visually-hidden">{focusText ? `${word} (pattern: ${focusText})` : word}</span>
      <span aria-hidden="true">
        {segments.map((segment) => (
          <span key={segment.start} className="grapheme" data-color={segment.colorIndex} data-focus={segment.inFocus}>
            {segment.text}
          </span>
        ))}
      </span>
    </span>
  )
}
