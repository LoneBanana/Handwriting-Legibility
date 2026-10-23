import { describe, expect, it } from 'vitest'
import {
  buildVisemeTimeline,
  CLAUSE_PAUSE_MS,
  SENTENCE_PAUSE_MS,
  timeForChar,
  visemeAt,
  wordAt,
  wordVisemes,
} from './visemes'

describe('wordVisemes', () => {
  it('maps letters and common spellings to mouth shapes', () => {
    expect(wordVisemes('mom')).toEqual(['MBP', 'OH', 'MBP'])
    expect(wordVisemes('phone')).toEqual(['FV', 'OH', 'DNT']) // ph = f, silent final e
    expect(wordVisemes('think')).toEqual(['TH', 'IH', 'DNT', 'KG'])
    expect(wordVisemes('queen')).toEqual(['KG', 'WQ', 'EE', 'DNT'])
    expect(wordVisemes('city')).toEqual(['SZ', 'IH', 'DNT', 'EE']) // soft c
    expect(wordVisemes('ghost')).toEqual(['KG', 'OH', 'SZ', 'DNT'])
    expect(wordVisemes('Who')).toEqual(['WQ', 'OH'])
  })

  it('merges repeated shapes and skips punctuation', () => {
    expect(wordVisemes('ball')).toEqual(['MBP', 'AA', 'L'])
    expect(wordVisemes("it's")).toEqual(['IH', 'DNT', 'SZ'])
    expect(wordVisemes('—')).toEqual([])
  })

  it('says digits as words', () => {
    expect(wordVisemes('3')).toEqual(['TH', 'R', 'EE'])
  })
})

describe('buildVisemeTimeline', () => {
  const text = 'Hello there, friend. How are you?'

  it('lasts about words ÷ words-per-minute, pauses included', () => {
    const timeline = buildVisemeTimeline(text, 150)
    expect(timeline.words).toHaveLength(6)
    expect(timeline.duration).toBeCloseTo((6 * 60_000) / 150, 0)
    const keyTimes = timeline.keys.map((k) => k.at)
    expect(keyTimes).toEqual([...keyTimes].sort((a, b) => a - b))
  })

  it('rests at punctuation', () => {
    const timeline = buildVisemeTimeline(text, 150)
    const [, there, friend, how] = timeline.words
    expect(friend.start - there.end).toBeCloseTo(CLAUSE_PAUSE_MS)
    expect(how.start - friend.end).toBeCloseTo(SENTENCE_PAUSE_MS)
    expect(visemeAt(timeline, there.end + 10)).toBe('rest')
  })

  it('finds the shape at a time and is at rest outside the utterance', () => {
    const timeline = buildVisemeTimeline(text, 150)
    expect(visemeAt(timeline, 0)).toBe('EE') // "hello": h is silent here
    expect(visemeAt(timeline, -5)).toBe('rest')
    expect(visemeAt(timeline, timeline.duration + 1)).toBe('rest')
  })

  it('maps SpeechSynthesis character offsets to word start times', () => {
    const timeline = buildVisemeTimeline(text, 150)
    const there = timeline.words[1]
    expect(timeForChar(timeline, text.indexOf('there'))).toBe(there.start)
    expect(timeForChar(timeline, text.indexOf(' there'))).toBe(there.start) // the space before it
    expect(timeForChar(timeline, text.length + 10)).toBe(timeline.duration)
    expect(wordAt(timeline, there.start + 1)).toBe(there)
    expect(wordAt(timeline, there.end + 50)).toBe(there) // in the pause after it
  })

  it('handles empty text and odd rates', () => {
    expect(buildVisemeTimeline('', 150)).toMatchObject({ duration: 0, words: [] })
    expect(buildVisemeTimeline('...', 150).duration).toBe(0)
    expect(buildVisemeTimeline('hello', Number.NaN).duration).toBeCloseTo(60_000 / 160)
    expect(buildVisemeTimeline('hello', 10_000).duration).toBeCloseTo(60_000 / 320)
  })
})
