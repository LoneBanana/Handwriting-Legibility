import { describe, expect, it } from 'vitest'
import type { Track, WeekSummary } from '../../api/types'
import {
  barPath,
  groupByModule,
  historyPoints,
  isBrandNew,
  labelledWeeks,
  levelsNote,
  shortDate,
  signed,
  streakText,
  thisWeek,
  weekOverview,
  weeklyGeometry,
  weeklySummary,
} from './progress'

const WEEK = '2026-10-19'

const week = (week_start: string, sessions: number, perfect = 0, levels = 0): WeekSummary => ({
  week_start,
  sessions,
  perfect_sessions: perfect,
  levels_gained: levels,
  average_score: sessions ? 0.8 : null,
})

function track(overrides: Partial<Track> = {}): Track {
  return {
    key: 'literacy.reading',
    module: 'literacy',
    label: 'Reading',
    description: 'Reading words quickly',
    level: 3,
    max_level: 60,
    progress: 0.05,
    streak: 2,
    best_streak: 4,
    sessions: 10,
    perfect_sessions: 4,
    last_promoted_on: '2026-10-20',
    weekly_goal: 5,
    parameters: {},
    history: [
      { date: '2026-10-19', level: 2, score: 1 },
      { date: '2026-10-20', level: 3, score: 1 },
    ],
    weekly: [week('2026-10-12', 4, 1, 1), week(WEEK, 3, 2, 1)],
    ...overrides,
  }
}

describe('grouping and summaries', () => {
  it('groups tracks in module order and skips empty modules', () => {
    const groups = groupByModule([
      track({ key: 'converse.listening', module: 'converse' }),
      track({ key: 'handwriting.legibility', module: 'handwriting' }),
      track({ key: 'literacy.spelling', module: 'literacy' }),
    ])
    expect(groups.map((g) => g.module)).toEqual(['handwriting', 'literacy', 'converse'])
  })

  it('knows a brand-new learner', () => {
    expect(isBrandNew([])).toBe(true)
    expect(isBrandNew([track({ sessions: 0 }), track({ sessions: 0 })])).toBe(true)
    expect(isBrandNew([track({ sessions: 0 }), track()])).toBe(false)
  })

  it('finds this week, or an empty one', () => {
    expect(thisWeek(track(), WEEK).sessions).toBe(3)
    expect(thisWeek(track({ weekly: [] }), WEEK)).toEqual({
      week_start: WEEK,
      sessions: 0,
      perfect_sessions: 0,
      levels_gained: 0,
      average_score: null,
    })
  })

  it('adds up the week across tracks, counting each track up to its own goal', () => {
    const overview = weekOverview(
      [
        track({ weekly: [week(WEEK, 8, 3, 1)], weekly_goal: 5, streak: 6, label: 'Reading' }),
        track({ weekly: [week(WEEK, 1, 0, -1)], weekly_goal: 5, streak: 0, label: 'Spelling' }),
        track({ weekly: [], weekly_goal: 3, streak: 1, label: 'Math' }),
      ],
      WEEK,
    )
    expect(overview).toMatchObject({
      sessions: 9,
      perfectSessions: 3,
      levelsGained: 0,
      goal: 13,
      tracksPractised: 2,
      tracksTotal: 3,
      longestStreak: { label: 'Reading', streak: 6 },
    })
    expect(overview.goalProgress).toBeCloseTo((5 + 1) / 13)
  })

  it('turns history into chart points', () => {
    const points = historyPoints(track())
    expect(points.map((p) => p.y)).toEqual([2, 3])
    expect(points[0].label).toBe(shortDate('2026-10-19'))
  })
})

describe('wording', () => {
  it('reads ISO dates as local calendar days', () => {
    const label = shortDate('2026-10-19')
    expect(label).toMatch(/19/)
    expect(shortDate('not a date')).toBe('not a date')
  })

  it('describes streaks and level changes kindly', () => {
    expect(streakText({ streak: 1, best_streak: 1 })).toBe('1 perfect session in a row · best 1')
    expect(streakText({ streak: 0, best_streak: 0 })).toBe('No perfect streak right now')
    expect(signed(2)).toBe('+2')
    expect(signed(-1)).toBe('−1')
    expect(signed(0)).toBe('0')
    expect(levelsNote(-1)).toMatch(/comfortable/)
  })
})

describe('weekly chart', () => {
  const box = { width: 560, height: 180, padding: { top: 14, right: 44, bottom: 30, left: 30 } }
  const weeks = Array.from({ length: 12 }, (_, i) => week(`2026-08-${String(3 + i).padStart(2, '0')}`, i % 4))

  it('lays out thin columns from a shared baseline, scaled to fit the goal', () => {
    const geometry = weeklyGeometry(weeks, 5, box)
    expect(geometry.bars).toHaveLength(12)
    expect(geometry.yMax).toBeGreaterThanOrEqual(5)
    for (const bar of geometry.bars) {
      expect(bar.width).toBeLessThanOrEqual(24)
      expect(bar.y + bar.height).toBeCloseTo(geometry.plot.bottom)
      expect(bar.x).toBeGreaterThanOrEqual(bar.bandLeft)
    }
    const [zero, , two, three] = geometry.bars
    expect(zero.path).toBe('') // no bar for an empty week
    expect(three.height).toBeCloseTo((two.height * 3) / 2)
    expect(geometry.goalY).toBeCloseTo(geometry.plot.bottom - (5 / geometry.yMax) * (geometry.plot.bottom - geometry.plot.top))
    expect(geometry.ticks[0]).toMatchObject({ value: 0 })
  })

  it('rounds only the data end of a bar', () => {
    expect(barPath(0, 10, 20, 50)).toBe('M0,60 V14 Q0,10 4,10 H16 Q20,10 20,14 V60 Z')
    expect(barPath(0, 10, 20, 0)).toBe('')
  })

  it('labels every third week, always including this week', () => {
    expect([...labelledWeeks(12)].sort((a, b) => a - b)).toEqual([2, 5, 8, 11])
  })

  it('summarises the chart in a sentence', () => {
    expect(weeklySummary([], 5)).toBe('No weeks to show yet.')
    expect(weeklySummary([week(WEEK, 0)], 5)).toMatch(/No practice sessions/)
    const text = weeklySummary([week('2026-10-12', 6, 2), week(WEEK, 2, 1)], 5)
    expect(text).toMatch(/^8 sessions in the last 2 weeks, 3 of them perfect\./)
    expect(text).toMatch(/reached in 1 week\./)
  })
})
