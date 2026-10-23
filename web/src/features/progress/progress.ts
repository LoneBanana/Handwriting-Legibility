// Pure helpers for the Progress page: grouping, this-week summaries, chart
// geometry and learner-facing wording.

import type { Track, TrackModule, WeekSummary } from '../../api/types'
import type { ChartPoint } from '../../lib/chart'
import { niceStep } from '../../lib/chart'

export const MODULE_ORDER: readonly TrackModule[] = ['handwriting', 'literacy', 'numeracy', 'converse']

export const MODULE_INFO: Record<TrackModule, { title: string; to: string; action: string }> = {
  handwriting: { title: 'Handwriting', to: '/handwriting', action: 'Transcribe writing' },
  literacy: { title: 'Reading & spelling', to: '/reading', action: 'Start cards' },
  numeracy: { title: 'Math facts', to: '/math', action: 'Start a round' },
  converse: { title: 'Talk & faces', to: '/talk', action: 'Start talking' },
}

export type ModuleGroup = { module: TrackModule; tracks: Track[] }

/** Tracks grouped by module, in the app's usual module order. */
export function groupByModule(tracks: readonly Track[]): ModuleGroup[] {
  return MODULE_ORDER.map((module) => ({ module, tracks: tracks.filter((t) => t.module === module) })).filter(
    (group) => group.tracks.length > 0,
  )
}

/** Nobody has finished a session yet. */
export function isBrandNew(tracks: readonly Track[]): boolean {
  return tracks.every((t) => t.sessions === 0)
}

export function emptyWeek(weekStart: string): WeekSummary {
  return { week_start: weekStart, sessions: 0, perfect_sessions: 0, levels_gained: 0, average_score: null }
}

/** This week's summary for a track (all zeros if it wasn't practised this week). */
export function thisWeek(track: Track, weekStart: string): WeekSummary {
  return track.weekly.find((w) => w.week_start === weekStart) ?? emptyWeek(weekStart)
}

export type WeekOverview = {
  sessions: number
  perfectSessions: number
  /** Net levels this week (a step back counts −1). */
  levelsGained: number
  /** Sum of every track's weekly goal. */
  goal: number
  /** 0..1: sessions toward the goals, each track counted up to its own goal. */
  goalProgress: number
  tracksPractised: number
  tracksTotal: number
  longestStreak: { label: string; streak: number } | null
}

export function weekOverview(tracks: readonly Track[], weekStart: string): WeekOverview {
  let sessions = 0
  let perfectSessions = 0
  let levelsGained = 0
  let goal = 0
  let towardGoal = 0
  let tracksPractised = 0
  let longestStreak: WeekOverview['longestStreak'] = null
  for (const track of tracks) {
    const week = thisWeek(track, weekStart)
    sessions += week.sessions
    perfectSessions += week.perfect_sessions
    levelsGained += week.levels_gained
    const trackGoal = Math.max(0, track.weekly_goal)
    goal += trackGoal
    towardGoal += Math.min(week.sessions, trackGoal)
    if (week.sessions > 0) tracksPractised += 1
    if (track.streak > 0 && (!longestStreak || track.streak > longestStreak.streak)) {
      longestStreak = { label: track.label, streak: track.streak }
    }
  }
  return {
    sessions,
    perfectSessions,
    levelsGained,
    goal,
    goalProgress: goal > 0 ? towardGoal / goal : 0,
    tracksPractised,
    tracksTotal: tracks.length,
    longestStreak,
  }
}

/** Parses an ISO date ("2026-10-19") as a local calendar date. */
function localDate(iso: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  const date = match ? new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3])) : new Date(iso)
  return Number.isNaN(date.getTime()) ? null : date
}

/** "Oct 19". */
export function shortDate(iso: string): string {
  const date = localDate(iso)
  return date ? date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : iso
}

/** "Monday, October 19". */
export function longDate(iso: string): string {
  const date = localDate(iso)
  return date ? date.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' }) : iso
}

export function plural(count: number, one: string, many = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`
}

/** Level history as chart points (one per practice day). */
export function historyPoints(track: Track): ChartPoint[] {
  return track.history.map((entry) => ({ label: shortDate(entry.date), y: entry.level }))
}

export function streakText(track: Pick<Track, 'streak' | 'best_streak'>): string {
  const now = track.streak > 0 ? `${plural(track.streak, 'perfect session')} in a row` : 'No perfect streak right now'
  return track.best_streak > 0 ? `${now} · best ${track.best_streak}` : now
}

/** "+2", "−1" (a true minus sign), "0". */
export function signed(value: number): string {
  if (value > 0) return `+${value}`
  if (value < 0) return `\u2212${-value}`
  return '0'
}

/** A short, encouraging note about this week's net level change. */
export function levelsNote(levels: number): string {
  if (levels > 0) return 'gained this week'
  if (levels < 0) return 'eased back to keep practice comfortable'
  return 'a perfect session moves a level up'
}

// ------------------------------------------------------------- weekly chart

export type ChartBox = {
  width: number
  height: number
  padding: { top: number; right: number; bottom: number; left: number }
}

export type WeeklyBar = {
  index: number
  week: WeekSummary
  /** Centre of the week's band. */
  cx: number
  x: number
  y: number
  width: number
  height: number
  bandLeft: number
  bandWidth: number
  /** Path with a 4px rounded top and a square base. */
  path: string
}

export type WeeklyGeometry = {
  bars: WeeklyBar[]
  plot: { left: number; right: number; top: number; bottom: number }
  yMax: number
  ticks: { value: number; y: number }[]
  goalY: number | null
}

/** A bar growing from the baseline with rounded data-end corners. */
export function barPath(x: number, y: number, width: number, height: number, radius = 4): string {
  if (!(height > 0) || !(width > 0)) return ''
  const r = Math.min(radius, width / 2, height)
  const f = (n: number) => Number(n.toFixed(2))
  return (
    `M${f(x)},${f(y + height)} V${f(y + r)} Q${f(x)},${f(y)} ${f(x + r)},${f(y)} ` +
    `H${f(x + width - r)} Q${f(x + width)},${f(y)} ${f(x + width)},${f(y + r)} V${f(y + height)} Z`
  )
}

/** Column geometry for sessions per week, with the weekly goal as a reference line. */
export function weeklyGeometry(weeks: readonly WeekSummary[], goal: number, box: ChartBox): WeeklyGeometry {
  const plot = {
    left: box.padding.left,
    right: box.width - box.padding.right,
    top: box.padding.top,
    bottom: box.height - box.padding.bottom,
  }
  const most = Math.max(1, goal, ...weeks.map((w) => w.sessions))
  const step = Math.max(1, Math.round(niceStep(most, 3)))
  const yMax = Math.ceil(most / step) * step
  const yFor = (value: number) => plot.bottom - (value / yMax) * (plot.bottom - plot.top)
  const bandWidth = weeks.length > 0 ? (plot.right - plot.left) / weeks.length : 0
  const barWidth = Math.min(24, bandWidth * 0.6)

  const bars = weeks.map((week, index) => {
    const bandLeft = plot.left + index * bandWidth
    const cx = bandLeft + bandWidth / 2
    const y = yFor(week.sessions)
    const height = plot.bottom - y
    const x = cx - barWidth / 2
    return { index, week, cx, x, y, width: barWidth, height, bandLeft, bandWidth, path: barPath(x, y, barWidth, height) }
  })
  const ticks: { value: number; y: number }[] = []
  for (let value = 0; value <= yMax; value += step) ticks.push({ value, y: yFor(value) })
  return { bars, plot, yMax, ticks, goalY: goal > 0 ? yFor(goal) : null }
}

/** Which weeks get an x-axis label: every third, counted back from this week. */
export function labelledWeeks(count: number, every = 3): Set<number> {
  const labelled = new Set<number>()
  for (let i = count - 1; i >= 0; i -= every) labelled.add(i)
  return labelled
}

/** One sentence describing the weekly chart for screen readers. */
export function weeklySummary(weeks: readonly WeekSummary[], goal: number): string {
  if (weeks.length === 0) return 'No weeks to show yet.'
  const total = weeks.reduce((sum, w) => sum + w.sessions, 0)
  const perfect = weeks.reduce((sum, w) => sum + w.perfect_sessions, 0)
  if (total === 0) return `No practice sessions in the last ${plural(weeks.length, 'week')} yet.`
  const busiest = weeks.reduce((best, w) => (w.sessions > best.sessions ? w : best), weeks[0])
  const metGoal = goal > 0 ? weeks.filter((w) => w.sessions >= goal).length : 0
  return (
    `${plural(total, 'session')} in the last ${plural(weeks.length, 'week')}, ${perfect} of them perfect. ` +
    `Busiest week: ${shortDate(busiest.week_start)} with ${plural(busiest.sessions, 'session')}.` +
    (goal > 0 ? ` The weekly goal of ${goal} was reached in ${plural(metGoal, 'week')}.` : '')
  )
}

export function weekDescription(week: WeekSummary): string {
  const score = week.average_score === null ? '' : `, average score ${Math.round(week.average_score * 100)}%`
  return `Week of ${shortDate(week.week_start)}: ${plural(week.sessions, 'session')}, ${week.perfect_sessions} perfect${score}.`
}
