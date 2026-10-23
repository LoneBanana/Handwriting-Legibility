// Mirrors the HTTP contract in docs/ARCHITECTURE.md. Keep the two in sync.

export type FontChoice = 'default' | 'lexend' | 'atkinson' | 'opendyslexic'
export type ThemeChoice = 'light' | 'dark' | 'cream' | 'high-contrast'

export type UserSettings = {
  font: FontChoice
  font_scale: number
  line_spacing: number
  theme: ThemeChoice
  reduce_motion: boolean
  tts_voice: string | null
  store_handwriting_samples: boolean
}

export type User = {
  id: string
  email: string
  display_name: string
  created_at: string
  settings: UserSettings
}

export type AuthResponse = {
  access_token: string
  token_type: 'bearer'
  user: User
}

export type TrackParameter = {
  value: number
  start: number
  target: number
  unit: string
  label: string
}

export type ProgressionReason = 'perfect' | 'held' | 'daily_cap' | 'struggling' | 'max_level'

export type ProgressionUpdate = {
  track: string
  previous_level: number
  level: number
  promoted: boolean
  demoted: boolean
  reason: ProgressionReason
  message: string
  progress: number
  parameters: Record<string, TrackParameter>
}

export type TrackModule = 'handwriting' | 'literacy' | 'numeracy' | 'converse'

export type TrackSnapshot = {
  track: string
  level: number
  parameters: Record<string, TrackParameter>
}

export type WeekSummary = {
  week_start: string
  sessions: number
  perfect_sessions: number
  levels_gained: number
  average_score: number | null
}

export type Track = {
  key: string
  module: TrackModule
  label: string
  description: string
  level: number
  max_level: number
  progress: number
  streak: number
  best_streak: number
  sessions: number
  perfect_sessions: number
  last_promoted_on: string | null
  weekly_goal: number
  parameters: Record<string, TrackParameter>
  history: { date: string; level: number; score: number }[]
  weekly: WeekSummary[]
}

export type ProgressSummary = {
  week_start: string
  tracks: Track[]
}

// ---------------------------------------------------------------- handwriting

export type TranscribeMode = 'auto' | 'page' | 'line'
export type DecoderChoice = 'fast' | 'accurate'

export type CharConfidence = { char: string; confidence: number }

export type TranscribedLine = {
  index: number
  text: string
  confidence: number
  bbox: [number, number, number, number]
  chars: CharConfidence[]
}

export type ClarifiedText = {
  text: string
  changes: { original: string; corrected: string; reason: string }[]
  uncertain: string[]
}

export type TranscribeResponse = {
  sample_id: string | null
  text: string
  lines: TranscribedLine[]
  legibility: number
  clarified: ClarifiedText | null
  model_version: string
  personalized: boolean
  elapsed_ms: number
  progressions: ProgressionUpdate[]
}

export type HandwritingStatus = {
  model_loaded: boolean
  model_version: string
  device: string
  personalized: boolean
}

// ------------------------------------------------------------------- literacy

export type Card = {
  card_id: string
  word: string
  mode: 'read' | 'spell'
  skill: string
  skill_label: string
  graphemes: string[]
  focus: [number, number] | null
  prompt: string
  choices: string[] | null
  sentence: string | null
  exposure_ms: number | null
  hint: SpellingHint | null
}

export type SpellingHint = {
  length: number
  first_letter: string | null
}

export type LiteracySession = {
  session_id: string
  skill: string
  skill_label: string
  tracks: TrackSnapshot[]
  cards: Card[]
}

export type SpellingErrorType =
  | 'reversal'
  | 'transposition'
  | 'omission'
  | 'insertion'
  | 'substitution'
  | 'vowel'
  | 'doubling'

export type AnswerResult = {
  correct: boolean
  expected: string
  phonetically_plausible: boolean
  feedback: string
  errors: { type: SpellingErrorType; position: number; detail: string }[]
}

export type HandwritingAnswerResult = AnswerResult & { recognized: string }

export type LiteracyCompletion = {
  score: number
  perfect: boolean
  correct: number
  total: number
  progressions: ProgressionUpdate[]
}

// ------------------------------------------------------------------- numeracy

export type Visual =
  | { type: 'dots'; count: number; layout: 'dice' | 'random' | 'line' }
  | { type: 'ten_frame'; filled: number; frames: number }
  | { type: 'array'; rows: number; cols: number }
  | { type: 'number_line'; min: number; max: number; target: number | null }

export type ProblemKind =
  | 'subitize'
  | 'compare'
  | 'bond'
  | 'add'
  | 'sub'
  | 'mul'
  | 'div'
  | 'number_line'

export type Problem = {
  problem_id: string
  kind: ProblemKind
  prompt: string
  visual: Visual | null
  answer_format: 'integer' | 'choice'
  choices: string[] | null
}

export type NumeracyRound = {
  round_id: string
  track: string
  skill: string
  skill_label: string
  level: number
  time_limit_ms: number
  parameters: Record<string, TrackParameter>
  problems: Problem[]
}

export type ProblemAnswer = { problem_id: string; answer: string; latency_ms: number }

export type RoundResult = {
  score: number
  perfect: boolean
  fluent_count: number
  total: number
  results: {
    problem_id: string
    correct: boolean
    expected: string
    fluent: boolean
    strategy_hint: string | null
  }[]
  progressions: ProgressionUpdate[]
}

// ------------------------------------------------------------------- converse

export const EMOTIONS = [
  'neutral',
  'content',
  'excited',
  'elated',
  'surprised',
  'sad',
  'dejected',
  'scared',
  'angry',
  'disgusted',
  'contempt',
] as const

export type Emotion = (typeof EMOTIONS)[number]

export type AvatarProvider = 'local' | 'tavus'
export type CaptionMode = 'full' | 'delayed' | 'off'

export type ConverseSession = {
  session_id: string
  provider: AvatarProvider
  assistant: 'claude' | 'demo'
  words_per_minute: number
  expression_intensity: number
  captions: CaptionMode
  tracks: TrackSnapshot[]
  tavus: null | { conversation_id: string; conversation_url: string; meeting_token: string | null }
}

export type SegmentEvent = { turn_id: string; index: number; emotion: Emotion; text: string }
export type DoneEvent = { turn_id: string; text: string; echo_text: string }

export type CheckQuestion = {
  question_id: string
  kind: 'main_idea' | 'detail' | 'emotion' | 'inference'
  prompt: string
  choices: string[]
}

export type ComprehensionCheck = { check_id: string; questions: CheckQuestion[] }

export type CheckResult = {
  score: number
  perfect: boolean
  results: { question_id: string; correct: boolean; expected: string; explanation: string }[]
  progressions: ProgressionUpdate[]
}
