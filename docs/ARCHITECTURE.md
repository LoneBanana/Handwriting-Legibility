# Architecture

Legible is one app with four learning modules that share accounts, an adaptive
progression engine, and accessibility settings.

```
                ┌──────────────────────────── web/ (Vite + React + TS) ───────────────────────────┐
                │  Handwriting   Reading (dyslexia)   Math (dyscalculia)   Talk (APD/LPD/NVLD)    │
                │      │               │                    │              │  avatar + face panel │
                └──────┼───────────────┼────────────────────┼──────────────┼──────────────────────┘
                       │  /api (JSON, SSE)                   │              │ WebRTC (Tavus/Daily)
                ┌──────▼───────────────▼────────────────────▼──────────────▼──────────────────────┐
                │ src/legible/api  (FastAPI)                                                       │
                │   auth · handwriting · literacy · numeracy · converse · progress                 │
                ├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────────┤
                │ legible.htr  │ legible.     │ legible.     │ legible.     │ legible.progression │
                │ CNN-ViT HTR  │ literacy     │ numeracy     │ converse     │ mastery-gated,      │
                │ (PyTorch)    │ HLR memory   │ fact fluency │ Claude chat  │ compounding         │
                │              │ model        │ generator    │ + avatars    │ difficulty          │
                └──────────────┴──────────────┴──────────────┴──────────────┴─────────────────────┘
                                         SQLAlchemy (SQLite dev / Postgres prod)
```

| Phase | Module | Package | What is built |
|------:|--------|---------|--------|
| 1 | Dysgraphia handwriting → text | `legible.htr` | ConvNeXt + ViT/Conformer encoder with joint CTC/attention and intermediate CTC; synthetic dysgraphic data; DDP/SLURM trainer with preemption-safe resume; line tracking segmentation; beam + CTC decoding; LoRA personalization; ONNX export ([TRAINING.md](TRAINING.md)) |
| 2 | Dyslexia flash cards | `legible.literacy` | 38-skill structured-literacy sequence, 455-word bank with grapheme splits and sentences, half-life memory for spaced review, look-alike distractors, named spelling errors and a "sounds right" check |
| 3 | Dyscalculia fact fluency | `legible.numeracy` | 26-skill ladder from subitizing to division, pictures that fade with level, weak-fact review, strategy hints, accuracy-first fluency scoring |
| 4 | APD / LPD / NVLD conversation | `legible.converse` + `web/src/features/talk` | Claude answers in emotion-tagged sentences (offline demo without a key), built-in SVG avatar with lip-sync or Tavus video in echo mode, live FACS action-unit panel, comprehension and feelings checks |
| 5 | Progress | `legible.progression` | per-track levels, perfect-only promotion (one a day), struggle step-back, geometric parameter curves, weekly meters ([PROGRESSION.md](PROGRESSION.md)) |
| 6 | Website | `legible.api` + `web/` | FastAPI backend (auth, rate limits, privacy deletes, SSE), accessible React app, Docker/Postgres deployment |

## Phase 5 progression semantics (shared by every module)

Every module owns one or more **tracks** (e.g. `converse.listening`). A track has an
integer **level** and one or more **parameters** (speech rate, time limit, exposure
time, ...). The parameter value is a deterministic function of the level, so the
level is the only state that matters.

* A session that scores **perfectly** (score ≥ `promote_threshold`, default 1.0)
  promotes the track by one level, at most `max_promotions_per_day` (default 1) per
  calendar day, so day-to-day change stays below the just-noticeable difference.
* Any imperfect session **holds** the level.
* `struggle_window` (default 3) consecutive sessions below `struggle_threshold`
  (default 0.6) step the level **back** by one, so a learner never gets stuck in
  frustration.
* Parameter curves compound geometrically from `start` to `target` over
  `levels_to_target` promotions (about a year of near-daily perfect sessions),
  then ease into the target (see `docs/PROGRESSION.md`).

## HTTP API contract

Base path `/api`. JSON everywhere except uploads (multipart) and chat streams (SSE).
Errors use FastAPI's shape: `{"detail": "message"}`. Auth is a bearer JWT:
`Authorization: Bearer <access_token>`; a 401 means the client should log in again.
Every request also sends `X-Timezone: <IANA zone>` (e.g. `America/New_York`): "one
promotion per day" and the weekly meters use the learner's own calendar.

### Shared types

```ts
type UserSettings = {
  font: "default" | "lexend" | "atkinson" | "opendyslexic";
  font_scale: number;          // 0.8 – 2.0
  line_spacing: number;        // 1.2 – 2.5
  theme: "light" | "dark" | "cream" | "high-contrast";
  reduce_motion: boolean;
  tts_voice: string | null;    // SpeechSynthesis voice name
  store_handwriting_samples: boolean;  // consent to keep uploads for personalization
  share_samples_for_research: boolean; // separate opt-in: corrected lines may train the shared model
};                                     // (forced false while store_handwriting_samples is false)

type User = { id: string; email: string; display_name: string; created_at: string; settings: UserSettings };

type TrackParameter = { value: number; start: number; target: number; unit: string; label: string };

type ProgressionUpdate = {           // one per track a completed session touched
  track: string;
  previous_level: number;
  level: number;
  promoted: boolean;
  demoted: boolean;
  reason: "perfect" | "held" | "daily_cap" | "struggling" | "max_level";
  message: string;                 // learner-facing, encouraging, one sentence
  progress: number;                // 0..1 toward the yearly target
  parameters: Record<string, TrackParameter>;
};

type TrackSnapshot = { track: string; level: number; parameters: Record<string, TrackParameter> };

type WeekSummary = {
  week_start: string;              // ISO date of the Monday
  sessions: number;
  perfect_sessions: number;
  levels_gained: number;           // net (a step back counts -1)
  average_score: number | null;
};

type Track = {
  key: string;                     // e.g. "converse.listening"
  module: "handwriting" | "literacy" | "numeracy" | "converse";
  label: string;
  description: string;
  level: number;
  max_level: number;
  progress: number;                // 0..1 toward the yearly target
  streak: number;                  // consecutive perfect sessions
  best_streak: number;
  sessions: number;
  perfect_sessions: number;
  last_promoted_on: string | null; // ISO date
  weekly_goal: number;             // sessions per week for the weekly meter
  parameters: Record<string, TrackParameter>;
  history: { date: string; level: number; score: number }[];  // one per practice day, oldest first, ≤ 365
  weekly: WeekSummary[];           // last 12 weeks, oldest first; the last entry is this week
};
```

### Health and auth

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/health` | – | `{status: "ok", version: string}` |
| POST | `/api/auth/register` | `{email, password, display_name}` | `201 {access_token, token_type: "bearer", user: User}` |
| POST | `/api/auth/login` | `{email, password}` | `{access_token, token_type: "bearer", user: User}` |
| GET | `/api/auth/me` | – | `User` |
| PATCH | `/api/auth/me/settings` | `Partial<UserSettings>` | `User` |
| DELETE | `/api/auth/me` | – | `204`; deletes the account, progress, samples, and adapters |

Passwords must be at least 8 characters (422 otherwise). Duplicate email → 409.

### Handwriting (Phase 1)

`POST /api/handwriting/transcribe` — multipart fields:
`file` (image/png, image/jpeg, image/webp, ≤ 15 MB), `mode` (`auto` | `page` | `line`, default `auto`),
`clarify` (`true`/`false`, default `false`), `decoder` (`fast` | `accurate`, default `accurate`),
`track_progress` (`true`/`false`, default `true`: "this is my own handwriting", feeds the clarity track).

```ts
type TranscribeResponse = {
  sample_id: string | null;        // set only when the user consented to storing samples
  text: string;                    // literal transcription, lines joined by "\n"
  lines: {
    index: number;
    text: string;
    confidence: number;            // 0..1
    bbox: [number, number, number, number];  // x0, y0, x1, y1 in source-image pixels
    chars: { char: string; confidence: number }[];
  }[];
  legibility: number;              // 0..1, character-confidence-weighted
  clarified: null | {              // present when clarify=true and the LLM is configured
    text: string;                  // what the writer most likely meant (spelling/grammar fixed, nothing invented)
    changes: { original: string; corrected: string; reason: string }[];
    uncertain: string[];           // words that could not be resolved confidently
  };
  model_version: string;
  personalized: boolean;           // a personal adapter trained on this user's corrections was used
  elapsed_ms: number;
  progressions: ProgressionUpdate[];  // handwriting.legibility, when track_progress=true
};
```

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/handwriting/transcribe` | multipart (above) | `TranscribeResponse` |
| POST | `/api/handwriting/samples/{sample_id}/correction` | `{text}` (lines joined by `\n`) | `{ok: true}` |
| GET | `/api/handwriting/status` | – | `{model_loaded, model_version, device, personalized}` |
| GET | `/api/handwriting/samples/count` | – | `{samples, corrected}` |
| DELETE | `/api/handwriting/samples` | – | `204`; deletes stored photos, corrections, and the personal adapter |

Corrections whose line count matches the detected lines become training pairs for the
user's personal adapter (see `docs/TRAINING.md`, "Personalization"). A 503 means no
handwriting model is configured; a 429 means the hourly limit for Claude calls was hit.

With `track_progress=true` a transcription feeds the `handwriting.legibility` track with
score `min(1, legibility / goal)`, where `goal` is the track's rising clarity goal.

### Literacy / dyslexia flash cards (Phase 2)

```ts
type Card = {
  card_id: string;
  word: string;
  mode: "read" | "spell";
  skill: string;                   // e.g. "digraph_ph"
  skill_label: string;             // e.g. "ph says /f/"
  graphemes: string[];             // ["ph","o","n","e"] — for color-coded display
  focus: [number, number] | null;  // char span of the skill's pattern within word
  prompt: string;                  // instruction shown/read to the learner
  choices: string[] | null;        // read mode: the word + look-alike/sound-alike distractors (shuffled)
  sentence: string | null;         // example sentence for spell mode (spoken via TTS)
  exposure_ms: number | null;      // read mode: how long the word is flashed
  hint: null | {                   // spell mode, fades with the spelling track's `hints` parameter
    length: number;                // show this many letter boxes
    first_letter: string | null;   // and pre-fill the first one (hints = 2)
  };
};
```

A session mixes read and spell cards for the learner's current phonics pattern plus
spaced review of earlier patterns. Completing it updates `literacy.reading` (read cards)
and `literacy.spelling` (spell cards) separately.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/literacy/session` | – | `{session_id, skill, skill_label, tracks: TrackSnapshot[], cards: Card[]}` |
| POST | `/api/literacy/session/{session_id}/answer` | `{card_id, response, latency_ms}` | `AnswerResult` |
| POST | `/api/literacy/session/{session_id}/answer-handwriting` | multipart `file`, `card_id`, `latency_ms` | `AnswerResult & {recognized: string}` |
| POST | `/api/literacy/session/{session_id}/complete` | – | `{score, perfect, correct, total, progressions: ProgressionUpdate[]}` |

```ts
type AnswerResult = {
  correct: boolean;
  expected: string;
  phonetically_plausible: boolean; // e.g. "fone" for "phone"
  feedback: string;                // one encouraging sentence
  errors: { type: "reversal" | "transposition" | "omission" | "insertion" | "substitution" | "vowel" | "doubling";
            position: number; detail: string }[];
};
```

### Numeracy / dyscalculia (Phase 3)

```ts
type Visual =
  | { type: "dots"; count: number; layout: "dice" | "random" | "line" }
  | { type: "ten_frame"; filled: number; frames: number }
  | { type: "array"; rows: number; cols: number }
  | { type: "number_line"; min: number; max: number; target: number | null };

type Problem = {
  problem_id: string;
  kind: "subitize" | "compare" | "bond" | "add" | "sub" | "mul" | "div" | "number_line";
  prompt: string;                  // e.g. "7 × 5"
  visual: Visual | null;
  answer_format: "integer" | "choice";
  choices: string[] | null;
};
```

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/numeracy/round` | – | `{round_id, track, skill, skill_label, level, time_limit_ms, parameters, problems: Problem[]}` |
| POST | `/api/numeracy/round/{round_id}/submit` | `{answers: {problem_id, answer, latency_ms}[]}` | `RoundResult` |

```ts
type RoundResult = {
  score: number; perfect: boolean; fluent_count: number; total: number;
  results: { problem_id: string; correct: boolean; expected: string; fluent: boolean; strategy_hint: string | null }[];
  progressions: ProgressionUpdate[];   // numeracy.fluency
};
```

A problem counts toward a perfect round only if it is correct **and** answered within `time_limit_ms`.

### Conversation (Phase 4)

Emotion vocabulary (shared by the LLM, the avatars, and the face panel; it is exactly
the Tavus echo-mode set):
`neutral, content, excited, elated, surprised, sad, dejected, scared, angry, disgusted, contempt`.

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/converse/sessions` | `{provider?: "local" \| "tavus"}` | `ConverseSession` |
| POST | `/api/converse/sessions/{session_id}/messages` | `{text}` | **SSE stream** (below) |
| POST | `/api/converse/sessions/{session_id}/turns/{turn_id}/check` | – | `{check_id, questions: CheckQuestion[]}` |
| POST | `/api/converse/checks/{check_id}/submit` | `{answers: {question_id, choice}[]}` | `CheckResult` |
| POST | `/api/converse/sessions/{session_id}/end` | – | `{ok: true}` |

```ts
type ConverseSession = {
  session_id: string;
  provider: "local" | "tavus";     // server falls back to "local" if Tavus is not configured
  assistant: "claude" | "demo";    // "demo" = offline canned answers (no Claude credentials on the server)
  words_per_minute: number;        // target speaking rate for TTS (progression-controlled)
  expression_intensity: number;    // 0.3..1.0 (1.0 = exaggerated, lower = subtler; progression-controlled)
  captions: "full" | "delayed" | "off";
  tracks: TrackSnapshot[];         // converse.listening and converse.expressions
  tavus: null | { conversation_id: string; conversation_url: string; meeting_token: string | null };
};
```

SSE events from `.../messages` (`Content-Type: text/event-stream`):

```
event: segment
data: {"turn_id":"…","index":0,"emotion":"excited","text":"Great question!"}

event: segment
data: {"turn_id":"…","index":1,"emotion":"content","text":"Octopuses have three hearts."}

event: done
data: {"turn_id":"…","text":"Great question! Octopuses have three hearts.",
       "echo_text":"<emotion value=\"excited\"/> Great question! <emotion value=\"content\"/> Octopuses have three hearts."}

event: error
data: {"detail":"…"}
```

`echo_text` is ready to send to a Tavus replica through the Daily data channel:
`call.sendAppMessage({message_type:"conversation", event_type:"conversation.echo", conversation_id, properties:{modality:"text", text: echo_text, done:true}}, "*")`.

Tavus (verified against docs.tavus.io, October 2026):

* Replicas are now called *faces* and personas *PALs*. `LEGIBLE_TAVUS_REPLICA_ID` is a face
  id; `LEGIBLE_TAVUS_PERSONA_ID` must be a PAL created once with `"pipeline_mode": "echo"`
  (`POST /v2/pals`), so Tavus renders face and voice for text that Legible supplies, with
  its own listening, LLM, and perception switched off.
* Rooms are created private: join with `call.join({url: conversation_url, token: meeting_token})`.
* Send the `done` event's `echo_text` once per answer. Phoenix faces act on the
  `<emotion value="…"/>` tags (the same 11 values as our vocabulary).
* Speech rate and expression strength can't be set per conversation in Tavus, so
  `words_per_minute` and `expression_intensity` apply to the built-in avatar only;
  `captions` controls the app's own captions in both modes (Tavus captions are off).

Behaviour clients rely on: an `error` event can follow some `segment` events with no
`done`, and that turn cannot be checked. Asking for a check again before submitting
returns the same check; the check endpoint returns 409 once it is submitted or if the
answer never finished; submitting twice is 409; checks still work after the session
ended; `/end` is idempotent. `/messages` and the check endpoint return 429 when the
learner's hourly Claude limit is reached (no limit applies to the offline demo).

```ts
type CheckQuestion = { question_id: string; kind: "main_idea" | "detail" | "emotion" | "inference";
                       prompt: string; choices: string[] };
type CheckResult = { score: number; perfect: boolean;
                     results: { question_id: string; correct: boolean; expected: string; explanation: string }[];
                     progressions: ProgressionUpdate[] };
```

`emotion` questions ("How did the avatar feel when it said …?") score the
`converse.expressions` track and offer `emotion_choices` feelings; the other kinds score
`converse.listening`, which also sets how many questions a check has.

### Progress (Phase 5)

| Method | Path | Response |
|---|---|---|
| GET | `/api/progress` | `{week_start: string, tracks: Track[]}` |

Track keys: `handwriting.legibility`, `literacy.reading`, `literacy.spelling`,
`numeracy.fluency`, `converse.listening`, `converse.expressions`.

## Facial-expression panel (Phase 4, client side)

The panel consumes 52 ARKit-style blendshape scores (`browInnerUp`, `mouthSmileLeft`, …),
either from MediaPipe Face Landmarker running on the avatar's `<video>` (realistic
providers) or directly from the local avatar's animation state (ground truth). Scores
map to FACS action units, and action units map to emotion prototypes (EMFACS):

| AU | Meaning (learner-facing) | Blendshapes |
|---|---|---|
| 1 | inner eyebrows raised | `browInnerUp` |
| 2 | outer eyebrows raised | `browOuterUpLeft/Right` |
| 4 | eyebrows pulled down and together | `browDownLeft/Right` |
| 5 | eyes opened wide | `eyeWideLeft/Right` |
| 6 | cheeks raised (eyes crinkle) | `cheekSquintLeft/Right` |
| 7 | eyelids tightened | `eyeSquintLeft/Right` |
| 9 | nose wrinkled | `noseSneerLeft/Right` |
| 10 | upper lip raised | `mouthUpperUpLeft/Right` |
| 12 | mouth corners pulled up | `mouthSmileLeft/Right` |
| 14 | dimples (often one-sided) | `mouthDimpleLeft/Right` |
| 15 | mouth corners pulled down | `mouthFrownLeft/Right` |
| 20 | lips stretched sideways | `mouthStretchLeft/Right` |
| 23/24 | lips pressed or tightened | `mouthPressLeft/Right` |
| 26 | jaw dropped | `jawOpen` |

| Emotion | Prototype |
|---|---|
| happy (content / excited / elated) | AU6 + AU12 |
| sad (sad / dejected) | AU1 + AU4 + AU15 |
| surprised | AU1 + AU2 + AU5 + AU26 |
| scared | AU1 + AU2 + AU4 + AU5 + AU7 + AU20 + AU26 |
| angry | AU4 + AU5 + AU7 + AU23/24 |
| disgusted | AU9 + AU10 + AU15 |
| contempt | one-sided AU12 + AU14 |
