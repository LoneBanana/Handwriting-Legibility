# Progression (Phase 5)

Every practice module shares one rule set (`src/legible/progression/`). The goal is
growth that feels *constructive*: difficulty rises only after the learner has shown
they are ready, in steps too small to notice from one day to the next, and it eases off
before practice turns into frustration.

## Tracks and levels

Each module owns one or more **tracks**. A track has an integer **level** (0 to about
300) and a few **parameters** (speech rate, time limit, flash duration, …). Parameters
are pure functions of the level, so the level is the only thing stored per learner, and
changing a curve later changes everyone's settings consistently.

After every completed session the track receives a score from 0 to 1:

| Outcome | Rule | Effect |
|---|---|---|
| **perfect** | score ≥ `promote_threshold` (1.0: every item right, and on time where timed) | level + 1 |
| **daily cap** | perfect, but the track already went up today (learner's own time zone) | level kept; streak still grows |
| **held** | any score below perfect | level kept |
| **struggling** | `struggle_window` (3) sessions in a row below `struggle_threshold` (0.6) | level − 1 (never below 0) |
| **max level** | perfect at the top level | level kept |

One promotion per day at most means even a learner who practises a lot climbs at about
one level a day, so ~300 levels is about a year of near-daily perfect practice, the
time frame the targets below are designed for. A middling session (between 0.6 and
perfect) breaks a "struggling" run, so only sustained difficulty steps the level back.

## Why the curves are geometric

Most parameters follow a geometric curve from `start` (level 0) to `target`
(`levels_to_target`): every level changes the value by the same *ratio*. How noticeable
a change in speed or duration feels depends on its relative size (Weber's law), so a
constant ratio keeps every step equally gentle: +0.16 % speech rate per level is as
imperceptible at 170 words per minute as it was at 110. Counts (choices, hints, cards)
and percentages that reach zero use linear curves.

| Track | Parameter | Start → target | Level 100 | Level 200 | Per level |
|---|---|---|---:|---:|---:|
| `handwriting.legibility` (150 levels) | clarity goal | 60 % → 90 % | 80 % | 90 % | +0.2 points |
| `literacy.reading` | word flash time | 2000 → 500 ms | 1260 ms | 794 ms | −0.46 % |
| | answer choices | 3 → 5 | 4 | 4 | linear |
| | reading cards per session | 5 → 8 | 6 | 7 | linear |
| `literacy.spelling` | spelling cards per session | 4 → 8 | 5 | 7 | linear |
| | hints (2 = letter boxes + first letter, 1 = boxes, 0 = none) | 2 → 0 | 1 | 1 | linear |
| `numeracy.fluency` | base time per problem (× 0.6–1.5 by skill) | 10 → 2.5 s | 6.3 s | 4.0 s | −0.46 % |
| | problems per round | 8 → 15 | 10 | 13 | linear |
| | picture support | 100 % → 0 % | 67 % | 33 % | linear |
| `converse.listening` | speaking speed | 110 → 180 wpm | 130 wpm | 153 wpm | +0.16 % |
| | caption support (≥ 67 full, ≥ 34 after each sentence, else off) | 100 → 0 | delayed | off | linear |
| | comprehension questions | 2 → 4 | 3 | 3 | linear |
| `converse.expressions` | expression strength | 100 % → 30 % | 67 % | 45 % | −0.40 % |
| | feelings to choose from | 3 → 6 | 4 | 5 | linear |

Speech starts slower than typical conversation (about 150 wpm) and ends a little faster;
expressions start exaggerated and end at everyday subtlety; the math timer never cuts a
learner off. It only decides whether an answer counts as fluent.

Modules also move *content* forward with the level: the literacy pattern (phonics
skill) follows the lower of the reading and spelling levels, and the numeracy skill
ladder follows the fluency level, always mixed with spaced review of earlier material.

## Weekly meter

`GET /api/progress` returns, per track, the last 12 weeks (Monday-based, in the
learner's time zone): sessions, perfect sessions, net levels gained, and average score,
plus a `weekly_goal` (sessions per week, default 5; 3 for handwriting). The Progress
page shows this week's sessions against the goal next to the yearly progress bar, so
the learner sees steady effort being rewarded even in weeks when the level holds.

## Changing the rules

Edit `src/legible/progression/tracks.py`: each `TrackSpec` lists its parameters as
`Curve(start, target, unit, label, shape, integer)` plus `levels_to_target`,
`promote_threshold`, `struggle_threshold`, `struggle_window`, `max_promotions_per_day`,
and `weekly_goal`. The engine (`engine.py`) is pure and fully unit-tested
(`tests/progression/`).
