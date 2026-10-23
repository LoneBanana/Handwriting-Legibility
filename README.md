# Legible

Legible is an assistive learning app for people with dysgraphia, dyslexia,
dyscalculia, auditory and language processing disorders, and nonverbal learning
disorder. It is one web app with one account, where every module shares a gentle,
mastery-based progress system.

| | Module | What it does |
|---|---|---|
| 1 | **Handwriting** (dysgraphia) | Photograph handwriting and get readable text. A CNN-ViT hybrid model reads it, uncertain letters are highlighted, Claude suggests what was probably meant (fixing form, never adding ideas), and the text can be read aloud. With consent, the model adapts to *your* handwriting from your corrections. |
| 2 | **Reading & spelling** (dyslexia) | Short flash-card sessions in a structured-literacy sequence: words flashed for reading, spoken for spelling, colour-coded by sound chunks, with error analysis ("b and d swapped") and spaced review. |
| 3 | **Math facts** (dyscalculia) | Quick rounds from subitizing dots and number bonds up to times tables, with dot/ten-frame/array/number-line pictures that fade over time, and concrete strategy tips. |
| 4 | **Talk & faces** (APD, LPD, NVLD) | Ask anything, as you would ask a search engine. A talking avatar answers in short spoken segments with facial expressions while a live panel shows which facial movements it is making and what feeling they signal. Short comprehension checks follow. |
| 5 | **Progress** | Per-skill levels and a weekly meter. Speed, subtlety, and difficulty rise only after *perfect* sessions (at most one step a day) and ease back if practice gets hard, reaching their targets over about a year. |
| 6 | **Website** | React app with accessibility at its core: OpenDyslexic/Lexend/Atkinson Hyperlegible fonts, adjustable size and spacing, light/dark/cream/high-contrast themes, reduced motion, and read-aloud everywhere. |

Architecture and the full HTTP contract: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Model training on a GPU cluster: [docs/TRAINING.md](docs/TRAINING.md).
Progression rules and curves: [docs/PROGRESSION.md](docs/PROGRESSION.md).

## Quick start (development)

Requires Python 3.10+ and Node 20+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,train,export,dev]"
legible-api --reload                     # API on http://localhost:8000 (docs at /api/docs)
```

In a second terminal:

```bash
cd web && npm install && npm run dev     # app on http://localhost:5173 (proxies /api)
```

Everything runs without external services. Without Claude credentials, the Talk module
answers from an offline demo set and "what was probably meant" is skipped. Without a
handwriting checkpoint, the Handwriting page explains that reading isn't set up yet.
To try the handwriting pipeline end to end, train the tiny model for a few minutes (it
learns synthetic handwriting only; it is not a useful reader of real handwriting):

```bash
legible-train --config configs/htr/tiny.yaml
LEGIBLE_HTR_CHECKPOINT=runs/htr-tiny/best.pt legible-api --reload
```

## Configuration

Settings are environment variables (or a `.env` file; see [.env.example](.env.example)).

| Variable | Purpose |
|---|---|
| `LEGIBLE_HTR_CHECKPOINT` | Handwriting model checkpoint (`best.pt`/`final.pt` from training) |
| `LEGIBLE_HTR_DEVICE` | `auto` (CUDA, then Apple MPS, then CPU), or e.g. `cuda:1` |
| `LEGIBLE_HTR_ONNX` | Optional `legible-export` file: ~3× faster "Quick" reading on CPU-only servers |
| `ANTHROPIC_API_KEY` | Enables Claude for clarification, conversation, and comprehension questions |
| `LEGIBLE_LLM_MODEL` | Claude model id (default `claude-opus-5-5`) |
| `LEGIBLE_TAVUS_API_KEY`, `LEGIBLE_TAVUS_REPLICA_ID`, `LEGIBLE_TAVUS_PERSONA_ID` | Optional realistic video avatar (Tavus); otherwise the built-in avatar |
| `LEGIBLE_DATABASE_URL` | Defaults to SQLite in `data/`; use Postgres in production (`pip install -e .[postgres]`) |
| `LEGIBLE_JWT_SECRET` | Required in production |
| `LEGIBLE_DATA_DIR` | Database, consented uploads, and personal adapters |
| `LEGIBLE_AUTO_MIGRATE` | Apply database migrations at startup (default `true`); turn off with several replicas and run `legible-db upgrade` per release |
| `LEGIBLE_CORS_ORIGINS` | JSON list of allowed browser origins when the web app is served elsewhere |

## Training the handwriting model

```bash
python scripts/download_fonts.py                         # open-licensed handwriting fonts
legible-prepare hf Teklia/IAM-line --out data/iam        # real handwriting (check licenses)
sbatch --nodes=4 scripts/slurm/train_htr.sbatch          # multi-node, preemption-safe
legible-eval --checkpoint runs/htr-large/best.pt --manifest data/iam/test.jsonl
```

The model is a ConvNeXt front end (local stroke features) feeding a ViT encoder with
2-D rotary attention (global context across the line), trained jointly with a CTC head
and an attention decoder. Synthetic training lines simulate dysgraphic traits (uneven
size, floating letters, irregular spacing, reversals, crossed-out false starts). See
[docs/TRAINING.md](docs/TRAINING.md) for data, recipes, scaling, and evaluation.

## Tests

```bash
python -m pytest                         # backend, model, data pipeline, progression
cd web && npm run typecheck && npm run lint && npm test && npm run build
```

## Deployment

```bash
cp .env.example .env                     # set LEGIBLE_JWT_SECRET, POSTGRES_PASSWORD, keys
docker compose up --build                # Postgres + API + built web app on :8000
```

The image serves the API and the built web app from one origin. Mount trained
checkpoints at `/models` and set `LEGIBLE_HTR_CHECKPOINT`. For GPU inference use a CUDA
base image (see the `Dockerfile` comments). Run one worker per process (the model is
loaded once per process) and scale with replicas. Database migrations (Alembic) run at
startup; with several replicas set `LEGIBLE_AUTO_MIGRATE=false` and run
`legible-db upgrade` once per release. After changing `models.py`, generate a migration
with `legible-db revision -m "what changed"` and review it before committing.

## Privacy and scope

* Handwriting photos are stored only if the learner opts in (Settings), and only to
  personalise *their own* model. A second, separate opt-in shares corrected lines
  (pseudonymously) to improve the shared model for everyone; see docs/TRAINING.md.
  Settings can delete all samples, or the whole account and everything attached to it.
* Text sent to Claude (transcriptions to clarify, questions asked in Talk) is processed
  by Anthropic's API; the Talk avatar, if Tavus is enabled, is rendered by Tavus.
* Legible supports practice and access. It is not a diagnostic tool or a medical
  device, and it does not replace assessment or support from qualified professionals.

## Project layout

```
src/legible/
  htr/          Phase 1 model, data pipeline, trainer, segmentation, inference, LoRA personalization, ONNX export
  literacy/     Phase 2 phonics sequence, word bank, memory model, spelling error analysis
  numeracy/     Phase 3 skill ladder, problem generators, strategy hints, fluency scoring
  converse/     Phase 4 emotion-tagged answers, Tavus client, comprehension checks
  progression/  Phase 5 tracks, curves, and the promotion engine
  api/          Phase 6 FastAPI app: auth, routers, services, database models
web/            React + TypeScript frontend (Vite)
configs/htr/    training configs (tiny, base, large, dysgraphia fine-tune)
scripts/        SLURM jobs and the font downloader
docs/           architecture, training, progression
tests/          pytest suite
```
