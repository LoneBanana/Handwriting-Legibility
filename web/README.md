# Legible web app

React 19 + TypeScript + Vite frontend for Legible. It talks only to the FastAPI backend
under `/api` (see `docs/ARCHITECTURE.md` for the contract; `src/api/types.ts` mirrors it).

```bash
npm install
npm run dev          # http://localhost:5173, proxies /api to http://localhost:8000 (LEGIBLE_API_URL overrides)
npm run typecheck && npm run lint && npm test
npm run build        # -> dist/, served by the API in production
```

`npm run dev` and `npm run build` first copy the MediaPipe vision WASM runtime into
`public/mediapipe/wasm` so the facial-expression panel is self-hosted. Optional settings
are in `.env.example`.

Layout: `src/pages/` (one page per module), `src/features/<module>/` (module-specific
components and pure logic with `*.test.ts` unit tests), `src/components/` (shared UI),
`src/lib/` (speech, settings, formatting, charts), `src/api/` (typed client, SSE parser).
Accessibility is a requirement, not a polish step: keyboard operation, visible focus,
screen-reader labels and live regions, reduced motion, and the user's font, size,
spacing, and theme settings apply everywhere.
