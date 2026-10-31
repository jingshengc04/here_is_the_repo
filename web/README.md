# VibeTrader — web

Next.js 16 + shadcn/ui + Recharts frontend for the portfolio-agents product.
API contract (source of truth): `../docs/API_CONTRACT.md` · wireframes: `../frontend/wireframes.html`.

```bash
pnpm install
cp .env.example .env.local
pnpm dev                      # http://localhost:3000
```

## Flow
Onboarding chat (`POST /api/intake/messages`) → confirm card (editable; also the plain-form fallback) → **Run** (`POST /api/runs`)
→ live agent lanes from SSE (`GET /api/runs/{id}/events`, polling `GET /api/runs/{id}` every 3 s if the stream drops)
→ report: round 1 → round 2, sunburst, holdings + evidence, backtest, disclaimer. `?run=<id>` restores a run.

Contract types: `lib/types.ts` (copied verbatim from the contract). Client: `lib/api.ts`. Event reducer: `lib/run-state.ts`.

## Backends
| How | What |
|---|---|
| `NEXT_PUBLIC_USE_MOCK=true` or `?mock=1` | In-browser replay of `lib/mocks/report.example.json` (`pnpm sync:fixture` re-copies it from `docs/fixtures`). No backend needed. |
| `pnpm fake-backend` + `?mock=0&backend=http://localhost:8787` | Node server implementing the whole contract (CORS, SSE with `id: runId:seq` + `Last-Event-ID` replay, 422 / 404 / 429). `POST /api/runs?scenario=agent_failed\|run_failed` for error paths. `SPEED=4` replays faster. |
| `?mock=0&backend=https://<cloud-run-or-tunnel>` | Real backend. `NEXT_PUBLIC_BACKEND_URL` sets the default at build time. |

The browser calls the backend directly (no Next.js proxy, so SSE isn't buffered). A deployed https page can't call an `http://` backend.

## E2E (Playwright)
```bash
pnpm e2e                                                              # local: mock UI + fake-backend live path
E2E_BASE_URL=https://<preview>.vercel.app vercel env run -- pnpm e2e  # protected Vercel preview (live-backend specs skip)
```

## Deploy
```bash
vercel deploy --environment preview --build-env NEXT_PUBLIC_USE_MOCK=true
```
