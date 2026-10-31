# Backend Implementation Plan

## Overview

Python backend that implements [`API_CONTRACT.md`](API_CONTRACT.md): it takes an intake (amount, horizon, risk tolerance), then runs 7 Gemini agents that research the market, build a two-level portfolio, backtest it in the Gemini code-execution sandbox, and check that backtest against an independent pandas recompute. If the Verifier rejects the portfolio, it is rebalanced once. Agent progress streams over SSE. Hosted on Cloud Run. **Ships today.**

## Architecture decisions

- **FastAPI + uvicorn, one process, runs stored in memory** (a dict of runId → event list). SSE replays the list, which gives `Last-Event-ID` resume for free.
  `ponytail:` Cloud Run must be set to `--max-instances=1` and `--min-instances=1` because of the in-memory state. Swap in Firestore if we ever need more than one instance.
- **`google-genai` Interactions API** (`client.interactions.create`), which the current docs use for tools and for structured output.
- **Models:**
  - `gemini-3.8-flash` for Discovery, Financials, Landscape, News and Social
  - `gemini-4-argon` for the Orchestrator and Verifier
  - Model names live in a single `config.py`.
- **Real numbers come from data, not model memory.** yfinance supplies prices and financial statements. The LLM interprets the data; it doesn't recall figures.
- **Backtest:**
  - Gemini code execution computes the metrics from weekly price data we send as a CSV (the sandbox has pandas and numpy and a 30-second limit).
  - Our pandas engine recomputes the same numbers independently and produces the equity curve.
  - `crossCheck.isMatch` compares the two results.
  - If they don't match, we report our own numbers and the check fails visibly.
- **Verifier = deterministic checks + LLM judgment.**
  - Python checks the hard limits:
    - drawdown limit by risk level: CONSERVATIVE -20%, MODERATE -35%, AGGRESSIVE -55%
    - weights sum to 100
    - no single holding above 25%
    - cross-check match
  - Argon writes `changesRequested` from the failures, and the Orchestrator rebalances. Maximum 2 rounds.
- **Call budget:** about 9–12 Gemini calls per run (each agent makes one batched call covering all symbols, not one per symbol). That's well under the hackathon's abuse threshold.
- **Weights:** the LLM proposes them, and Python normalises them to exactly 100 and rounds them before anything is emitted. The UI never recomputes them.
- **Secrets:** `GEMINI_API_KEY` comes from the environment only (`.env` locally, a Cloud Run environment variable or secret in production). `.gitignore` already excludes `.env`.

## Verified API facts (live-tested 2026-10-31 against the hackathon key; workers: use these exactly)

```python
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
i = client.interactions.create(
    model="gemini-3.8-flash",               # or "gemini-4-argon"
    input="..." ,                            # or list of parts (see CSV below)
    tools=[{"type": "google_search"}],       # and/or {"type": "code_execution"}
    response_format={"type": "text", "mime_type": "application/json", "schema": Model.model_json_schema()},
)
Model.model_validate_json(i.output_text)
# i.steps[*].type ∈ thought | model_output | google_search_call | google_search_result
#                    | code_execution_call (.arguments.code) | code_execution_result (.result)
# CSV into sandbox: input=[{"type":"text","text":"..."},
#                          {"type":"document","mime_type":"text/csv","data":base64_str}]
```
- ✅ Search grounding plus a schema on `gemini-4-argon` (about 11s). ✅ A CSV in the sandbox with pandas computing a 60/40 weekly-rebalanced return exactly (about 18s).
- Latency: about 5s for a basic call, about 10s with search or code execution. Budget about 2–3 minutes for a full run.
- ✅ yfinance works: `yf.download([...], period="5y", interval="1wk", auto_adjust=True)["Close"]` returns 262 rows. `Ticker(s).info["trailingPE"]` works too.

## Inherited from Vibe-Trading (`HKUDS/Vibe-Trading`, MIT)

The clone is at `../salmon/Vibe-Trading` locally. I live-tested the vendored modules on real yfinance data (NVDA/MSFT/BTC/GLD):
max drawdown -48.6%, Sharpe ratio 1.01 with a 95% bootstrap confidence interval of **0.07–1.89**, and risk-parity weights GLD 47 / MSFT 25 / BTC 15 / NVDA 14.

| What | Source | How we use it |
|---|---|---|
| **Risk x-ray** `compute_risk_xray(closes, weights, periods_per_year=52)` | `agent/backtest/risk_xray.py` | **Vendor.** Verifier's drawdown, volatility, VaR/ES tails, concentration (HHI) and correlation. It's the independent recompute side of `crossCheck` |
| **Bootstrap Sharpe CI** `bootstrap_sharpe_ci(equity, bars_per_year=52)` | `agent/backtest/validation.py` | **Vendor.** Fills `Backtest.sharpeCi`, and the Verifier rejects the portfolio if the lower bound is ≤ 0. Strip `sharpe_samples` before emitting |
| **Metrics helpers** (`period_returns`, `effective_bars_per_year`) | `agent/backtest/metrics.py` (+ `models.py`) | **Vendor.** `risk_xray` and `validation` depend on them |
| **Risk-parity / inverse-vol optimizers** `RiskParityOptimizer()._calc_weights({"cov": cov})` | `agent/backtest/optimizers/` | **Vendor.** Gives the Orchestrator a quant baseline ("risk parity would say GLD 47%") to anchor or justify its weights |
| **Agent prompts** | `agent/src/swarm/presets/*.yaml` | **Adapt the text into `agents.py`:** `social_alpha_team` (Twitter/Reddit analysts → SOCIAL), `fundamental_research_team` (Financial, Valuation and Quality analysts → FINANCIALS), `sentiment_intelligence_team` (→ NEWS), `global_allocation_committee` allocator (→ ORCHESTRATOR), `risk_committee` Head of Risk (→ VERIFIER) |
| **Domain checklists** | `agent/src/skills/{financial-statement,valuation-model,sentiment-analysis,social-media-intelligence,asset-allocation,risk-analysis}/` | **Adapt:** pull the key bullet points into each agent's prompt (one paragraph at most, to keep prompts short) |

**Not inherited, and why:**
- **Swarm runtime** (langchain/langgraph, 50 iterations and 1,800s timeouts per agent, file-based task store): far too slow for a 3-minute demo. We use one Gemini call per agent instead.
- **`api_server.py`:** a different API shape; we follow our contract.
- **The vibe-trading React frontend:** the team is building in Next.js.
- **The 40+ loaders and registry:** yfinance alone covers stocks, crypto and gold.
- **The grounding/evidence ledger:** our `Finding.sources` covers it.

`ponytail:` we vendor about 1.6k lines instead of `pip install vibe-trading-ai`, because that package pulls in langchain, tushare, akshare, ccxt, duckdb and more (a slow Cloud Run build). Re-sync the copy if upstream fixes a metric bug.

## File layout (target)

```
backend/
  app/
    main.py       # FastAPI routes, SSE, request validation
    models.py     # Pydantic mirror of the contract types
    config.py     # model names, risk limits, budgets
    llm.py        # one ask() helper: tools, schema, retries, logging
    data.py       # yfinance prices and financial statements, with on-disk cache
    agents.py     # 7 agent functions (prompt + schema each)
    engine.py     # equity curves, benchmarks, sandbox call, crossCheck (uses vendored backtest/)
    pipeline.py   # orchestration and event emission
  backtest/       # VENDORED from Vibe-Trading, unmodified: __init__, models, metrics, validation, risk_xray, optimizers/
  NOTICE          # MIT attribution for backtest/
  tests/test_engine.py
  pyproject.toml  # pandas<3 (vendored code and yfinance break on pandas 3), numpy, scipy
Dockerfile
```

## Task list

### Phase 1 — Foundation (unblocks the frontend)

**Task 0: Vendor the Vibe-Trading modules** · XS · *coordinator (me)*
- [ ] Copy them into `backend/backtest/` unchanged and add `NOTICE`.
- [ ] Smoke check: `compute_risk_xray`, `bootstrap_sharpe_ci` and `RiskParityOptimizer` all run on yfinance data.


**Task 1: Contract-compliant server with a fake pipeline** · M
Builds all 4 endpoints. A fake pipeline replays `fixtures/report.example.json` as timed SSE events, so the frontend can integrate against a real server within the first hour.
- [ ] `POST /api/runs` returns 202 `Run`. A bad intake returns 422 `VALIDATION_ERROR` with a `{field: msg}` map.
- [ ] `GET /api/runs/{id}/events` streams named events with `id: runId:seq`, replays from `Last-Event-ID`, and closes after `run.completed`.
- [ ] Unknown run → 404 `RUN_NOT_FOUND`. A second concurrent run → 429 `RATE_LIMITED`.
- Verify: run `curl -N` against the stream and see the full event sequence. Then point the Next.js frontend's `NEXT_PUBLIC_BACKEND_URL` at it. CORS is enabled.
- Files: `main.py`, `models.py`, `pipeline.py` (fake), `pyproject.toml`

**Task 2: Deploy to Cloud Run** · S
- [ ] `Dockerfile` added. Deployed with `--max-instances=1 --min-instances=1` and the key set as an environment variable.
- [ ] Public URL returns `/api/health` 200. URL shared with the frontend team.
- Verify: `curl https://<url>/api/health`
- Blocked on: `gcloud` CLI (not installed here) **or** deploying from Cloud Run's web console from a GitHub repo.

**Task 3: Gemini helper** · S
`ask(model, prompt, schema=None, tools=[])` → a validated Pydantic object. It handles retries with backoff on 429/503 and falls back from `gemini-4-argon` to `gemini-3.8-flash` on repeated 503s. Every call is logged with model, prompt and output.
- [ ] A smoke script makes one live call each with `google_search`, `code_execution` and a schema, and all three succeed.
- Files: `config.py`, `llm.py`

### Checkpoint A
- [ ] The frontend is consuming the deployed fake stream. The Gemini helper works live. **Review with the team.**

### Phase 2 — Research agents (each one replaces part of the fake pipeline)

**Task 4: Data layer** · S
`prices(symbols, years=5)` → **weekly** closes as a DataFrame (one frequency everywhere: the equity curve, the sandbox CSV, and `risk_xray`/bootstrap with 52 periods per year). `fundamentals(symbol)` → a dict of revenue, margins, debt, cash, P/E and so on. Results are cached to disk.
- [ ] Works for `NVDA`, `BTC-USD`, `GLD`, `SPY`, `AGG`. Cached files survive a restart.
- Files: `data.py`

**Task 5: Discovery agent** · S
Search grounding → 4–6 stocks and 2–3 coins, each with a finding and sources. Gold is always `GLD`.
- [ ] Every symbol it returns resolves in yfinance; any that don't are dropped with an `agent.progress` note.
- Files: `agents.py`, `pipeline.py`

**Task 6: Financials agent** · S
The financial statements for all stocks go to code execution in one call, which returns a 1–10 score and a verdict per stock. The prompt is adapted from `fundamental_research_team.yaml`: the Financial, Valuation and Quality analyst lenses folded into one call.
- [ ] The `agent.progress` events show the sandbox code ran.
- Files: `agents.py`

**Task 7: Landscape, News and Social agents in parallel** · M
Same pattern for all three. The prompts are adapted from `social_alpha_team.yaml` (SOCIAL), `sentiment_intelligence_team.yaml` (NEWS) and `global_equities_desk.yaml` (LANDSCAPE). They run with `asyncio.gather`. If one fails, it emits `agent.failed` and the run continues without it.
- [ ] Their events interleave in the stream. Killing one agent doesn't kill the run.
- Files: `agents.py`, `pipeline.py`

### Checkpoint B
- [ ] Real findings from 5 agents stream to the frontend. Calls per run ≤ 6 so far.

### Phase 3 — Decision and verification

**Task 8: Orchestrator** · S
All findings, the intake **and the vendored risk-parity weights for the candidates** go to Argon, which returns an `Allocation`. The prompt is adapted from the allocator in `global_allocation_committee.yaml`. Python then normalises the weights and computes `amountUsd`.
- [ ] Slice weights sum to exactly 100.0, and each slice equals the sum of its holdings.
- Files: `agents.py`

**Task 9: Backtest engine and sandbox cross-check** · M
- `engine.py` builds weekly-rebalanced equity curves for the portfolio and the 3 benchmarks (SPY, a 60/40 SPY/AGG mix, equal-weight).
- `Metrics` comes from vendored `compute_risk_xray` (drawdown, volatility) plus our total return and CAGR.
- `sharpeCi` comes from vendored `bootstrap_sharpe_ci`.
- The same weekly CSV goes to Gemini code execution, and `crossCheck` compares the sandbox's total return and max drawdown with ours.
- [ ] `tests/test_engine.py`: a synthetic series [100,120,90,130,65,140] gives max drawdown -50% and the expected total return. The 60/40 weekly-rebalanced fixture from the live test gives 22.12%.
- [ ] The sandbox and our total returns agree within 0.5 percentage points on a real portfolio.
- Files: `engine.py`, `tests/test_engine.py` (do **not** edit `backtest/`)

**Task 10: Verifier, rebalance loop and final report** · M
Deterministic checks (drawdown limit for the risk level, weights sum to 100, max single holding 25%, `crossCheck.isMatch`, **Sharpe CI lower bound > 0**, and risk-x-ray concentration) plus Argon writing `changesRequested`. The prompt is adapted from the Head of Risk in `risk_committee.yaml`. Then round 2 if needed, then `Report` assembly.
- [ ] An AGGRESSIVE-heavy crypto proposal under a MODERATE intake gets rejected in round 1 and approved in round 2.
- [ ] `run.completed.report` validates against `models.Report`.
- Files: `agents.py`, `pipeline.py`

**Task 13: Intake chat agent** · S
`POST /api/intake/messages` is stateless: the whole history goes in, and an `IntakeTurn` comes back from one Flash call with a structured schema. Python validates the draft against `IntakeInput` and only sets `isComplete` when the required fields are valid.
- [ ] A test with a scripted 3-turn conversation ("10k" → "2 years" → "moderate") ends with `isComplete` and `amountUsd == 10000`.
- [ ] 21 messages → 422.
- Files: `main.py` (route only), `agents.py` (`intake_turn`)

### Checkpoint C — end to end
- [ ] A real run from the deployed frontend finishes in under about 3 minutes with 12 or fewer Gemini calls.
- [ ] The round-1 rejection is visible in the UI.

### Phase 4 — Demo safety

**Task 11: Saved-run replay** · S
`POST /api/runs?replay=<file>` replays a saved real run, which is our fallback if the API returns 503 on stage. Save 2–3 good real runs as fixtures.
- [ ] Replay is indistinguishable from a live run in the UI.

**Task 12: Export everything** · XS
Everything gets pushed to GitHub and the demo video is recorded before the account is deleted.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Yahoo blocks or rate-limits Cloud Run IPs | High | Disk cache plus pre-fetched data for the likely tickers committed to the repo |
| `gemini-4-argon` 503 under hackathon load | High | Retry, then fall back to `gemini-3.8-flash`; Task 11 replay |
| Sandbox 30s limit on the backtest | Med | Weekly (not daily) prices for 12 or fewer symbols over 5 years, about 260 rows |
| Combining structured output with tools is "Preview" | Low | ✅ Live-tested and works. Fallback if it breaks: make two calls, one with tools and one with the schema |
| In-memory state lost on Cloud Run restart | Low | `min-instances=1`; the frontend falls back to `GET /api/runs/{id}` |
| Look-ahead bias accusation | Med | We only backtest *today's* weights over history and say so in the disclaimer |

## Open questions

1. **Deploy path:** do we install `gcloud` here (`brew install --cask google-cloud-sdk`), or does a teammate deploy from the Cloud Run console?
2. **Social data:** Search grounding only (planned). Is that enough for you, or does someone have Reddit API keys?
