# Portfolio Research API — Contract v1

> **Audience:** frontend teammates and their coding agents. **This file is the source of truth and supersedes `API_SPECIFICATION.md`** (`/py/parse`, `/py/run` are gone).
> Frontend stack stays as in `frontend/frontend_spec.md` (Next.js + Tailwind + shadcn + Recharts, light theme). Only the API, types and screens change.
> Backend implements exactly this. Changes are **additive only** (new optional fields / new event types);
> nothing here gets renamed or removed without a heads-up in the team chat.

## What the product does

User fills an intake (amount, horizon, risk) → backend runs a team of Gemini agents →
streams their progress live → returns a **verified two-level portfolio**:

- **Inner ring:** asset classes: `STOCKS` / `CRYPTO` / `GOLD` / `BONDS` with %.
- **Outer ring:** holdings inside each class (e.g. `NVDA 14%`, `BTC-USD 15%`, `GLD 25%`).
- Every holding has a rationale + evidence links from the agents.
- A backtest compares the portfolio to benchmarks; a Verifier may **reject** round 1 and
  the Orchestrator rebalances in round 2. **The UI should make that rejection visible** — it's the demo moment.

## Agents (fixed set — render one card/lane per agent)

| `AgentId`      | Model           | Job |
|----------------|-----------------|-----|
| `INTAKE`       | Gemini 3.8 Flash (`gemini-3.8-flash`) | Onboarding chat → fills `IntakeInput` (not part of a run, so no lane) |
| `DISCOVERY`    | Gemini 3.8 Flash (`gemini-3.8-flash`) | Finds candidate stocks + crypto (Google Search grounding) |
| `FINANCIALS`   | Gemini 3.8 Flash (`gemini-3.8-flash`) | Scores each stock 1–10 from real statements (code execution) |
| `LANDSCAPE`    | Gemini 3.8 Flash (`gemini-3.8-flash`) | Competitors / market position (Search grounding) |
| `NEWS`         | Gemini 3.8 Flash (`gemini-3.8-flash`) | Recent news per holding (Search grounding) |
| `SOCIAL`       | Gemini 3.8 Flash (`gemini-3.8-flash`) | Reddit/X sentiment (Search grounding) |
| `ORCHESTRATOR` | Gemini 4 Argon (`gemini-4-argon`) | Builds the allocation from all findings |
| `VERIFIER`     | Gemini 4 Argon (`gemini-4-argon`) | Runs backtest in Gemini sandbox, cross-checks numbers, approves/rejects |

Order: `DISCOVERY` first → `FINANCIALS`, `LANDSCAPE`, `NEWS`, `SOCIAL` **in parallel** →
`ORCHESTRATOR` → `VERIFIER` → (if rejected) `ORCHESTRATOR` → `VERIFIER` again. Max 2 rounds.

---

## Endpoints

Base path: `/api`. Backend sends CORS headers for any origin, so the frontend calls it **directly**:

```
# web/.env.local
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000   # later: Cloud Run URL from team chat
NEXT_PUBLIC_USE_MOCK=true                       # true until backend is live → use fixtures/report.example.json
```

Call `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/...`. Don't go through a Next.js `rewrites()` proxy for
the SSE stream, because proxies can buffer it and the agents' progress would arrive all at once.

### `POST /api/intake/messages` — chat onboarding (Intake agent)

**Stateless:** the frontend keeps the chat history and sends all of it each turn. Response `200`: `IntakeTurn`.
One Gemini Flash call per turn, typically 2–4 turns.

```http
POST /api/intake/messages
Content-Type: application/json

{ "messages": [
    { "role": "user", "content": "I have about 10k, want growth but can't stomach crypto crashes" }
] }
```
```json
{ "reply": "Got it: $10,000 with moderate risk. How long can you leave it invested, 1 year or more like 5?",
  "intake": { "amountUsd": 10000, "horizonYears": null, "riskTolerance": "MODERATE", "theme": null },
  "isComplete": false }
```

- Append `{role:"assistant", content: reply}` to the history and send the next user message.
- When `isComplete === true`, every required field in `intake` is non-null. Show a **confirm card** with editable fields
  and a **Run** button that sends `POST /api/runs` with that `IntakeInput`. Never auto-start a run.
- Maximum 20 messages per request (`422` beyond that). Each `content` is at most 2,000 chars.
- If this endpoint fails, show the plain form, which uses the same fields. The demo must not depend on the chat working.

### `POST /api/runs` — start a run

Request body: `IntakeInput`. Response `202`: `Run` (status `QUEUED`, `report: null`).

```http
POST /api/runs
Content-Type: application/json

{ "amountUsd": 10000, "horizonYears": 2, "riskTolerance": "MODERATE", "theme": "AI infrastructure" }
```

### `GET /api/runs/{runId}/events` — live stream (SSE)

`text/event-stream`. Each message has:
- `event:` = the event `type` (see `RunEvent` below) — subscribe with `new EventSource(url)` + `addEventListener(type, ...)` per type.
- `id:` = `"{runId}:{seq}"` — supports `Last-Event-ID` reconnect; backend replays missed events.
- `data:` = JSON of the event.

Stream closes after `run.completed` or `run.failed`. Connecting after the run finished replays everything, then closes.

### `GET /api/runs/{runId}` — snapshot

Response `200`: `Run`. `report` is non-null once `status === "COMPLETED"`.
Use this for page refresh / deep links; use the stream for the live view.

### `GET /api/health`

`200 { "status": "ok" }`

### Errors (every endpoint, same shape)

```ts
interface APIError {
  error: { code: ErrorCode; message: string; details?: unknown };
}
type ErrorCode = "VALIDATION_ERROR" | "RUN_NOT_FOUND" | "RATE_LIMITED" | "INTERNAL";
```

| Status | `code` | When |
|--------|--------|------|
| 422 | `VALIDATION_ERROR` | Bad intake. `details` = `{ field: message }` map — show inline on the form |
| 404 | `RUN_NOT_FOUND` | Unknown `runId` |
| 429 | `RATE_LIMITED` | A run is already in progress (we cap concurrency to protect the hackathon quota) |
| 500 | `INTERNAL` | Anything else. `message` is safe to show |

---

## Types (copy into `web/lib/types.ts`, replacing the old VibeTrader types)

```ts
// ---------- Input ----------
export type RiskTolerance = "CONSERVATIVE" | "MODERATE" | "AGGRESSIVE";

export type Style = "GROWTH" | "VALUE" | "DIVIDEND" | "MOMENTUM" | "MEME" | "ESG" | "DEFENSIVE";

// Personalisation, filled by the Intake chat (show on the confirm card as editable chips)
export interface Preferences {
  mustInclude?: string[];   // ≤ 10 symbols the user insists on, e.g. ["DOGE-USD","NVDA"]
  exclude?: string[];       // ≤ 20 symbols OR plain words, e.g. ["TSLA","oil","tobacco"]
  assetClassTargets?: Partial<Record<AssetClass, number>> | null; // e.g. {CRYPTO: 30}; each 0–100, sum ≤ 100
  styles?: Style[];
  notes?: string | null;    // ≤ 500 chars free text, e.g. "companies I use every day"
  onlyInclude?: string[];   // NEW: restrict the universe, any words: ["biotech"], ["European banks"], ["companies I can name"]
}

export interface IntakeInput {
  amountUsd: number;        // 100 – 10_000_000
  horizonYears: number;     // 0.25 – 10 in steps of 0.25 (0.25 = 3 months, 1.5 = 18 months). Was integer 1–10; whole years still valid
  riskTolerance: RiskTolerance;
  theme?: string;           // optional focus, ≤ 200 chars, e.g. "AI infrastructure"
  preferences?: Preferences | null;  // NEW (optional)
}

// ---------- Chat onboarding ----------
export interface ChatMessage { role: "user" | "assistant"; content: string }
export interface IntakeTurnRequest { messages: ChatMessage[] }   // ≤ 20, content ≤ 2000 chars
export interface IntakeTurn {
  reply: string;                     // assistant message to render, markdown allowed
  intake: {                          // partially filled draft, null = not known yet
    amountUsd: number | null;
    horizonYears: number | null;
    riskTolerance: RiskTolerance | null;
    theme: string | null;
    preferences: Preferences | null;   // NEW: draft preferences gathered so far
  };
  isComplete: boolean;               // true ⇒ amountUsd, horizonYears, riskTolerance non-null
}

// ---------- Run ----------
export type RunStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";

export interface Run {
  runId: string;
  status: RunStatus;
  intake: IntakeInput;
  createdAt: string;        // ISO 8601
  report: Report | null;
  error: APIError["error"] | null;
}

// ---------- Agents ----------
export type AgentId =
  | "DISCOVERY" | "FINANCIALS" | "LANDSCAPE" | "NEWS" | "SOCIAL"
  | "ORCHESTRATOR" | "VERIFIER";

export interface Source { title: string; url: string }

export interface Finding {
  id: string;               // "f_12" — referenced by Holding.findingIds
  agent: AgentId;
  symbol: string | null;    // null = market-wide finding
  claim: string;            // one sentence, human readable
  sentiment: "POSITIVE" | "NEGATIVE" | "NEUTRAL";
  sources: Source[];        // may be empty for FINANCIALS (data from statements)
  numbersVerified?: boolean | null; // true = every number in `claim` matched SEC/yfinance data (±2%). null = no numbers. Show a ✓ or ⚠ badge
}

// ---------- Allocation (the sunburst) ----------
export type AssetClass = "STOCKS" | "CRYPTO" | "GOLD" | "BONDS";  // BONDS is NEW: bond ETFs (BND, SHY…) for Conservative/Moderate. Give it a 4th pie colour

export interface Holding {
  symbol: string;           // "NVDA", "BTC-USD", "GLD"
  name: string;             // "NVIDIA Corp."
  weightPct: number;        // % of WHOLE portfolio, 1 decimal
  amountUsd: number;        // weightPct * intake.amountUsd / 100
  rationale: string;        // 1–3 sentences
  findingIds: string[];     // evidence → look up in Report.findings
  financialScore: number | null;  // 1–10, stocks only
  sentimentScore: number | null;  // -1.0 … 1.0 (SOCIAL + NEWS blend)
  isUserRequested?: boolean;      // NEW: true if from preferences.mustInclude, so show a "you asked for this" tag
}

export interface AssetClassSlice {
  assetClass: AssetClass;
  weightPct: number;        // = sum of holdings[].weightPct
  rationale: string;
  holdings: Holding[];
}

export interface Allocation {
  slices: AssetClassSlice[]; // all slice weightPct sum to 100
}

// ---------- Backtest ----------
export interface Metrics {
  totalReturnPct: number;
  cagrPct: number;
  volatilityPct: number;    // annualized
  sharpe: number | null;    // null if undefined
  maxDrawdownPct: number;   // negative number, e.g. -28.4
  worstMonthPct: number;    // negative number
}

export type SeriesKey = "PORTFOLIO" | "SPY" | "SIXTY_FORTY" | "EQUAL_WEIGHT";

export interface Backtest {
  startDate: string;        // "2021-10-29"
  endDate: string;
  metrics: Record<SeriesKey, Metrics>;
  equityCurve: {
    dates: string[];        // weekly points, ~260 for 5y
    series: Record<SeriesKey, number[]>; // growth of $1, same length as dates
  };
  sharpeCi?: {              // PORTFOLIO only. 95% bootstrap CI on Sharpe (1000 resamples). Show as "Sharpe 0.74 (95% CI 0.07–1.41)"
    lower: number; upper: number;
    probPositive: number;   // 0–1, share of resamples with Sharpe > 0
  } | null;
  crossCheck: {             // sandbox result vs. our independent recompute
    sandboxTotalReturnPct: number;
    independentTotalReturnPct: number;
    isMatch: boolean;       // |diff| < 0.5pp
  };
}

// ---------- Verification ----------
export interface Check { name: string; isPassed: boolean; detail: string }

export interface Verification {
  isApproved: boolean;
  checks: Check[];          // e.g. "Max drawdown within MODERATE limit (-35%)"
  changesRequested: string[]; // empty when approved
}

export interface Round {
  round: 1 | 2;
  allocation: Allocation;
  backtest: Backtest;
  verification: Verification;
}

// ---------- Report (final output) ----------
export interface Report {
  runId: string;
  generatedAt: string;
  intake: IntakeInput;
  summary: string;          // 3–5 sentence executive summary, markdown allowed
  finalAllocation: Allocation;   // = last round's allocation
  rounds: Round[];          // 1 or 2 — show round 1 → 2 diff when length is 2
  findings: Finding[];
  disclaimer: string;       // always render
  preferenceNotes?: string[];    // NEW: how each preference was honoured or why it was capped, e.g. "DOGE capped at 2%: Conservative -20% drawdown limit". Render under the summary
}

// ---------- SSE events ----------
interface Base { runId: string; seq: number; ts: string }

export type RunEvent =
  | (Base & { type: "run.started"; agents: AgentId[] })
  | (Base & { type: "agent.started"; agent: AgentId; model: string; task: string })
  | (Base & { type: "agent.progress"; agent: AgentId; message: string })   // "Searching: NVDA earnings"
  | (Base & { type: "agent.finding"; finding: Finding })
  | (Base & { type: "agent.completed"; agent: AgentId; summary: string; durationMs: number })
  | (Base & { type: "agent.failed"; agent: AgentId; message: string })     // run continues without it
  | (Base & { type: "allocation.proposed"; round: 1 | 2; allocation: Allocation })
  | (Base & { type: "backtest.completed"; round: 1 | 2; backtest: Backtest })
  | (Base & { type: "verification.completed"; round: 1 | 2; verification: Verification })
  | (Base & { type: "run.completed"; report: Report })
  | (Base & { type: "run.failed"; error: APIError["error"] });
```

### Example event sequence (abridged)

```
event: run.started          data: {"agents":["DISCOVERY","FINANCIALS",...]}
event: agent.started        data: {"agent":"DISCOVERY","model":"gemini-3.8-flash","task":"Find candidates for AI infrastructure"}
event: agent.progress       data: {"agent":"DISCOVERY","message":"Searching: top AI infrastructure stocks 2026"}
event: agent.finding        data: {"finding":{"id":"f_1","agent":"DISCOVERY","symbol":"NVDA",...}}
event: agent.completed      data: {"agent":"DISCOVERY","summary":"Found 5 stocks, 3 coins","durationMs":8200}
event: agent.started        data: {"agent":"FINANCIALS",...}     ← these four interleave
event: agent.started        data: {"agent":"NEWS",...}
...
event: allocation.proposed  data: {"round":1,"allocation":{...}}
event: backtest.completed   data: {"round":1,"backtest":{...}}
event: verification.completed data: {"round":1,"verification":{"isApproved":false,"changesRequested":["Cut crypto below 30%: max drawdown -41% exceeds MODERATE limit -35%"]}}
event: allocation.proposed  data: {"round":2,...}
event: backtest.completed   data: {"round":2,...}
event: verification.completed data: {"round":2,"verification":{"isApproved":true,...}}
event: run.completed        data: {"report":{...}}
```

A full example `GET /api/runs/{runId}` response (a completed `Run` with a 2-round `Report`) lives in
[`fixtures/report.example.json`](fixtures/report.example.json). Equity curves are synthetic, but the shape is real. Build the
report page against it before the backend is live.

---

## UI notes for frontend agents

Screens (one page, top to bottom, replacing wireframes ① and ②):
1. **Onboarding chat**: a chat with the Intake agent (`POST /api/intake/messages`). Its first message is hard-coded in the frontend: *"Hi! How much are you looking to invest, and what are you hoping for?"* When `isComplete`, show the **confirm card** (amount, horizon 3 months–10 years, risk toggle, theme; all editable) with a **Run** button. Show 422 `details` inline. If the chat endpoint errors, fall back to the same card as a plain form.
2. **Agents at work**: 7 agent lanes that animate live from the stream.
3. **Results**: the two-level pie, a holdings table (symbol, %, $, scores, rationale), the round-1 → round-2 diff, the backtest chart and metrics, findings with sources, and the disclaimer.

- **Two-level pie (Recharts):** two concentric `<Pie>`s in one `<PieChart>`. Inner `<Pie data={slices} dataKey="weightPct" outerRadius="45%">`; outer `<Pie data={slices.flatMap(s => s.holdings)} dataKey="weightPct" innerRadius="50%" outerRadius="80%">`. Holdings are already ordered by slice, so they line up with their parent class. Colour holdings as shades of their class colour.
- **Agent lanes:** one card per `AgentId`, states `idle → running → done | failed` driven by `agent.*` events; show latest `agent.progress.message`; findings append live.
- **Round diff:** when `rounds.length === 2`, show round-1 sunburst greyed with the Verifier's `changesRequested`, then round 2. This is the money shot.
- **Evidence:** clicking a holding shows its `rationale` + `findingIds` → findings with source links.
- **Backtest:** line chart of `equityCurve` (4 series) + metrics table; show `crossCheck.isMatch` as a "✓ Verified" badge.
- **Stream:** native `EventSource`. Handle `onerror` by falling back to polling `GET /api/runs/{runId}` every 3 s.
- **Mock mode:** with `NEXT_PUBLIC_USE_MOCK=true`, replay the fixture's `rounds` on a timer so the agent lanes, round-1 rejection and round-2 approval all animate. The demo has to work with the backend off.
- **Personalisation is open-ended.** Users can say anything ("no tech", "only biotech", "no gold", "no Chinese companies", "dividend payers"). The backend maps their words onto Yahoo's sectors, industries, countries and asset classes, and puts how it understood each phrase in `preferenceNotes` (e.g. "'bio stocks' → industry Biotechnology"). Show these notes prominently; that's how the user sees they were understood.
- **Personalisation:** the confirm card shows `preferences` as editable chips (must include / exclude / styles / class targets / notes). The results page tags holdings with `isUserRequested`, and lists `preferenceNotes` under the summary, so the user sees their wishes were heard even when risk limits capped them.
- **Confirm card stays once shown:** `isComplete` can briefly go back to `false` while the chat asks an optional follow-up (e.g. a risk trade-off or "should bonds still be included?"). Once the card has appeared, keep it visible and editable; the user can press Run at any time. The reply ends with a **"Saved: …"** line generated from the stored preferences, which is the source of truth for the chips.
- **Retry buttons (frontend-only, no new endpoint):**
  - On `run.failed`, a stream error or a 5xx, show **Try again**, which re-sends `POST /api/runs` with the same intake. On 429, wait 5 seconds and retry.
  - On the results page, show **Run again** (same intake) and **Edit & run again** (back to the confirm card).
  - The Orchestrator→Verifier redo inside a run is automatic (max 2 rounds). The buttons restart the whole run, research included.
- All numbers arrive pre-rounded (1 decimal for %, 2 for sharpe). Don't recompute weights client-side.
