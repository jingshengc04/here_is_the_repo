// API contract types, copied verbatim from docs/API_CONTRACT.md (source of truth). Do not rename fields.
// Regenerate by re-copying the "Types" and "Errors" code blocks if the contract changes.

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

// ---------- Errors ----------
export interface APIError {
  error: { code: ErrorCode; message: string; details?: unknown };
}
export type ErrorCode = "VALIDATION_ERROR" | "RUN_NOT_FOUND" | "RATE_LIMITED" | "INTERNAL";
