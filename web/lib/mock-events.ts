// Turns a finished Report (docs/fixtures/report.example.json) into a timed RunEvent sequence,
// so mock mode can animate the agent lanes, the round-1 rejection and the round-2 approval.
// Pure and type-only imports: also used by scripts/fake-backend.ts under Node type stripping.
import type { AgentId, Finding, IntakeInput, Report, RunEvent } from "./types";

export const AGENTS: AgentId[] = ["DISCOVERY", "FINANCIALS", "LANDSCAPE", "NEWS", "SOCIAL", "ORCHESTRATOR", "VERIFIER"];
export const AGENT_MODEL: Record<AgentId, string> = {
  DISCOVERY: "gemini-3.8-flash",
  FINANCIALS: "gemini-3.8-flash",
  LANDSCAPE: "gemini-3.8-flash",
  NEWS: "gemini-3.8-flash",
  SOCIAL: "gemini-3.8-flash",
  ORCHESTRATOR: "gemini-4-argon",
  VERIFIER: "gemini-4-argon",
};

/** Fixture report re-labelled for this run: new runId, the user's intake, amounts scaled to their amount. */
export function adaptReport(report: Report, intake: IntakeInput, runId: string, now = new Date()): Report {
  const scale = <T extends { slices: Report["finalAllocation"]["slices"] }>(a: T): T => ({
    ...a,
    slices: a.slices.map((s) => ({
      ...s,
      holdings: s.holdings.map((h) => ({ ...h, amountUsd: Math.round((h.weightPct * intake.amountUsd) / 100) })),
    })),
  });
  // Fixture predates Finding.numbersVerified: assume the backend's audit passed every claim that has a number.
  const findings = report.findings.map((f) =>
    f.numbersVerified !== undefined ? f : { ...f, numbersVerified: /\d/.test(f.claim) ? true : null },
  );
  // Mock stand-in for the backend's preferenceNotes: one plain line per preference.
  const p = intake.preferences;
  const preferenceNotes = p
    ? [
        ...(p.mustInclude ?? []).map((s) => `${s} requested: kept if it fits your risk limit`),
        ...(p.exclude ?? []).map((w) => `"${w}" excluded from every stage of the research`),
        ...(p.onlyInclude ?? []).map((w) => `Research limited to "${w}"`),
        ...(p.styles ?? []).map((s) => `${s.charAt(0)}${s.slice(1).toLowerCase()} style favoured when ranking candidates`),
      ]
    : report.preferenceNotes;
  return {
    ...report,
    findings,
    preferenceNotes,
    runId,
    generatedAt: now.toISOString(),
    intake,
    finalAllocation: scale(report.finalAllocation),
    rounds: report.rounds.map((r) => ({ ...r, allocation: scale(r.allocation) })),
  };
}

type Draft = { at: number; event: DistributiveOmit<RunEvent, "runId" | "seq" | "ts"> };
type DistributiveOmit<T, K extends keyof never> = T extends unknown ? Omit<T, K> : never;
export interface TimedEvent {
  at: number; // ms after run start
  event: RunEvent;
}

const TASK: Partial<Record<AgentId, (theme: string) => string>> = {
  DISCOVERY: (t) => `Find stock and crypto candidates for ${t}`,
  FINANCIALS: () => "Score each stock 1–10 from its financial statements",
  LANDSCAPE: () => "Map competitors and market position",
  NEWS: () => "Collect recent news per holding",
  SOCIAL: () => "Gauge Reddit / X sentiment",
};
const PROGRESS: Partial<Record<AgentId, (syms: string[], theme: string) => string[]>> = {
  DISCOVERY: (_, t) => [`Searching: ${t} leaders, suppliers and mid caps`, "Searching: large-cap crypto with ETF flows", "Resolving tickers in yfinance / OKX"],
  FINANCIALS: (s) => [`Loading SEC EDGAR filings for ${s.filter((x) => !x.includes("-") && x !== "GLD").slice(0, 4).join(", ")}`, "Running scoring code in sandbox", "Cross-checking every figure against SEC + yfinance (±2%)"],
  LANDSCAPE: (s) => [`Searching: ${s[0] ?? "NVDA"} competitors market share`, "Comparing moats"],
  NEWS: (s) => [`Searching: ${s[0] ?? "NVDA"} news this month`, `Searching: ${s[s.length - 1] ?? "GLD"} headlines`],
  SOCIAL: () => ["Searching: r/wallstreetbets sentiment", "Searching: X crypto sentiment"],
};

/**
 * @param speed 1 = ~22 s total. Larger is faster.
 */
export function buildMockEvents(report: Report, speed = 1): TimedEvent[] {
  const drafts: Draft[] = [];
  const add = (at: number, event: Draft["event"]) => drafts.push({ at: at / speed, event });
  const theme = report.intake.theme || "a diversified portfolio";
  const byAgent = (a: AgentId) => report.findings.filter((f) => f.agent === a);
  const symbols = report.finalAllocation.slices.flatMap((s) => s.holdings.map((h) => h.symbol));

  add(0, { type: "run.started", agents: AGENTS });

  // Discovery alone, then the four research agents interleaved.
  const research = (agent: AgentId, start: number, duration: number, summary: string) => {
    add(start, { type: "agent.started", agent, model: AGENT_MODEL[agent], task: TASK[agent]!(theme) });
    const msgs = PROGRESS[agent]!(symbols, theme);
    const findings: Finding[] = byAgent(agent);
    const steps = msgs.length + findings.length;
    let i = 1;
    for (const message of msgs) add(start + (duration * i++) / (steps + 1), { type: "agent.progress", agent, message });
    for (const finding of findings) add(start + (duration * i++) / (steps + 1), { type: "agent.finding", finding });
    add(start + duration, { type: "agent.completed", agent, summary, durationMs: Math.round(duration / speed) });
  };

  const stocks = report.finalAllocation.slices.find((s) => s.assetClass === "STOCKS")?.holdings.length ?? 0;
  const coins = report.finalAllocation.slices.find((s) => s.assetClass === "CRYPTO")?.holdings.length ?? 0;
  research("DISCOVERY", 300, 3200, `Found ${stocks} stocks, ${coins} coins + GLD`);
  research("FINANCIALS", 3700, 4200, `Scored ${stocks} stocks from statements`);
  research("LANDSCAPE", 3800, 3600, "Mapped competitors and moats");
  research("NEWS", 3900, 3300, `Checked news for ${symbols.length} holdings`);
  research("SOCIAL", 4000, 2900, "Read Reddit / X sentiment");

  let t = 8400;
  for (const round of report.rounds) {
    const r = round.round;
    const prev = report.rounds.find((x) => x.round === r - 1);
    add(t, {
      type: "agent.started",
      agent: "ORCHESTRATOR",
      model: AGENT_MODEL.ORCHESTRATOR,
      task: r === 1 ? "Build an allocation from all findings" : "Rebalance using the Verifier's requested changes",
    });
    const orchMsgs = r === 1 ? [`Weighing ${report.findings.length} findings`, "Sizing asset classes for the risk limit"] : (prev?.verification.changesRequested ?? []).map((c) => `Applying: ${c}`);
    orchMsgs.forEach((message, i) => add(t + 500 + i * 700, { type: "agent.progress", agent: "ORCHESTRATOR", message }));
    t += 600 + orchMsgs.length * 700;
    add(t, { type: "allocation.proposed", round: r, allocation: round.allocation });
    const summary = round.allocation.slices.map((s) => `${s.assetClass[0]}${s.assetClass.slice(1).toLowerCase()} ${s.weightPct}%`).join(" · ");
    add(t + 100, { type: "agent.completed", agent: "ORCHESTRATOR", summary, durationMs: Math.round((600 + orchMsgs.length * 700) / speed) });

    t += 400;
    const vStart = t;
    add(t, { type: "agent.started", agent: "VERIFIER", model: AGENT_MODEL.VERIFIER, task: `Backtest and check round ${r}` });
    add(t + 500, { type: "agent.progress", agent: "VERIFIER", message: "Running backtest in Gemini sandbox" });
    add(t + 1100, { type: "agent.progress", agent: "VERIFIER", message: "Bootstrapping Sharpe 95% CI (1,000 resamples)" });
    add(t + 1700, { type: "backtest.completed", round: r, backtest: round.backtest });
    add(t + 1900, { type: "agent.progress", agent: "VERIFIER", message: "Cross-checking sandbox vs independent recompute; auditing every number" });
    add(t + 2700, { type: "verification.completed", round: r, verification: round.verification });
    const v = round.verification;
    add(t + 2800, {
      type: "agent.completed",
      agent: "VERIFIER",
      summary: v.isApproved ? "Approved" : `Rejected: ${v.checks.filter((c) => !c.isPassed).map((c) => c.name).join("; ")}`,
      durationMs: Math.round(2800 / speed),
    });
    t = vStart + 3300;
  }
  add(t, { type: "run.completed", report });

  return drafts
    .sort((a, b) => a.at - b.at)
    .map((d, seq) => ({
      at: Math.round(d.at),
      event: { ...d.event, runId: report.runId, seq, ts: "" } as RunEvent,
    }));
}
