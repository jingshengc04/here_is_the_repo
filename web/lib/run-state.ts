import type { AgentId, Allocation, APIError, Backtest, Finding, Report, RunEvent, Verification } from "./types";

export interface AgentAttempt {
  state: "running" | "done" | "failed";
  model: string;
  task: string;
  message: string | null; // latest agent.progress (or failure message)
  summary: string | null;
  durationMs: number | null;
}

export interface PartialRound {
  allocation?: Allocation;
  backtest?: Backtest;
  verification?: Verification;
}

export interface RunState {
  status: "RUNNING" | "COMPLETED" | "FAILED";
  agents: AgentId[];
  /** One attempt per agent.started; ORCHESTRATOR / VERIFIER get one per round. */
  attempts: Partial<Record<AgentId, AgentAttempt[]>>;
  findings: Finding[];
  rounds: PartialRound[]; // index = round - 1
  report: Report | null;
  error: APIError["error"] | null;
  seen: number[]; // seq already applied (SSE replays after reconnect)
  startedAt: number; // from run.started ts; 0 until then
}

export const initialRunState = (): RunState => ({
  status: "RUNNING",
  agents: [],
  attempts: {},
  findings: [],
  rounds: [],
  report: null,
  error: null,
  seen: [],
  startedAt: 0,
});

function updateLast(s: RunState, agent: AgentId, patch: Partial<AgentAttempt>): RunState["attempts"] {
  const list = s.attempts[agent] ?? [];
  if (!list.length) return s.attempts;
  return { ...s.attempts, [agent]: [...list.slice(0, -1), { ...list[list.length - 1], ...patch }] };
}

function setRound(s: RunState, round: number, patch: PartialRound): PartialRound[] {
  const rounds = [...s.rounds];
  rounds[round - 1] = { ...rounds[round - 1], ...patch };
  return rounds;
}

export function reduceRun(s: RunState, e: RunEvent): RunState {
  if (s.seen.includes(e.seq)) return s;
  const next = { ...s, seen: [...s.seen, e.seq] };
  switch (e.type) {
    case "run.started":
      return { ...next, agents: e.agents, startedAt: Date.parse(e.ts) || 0 };
    case "agent.started": {
      const attempt: AgentAttempt = { state: "running", model: e.model, task: e.task, message: null, summary: null, durationMs: null };
      return { ...next, attempts: { ...s.attempts, [e.agent]: [...(s.attempts[e.agent] ?? []), attempt] } };
    }
    case "agent.progress":
      return { ...next, attempts: updateLast(s, e.agent, { message: e.message }) };
    case "agent.finding":
      return s.findings.some((f) => f.id === e.finding.id) ? next : { ...next, findings: [...s.findings, e.finding] };
    case "agent.completed":
      return { ...next, attempts: updateLast(s, e.agent, { state: "done", summary: e.summary, durationMs: e.durationMs }) };
    case "agent.failed":
      return { ...next, attempts: updateLast(s, e.agent, { state: "failed", message: e.message }) };
    case "allocation.proposed":
      return { ...next, rounds: setRound(s, e.round, { allocation: e.allocation }) };
    case "backtest.completed":
      return { ...next, rounds: setRound(s, e.round, { backtest: e.backtest }) };
    case "verification.completed":
      return { ...next, rounds: setRound(s, e.round, { verification: e.verification }) };
    case "run.completed":
      return { ...next, status: "COMPLETED", report: e.report, findings: e.report.findings, rounds: e.report.rounds };
    case "run.failed":
      return { ...next, status: "FAILED", error: e.error };
  }
}

/** State for a run that was already finished when we opened it (deep link / refresh). */
export function completedRunState(report: Report): RunState {
  return { ...initialRunState(), status: "COMPLETED", report, findings: report.findings, rounds: report.rounds };
}
