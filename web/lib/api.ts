import { backendUrl, isMock, replayName } from "./config";
import { adaptReport, buildMockEvents } from "./mock-events";
import { mockIntakeTurn, validateIntake, validateMessages } from "./mock-intake";
import fixture from "./mocks/report.example.json";
import type { ChatMessage, ErrorCode, IntakeInput, IntakeTurn, Report, Run, RunEvent } from "./types";

const fixtureRun = fixture as unknown as Run;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class ApiError extends Error {
  code: ErrorCode;
  status: number;
  details?: unknown;
  constructor(status: number, code: ErrorCode, message: string, details?: unknown) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
  /** 422 `details` as a {field: message} map for inline form errors. */
  get fieldErrors(): Record<string, string> {
    return this.code === "VALIDATION_ERROR" && this.details && typeof this.details === "object"
      ? (this.details as Record<string, string>)
      : {};
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const base = backendUrl();
  let res: Response;
  try {
    res = await fetch(`${base}${path}`, {
      ...init,
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    });
  } catch {
    throw new ApiError(0, "INTERNAL", `Can't reach the backend at ${base}. Try mock mode (?mock=1).`);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const e = body?.error;
    // Non-contract bodies (e.g. FastAPI's default {"detail": "..."}) still get a readable message.
    const fallback = typeof body?.detail === "string" ? body.detail : `Backend error ${res.status}`;
    throw new ApiError(res.status, e?.code ?? (res.status === 404 ? "RUN_NOT_FOUND" : "INTERNAL"), e?.message ?? fallback, e?.details);
  }
  return body as T;
}

// ---------------- mock backend (fixture replay) ----------------

interface MockRun {
  run: Run;
  events: { at: number; event: RunEvent }[];
  startedAt: number;
}
const mockRuns = new Map<string, MockRun>();

function mockSnapshot(m: MockRun): Run {
  const elapsed = Date.now() - m.startedAt;
  const done = elapsed >= m.events[m.events.length - 1].at;
  return { ...m.run, status: done ? "COMPLETED" : "RUNNING", report: done ? m.run.report : null };
}

// ---------------- endpoints ----------------

export async function sendIntakeMessages(messages: ChatMessage[]): Promise<IntakeTurn> {
  if (isMock()) {
    const err = validateMessages(messages);
    if (err) throw new ApiError(422, "VALIDATION_ERROR", err, { messages: err });
    await sleep(700);
    return mockIntakeTurn(messages);
  }
  return request<IntakeTurn>("/api/intake/messages", { method: "POST", body: JSON.stringify({ messages }) });
}

export async function createRun(intake: IntakeInput): Promise<Run> {
  if (isMock()) {
    const errors = validateIntake(intake);
    if (Object.keys(errors).length) throw new ApiError(422, "VALIDATION_ERROR", "Please fix the highlighted fields.", errors);
    await sleep(300);
    const runId = `mock_${Date.now().toString(36)}`;
    const report = adaptReport(fixtureRun.report as Report, intake, runId);
    const run: Run = { runId, status: "QUEUED", intake, createdAt: new Date().toISOString(), report, error: null };
    mockRuns.set(runId, { run, events: buildMockEvents(report), startedAt: Date.now() });
    return { ...run, report: null };
  }
  const replay = replayName();
  const qs = replay ? `?replay=${encodeURIComponent(replay)}` : "";
  return request<Run>(`/api/runs${qs}`, { method: "POST", body: JSON.stringify(intake) });
}

export async function getRun(runId: string): Promise<Run> {
  if (isMock()) {
    const m = mockRuns.get(runId);
    if (m) return mockSnapshot(m);
    // Deep link / refresh in mock mode: serve the fixture as a finished run.
    return { ...fixtureRun, runId, report: adaptReport(fixtureRun.report as Report, fixtureRun.intake, runId) };
  }
  return request<Run>(`/api/runs/${encodeURIComponent(runId)}`);
}

export type StreamMode = "live" | "polling" | "closed";

const EVENT_TYPES: RunEvent["type"][] = [
  "run.started",
  "agent.started",
  "agent.progress",
  "agent.finding",
  "agent.completed",
  "agent.failed",
  "allocation.proposed",
  "backtest.completed",
  "verification.completed",
  "run.completed",
  "run.failed",
];
const POLL_MS = 3000;

/**
 * Streams a run's events. Live: native EventSource (one listener per event type).
 * On stream error before the run ends: fall back to polling GET /api/runs/{id} every 3 s.
 * Events can arrive twice after a reconnect replay; the reducer drops seq it has already seen.
 * Returns a stop function.
 */
export function watchRun(runId: string, onEvent: (e: RunEvent) => void, onMode: (m: StreamMode) => void): () => void {
  let stopped = false;
  let finished = false;
  const timers: ReturnType<typeof setTimeout>[] = [];
  const emit = (e: RunEvent) => {
    if (stopped) return;
    if (e.type === "run.completed" || e.type === "run.failed") {
      finished = true;
      onMode("closed");
    }
    onEvent(e);
  };

  if (isMock()) {
    onMode("live");
    const m = mockRuns.get(runId);
    if (!m) {
      getRun(runId).then((run) => run.report && emit({ type: "run.completed", report: run.report, runId, seq: 0, ts: run.createdAt }));
      return () => void (stopped = true);
    }
    const elapsed = Date.now() - m.startedAt;
    for (const { at, event } of m.events) {
      timers.push(setTimeout(() => emit({ ...event, ts: new Date().toISOString() }), Math.max(0, at - elapsed)));
    }
    return () => {
      stopped = true;
      timers.forEach(clearTimeout);
    };
  }

  const poll = async () => {
    if (stopped || finished) return;
    try {
      const run = await getRun(runId);
      if (run.status === "COMPLETED" && run.report) return emit({ type: "run.completed", report: run.report, runId, seq: Number.MAX_SAFE_INTEGER, ts: new Date().toISOString() });
      if (run.status === "FAILED" && run.error) return emit({ type: "run.failed", error: run.error, runId, seq: Number.MAX_SAFE_INTEGER, ts: new Date().toISOString() });
    } catch {
      // keep polling; a transient backend blip should not end the run view
    }
    timers.push(setTimeout(poll, POLL_MS));
  };

  const es = new EventSource(`${backendUrl()}/api/runs/${encodeURIComponent(runId)}/events`);
  onMode("live");
  for (const type of EVENT_TYPES) {
    es.addEventListener(type, (msg) => {
      try {
        emit(JSON.parse((msg as MessageEvent).data) as RunEvent);
      } catch {
        // ignore malformed event
      }
      if (finished) es.close();
    });
  }
  es.onerror = () => {
    es.close();
    if (stopped || finished) return;
    onMode("polling");
    poll();
  };

  return () => {
    stopped = true;
    es.close();
    timers.forEach(clearTimeout);
  };
}
