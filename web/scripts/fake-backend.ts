// Contract-compliant fake backend (docs/API_CONTRACT.md) that replays the fixture as timed SSE events.
// Lets the frontend run against real HTTP + EventSource before the Python backend is live, and is a
// reference for what the backend should send.
//
//   node scripts/fake-backend.ts                 # http://localhost:8787
//   PORT=8000 SPEED=2 node scripts/fake-backend.ts
//   then open the app with ?mock=0&backend=http://localhost:8787
//
// Dev-only scenarios: POST /api/runs?scenario=agent_failed | run_failed | unverified_number | no_sandbox | personalised | not_approved
// Replay like the real backend: POST /api/runs?replay=demo (any other name → 404)
import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { adaptReport, buildMockEvents, type TimedEvent } from "../lib/mock-events.ts";
import { mockIntakeTurn, validateIntake, validateMessages } from "../lib/mock-intake.ts";
import type { IntakeInput, Report, Run, RunEvent } from "../lib/types.ts";

const PORT = Number(process.env.PORT ?? 8787);
const SPEED = Number(process.env.SPEED ?? 1);
const fixture = JSON.parse(readFileSync(new URL("../lib/mocks/report.example.json", import.meta.url), "utf8")) as Run;

interface Stored {
  run: Run;
  report: Report;
  events: TimedEvent[];
  startedAt: number;
}
const runs = new Map<string, Stored>();

const elapsed = (s: Stored) => Date.now() - s.startedAt;
const finished = (s: Stored) => elapsed(s) >= s.events[s.events.length - 1].at;
const terminal = (s: Stored) => s.events[s.events.length - 1].event;

function snapshot(s: Stored): Run {
  if (!finished(s)) return { ...s.run, status: "RUNNING", report: null, error: null };
  const last = terminal(s);
  return last.type === "run.failed"
    ? { ...s.run, status: "FAILED", report: null, error: last.error }
    : { ...s.run, status: "COMPLETED", report: s.report, error: null };
}

function applyScenario(events: TimedEvent[], scenario: string | null): TimedEvent[] {
  if (scenario === "agent_failed") {
    return events
      .filter((e) => !(e.event.type === "agent.finding" && e.event.finding.agent === "SOCIAL"))
      .map((e) =>
        e.event.type === "agent.completed" && e.event.agent === "SOCIAL"
          ? { ...e, event: { ...e.event, type: "agent.failed", message: "Search quota exceeded — continuing without social sentiment" } as RunEvent }
          : e,
      );
  }
  if (scenario === "run_failed") {
    const cut = events.findIndex((e) => e.event.type === "backtest.completed");
    const head = events.slice(0, cut);
    const last = head[head.length - 1];
    return [
      ...head,
      {
        at: last.at + 800,
        event: { type: "run.failed", runId: last.event.runId, seq: last.event.seq + 1, ts: "", error: { code: "INTERNAL", message: "Gemini sandbox timed out after 30s. Please try again." } },
      },
    ];
  }
  return events;
}

// ---------------- HTTP helpers ----------------

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Last-Event-ID",
};

function send(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { ...CORS, "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}
const fail = (res: ServerResponse, status: number, code: string, message: string, details?: unknown) =>
  send(res, status, { error: { code, message, ...(details ? { details } : {}) } });

async function readJson(req: IncomingMessage): Promise<unknown> {
  let raw = "";
  for await (const chunk of req) raw += chunk;
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
}

function stream(req: IncomingMessage, res: ServerResponse, s: Stored) {
  res.writeHead(200, {
    ...CORS,
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });
  // Resume after `Last-Event-ID: {runId}:{seq}`.
  const lastId = req.headers["last-event-id"];
  const after = typeof lastId === "string" ? Number(lastId.split(":").pop()) : -1;
  const timers: ReturnType<typeof setTimeout>[] = [];
  const write = ({ at, event: e }: TimedEvent) => {
    const event = { ...e, ts: new Date(s.startedAt + at).toISOString() };
    res.write(`event: ${e.type}\nid: ${e.runId}:${e.seq}\ndata: ${JSON.stringify(event)}\n\n`);
    if (e.type === "run.completed" || e.type === "run.failed") {
      timers.forEach(clearTimeout);
      clearInterval(heartbeat);
      res.end();
    }
  };
  const heartbeat = setInterval(() => res.write(": keep-alive\n\n"), 15_000);
  if (terminal(s).seq <= after) {
    // Client already has the terminal event: nothing left to replay.
    clearInterval(heartbeat);
    return res.end();
  }
  const now = elapsed(s);
  for (const t of s.events) {
    if (t.event.seq <= after) continue;
    if (t.at <= now) write(t);
    else timers.push(setTimeout(() => write(t), t.at - now));
  }
  req.on("close", () => {
    timers.forEach(clearTimeout);
    clearInterval(heartbeat);
  });
}

// ---------------- routes ----------------

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
  const path = url.pathname;
  console.log(req.method, url.pathname + url.search);

  if (req.method === "OPTIONS") {
    res.writeHead(204, CORS);
    return res.end();
  }
  if (req.method === "GET" && path === "/api/health") return send(res, 200, { status: "ok" });

  if (req.method === "POST" && path === "/api/intake/messages") {
    const body = (await readJson(req)) as { messages?: unknown } | undefined;
    const err = validateMessages(body?.messages);
    if (err) return fail(res, 422, "VALIDATION_ERROR", err, { messages: err });
    await new Promise((r) => setTimeout(r, 500));
    return send(res, 200, mockIntakeTurn(body!.messages as never));
  }

  if (req.method === "POST" && path === "/api/runs") {
    const intake = (await readJson(req)) as IntakeInput | undefined;
    const errors = validateIntake(intake ?? {});
    if (!intake || Object.keys(errors).length) return fail(res, 422, "VALIDATION_ERROR", "Invalid intake", errors);
    const replay = url.searchParams.get("replay");
    if (replay && replay !== "demo") return fail(res, 404, "RUN_NOT_FOUND", `Replay '${replay}' not found`);
    if ([...runs.values()].some((s) => !finished(s))) return fail(res, 429, "RATE_LIMITED", "A run is already in progress");
    const runId = `run_${Date.now().toString(36)}`;
    const scenario = url.searchParams.get("scenario");
    let report = adaptReport(fixture.report as Report, intake, runId);
    if (scenario === "personalised") {
      // GOLD 25 → GLD 15 + BONDS (BND 10); NVDA marked as user-requested; preference notes like the real backend's.
      const tweak = (a: Report["finalAllocation"]): Report["finalAllocation"] => ({
        slices: [
          ...a.slices
            .filter((s) => s.assetClass !== "GOLD")
            .map((s) => ({ ...s, holdings: s.holdings.map((h) => (h.symbol === "NVDA" ? { ...h, isUserRequested: true } : h)) })),
          { assetClass: "GOLD", weightPct: 15, rationale: "Drawdown hedge.", holdings: [{ ...a.slices.find((s) => s.assetClass === "GOLD")!.holdings[0], weightPct: 15, amountUsd: Math.round(intake.amountUsd * 0.15) }] },
          {
            assetClass: "BONDS",
            weightPct: 10,
            rationale: "Low-drawdown ballast.",
            holdings: [{ symbol: "BND", name: "Vanguard Total Bond Market ETF", weightPct: 10, amountUsd: Math.round(intake.amountUsd * 0.1), rationale: "Lowest 5-year drawdown in the bond pool.", findingIds: [], financialScore: null, sentimentScore: null }],
          },
        ],
      });
      report = {
        ...report,
        finalAllocation: tweak(report.finalAllocation),
        rounds: report.rounds.map((r) => ({ ...r, allocation: tweak(r.allocation) })),
        preferenceNotes: ["NVDA kept at 14% as you asked", "'no tobacco' → excluded industry Tobacco"],
      };
    }
    if (scenario === "not_approved") {
      // Both rounds fail the drawdown check; reasons ride first in preferenceNotes like the real backend.
      const fail = (v: Report["rounds"][number]["verification"]) => ({
        ...v,
        isApproved: false,
        checks: v.checks.map((c, i) => (i === 0 ? { ...c, isPassed: false, detail: "Portfolio max drawdown -38.2%" } : c)),
        changesRequested: v.changesRequested.length ? v.changesRequested : ["Cut crypto further"],
      });
      report = {
        ...report,
        rounds: report.rounds.map((r) => ({ ...r, verification: fail(r.verification) })),
        preferenceNotes: ["Even after rebalancing, this mix fell 38.2% at its worst in the last 5 years, beyond the 35% your moderate profile allows. Consider a higher risk level or fewer volatile assets."],
      };
    }
    if (scenario === "no_sandbox") {
      const unchecked = (b: Report["rounds"][number]["backtest"]) => ({
        ...b,
        crossCheck: { sandboxTotalReturnPct: b.crossCheck.independentTotalReturnPct, independentTotalReturnPct: b.crossCheck.independentTotalReturnPct, isMatch: false },
      });
      report = { ...report, rounds: report.rounds.map((r) => ({ ...r, backtest: unchecked(r.backtest) })) };
    }
    if (scenario === "unverified_number") {
      // First FINANCIALS claim fails the SEC/yfinance number audit.
      const bad = report.findings.find((f) => f.agent === "FINANCIALS")?.id;
      report = { ...report, findings: report.findings.map((f) => (f.id === bad ? { ...f, numbersVerified: false } : f)) };
    }
    const events = applyScenario(buildMockEvents(report, SPEED), scenario);
    const run: Run = { runId, status: "QUEUED", intake, createdAt: new Date().toISOString(), report: null, error: null };
    runs.set(runId, { run, report, events, startedAt: Date.now() });
    return send(res, 202, run);
  }

  const m = path.match(/^\/api\/runs\/([^/]+)(\/events)?$/);
  if (req.method === "GET" && m) {
    const s = runs.get(decodeURIComponent(m[1]));
    if (!s) return fail(res, 404, "RUN_NOT_FOUND", `Unknown runId ${m[1]}`);
    return m[2] ? stream(req, res, s) : send(res, 200, snapshot(s));
  }

  fail(res, 404, "RUN_NOT_FOUND", `No route ${req.method} ${path}`);
});

server.listen(PORT, () => console.log(`fake backend on http://localhost:${PORT} (speed ×${SPEED})`));
