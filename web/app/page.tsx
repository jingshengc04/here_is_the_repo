"use client";

import { useEffect, useReducer, useRef, useState, useSyncExternalStore } from "react";
import { toast } from "sonner";
import { ConfirmCard } from "@/components/intake/ConfirmCard";
import { GREETING, IntakeChat } from "@/components/intake/IntakeChat";
import { ReportView } from "@/components/report/ReportView";
import { AgentLanes } from "@/components/run/AgentLanes";
import { FindingsFeed } from "@/components/run/FindingsFeed";
import { RunStatusBar } from "@/components/run/RunStatusBar";
import { Button } from "@/components/ui/button";
import { ApiError, createRun, getRun, sendIntakeMessages, watchRun, type StreamMode } from "@/lib/api";
import { backendUrl, isMock, replayName } from "@/lib/config";
import { horizonLabel, usd } from "@/lib/format";
import { completedRunState, initialRunState, reduceRun, type RunState } from "@/lib/run-state";
import type { ChatMessage, IntakeInput, IntakeTurn, RunEvent } from "@/lib/types";

type Action = { type: "event"; event: RunEvent } | { type: "reset" } | { type: "load"; state: RunState };
const runReducer = (s: RunState, a: Action): RunState =>
  a.type === "event" ? reduceRun(s, a.event) : a.type === "load" ? a.state : initialRunState();

const noopSubscribe = () => () => {};
const RISK = { CONSERVATIVE: "Conservative", MODERATE: "Moderate", AGGRESSIVE: "Aggressive" } as const;

function setRunParam(runId: string | null) {
  const url = new URL(window.location.href);
  if (runId) url.searchParams.set("run", runId);
  else url.searchParams.delete("run");
  window.history.replaceState(null, "", url);
}

export default function Home() {
  // Onboarding
  const [messages, setMessages] = useState<ChatMessage[]>([{ role: "assistant", content: GREETING }]);
  const [draft, setDraft] = useState<IntakeTurn["intake"] | null>(null);
  const [complete, setComplete] = useState(false);
  const [chatPending, setChatPending] = useState(false);
  const [formOnly, setFormOnly] = useState(false);
  const [chatFailed, setChatFailed] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  // Run
  const [intake, setIntake] = useState<IntakeInput | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [run, dispatch] = useReducer(runReducer, undefined, initialRunState);
  const [mode, setMode] = useState<StreamMode>("closed");
  const stopRef = useRef<(() => void) | null>(null);
  const topRef = useRef<HTMLDivElement>(null);

  const modeLabel = useSyncExternalStore(
    noopSubscribe,
    () => (isMock() ? "Mock data" : `Backend: ${backendUrl()}${replayName() ? ` · replay ${replayName()}` : ""}`),
    () => null,
  );

  function watch(id: string) {
    stopRef.current?.();
    stopRef.current = watchRun(id, (event) => dispatch({ type: "event", event }), setMode);
  }

  // Deep link / refresh: ?run=<id> → snapshot, then stream if still running.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("run");
    if (!id) return;
    let cancelled = false;
    getRun(id)
      .then((r) => {
        if (cancelled) return;
        setRunId(r.runId);
        setIntake(r.intake);
        if (r.status === "COMPLETED" && r.report) dispatch({ type: "load", state: completedRunState(r.report) });
        else if (r.status === "FAILED") dispatch({ type: "load", state: { ...initialRunState(), status: "FAILED", error: r.error } });
        else watch(r.runId);
      })
      .catch((e: Error) => {
        if (cancelled) return;
        toast.error(e.message);
        setRunParam(null);
      });
    return () => {
      cancelled = true;
      stopRef.current?.();
    };
  }, []);

  useEffect(() => {
    if (run.status === "COMPLETED") topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [run.status]);

  async function onSend(text: string) {
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(history);
    setChatPending(true);
    try {
      // The greeting is hard-coded in the frontend, so it is not part of the history the agent sees.
      const turn = await sendIntakeMessages(history.slice(1));
      setMessages([...history, { role: "assistant", content: turn.reply }]);
      // The backend may ask an optional follow-up (isComplete flips back to false, required fields may come back null).
      // Keep what we already know, take the latest preferences ("Saved:" line is their source of truth),
      // and never hide the confirm card once it has appeared.
      setDraft((prev) => ({
        amountUsd: turn.intake.amountUsd ?? prev?.amountUsd ?? null,
        horizonYears: turn.intake.horizonYears ?? prev?.horizonYears ?? null,
        riskTolerance: turn.intake.riskTolerance ?? prev?.riskTolerance ?? null,
        theme: turn.intake.theme ?? prev?.theme ?? null,
        preferences: turn.intake.preferences ?? null,
      }));
      if (turn.isComplete) setComplete(true);
      setChatFailed(false);
    } catch (e) {
      toast.error(`Chat unavailable (${(e as Error).message}). Fill in the form instead.`);
      setMessages(messages); // drop the unanswered message so the chat is clean if the user comes back
      setChatFailed(true);
      setFormOnly(true);
    } finally {
      setChatPending(false);
    }
  }

  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const RETRY_MS = 5000;
  const MAX_RETRIES = 24; // ~2 minutes of waiting on 429

  async function onRun(input: IntakeInput, attempt = 0) {
    if (retryRef.current) clearTimeout(retryRef.current);
    setSubmitting(true);
    setFieldErrors({});
    try {
      const r = await createRun(input);
      toast.dismiss("rate-limited");
      setIntake(input);
      setRunId(r.runId);
      setRunParam(r.runId);
      dispatch({ type: "reset" });
      watch(r.runId);
      window.scrollTo({ top: 0, behavior: "smooth" });
      setSubmitting(false);
    } catch (e) {
      // 429: another run holds the backend; wait 5 s and retry the same intake (contract UI notes).
      if (e instanceof ApiError && e.code === "RATE_LIMITED" && attempt < MAX_RETRIES) {
        toast.loading("Another run is in progress. Retrying in 5 seconds…", { id: "rate-limited" });
        retryRef.current = setTimeout(() => onRun(input, attempt + 1), RETRY_MS);
        return;
      }
      toast.dismiss("rate-limited");
      setSubmitting(false);
      if (e instanceof ApiError && e.code === "VALIDATION_ERROR") {
        setFieldErrors(e.fieldErrors);
        toast.error(e.message);
      } else {
        toast.error((e as Error).message, { action: { label: "Try again", onClick: () => onRun(input) } });
      }
    }
  }

  /** Back to the confirm card with the last intake filled in. */
  function editAndRunAgain() {
    if (intake) {
      setDraft({
        amountUsd: intake.amountUsd,
        horizonYears: intake.horizonYears,
        riskTolerance: intake.riskTolerance,
        theme: intake.theme ?? null,
        preferences: intake.preferences ?? null,
      });
      setFormOnly(true);
    }
    newPortfolio();
  }

  function newPortfolio() {
    if (retryRef.current) clearTimeout(retryRef.current);
    toast.dismiss("rate-limited");
    stopRef.current?.();
    setRunId(null);
    setRunParam(null);
    dispatch({ type: "reset" });
  }

  const badge = modeLabel && (
    <span data-testid="mode-badge" className="max-w-full truncate rounded-full bg-neutral-900 px-2.5 py-0.5 text-xs font-semibold text-white">
      {modeLabel}
    </span>
  );

  // ---------------- Onboarding ----------------
  if (!runId) {
    return (
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h1 className="text-2xl font-extrabold tracking-tight">Build a portfolio with a team of AI agents</h1>
          {badge}
        </div>
        {formOnly ? (
          <div className="mx-auto max-w-xl space-y-2">
            <button
              type="button"
              onClick={() => setFormOnly(false)}
              className="text-sm font-semibold text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
            >
              ← {chatFailed ? "Try the chat again" : "Back to chat"}
            </button>
            <ConfirmCard initial={draft} errors={fieldErrors} submitting={submitting} onRun={onRun} title="Your plan" />
          </div>
        ) : (
          <div className="grid gap-5 lg:grid-cols-[1.3fr_1fr]">
            <IntakeChat messages={messages} draft={draft} pending={chatPending} onSend={onSend} onSkip={() => setFormOnly(true)} />
            {complete ? (
              <ConfirmCard key={JSON.stringify(draft)} initial={draft} errors={fieldErrors} submitting={submitting} onRun={onRun} />
            ) : (
              <div className="flex flex-col justify-center rounded-xl border-2 border-dashed bg-card/60 p-6 text-center text-sm text-muted-foreground">
                <p className="text-base font-semibold text-foreground">Your plan appears here</p>
                <p className="mt-1">once the agent knows your amount, horizon and risk tolerance.</p>
                <p className="mt-4">The agents never start on their own: you confirm and press Run.</p>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // ---------------- Run / report ----------------
  const done = run.status === "COMPLETED" && run.report;
  return (
    <div ref={topRef} className="scroll-mt-20 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b pb-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {intake && (
            <span className="font-semibold">
              {usd(intake.amountUsd)}, {horizonLabel(intake.horizonYears)}, {RISK[intake.riskTolerance]}
              {intake.theme ? `, ${intake.theme}` : ""}
            </span>
          )}
          <RunStatusBar state={run} mode={mode} />
        </div>
        <div className="flex items-center gap-2">
          {badge}
          <Button variant="outline" onClick={newPortfolio}>
            New portfolio
          </Button>
        </div>
      </div>

      {run.status === "FAILED" && (
        <div role="alert" data-testid="run-failed" className="rounded-xl border-2 border-no-edge bg-no-edge-bg p-4">
          <b className="text-no-edge">Run failed.</b> {run.error?.message ?? "Unknown error."}
          <div className="mt-3 flex flex-wrap gap-2">
            <Button onClick={() => intake && onRun(intake)} disabled={!intake || submitting}>
              Try again
            </Button>
            <Button variant="outline" onClick={editAndRunAgain}>
              Edit and run again
            </Button>
          </div>
        </div>
      )}

      {done && (
        <ReportView
          report={run.report!}
          onRunAgain={intake ? () => onRun(intake) : undefined}
          onEdit={intake ? editAndRunAgain : undefined}
          activity={
            run.agents.length > 0 ? (
              <div className="grid gap-4 lg:grid-cols-[1.75fr_1fr]">
                <AgentLanes state={run} />
                <FindingsFeed findings={run.findings} live={false} />
              </div>
            ) : undefined
          }
        />
      )}

      {!done && run.status !== "FAILED" && (
        <div className="grid gap-4 lg:grid-cols-[1.75fr_1fr]">
          <AgentLanes state={run} />
          <FindingsFeed findings={run.findings} live={run.status === "RUNNING"} />
        </div>
      )}
    </div>
  );
}
