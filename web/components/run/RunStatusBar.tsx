"use client";

import { useEffect, useState } from "react";
import type { StreamMode } from "@/lib/api";
import type { RunState } from "@/lib/run-state";

export function RunStatusBar({ state, mode }: { state: RunState; mode: StreamMode }) {
  const [now, setNow] = useState(() => Date.now());
  const running = state.status === "RUNNING";
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);

  const secs = state.startedAt ? Math.max(0, Math.round((now - state.startedAt) / 1000)) : 0;
  const round = Math.max(1, state.attempts.ORCHESTRATOR?.length ?? 0, state.rounds.filter(Boolean).length);

  return (
    <div data-testid="run-status" data-status={state.status} className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
      <span className="font-semibold">{state.status.charAt(0) + state.status.slice(1).toLowerCase()}</span>
      {state.agents.length > 0 && (
        <span>
          Round <b>{round}</b> / 2
        </span>
      )}
      {running && <span className="font-mono text-muted-foreground tabular-nums">{`${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`}</span>}
      {running && mode === "live" && <span className="font-semibold text-edge">● Live</span>}
      {running && mode === "polling" && <span className="font-semibold text-inconclusive">● Reconnecting · polling every 3s</span>}
    </div>
  );
}
