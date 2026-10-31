import { Loader2Icon } from "lucide-react";
import { cn } from "@/lib/utils";
import { AGENT_MODEL } from "@/lib/mock-events";
import type { AgentAttempt, RunState } from "@/lib/run-state";
import type { AgentId, Verification } from "@/lib/types";

const META: Record<AgentId, { icon: string; label: string; tool: string }> = {
  DISCOVERY: { icon: "🔎", label: "Discovery", tool: "Google Search" },
  FINANCIALS: { icon: "📊", label: "Financials", tool: "SEC EDGAR · code execution" },
  LANDSCAPE: { icon: "🗺️", label: "Landscape", tool: "Search" },
  NEWS: { icon: "📰", label: "News", tool: "Search" },
  SOCIAL: { icon: "💬", label: "Social", tool: "Search" },
  ORCHESTRATOR: { icon: "🧭", label: "Orchestrator", tool: "allocation" },
  VERIFIER: { icon: "🛡️", label: "Verifier", tool: "sandbox backtest · number audit" },
};

type CardState = "idle" | "running" | "done" | "failed" | "rejected" | "approved";

function AgentCard({
  agent,
  attempt,
  roundLabel,
  verification,
  faded,
}: {
  agent: AgentId;
  attempt?: AgentAttempt;
  roundLabel?: string;
  verification?: Verification;
  faded?: boolean;
}) {
  const m = META[agent];
  let state: CardState = attempt?.state ?? "idle";
  if (verification && attempt?.state === "done") state = verification.isApproved ? "approved" : "rejected";
  const secs = attempt?.durationMs != null ? ` · ${(attempt.durationMs / 1000).toFixed(1)}s` : "";
  const badge: Record<CardState, string> = {
    idle: roundLabel ? `Waiting · ${roundLabel}` : "Waiting",
    running: roundLabel ?? "Working",
    done: `✓${roundLabel ? ` ${roundLabel}` : ""}`,
    failed: "✕ Failed",
    rejected: `✕ Rejected · ${roundLabel}`,
    approved: `✓ Approved · ${roundLabel}`,
  };
  // With a verification the checks below say it all; skip the one-line summary.
  const line = state === "failed" ? attempt?.message : attempt?.state === "running" ? attempt.message ?? attempt.task : verification ? null : attempt?.summary;

  return (
    <div
      data-testid={`agent-${agent}`}
      data-state={state}
      className={cn(
        "relative rounded-xl border-2 bg-card p-3 text-[13px] transition-colors",
        state === "idle" && "border-dashed border-neutral-300 text-neutral-400",
        state === "running" && "border-neutral-900 shadow-[0_0_0_3px_rgba(17,17,17,0.08)]",
        (state === "done" || state === "approved") && "border-edge",
        (state === "failed" || state === "rejected") && "border-no-edge bg-[#fffafa]",
        faded && "opacity-55",
      )}
    >
      <span
        className={cn(
          "absolute top-2.5 right-2.5 rounded-full px-2 py-0.5 text-[11px] font-extrabold",
          state === "idle" && "bg-neutral-100 text-neutral-500",
          state === "running" && "bg-neutral-900 text-white",
          (state === "done" || state === "approved") && "bg-edge-bg text-edge",
          state === "failed" && "bg-no-edge-bg text-no-edge",
          state === "rejected" && "bg-no-edge text-white",
        )}
      >
        {state === "running" && <Loader2Icon className="mr-1 inline size-3 animate-spin" />}
        {badge[state]}
      </span>
      <div className="pr-20 text-[15px] font-semibold">
        {m.icon} {m.label}
      </div>
      <div className="text-[11px] text-muted-foreground">
        {attempt?.model ?? AGENT_MODEL[agent]} · {m.tool}
        {secs}
      </div>
      {line && <p className={cn("mt-1.5 leading-snug", state === "running" && "italic")}>{line}</p>}
      {verification && (
        <div className="mt-2 space-y-0.5 text-[12.5px] leading-snug">
          {verification.checks.map((c) => (
            <div key={c.name}>
              <span className={c.isPassed ? "text-edge" : "text-no-edge"}>{c.isPassed ? "✓" : "✕"}</span> {c.name}
              {!c.isPassed && <span className="text-muted-foreground"> — {c.detail}</span>}
            </div>
          ))}
          {verification.changesRequested.length > 0 && (
            <div className="mt-1.5 rounded-md bg-no-edge-bg px-2.5 py-1.5">
              <b>Changes requested</b>
              <ul className="list-disc pl-4">
                {verification.changesRequested.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const Arrow = ({ label }: { label?: string }) => (
  <div className="py-1 text-center text-lg leading-none text-neutral-300">
    ↓{label && <span className="mx-2 align-middle text-[11px] text-muted-foreground">{label}</span>}
    {label && "↓"}
  </div>
);

export function AgentLanes({ state }: { state: RunState }) {
  const a = state.attempts;
  const rounds = Math.max(1, (a.ORCHESTRATOR?.length ?? 0), (a.VERIFIER?.length ?? 0));
  const roundIdx = Array.from({ length: Math.min(2, rounds) }, (_, i) => i);
  const verifierIdle = (a.VERIFIER?.length ?? 0) < (a.ORCHESTRATOR?.length ?? 0);

  return (
    <section data-testid="agent-lanes" aria-label="Agents at work" className="rounded-xl border bg-card p-4 shadow-xs">
      <div className="mx-auto w-full max-w-md">
        <AgentCard agent="DISCOVERY" attempt={a.DISCOVERY?.[0]} />
      </div>
      <Arrow label="4 in parallel" />
      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
        {(["FINANCIALS", "LANDSCAPE", "NEWS", "SOCIAL"] as const).map((id) => (
          <AgentCard key={id} agent={id} attempt={a[id]?.[0]} />
        ))}
      </div>
      <Arrow />
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          {roundIdx.map((i) => (
            <div key={i}>
              {i > 0 && <Arrow />}
              <AgentCard
                agent="ORCHESTRATOR"
                attempt={a.ORCHESTRATOR?.[i]}
                roundLabel={`Round ${i + 1}`}
                faded={i < roundIdx.length - 1}
              />
            </div>
          ))}
        </div>
        <div>
          {roundIdx.map((i) => (
            <div key={i}>
              {i > 0 && <Arrow />}
              <AgentCard agent="VERIFIER" attempt={a.VERIFIER?.[i]} roundLabel={`Round ${i + 1}`} verification={state.rounds[i]?.verification} />
            </div>
          ))}
          {verifierIdle && roundIdx.length === (a.VERIFIER?.length ?? 0) && (
            <>
              <Arrow />
              <AgentCard agent="VERIFIER" roundLabel={`Round ${roundIdx.length + 1}`} />
            </>
          )}
        </div>
      </div>
    </section>
  );
}
