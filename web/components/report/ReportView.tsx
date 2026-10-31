"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { DRAWDOWN_LIMIT, horizonLabel, pct, usd } from "@/lib/format";
import { Md } from "@/lib/markdown";
import type { Finding, Report } from "@/lib/types";
import { GrowthChart, MetricsTable } from "./Backtest";
import { Comparisons } from "./Comparisons";
import { AllocationDonut } from "./AllocationDonut";
import { Fold } from "./Fold";
import { FindingItem, HoldingsList } from "./Holdings";
import { VerifierDetails } from "./VerifierDetails";

const RISK = { CONSERVATIVE: "conservative", MODERATE: "moderate", AGGRESSIVE: "aggressive" } as const;
const AGENT_ORDER: Finding["agent"][] = ["DISCOVERY", "FINANCIALS", "LANDSCAPE", "NEWS", "SOCIAL"];
const AGENT_NAME: Record<string, string> = { DISCOVERY: "Discovery", FINANCIALS: "Financials", LANDSCAPE: "Landscape", NEWS: "News", SOCIAL: "Social" };

type FoldId = "verifier" | "metrics" | "findings" | "activity";

export function ReportView({
  report,
  activity,
  onRunAgain,
  onEdit,
}: {
  report: Report;
  activity?: React.ReactNode;
  onRunAgain?: () => void;
  onEdit?: () => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [open, setOpen] = useState<Record<FoldId, boolean>>({ verifier: false, metrics: false, findings: false, activity: false });
  const [fullSummary, setFullSummary] = useState(false);
  const toggle = (id: FoldId) => (v: boolean) => setOpen((o) => ({ ...o, [id]: v }));

  const final = report.rounds[report.rounds.length - 1];
  const r1 = report.rounds.length === 2 ? report.rounds[0] : null;
  const i = report.intake;
  const approved = final?.verification.isApproved ?? false;
  const checks = final?.verification.checks ?? [];
  const passed = checks.filter((c) => c.isPassed).length;
  const r1Failed = r1?.verification.checks.filter((c) => !c.isPassed) ?? [];

  const openVerifier = () => {
    setOpen((o) => ({ ...o, verifier: true }));
    requestAnimationFrame(() => document.getElementById("verifier")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };

  return (
    <article data-testid="report" aria-label="Your portfolio" className="space-y-8">
      {/* Verdict */}
      <header className="space-y-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span
            data-testid="approval"
            data-approved={approved}
            className={`rounded-full px-3 py-1 text-[15px] font-semibold ${approved ? "bg-edge-bg text-edge" : "bg-no-edge-bg text-no-edge"}`}
          >
            {approved ? "✓ Approved by the Verifier" : "✕ The Verifier did not approve this portfolio"}
          </span>
          {checks.length > 0 && (
            <button type="button" onClick={openVerifier} className="text-[15px] text-slate underline-offset-2 hover:text-ink hover:underline">
              {passed} of {checks.length} checks passed
            </button>
          )}
        </div>
        <h1 className="font-display text-4xl leading-tight font-semibold tracking-tight md:text-5xl">
          Your {usd(i.amountUsd)} {RISK[i.riskTolerance]} portfolio{i.theme ? <>, focused on {i.theme}</> : null}
        </h1>
        <p className="text-[15px] text-slate">Planned for {horizonLabel(i.horizonYears)}.</p>
        {r1 && (
          <p data-testid="rejection-note" className="max-w-3xl text-[17px] text-ink">
            The first proposal failed {r1Failed.length === 1 ? "one check" : `${r1Failed.length} checks`} ({r1Failed.map((c) => c.name).join("; ")}), so the allocator rebalanced
            {approved ? " and the second proposal passed." : ", but the second proposal still failed."}{" "}
            <button type="button" onClick={openVerifier} className="font-semibold text-[#2f5bea] hover:underline">
              See what changed
            </button>
          </p>
        )}
        <div className="max-w-3xl text-[17px] leading-relaxed text-slate">
          <div className={fullSummary ? undefined : "line-clamp-3"}>
            <Md text={report.summary} />
          </div>
          <button type="button" onClick={() => setFullSummary((f) => !f)} className="mt-1 text-[15px] font-semibold text-[#2f5bea] hover:underline">
            {fullSummary ? "Show less" : "Read the full summary"}
          </button>
        </div>
        {!!report.preferenceNotes?.length && (
          <section
            data-testid="preference-notes"
            aria-label="Notes on this portfolio"
            className={`max-w-3xl rounded-xl border px-4 py-3 ${approved ? "border-[#2f5bea]/25 bg-[#2f5bea]/[0.04]" : "border-no-edge/30 bg-no-edge-bg/60"}`}
          >
            <h2 className="font-semibold">{approved ? "How we read your preferences" : "Why it wasn't approved, and how we read your preferences"}</h2>
            <ul className="mt-1 list-disc space-y-0.5 pl-5 text-[15px]">
              {report.preferenceNotes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </section>
        )}
        {(onRunAgain || onEdit) && (
          <div className="flex flex-wrap gap-2 pt-1">
            {onRunAgain && (
              <Button variant="outline" onClick={onRunAgain}>
                Run again
              </Button>
            )}
            {onEdit && (
              <Button variant="outline" onClick={onEdit}>
                Edit and run again
              </Button>
            )}
          </div>
        )}
      </header>

      {/* The sculpture and what's in it */}
      <section aria-label="Allocation" className="grid items-start gap-8 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <AllocationDonut allocation={report.finalAllocation} selected={selected} onSelect={setSelected} />
        <div className="lg:max-h-[520px] lg:overflow-y-auto lg:pr-2">
          <HoldingsList allocation={report.finalAllocation} findings={report.findings} selected={selected} onSelect={setSelected} />
        </div>
      </section>

      {final && <Comparisons backtest={final.backtest} amount={i.amountUsd} risk={i.riskTolerance} />}
      {final && <GrowthChart backtest={final.backtest} amount={i.amountUsd} />}

      {/* Details for people who want them; all closed at first */}
      <div>
        {final && (
          <Fold
            id="verifier"
            title="How the Verifier checked it"
            hint={`${checks.length} checks, drawdown limit ${pct(DRAWDOWN_LIMIT[i.riskTolerance], 0)}${r1 ? ", first proposal vs revised" : ""}`}
            open={open.verifier}
            onToggle={toggle("verifier")}
          >
            <VerifierDetails report={report} />
          </Fold>
        )}
        {final && (
          <Fold id="metrics" title="Full backtest numbers" hint="All benchmarks, Sharpe confidence interval, sandbox cross-check" open={open.metrics} onToggle={toggle("metrics")}>
            <MetricsTable backtest={final.backtest} />
          </Fold>
        )}
        <Fold id="findings" title={`All research findings (${report.findings.length})`} hint="Grouped by the agent that found them" open={open.findings} onToggle={toggle("findings")}>
          <div className="grid gap-x-10 gap-y-6 md:grid-cols-2">
            {AGENT_ORDER.map((a) => {
              const fs = report.findings.filter((f) => f.agent === a);
              if (!fs.length) return null;
              const seen = new Set<string>();
              return (
                <div key={a}>
                  <h3 className="font-semibold">
                    {AGENT_NAME[a]} <span className="font-normal text-slate">({fs.length})</span>
                  </h3>
                  <ul className="divide-y divide-hairline">
                    {fs.map((f) => (
                      <FindingItem key={f.id} f={{ ...f, claim: f.symbol ? `${f.symbol}: ${f.claim}` : f.claim }} showSource={(u) => !seen.has(u) && !!seen.add(u)} />
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        </Fold>
        {activity && (
          <Fold id="activity" title="Agent activity" hint="How the seven agents worked through this run" open={open.activity} onToggle={toggle("activity")}>
            {activity}
          </Fold>
        )}
      </div>

      <p data-testid="disclaimer" className="text-sm text-slate">
        {report.disclaimer}
      </p>
    </article>
  );
}
