"use client";

import { useEffect, useRef } from "react";
import { ChevronDownIcon } from "lucide-react";
import { NumbersBadge } from "@/components/run/NumbersBadge";
import { CLASS_COLOR, CLASS_LABEL, num, short, usd, weight } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Allocation, Finding, Holding } from "@/lib/types";

const AGENT_NAME: Record<Finding["agent"], string> = {
  DISCOVERY: "Discovery",
  FINANCIALS: "Financials",
  LANDSCAPE: "Landscape",
  NEWS: "News",
  SOCIAL: "Social",
  ORCHESTRATOR: "Orchestrator",
  VERIFIER: "Verifier",
};
const MARK = { POSITIVE: ["▲", "text-edge"], NEUTRAL: ["●", "text-slate"], NEGATIVE: ["▼", "text-no-edge"] } as const;

export function FindingItem({ f, showSource }: { f: Finding; showSource: (url: string) => boolean }) {
  const sources = f.sources.filter((s) => showSource(s.url));
  return (
    <li className="py-2.5 text-[15px] leading-snug">
      <div className="mb-0.5 text-[13px] text-slate">
        <span className={MARK[f.sentiment][1]}>{MARK[f.sentiment][0]}</span> {AGENT_NAME[f.agent]}
        <NumbersBadge verified={f.numbersVerified} />
      </div>
      {f.claim}
      {sources.map((s) => (
        <a key={s.url} href={s.url} target="_blank" rel="noopener noreferrer" className="mt-0.5 block truncate text-[13px] text-[#2f5bea] hover:underline">
          {s.title}
        </a>
      ))}
    </li>
  );
}

function Evidence({ holding, findings }: { holding: Holding; findings: Finding[] }) {
  const evidence = holding.findingIds.map((id) => findings.find((f) => f.id === id)).filter((f): f is Finding => !!f);
  const bull = evidence.filter((f) => f.sentiment !== "NEGATIVE");
  const bear = evidence.filter((f) => f.sentiment === "NEGATIVE");
  // Same agent call → same source list; show each link once.
  const shown = new Set<string>();
  const firstTime = (url: string) => !shown.has(url) && !!shown.add(url);
  return (
    <div data-testid="evidence" className="space-y-3 pb-4 pl-6">
      <p className="text-[15px] leading-relaxed text-ink">{holding.rationale}</p>
      {(holding.financialScore !== null || holding.sentimentScore !== null) && (
        <p className="text-sm text-slate">
          {holding.financialScore !== null && <>Financial health {num(holding.financialScore, 0)}/10. </>}
          {holding.sentimentScore !== null && <>Sentiment {holding.sentimentScore > 0 ? "+" : ""}{num(holding.sentimentScore, 1)} on a −1 to +1 scale.</>}
        </p>
      )}
      {bull.length > 0 && (
        <div>
          <div className="text-sm font-semibold">Case for</div>
          <ul className="divide-y divide-hairline">{bull.map((f) => <FindingItem key={f.id} f={f} showSource={firstTime} />)}</ul>
        </div>
      )}
      {bear.length > 0 && (
        <div>
          <div className="text-sm font-semibold">Case against</div>
          <ul className="divide-y divide-hairline">{bear.map((f) => <FindingItem key={f.id} f={f} showSource={firstTime} />)}</ul>
        </div>
      )}
    </div>
  );
}

export function HoldingsList({
  allocation,
  findings,
  selected,
  onSelect,
}: {
  allocation: Allocation;
  findings: Finding[];
  selected: string | null;
  onSelect: (s: string | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (selected) ref.current?.querySelector(`[data-symbol="${CSS.escape(selected)}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selected]);
  const max = Math.max(...allocation.slices.flatMap((s) => s.holdings.map((h) => h.weightPct)));

  return (
    <div ref={ref} data-testid="holdings-list" className="space-y-5">
      {allocation.slices.map((s) => (
        <section key={s.assetClass} aria-label={CLASS_LABEL[s.assetClass]}>
          <h3 className="flex items-baseline gap-2 border-b border-hairline pb-1.5 text-[15px] font-semibold">
            <span className="inline-block size-2.5 rounded-full" style={{ background: CLASS_COLOR[s.assetClass] }} />
            {CLASS_LABEL[s.assetClass]}
            <span className="ml-auto font-normal text-slate tabular-nums">
              {weight(s.weightPct)}, {usd(s.holdings.reduce((t, h) => t + h.amountUsd, 0))}
            </span>
          </h3>
          <ul>
            {s.holdings.map((h) => {
              const open = selected === h.symbol;
              return (
                <li key={h.symbol} data-symbol={h.symbol} className="border-b border-hairline last:border-0">
                  <button
                    type="button"
                    aria-expanded={open}
                    onClick={() => onSelect(open ? null : h.symbol)}
                    className={cn("grid w-full grid-cols-[1fr_3.5rem_4.5rem_1rem] items-center gap-3 py-2.5 text-left transition-colors hover:bg-ink/[0.03] sm:grid-cols-[1fr_5.5rem_4rem_4.5rem_1rem]", open && "bg-ink/[0.03]")}
                  >
                    <span className="min-w-0 truncate">
                      <b className="font-semibold">{short(h.symbol)}</b> <span className="text-slate">{h.name}</span>
                      {h.isUserRequested && (
                        <span data-testid="user-requested" className="ml-2 rounded-full bg-[#2f5bea]/[0.08] px-2 py-0.5 text-xs font-semibold text-[#2f5bea]">
                          You asked for this
                        </span>
                      )}
                    </span>
                    <span className="hidden h-1.5 rounded-full bg-hairline sm:block" aria-hidden>
                      <span className="block h-full rounded-full" style={{ width: `${(h.weightPct / max) * 100}%`, background: CLASS_COLOR[s.assetClass] }} />
                    </span>
                    <span className="text-right font-semibold tabular-nums">{weight(h.weightPct)}</span>
                    <span className="text-right text-slate tabular-nums">{usd(h.amountUsd)}</span>
                    <ChevronDownIcon className={cn("size-4 text-slate transition-transform", open && "rotate-180")} />
                  </button>
                  {open && <Evidence holding={h} findings={findings} />}
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}
