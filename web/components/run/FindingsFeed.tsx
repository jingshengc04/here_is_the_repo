import { cn } from "@/lib/utils";
import { NumbersBadge } from "./NumbersBadge";
import type { Finding } from "@/lib/types";

export const SENTIMENT = {
  POSITIVE: { mark: "▲", cls: "text-edge" },
  NEUTRAL: { mark: "●", cls: "text-neutral-400" },
  NEGATIVE: { mark: "▼", cls: "text-no-edge" },
} as const;

export function FindingsFeed({ findings, live }: { findings: Finding[]; live: boolean }) {
  const newest = [...findings].reverse();
  return (
    <section data-testid="findings-feed" aria-label="Live findings" className="flex max-h-[640px] flex-col rounded-xl border bg-card p-4 shadow-xs">
      <h2 className="font-semibold">Live findings <span className="font-normal text-muted-foreground">({findings.length})</span></h2>
      {findings.length === 0 && <p className="py-6 text-center text-sm text-muted-foreground">Findings appear here as agents report.</p>}
      <ol className="mt-1 flex-1 overflow-y-auto" aria-live="polite">
        {newest.map((f, i) => (
          <li key={f.id} className={cn("border-t py-2 text-[13px] leading-snug", i === 0 && live && "-mx-2 rounded-md border-transparent bg-[#fffbe6] px-2")}>
            <div className="text-[11px] font-extrabold tracking-wide text-muted-foreground">
              {f.agent} · <span className="text-foreground">{f.symbol ?? "MARKET"}</span>{" "}
              <span className={SENTIMENT[f.sentiment].cls}>{SENTIMENT[f.sentiment].mark}</span>
              {i === 0 && live && <span className="ml-1 text-inconclusive">NEW</span>}
              <NumbersBadge verified={f.numbersVerified} />
            </div>
            <div>{f.claim}</div>
            {f.sources.slice(0, 1).map((s) => (
              <a key={s.url} href={s.url} target="_blank" rel="noopener noreferrer" className="block truncate text-xs text-[#2f5bea] hover:underline">
                ↗ {s.title}
              </a>
            ))}
          </li>
        ))}
      </ol>
    </section>
  );
}
