import { CLASS_COLOR, CLASS_LABEL, num, pct, short } from "@/lib/format";
import type { Allocation, Report, Round, Verification } from "@/lib/types";
import { SharpeCIBar } from "./SharpeCI";
import { Sunburst } from "./Sunburst";

function Checks({ v }: { v: Verification }) {
  return (
    <ul className="space-y-1 text-[14px] leading-snug">
      {v.checks.map((c) => (
        <li key={c.name}>
          <span className={c.isPassed ? "text-edge" : "text-no-edge"}>{c.isPassed ? "✓" : "✕"}</span> {c.name}{" "}
          <span className="text-muted-foreground">· {c.detail}</span>
        </li>
      ))}
    </ul>
  );
}

const Old = ({ children }: { children: React.ReactNode }) => <s className="text-muted-foreground">{children}</s>;

const weights = (a: Allocation) => new Map(a.slices.flatMap((s) => s.holdings.map((h) => [h.symbol, h.weightPct] as const)));

/** Display-only comparison of two rounds (weights come from the backend; nothing is re-derived). */
function RoundDiff({ r1, r2 }: { r1: Round; r2: Round }) {
  const classes = r2.allocation.slices.map((s) => ({
    s,
    before: r1.allocation.slices.find((x) => x.assetClass === s.assetClass)?.weightPct ?? 0,
  }));
  const w1 = weights(r1.allocation);
  const w2 = weights(r2.allocation);
  const symbols = [...new Set([...w1.keys(), ...w2.keys()])];
  const holdings = symbols.filter((sym) => w1.get(sym) !== w2.get(sym));
  const m1 = r1.backtest.metrics.PORTFOLIO;
  const m2 = r2.backtest.metrics.PORTFOLIO;

  return (
    <div data-testid="round-diff" className="flex flex-col justify-center gap-1 rounded-xl border bg-card p-4 text-sm leading-relaxed shadow-xs">
      <div className="mb-1 text-center font-semibold">What changed</div>
      {classes
        .filter(({ s, before }) => s.weightPct !== before)
        .map(({ s, before }) => (
          <div key={s.assetClass}>
            <span style={{ color: CLASS_COLOR[s.assetClass] }}>●</span> {CLASS_LABEL[s.assetClass]} <Old>{before}%</Old> → <b>{s.weightPct}%</b>
          </div>
        ))}
      {holdings.map((sym) => {
        const a = w1.get(sym);
        const b = w2.get(sym);
        return (
          <div key={sym} className="pl-4">
            {a === undefined ? (
              <b className="text-edge">+ {short(sym)} {b}%</b>
            ) : b === undefined ? (
              <b className="text-no-edge">− {short(sym)} removed</b>
            ) : (
              <>
                {short(sym)} <Old>{a}%</Old> → <b>{b}%</b>
              </>
            )}
          </div>
        );
      })}
      <div className="mt-1.5 border-t pt-1.5">
        Max DD <Old>{pct(m1.maxDrawdownPct)}</Old> → <b className={m2.maxDrawdownPct > m1.maxDrawdownPct ? "text-edge" : undefined}>{pct(m2.maxDrawdownPct)}</b>
      </div>
      <div>
        Sharpe <Old>{num(m1.sharpe)}</Old> → <b>{num(m2.sharpe)}</b>
      </div>
      <div>
        Return <Old>{pct(m1.totalReturnPct, 0)}</Old> → <b>{pct(m2.totalReturnPct, 0)}</b>
      </div>
      {r1.backtest.sharpeCi && r2.backtest.sharpeCi && (
        <div>
          Sharpe CI low <Old>{num(r1.backtest.sharpeCi.lower)}</Old> →{" "}
          <b className={r2.backtest.sharpeCi.lower > 0 ? "text-edge" : undefined}>{num(r2.backtest.sharpeCi.lower)}</b>
        </div>
      )}
    </div>
  );
}

export function VerifierDetails({ report }: { report: Report }) {
  const final = report.rounds[report.rounds.length - 1];
  const r1 = report.rounds.length === 2 ? report.rounds[0] : null;
  const passed = final.verification.checks.filter((c) => c.isPassed).length;

  return (
    <div className="space-y-6">
      {r1 && (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_190px_minmax(0,1fr)]">
          <div data-testid="round-1" className="min-w-0 rounded-xl border-2 border-no-edge bg-no-edge-bg p-4">
            <div className="flex items-center justify-between">
              <span className="font-semibold">First proposal</span>
              <b className="text-no-edge">✕ Rejected</b>
            </div>
            <div className="flex justify-center opacity-70 grayscale-[0.7]">
              <Sunburst allocation={r1.allocation} size={220} testId="sunburst-r1" />
            </div>
            <Checks v={r1.verification} />
            <div className="mt-2">
              <SharpeCIBar backtest={r1.backtest} compact />
            </div>
            {r1.verification.changesRequested.length > 0 && (
              <div className="mt-2 rounded-md border-[1.5px] border-no-edge bg-card px-2.5 py-1.5 text-[13px]">
                <b className="text-no-edge">Verifier asked for:</b> {r1.verification.changesRequested.join("; ")}
              </div>
            )}
          </div>
          <RoundDiff r1={r1} r2={final} />
          <div className="min-w-0 rounded-xl border-2 border-edge bg-edge-bg p-4">
            <div className="flex items-center justify-between">
              <span className="font-semibold">Revised proposal</span>
              <b className={final.verification.isApproved ? "text-edge" : "text-no-edge"}>{final.verification.isApproved ? "✓ Approved" : "✕ Not approved"}</b>
            </div>
            <div className="flex justify-center">
              <Sunburst allocation={final.allocation} size={220} testId="sunburst-final" />
            </div>
          </div>
        </div>
      )}
      <div data-testid="final-round" className="grid gap-6 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div>
          <div className="mb-2 font-semibold">
            {passed} of {final.verification.checks.length} checks passed{r1 ? " on the revised proposal" : ""}
          </div>
          <Checks v={final.verification} />
        </div>
        <SharpeCIBar backtest={final.backtest} />
      </div>
    </div>
  );
}
