import { DRAWDOWN_LIMIT } from "@/lib/format";
import { num, pct, usd } from "@/lib/format";
import type { Backtest, RiskTolerance } from "@/lib/types";

/** Three plain-language comparisons: against the S&P 500 and against the user's own risk limit. */
export function Comparisons({ backtest, amount, risk }: { backtest: Backtest; amount: number; risk: RiskTolerance }) {
  const p = backtest.metrics.PORTFOLIO;
  const spy = backtest.metrics.SPY;
  const limit = DRAWDOWN_LIMIT[risk];
  const end = amount * (1 + p.totalReturnPct / 100);
  const spyEnd = amount * (1 + spy.totalReturnPct / 100);
  const ddShare = Math.min(1, Math.abs(p.maxDrawdownPct) / Math.abs(limit));

  return (
    <div data-testid="comparisons" className="grid gap-x-10 gap-y-6 border-y border-hairline py-6 md:grid-cols-3">
      <div>
        <div className="text-[15px] text-slate">Backtested value</div>
        <div className="font-display text-4xl font-semibold tabular-nums">{usd(end)}</div>
        <div className="mt-1 text-[15px] text-slate">
          S&P 500 would be {usd(spyEnd)} ({pct(p.totalReturnPct, 0)} vs {pct(spy.totalReturnPct, 0)})
        </div>
      </div>
      <div>
        <div className="text-[15px] text-slate">Worst drop along the way</div>
        <div className={`font-display text-4xl font-semibold tabular-nums ${p.maxDrawdownPct < limit ? "text-no-edge" : ""}`}>{pct(p.maxDrawdownPct)}</div>
        <div className="mt-2 h-1.5 w-full max-w-60 rounded-full bg-hairline" aria-hidden>
          <div className={`h-full rounded-full ${p.maxDrawdownPct < limit ? "bg-no-edge" : "bg-ink"}`} style={{ width: `${ddShare * 100}%` }} />
        </div>
        <div className="mt-1 text-[15px] text-slate">Your {risk.toLowerCase()} limit is {pct(limit, 0)}</div>
      </div>
      <div>
        <div className="text-[15px] text-slate">Return per unit of risk (Sharpe)</div>
        <div className="font-display text-4xl font-semibold tabular-nums">{num(p.sharpe)}</div>
        <div className="mt-1 text-[15px] text-slate">
          S&P 500 {num(spy.sharpe)}
          {backtest.sharpeCi && (
            <>
              ; likely range {num(backtest.sharpeCi.lower)} to {num(backtest.sharpeCi.upper)}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
