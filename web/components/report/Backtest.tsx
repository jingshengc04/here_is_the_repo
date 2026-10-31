"use client";

import { useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { num, pct, usd } from "@/lib/format";
import type { Backtest, Metrics, SeriesKey } from "@/lib/types";
import { sharpeCiText } from "./SharpeCI";

const SERIES: { key: SeriesKey; label: string; color: string }[] = [
  { key: "PORTFOLIO", label: "Your portfolio", color: "#2f5bea" },
  { key: "SPY", label: "S&P 500", color: "#5b6170" },
  { key: "SIXTY_FORTY", label: "60/40 stocks/bonds", color: "#8fb1d6" },
  { key: "EQUAL_WEIGHT", label: "Equal weight", color: "#c9a6a6" },
];

const month = (d: string) => new Date(`${d}T00:00:00Z`).toLocaleString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
const kusd = (v: number) => (v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `$${Math.round(v / 1e3)}k` : `$${Math.round(v)}`);

/** Backtest equity curve shown as the user's own money (amount × growth of $1). */
export function GrowthChart({ backtest, amount }: { backtest: Backtest; amount: number }) {
  const [all, setAll] = useState(false);
  const { dates, series } = backtest.equityCurve;
  const rows = dates.map((date, i) => ({
    date,
    PORTFOLIO: series.PORTFOLIO[i] * amount,
    SPY: series.SPY[i] * amount,
    SIXTY_FORTY: series.SIXTY_FORTY[i] * amount,
    EQUAL_WEIGHT: series.EQUAL_WEIGHT[i] * amount,
  }));
  const yearTicks = dates.filter((d, i) => i > 0 && d.slice(0, 4) !== dates[i - 1].slice(0, 4));
  const shown = all ? SERIES : SERIES.slice(0, 2);

  return (
    <section data-testid="backtest-chart" aria-label="Backtest growth">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <h2 className="text-xl font-semibold">
          What {usd(amount)} would have become
          <span className="ml-2 text-base font-normal text-slate">
            {month(backtest.startDate)} to {month(backtest.endDate)}, today&apos;s weights
          </span>
        </h2>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
          {shown.map((s) => (
            <span key={s.key} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 rounded" style={{ background: s.color, height: s.key === "PORTFOLIO" ? 3 : 2 }} />
              {s.label}
            </span>
          ))}
          <button type="button" onClick={() => setAll((a) => !a)} aria-pressed={all} className="font-semibold text-[#2f5bea] hover:underline">
            {all ? "Fewer benchmarks" : "Compare more benchmarks"}
          </button>
        </div>
      </div>
      <div className="mt-3 h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="growthFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#2f5bea" stopOpacity={0.28} />
                <stop offset="1" stopColor="#2f5bea" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} stroke="#e3e5ea" />
            <XAxis dataKey="date" ticks={yearTicks} tickFormatter={(d: string) => d.slice(0, 4)} tickLine={false} axisLine={false} tick={{ fontSize: 13, fill: "#5b6170" }} />
            <YAxis width={56} tickLine={false} axisLine={false} domain={["auto", "auto"]} tick={{ fontSize: 13, fill: "#5b6170" }} tickFormatter={kusd} />
            <Tooltip
              labelFormatter={(d) => month(String(d))}
              formatter={(v, name) => [usd(Number(v)), SERIES.find((s) => s.key === name)?.label ?? name]}
              contentStyle={{ borderRadius: 10, border: "1px solid #e3e5ea", fontSize: 13 }}
            />
            {shown
              .slice(1)
              .reverse()
              .map((s) => (
                <Line key={s.key} dataKey={s.key} stroke={s.color} strokeWidth={2} strokeDasharray="5 4" type="monotone" dot={false} isAnimationActive={false} />
              ))}
            <Area dataKey="PORTFOLIO" stroke="#2f5bea" strokeWidth={3} fill="url(#growthFill)" type="monotone" dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

const ROWS: { label: string; get: (m: Metrics) => string }[] = [
  { label: "Total return", get: (m) => pct(m.totalReturnPct) },
  { label: "Annual growth (CAGR)", get: (m) => pct(m.cagrPct) },
  { label: "Volatility", get: (m) => pct(m.volatilityPct) },
  { label: "Sharpe ratio", get: (m) => num(m.sharpe) },
  { label: "Max drawdown", get: (m) => pct(m.maxDrawdownPct) },
  { label: "Worst month", get: (m) => pct(m.worstMonthPct) },
];

export function CrossCheckBadge({ backtest }: { backtest: Backtest }) {
  const cc = backtest.crossCheck;
  // Backend reports the independent number on both sides when the sandbox didn't run.
  const state = cc.isMatch ? "match" : cc.sandboxTotalReturnPct === cc.independentTotalReturnPct ? "unchecked" : "mismatch";
  const badge = {
    match: { cls: "border-edge bg-edge-bg", text: "text-edge", label: "✓ Verified", note: `sandbox ${pct(cc.sandboxTotalReturnPct)} vs ours ${pct(cc.independentTotalReturnPct)}` },
    unchecked: { cls: "border-[#e0b100] bg-inconclusive-bg", text: "text-inconclusive", label: "⚠ Not cross-checked", note: "sandbox unavailable" },
    mismatch: { cls: "border-no-edge bg-no-edge-bg", text: "text-no-edge", label: "⚠ Numbers differ", note: `sandbox ${pct(cc.sandboxTotalReturnPct)} vs ours ${pct(cc.independentTotalReturnPct)}` },
  }[state];
  return (
    <span data-testid="verified-badge" data-match={cc.isMatch} data-state={state} className={`inline-block rounded-full border-[1.5px] px-2.5 py-0.5 text-sm ${badge.cls}`}>
      <b className={badge.text}>{badge.label}</b> <span className="text-slate">{badge.note}</span>
    </span>
  );
}

export function MetricsTable({ backtest }: { backtest: Backtest }) {
  return (
    <div className="space-y-3">
      <CrossCheckBadge backtest={backtest} />
      <div className="overflow-x-auto">
        <table data-testid="metrics-table" className="w-full max-w-3xl text-[15px] tabular-nums">
          <thead>
            <tr className="text-sm text-slate">
              <th />
              {SERIES.map((s) => (
                <th key={s.key} className="px-2 py-1 text-right font-medium">
                  {s.key === "PORTFOLIO" ? "Portfolio" : s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.label} className="border-t border-hairline">
                <td className="py-1.5 pr-4">{r.label}</td>
                {SERIES.map((s) => (
                  <td key={s.key} className={`px-2 py-1.5 text-right ${s.key === "PORTFOLIO" ? "font-semibold" : "text-slate"}`}>
                    {r.get(backtest.metrics[s.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {backtest.sharpeCi && (
        <p data-testid="sharpe-ci-text" className="text-[15px]">
          <b>{sharpeCiText(backtest)}</b>{" "}
          <span className="text-slate">from 1,000 bootstrap resamples; {Math.round(backtest.sharpeCi.probPositive * 100)}% of them above zero.</span>
        </p>
      )}
      <p className="text-sm text-slate">The backtest applies today&apos;s weights to past prices, so it can&apos;t account for how the portfolio would have been chosen back then.</p>
    </div>
  );
}
