import { num } from "@/lib/format";
import type { Backtest } from "@/lib/types";

const short = (v: number) => String(Number(v.toFixed(2))).replace("-", "−");

/** "Sharpe 0.74 (95% CI 0.07–1.41)" — contract format for Backtest.sharpeCi (PORTFOLIO only). */
export const sharpeCiText = (b: Backtest) =>
  b.sharpeCi ? `Sharpe ${num(b.metrics.PORTFOLIO.sharpe)} (95% CI ${short(b.sharpeCi.lower)}–${short(b.sharpeCi.upper)})` : null;

/** Interval bar: red when the CI touches or crosses 0 (can't tell from luck), green when entirely above. */
export function SharpeCIBar({ backtest, compact }: { backtest: Backtest; compact?: boolean }) {
  const ci = backtest.sharpeCi;
  if (!ci) return null;
  const point = backtest.metrics.PORTFOLIO.sharpe;
  const crosses = ci.lower <= 0;
  const color = crosses ? "var(--no-edge)" : "var(--edge)";
  const span = Math.max(Math.abs(ci.lower), Math.abs(ci.upper), 0.5) * 1.3;
  const min = Math.min(-span * 0.45, ci.lower - 0.2);
  const max = Math.max(span, ci.upper + 0.2);
  const W = 320;
  const PAD = 16;
  const x = (v: number) => PAD + ((v - min) / (max - min)) * (W - 2 * PAD);

  return (
    <div data-testid="sharpe-ci" data-crosses-zero={crosses} className="text-[13px]">
      <div className="font-semibold">{sharpeCiText(backtest)}</div>
      <svg viewBox={`0 0 ${W} 46`} className={compact ? "w-full max-w-72" : "w-full max-w-80"} role="img" aria-label={`95% CI ${ci.lower} to ${ci.upper}`}>
        <line x1={PAD} x2={W - PAD} y1={22} y2={22} stroke="#d4d4d8" strokeWidth={2} />
        <line x1={x(0)} x2={x(0)} y1={6} y2={38} stroke="#55555c" strokeWidth={1.5} strokeDasharray="4 3" />
        <text x={x(0)} y={45} textAnchor="middle" fontSize={10} fontWeight={700} fill="#55555c">0</text>
        <line x1={x(ci.lower)} x2={x(ci.upper)} y1={22} y2={22} stroke={color} strokeWidth={9} strokeLinecap="round" />
        {point !== null && <circle cx={x(point)} cy={22} r={6} fill="#fff" stroke="#111114" strokeWidth={2.5} />}
        <text x={x(ci.lower)} y={10} textAnchor="middle" fontSize={10} fontWeight={600} fill="#33333a">{short(ci.lower)}</text>
        <text x={x(ci.upper)} y={10} textAnchor="middle" fontSize={10} fontWeight={600} fill="#33333a">{short(ci.upper)}</text>
      </svg>
      <div style={{ color }} className="font-medium">
        {crosses ? "Interval includes 0 → can't tell skill from luck" : "Entire interval above 0"} · {Math.round(ci.probPositive * 100)}% of resamples &gt; 0
      </div>
    </div>
  );
}
