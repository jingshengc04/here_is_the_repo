import type { AssetClass, RiskTolerance } from "./types";

/** Verifier drawdown limits per risk level (backend config.DRAWDOWN_LIMITS). */
export const DRAWDOWN_LIMIT: Record<RiskTolerance, number> = { CONSERVATIVE: -20, MODERATE: -35, AGGRESSIVE: -55 };

export const CLASS_COLOR: Record<AssetClass, string> = { STOCKS: "#2f5bea", CRYPTO: "#7c4dff", GOLD: "#c9a227", BONDS: "#1b8a9e" };
export const CLASS_LABEL: Record<AssetClass, string> = { STOCKS: "Stocks", CRYPTO: "Crypto", GOLD: "Gold", BONDS: "Bonds" };

/** horizonYears is 0.25–10 in quarter steps: 0.5 → "6 months", 1.5 → "18 months", 2.5 → "2.5 years". */
export function horizonLabel(y: number): string {
  if (y === 1) return "1 year";
  if (y < 1 || (y < 2 && !Number.isInteger(y))) return `${Math.round(y * 12)} months`;
  return `${y} years`;
}

/** Lighten a hex colour towards white by t (0–1). Holdings are shades of their class colour. */
export function shade(hex: string, t: number): string {
  const n = parseInt(hex.slice(1), 16);
  const m = (v: number) => Math.round(v + (255 - v) * t);
  return `rgb(${m(n >> 16)}, ${m((n >> 8) & 255)}, ${m(n & 255)})`;
}

const minus = (s: string) => s.replace("-", "−");
/** Numbers arrive pre-rounded; only format, never recompute. */
export const pct = (x: number, digits = 1) => minus(`${x.toFixed(digits)}%`);
/** Holding/slice weights arrive with 1 decimal: keep it when present (4.5% must not read as 5%). */
export const weight = (x: number) => (Number.isInteger(x) ? `${x}%` : pct(x, 1));
export const pctDelta = (x: number) => minus(`${x > 0 ? "+" : ""}${x.toFixed(1)}`);
export const usd = (x: number) => `$${Math.round(x).toLocaleString("en-US")}`;
export const num = (x: number | null, digits = 2) => (x === null ? "—" : minus(x.toFixed(digits)));
export const short = (symbol: string) => symbol.replace(/-USD$/, "");
