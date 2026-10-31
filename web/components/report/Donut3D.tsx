"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CLASS_COLOR, CLASS_LABEL, shade, short, usd, weight } from "@/lib/format";
import type { Allocation, AssetClass } from "@/lib/types";

// 2.5D extruded two-ring donut drawn in plain SVG (no WebGL): an ellipse projection of two cylinders.
// Inner raised ring = asset classes, outer ring = holdings (shades of their class colour).
// Drag to rotate; click a holding to select it. One grow-in on mount, skipped for reduced motion.

const W = 600;
const H = 420;
const CX = W / 2;
const CY = 190;
const TILT = 0.5; // ry / rx
const TAU = Math.PI * 2;

const OUTER = { r0: 168, r1: 262, depth: 34, lift: 0 };
const INNER = { r0: 70, r1: 158, depth: 30, lift: 30 };

function darken(rgb: string, t: number) {
  const m = rgb.match(/\d+/g)!.map(Number);
  return `rgb(${m.map((v) => Math.round(v * (1 - t))).join(",")})`;
}
const toRgb = (hex: string) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgb(${n >> 16},${(n >> 8) & 255},${n & 255})`;
};

const pt = (r: number, a: number, dy: number) => `${(CX + r * Math.cos(a)).toFixed(2)} ${(CY + r * TILT * Math.sin(a) + dy).toFixed(2)}`;

function top(r0: number, r1: number, a0: number, a1: number, dy: number) {
  const large = a1 - a0 > Math.PI ? 1 : 0;
  return `M${pt(r1, a0, dy)} A${r1} ${r1 * TILT} 0 ${large} 1 ${pt(r1, a1, dy)} L${pt(r0, a1, dy)} A${r0} ${r0 * TILT} 0 ${large} 0 ${pt(r0, a0, dy)} Z`;
}
function wall(r: number, a0: number, a1: number, dy: number, depth: number) {
  return `M${pt(r, a0, dy)} A${r} ${r * TILT} 0 0 1 ${pt(r, a1, dy)} L${pt(r, a1, dy + depth)} A${r} ${r * TILT} 0 0 0 ${pt(r, a0, dy + depth)} Z`;
}
/** Parts of [a0, a1] where sin(θ) has the given sign: front (+) faces the viewer on the outside, back (−) on the inside. */
function segments(a0: number, a1: number, front: boolean): [number, number][] {
  const out: [number, number][] = [];
  for (let k = Math.floor(a0 / TAU) - 1; k <= Math.ceil(a1 / TAU) + 1; k++) {
    const s = k * TAU + (front ? 0 : Math.PI);
    const lo = Math.max(a0, s);
    const hi = Math.min(a1, s + Math.PI);
    if (hi - lo > 1e-4) out.push([lo, hi]);
  }
  return out;
}

interface Slice {
  key: string;
  a0: number;
  a1: number;
  fill: string;
  side: string;
  label: string | null;
  symbol?: string;
}

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export function Donut3D({
  allocation,
  selected,
  onSelect,
  amountLabel,
}: {
  allocation: Allocation;
  selected: string | null;
  onSelect: (symbol: string) => void;
  amountLabel?: boolean;
}) {
  const [rot, setRot] = useState(-Math.PI / 2);
  const [grow, setGrow] = useState(0);
  const [hover, setHover] = useState<string | null>(null);
  const drag = useRef<{ x: number; rot: number; moved: boolean } | null>(null);

  // Single orchestrated entrance: rings rise and turn into place.
  useEffect(() => {
    const duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 1100;
    let raf = 0;
    const t0 = performance.now();
    const step = (now: number) => {
      const p = duration ? Math.min(1, (now - t0) / duration) : 1;
      setGrow(easeOut(p));
      setRot(-Math.PI / 2 - (1 - easeOut(p)) * 1.4);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, []);

  const holdings = useMemo(
    () => allocation.slices.flatMap((s) => s.holdings.map((h, i) => ({ ...h, assetClass: s.assetClass as AssetClass, i }))),
    [allocation],
  );

  const outerSlices: Slice[] = [];
  let a = rot;
  for (const h of holdings) {
    const span = (h.weightPct / 100) * TAU;
    const fill = shade(CLASS_COLOR[h.assetClass], 0.05 + h.i * 0.15);
    outerSlices.push({ key: h.symbol, symbol: h.symbol, a0: a, a1: a + span, fill, side: darken(fill, 0.28), label: h.weightPct >= 4 ? short(h.symbol) : null });
    a += span;
  }
  const innerSlices: Slice[] = [];
  a = rot;
  for (const s of allocation.slices) {
    const span = (s.weightPct / 100) * TAU;
    const fill = toRgb(CLASS_COLOR[s.assetClass]);
    innerSlices.push({ key: s.assetClass, a0: a, a1: a + span, fill, side: darken(fill, 0.32), label: s.weightPct >= 5 ? weight(s.weightPct) : null });
    a += span;
  }

  const active = hover ?? selected;
  const raise = (s: Slice) => (s.symbol && s.symbol === active ? -10 : 0);

  const ring = (slices: Slice[], g: typeof OUTER, gap: number) => {
    const lift = -g.lift * grow;
    const depth = Math.max(2, g.depth * grow);
    const parts: React.ReactNode[] = [];
    for (const s of slices) {
      const dy = lift + raise(s);
      const a0 = s.a0 + gap;
      const a1 = s.a1 - gap;
      if (a1 <= a0) continue;
      for (const [b0, b1] of segments(a0, a1, false)) parts.push(<path key={`${s.key}-in-${b0}`} d={wall(g.r0, b0, b1, dy, depth)} fill={darken(s.side, 0.15)} />);
    }
    for (const s of slices) {
      const dy = lift + raise(s);
      const a0 = s.a0 + gap;
      const a1 = s.a1 - gap;
      if (a1 <= a0) continue;
      for (const [b0, b1] of segments(a0, a1, true)) parts.push(<path key={`${s.key}-out-${b0}`} d={wall(g.r1, b0, b1, dy, depth)} fill={s.side} />);
    }
    for (const s of slices) {
      const dy = lift + raise(s);
      const a0 = s.a0 + gap;
      const a1 = s.a1 - gap;
      if (a1 <= a0) continue;
      const d = top(g.r0, g.r1, a0, a1, dy);
      parts.push(
        <g
          key={`${s.key}-top`}
          data-symbol={s.symbol}
          onPointerEnter={() => s.symbol && setHover(s.symbol)}
          onPointerLeave={() => setHover(null)}
          onClick={() => s.symbol && !drag.current?.moved && onSelect(s.symbol)}
          className={s.symbol ? "cursor-pointer" : undefined}
        >
          <path d={d} fill={s.fill} />
          <path d={d} fill="url(#sheen)" />
          {s.symbol === active && <path d={d} fill="none" stroke="#14161a" strokeWidth={2} />}
        </g>,
      );
    }
    return parts;
  };

  const labels = (slices: Slice[], g: typeof OUTER, at: number) => {
    if (grow < 0.85) return null;
    const lift = -g.lift * grow;
    const parts: React.ReactNode[] = [];
    for (const s of slices) {
      if (!s.label) continue;
      const m = (s.a0 + s.a1) / 2;
      const r = g.r0 + (g.r1 - g.r0) * at;
      const [x, y] = pt(r, m, lift + raise(s)).split(" ").map(Number);
      parts.push(
        <text key={`${s.key}-label`} x={x} y={y} textAnchor="middle" dominantBaseline="central" className="pointer-events-none select-none" fontSize={s.symbol ? 14 : 16} fontWeight={700} fill="#fff" stroke="rgba(20,22,26,0.35)" strokeWidth={3} paintOrder="stroke">
          {s.label}
        </text>,
      );
    }
    return parts;
  };

  const hovered = holdings.find((h) => h.symbol === active);

  return (
    <figure className="relative m-0 select-none">
      <svg
        data-testid="donut-3d"
        viewBox={`0 0 ${W} ${H}`}
        className="w-full cursor-grab touch-pan-y active:cursor-grabbing"
        role="img"
        aria-label={`Allocation: ${allocation.slices.map((s) => `${CLASS_LABEL[s.assetClass]} ${s.weightPct}%`).join(", ")}`}
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, rot, moved: false };
        }}
        onPointerMove={(e) => {
          if (!drag.current || e.buttons !== 1) return;
          const dx = e.clientX - drag.current.x;
          if (Math.abs(dx) > 3) drag.current.moved = true;
          setRot(drag.current.rot + dx * 0.012);
        }}
        onPointerUp={() => setTimeout(() => (drag.current = null))}
      >
        <defs>
          <linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#fff" stopOpacity="0.28" />
            <stop offset="0.55" stopColor="#fff" stopOpacity="0.04" />
            <stop offset="1" stopColor="#000" stopOpacity="0.08" />
          </linearGradient>
          <radialGradient id="floor" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0" stopColor="#14161a" stopOpacity="0.22" />
            <stop offset="1" stopColor="#14161a" stopOpacity="0" />
          </radialGradient>
        </defs>
        <ellipse cx={CX} cy={CY + OUTER.depth + 34} rx={300} ry={300 * TILT * 0.55} fill="url(#floor)" opacity={grow} />
        {ring(outerSlices, OUTER, 0.006)}
        {ring(innerSlices, INNER, 0.004)}
        {labels(outerSlices, OUTER, 0.68)}
        {labels(innerSlices, INNER, 0.5)}
      </svg>
      <figcaption className="-mt-4 min-h-6 text-center text-[15px] text-slate">
        {hovered ? (
          <>
            <b className="text-ink">{hovered.symbol}</b> {hovered.name}, {weight(hovered.weightPct)}
            {amountLabel !== false && <>, {usd(hovered.amountUsd)}</>}
          </>
        ) : (
          "Drag to turn. Select a holding to see why it's there."
        )}
      </figcaption>
    </figure>
  );
}
