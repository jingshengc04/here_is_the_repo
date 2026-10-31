"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip, type PieLabelRenderProps } from "recharts";
import { CLASS_COLOR, CLASS_LABEL, pct, shade, short, usd } from "@/lib/format";
import type { Allocation, AssetClass } from "@/lib/types";

const RAD = Math.PI / 180;

interface Props {
  allocation: Allocation;
  size?: number;
  onSelect?: (symbol: string) => void;
  testId?: string;
}

// Two concentric Pies (contract UI notes): inner = asset classes, outer = holdings in slice order.
export function Sunburst({ allocation, size = 360, onSelect, testId }: Props) {
  const slices = allocation.slices.map((s) => ({ ...s, name: CLASS_LABEL[s.assetClass] }));
  const holdings = allocation.slices.flatMap((s) =>
    s.holdings.map((h, i) => ({ ...h, assetClass: s.assetClass, fill: shade(CLASS_COLOR[s.assetClass], 0.12 + i * 0.16) })),
  );
  const small = size < 300;

  const label = (key: "symbol" | "class", min: number) =>
    function SliceLabel({ cx, cy, midAngle = 0, innerRadius, outerRadius, payload }: PieLabelRenderProps) {
      const p = payload as { symbol?: string; weightPct: number; assetClass: AssetClass };
      if (p.weightPct < min) return null;
      const r = (Number(innerRadius) + Number(outerRadius)) / 2;
      const x = Number(cx) + r * Math.cos(-midAngle * RAD);
      const y = Number(cy) + r * Math.sin(-midAngle * RAD);
      return (
        <text x={x} y={y} fill="#fff" textAnchor="middle" dominantBaseline="central" fontSize={small ? 10 : 13} fontWeight={800} pointerEvents="none">
          {key === "symbol" ? short(p.symbol!) : `${p.weightPct}%`}
        </text>
      );
    };

  return (
    <div data-testid={testId} style={{ width: size, height: size }} className="shrink-0">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={slices} dataKey="weightPct" nameKey="name" innerRadius="20%" outerRadius="46%" startAngle={90} endAngle={-270} stroke="#fff" strokeWidth={2} isAnimationActive={false} labelLine={false} label={label("class", 5)}>
            {slices.map((s) => (
              <Cell key={s.assetClass} fill={CLASS_COLOR[s.assetClass]} />
            ))}
          </Pie>
          <Pie
            data={holdings}
            dataKey="weightPct"
            nameKey="symbol"
            innerRadius="51%"
            outerRadius="82%"
            startAngle={90}
            endAngle={-270}
            stroke="#fff"
            strokeWidth={2}
            isAnimationActive={false}
            labelLine={false}
            label={label("symbol", small ? 6 : 4)}
            onClick={onSelect ? (d) => onSelect((d.payload as { symbol: string }).symbol) : undefined}
            className={onSelect ? "cursor-pointer" : undefined}
          >
            {holdings.map((h) => (
              <Cell key={h.symbol} fill={h.fill} />
            ))}
          </Pie>
          <Tooltip
            formatter={(v, _n, item) => {
              const p = item.payload as { name?: string; symbol?: string; amountUsd?: number; weightPct: number };
              return [`${pct(Number(v))}${p.amountUsd != null ? ` · ${usd(p.amountUsd)}` : ""}`, p.symbol ? `${p.symbol} ${p.name ?? ""}` : p.name];
            }}
            contentStyle={{ borderRadius: 10, border: "1px solid #e5e5e5", fontSize: 13 }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
