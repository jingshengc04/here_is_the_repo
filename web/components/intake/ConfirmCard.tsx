"use client";

import { useState } from "react";
import { Loader2Icon, XIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CLASS_LABEL, horizonLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AssetClass, IntakeInput, IntakeTurn, Preferences, RiskTolerance, Style } from "@/lib/types";

const RISKS: { value: RiskTolerance; label: string; limit: string }[] = [
  { value: "CONSERVATIVE", label: "Conservative", limit: "−20%" },
  { value: "MODERATE", label: "Moderate", limit: "−35%" },
  { value: "AGGRESSIVE", label: "Aggressive", limit: "−55%" },
];
const PRESETS = [0.25, 0.5, 1, 2, 3, 5, 10];
const STYLES: { value: Style; label: string }[] = [
  { value: "GROWTH", label: "Growth" },
  { value: "VALUE", label: "Value" },
  { value: "DIVIDEND", label: "Dividend" },
  { value: "MOMENTUM", label: "Momentum" },
  { value: "DEFENSIVE", label: "Defensive" },
  { value: "ESG", label: "ESG" },
  { value: "MEME", label: "Meme" },
];
const CLASSES: AssetClass[] = ["STOCKS", "BONDS", "GOLD", "CRYPTO"];

interface Props {
  initial: (IntakeTurn["intake"] & { preferences?: Preferences | null }) | null;
  errors: Record<string, string>;
  submitting: boolean;
  onRun: (intake: IntakeInput) => void;
  title?: string;
}

const FieldError = ({ msg, id }: { msg?: string; id: string }) =>
  msg ? (
    <p id={id} role="alert" className="mt-1 text-sm font-medium text-no-edge">
      {msg}
    </p>
  ) : null;

/** Editable chip list: Enter or comma adds, × removes. */
function ChipInput({
  label,
  hint,
  values,
  onChange,
  max,
  upper,
  placeholder,
}: {
  label: string;
  hint: string;
  values: string[];
  onChange: (v: string[]) => void;
  max: number;
  upper?: boolean;
  placeholder: string;
}) {
  const [text, setText] = useState("");
  const add = () => {
    const v = (upper ? text.toUpperCase() : text).trim().slice(0, 40);
    if (v && !values.includes(v) && values.length < max) onChange([...values, v]);
    setText("");
  };
  return (
    <div>
      <div className="text-sm font-semibold">
        {label} <span className="font-normal text-slate">{hint}</span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-1.5 rounded-lg border-2 border-input bg-card px-2 py-1.5 focus-within:border-ink">
        {values.map((v) => (
          <span key={v} className="flex items-center gap-1 rounded-full bg-ink/[0.06] py-0.5 pr-1 pl-2.5 text-sm">
            {v}
            <button type="button" aria-label={`Remove ${v}`} onClick={() => onChange(values.filter((x) => x !== v))} className="rounded-full p-0.5 hover:bg-ink/10">
              <XIcon className="size-3" />
            </button>
          </span>
        ))}
        <input
          aria-label={label}
          value={text}
          disabled={values.length >= max}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              add();
            } else if (e.key === "Backspace" && !text && values.length) onChange(values.slice(0, -1));
          }}
          onBlur={add}
          placeholder={values.length >= max ? `Up to ${max}` : placeholder}
          className="h-7 min-w-28 flex-1 bg-transparent px-1 text-[15px] outline-none"
        />
      </div>
    </div>
  );
}

const hasPrefs = (p?: Preferences | null) =>
  !!p &&
  !!(p.mustInclude?.length || p.exclude?.length || p.onlyInclude?.length || p.styles?.length || p.notes || Object.keys(p.assetClassTargets ?? {}).length);

export function ConfirmCard({ initial, errors, submitting, onRun, title = "Confirm your plan" }: Props) {
  const p0 = initial?.preferences ?? null;
  const [amount, setAmount] = useState(initial?.amountUsd != null ? String(initial.amountUsd) : "");
  const [horizon, setHorizon] = useState<number | null>(initial?.horizonYears ?? null);
  const [risk, setRisk] = useState<RiskTolerance | null>(initial?.riskTolerance ?? null);
  const [theme, setTheme] = useState(initial?.theme ?? "");
  const [mustInclude, setMustInclude] = useState<string[]>(p0?.mustInclude ?? []);
  const [exclude, setExclude] = useState<string[]>(p0?.exclude ?? []);
  const [onlyInclude, setOnlyInclude] = useState<string[]>(p0?.onlyInclude ?? []);
  const [styles, setStyles] = useState<Style[]>(p0?.styles ?? []);
  const [targets, setTargets] = useState<Partial<Record<AssetClass, string>>>(
    Object.fromEntries(Object.entries(p0?.assetClassTargets ?? {}).map(([k, v]) => [k, String(v)])),
  );
  const [notes, setNotes] = useState(p0?.notes ?? "");
  const [personalise, setPersonalise] = useState(hasPrefs(p0) || !!errors.preferences);

  const amountNum = Number(amount.replace(/[$,\s]/g, ""));
  const ready = amount.trim() !== "" && horizon !== null && risk !== null;
  const targetSum = Object.values(targets).reduce((t, v) => t + (Number(v) || 0), 0);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!ready || submitting) return;
    // Optional fields are omitted rather than sent as null / "" / [].
    const intake: IntakeInput = { amountUsd: amountNum, horizonYears: horizon!, riskTolerance: risk! };
    if (theme.trim()) intake.theme = theme.trim();
    const t = Object.fromEntries(Object.entries(targets).filter(([, v]) => v !== "" && v !== undefined).map(([k, v]) => [k, Number(v)]));
    const prefs: Preferences = {
      ...(mustInclude.length && { mustInclude }),
      ...(exclude.length && { exclude }),
      ...(onlyInclude.length && { onlyInclude }),
      ...(styles.length && { styles }),
      ...(Object.keys(t).length && { assetClassTargets: t }),
      ...(notes.trim() && { notes: notes.trim() }),
    };
    if (Object.keys(prefs).length) intake.preferences = prefs;
    onRun(intake);
  };

  return (
    <form data-testid="confirm-card" onSubmit={submit} className="space-y-5 rounded-xl border-2 border-ink bg-card p-5 shadow-sm">
      <h2 className="text-lg font-semibold">{title}</h2>

      <div>
        <label htmlFor="amount" className="text-sm font-semibold">Amount (USD)</label>
        <div className="mt-1 flex items-center rounded-lg border-2 border-input bg-card px-3 focus-within:border-ink">
          <span className="text-slate">$</span>
          <input
            id="amount"
            inputMode="numeric"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            aria-invalid={!!errors.amountUsd}
            aria-describedby="amount-err"
            placeholder="10,000"
            className="h-10 w-full bg-transparent px-2 text-base outline-none"
          />
        </div>
        <FieldError id="amount-err" msg={errors.amountUsd} />
      </div>

      <fieldset>
        <legend className="text-sm font-semibold">
          How long you can leave it invested <span className="font-normal text-slate">{horizon ? horizonLabel(horizon) : ""}</span>
        </legend>
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {PRESETS.map((y) => (
            <button
              key={y}
              type="button"
              aria-pressed={horizon === y}
              aria-label={horizonLabel(y)}
              onClick={() => setHorizon(y)}
              className={cn(
                "h-9 rounded-md border-2 px-3 text-sm font-semibold transition-colors",
                horizon === y ? "border-ink bg-ink text-white" : "border-input hover:border-slate",
              )}
            >
              {y < 1 ? `${y * 12} mo` : `${y} yr`}
            </button>
          ))}
        </div>
        <input
          type="range"
          aria-label="Horizon in years"
          min={0.25}
          max={10}
          step={0.25}
          value={horizon ?? 1}
          onChange={(e) => setHorizon(Number(e.target.value))}
          className="mt-2 w-full accent-ink"
        />
        <FieldError id="horizon-err" msg={errors.horizonYears} />
      </fieldset>

      <fieldset>
        <legend className="text-sm font-semibold">Risk tolerance</legend>
        <div className="mt-1.5 grid grid-cols-3 overflow-hidden rounded-lg border-2 border-input">
          {RISKS.map((r) => (
            <button
              key={r.value}
              type="button"
              aria-pressed={risk === r.value}
              onClick={() => setRisk(r.value)}
              className={cn("py-2 text-sm font-semibold transition-colors", risk === r.value ? "bg-ink text-white" : "hover:bg-ink/[0.04]")}
            >
              {r.label}
            </button>
          ))}
        </div>
        <p className="mt-1 text-sm text-slate">
          Largest drop the Verifier allows:{" "}
          {RISKS.map((r, i) => (
            <span key={r.value} className={risk === r.value ? "font-semibold text-ink" : undefined}>
              {i > 0 && ", "}
              {r.limit}
            </span>
          ))}
        </p>
        <FieldError id="risk-err" msg={errors.riskTolerance} />
      </fieldset>

      <div>
        <label htmlFor="theme" className="text-sm font-semibold">
          Theme <span className="font-normal text-slate">(optional)</span>
        </label>
        <div className="mt-1 flex items-center rounded-lg border-2 border-input bg-card px-3 focus-within:border-ink">
          <input id="theme" value={theme} maxLength={200} onChange={(e) => setTheme(e.target.value)} placeholder="e.g. AI infrastructure" className="h-10 w-full bg-transparent text-base outline-none" />
          <span className="shrink-0 text-xs text-slate tabular-nums">{theme.length} / 200</span>
        </div>
        <FieldError id="theme-err" msg={errors.theme} />
      </div>

      <details open={personalise} onToggle={(e) => setPersonalise((e.currentTarget as HTMLDetailsElement).open)} data-testid="preferences">
        <summary className="cursor-pointer text-sm font-semibold text-[#2f5bea] hover:underline">
          {personalise ? "Personal preferences" : hasPrefs(p0) ? "Personal preferences (from your chat)" : "Add personal preferences"}
        </summary>
        <div className="mt-3 space-y-3">
          <ChipInput label="Must include" hint="tickers, up to 10" values={mustInclude} onChange={setMustInclude} max={10} upper placeholder="NVDA, DOGE-USD…" />
          <ChipInput label="Exclude" hint="tickers or words, up to 20" values={exclude} onChange={setExclude} max={20} placeholder="TSLA, oil, tobacco…" />
          <ChipInput label="Only invest in" hint="any words, up to 10" values={onlyInclude} onChange={setOnlyInclude} max={10} placeholder="biotech, European banks…" />
          <div>
            <div className="text-sm font-semibold">Investing style</div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {STYLES.map((s) => {
                const on = styles.includes(s.value);
                return (
                  <button
                    key={s.value}
                    type="button"
                    aria-pressed={on}
                    onClick={() => setStyles(on ? styles.filter((x) => x !== s.value) : [...styles, s.value])}
                    className={cn("rounded-full border-2 px-3 py-0.5 text-sm transition-colors", on ? "border-ink bg-ink text-white" : "border-input hover:border-slate")}
                  >
                    {s.label}
                  </button>
                );
              })}
            </div>
          </div>
          <div>
            <div className="text-sm font-semibold">
              Target mix <span className="font-normal text-slate">% per asset class, leave blank for no target</span>
            </div>
            <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {CLASSES.map((c) => (
                <label key={c} className="flex items-center gap-1.5 rounded-lg border-2 border-input bg-card px-2 focus-within:border-ink">
                  <span className="text-sm text-slate">{CLASS_LABEL[c]}</span>
                  <input
                    inputMode="numeric"
                    aria-label={`${CLASS_LABEL[c]} target %`}
                    value={targets[c] ?? ""}
                    onChange={(e) => setTargets({ ...targets, [c]: e.target.value.replace(/[^\d.]/g, "") })}
                    className="h-9 w-full min-w-0 bg-transparent text-right text-[15px] outline-none"
                  />
                  <span className="text-sm text-slate">%</span>
                </label>
              ))}
            </div>
            {targetSum > 100 && <p className="mt-1 text-sm text-no-edge">Targets add up to {targetSum}%; keep them at 100% or less.</p>}
          </div>
          <div>
            <label htmlFor="notes" className="text-sm font-semibold">
              Anything else <span className="font-normal text-slate">{notes.length} / 500</span>
            </label>
            <textarea
              id="notes"
              rows={2}
              maxLength={500}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g. companies I use every day"
              className="mt-1 w-full resize-none rounded-lg border-2 border-input bg-card px-3 py-2 text-[15px] outline-none focus:border-ink"
            />
          </div>
        </div>
      </details>
      <FieldError id="prefs-err" msg={errors.preferences} />

      <div className="flex justify-end">
        <Button type="submit" disabled={!ready || submitting || targetSum > 100} className="h-12! px-7! text-base font-semibold">
          {submitting ? <Loader2Icon className="animate-spin" /> : "▶"} Run, build my portfolio
        </Button>
      </div>
    </form>
  );
}
