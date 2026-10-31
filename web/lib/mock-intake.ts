// Scripted stand-in for the Intake agent + the backend's IntakeInput validation.
// Pure and type-only imports: also used by scripts/fake-backend.ts under Node type stripping.
import type { ChatMessage, IntakeInput, IntakeTurn, Preferences, RiskTolerance } from "./types";

type Draft = IntakeTurn["intake"];

function parseAmount(text: string): number | null {
  const m = text.match(/\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|thousand|m|mm|million)?\b/i);
  if (!m) return null;
  let n = Number(m[1].replace(/,/g, ""));
  const unit = m[2]?.toLowerCase();
  if (unit === "k" || unit === "thousand") n *= 1_000;
  if (unit === "m" || unit === "mm" || unit === "million") n *= 1_000_000;
  // A bare small number ("2") is more likely years than dollars.
  return n >= 100 ? Math.round(n) : null;
}

const WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
const toNum = (w: string) => (WORDS.includes(w.toLowerCase()) ? WORDS.indexOf(w.toLowerCase()) + 1 : Number(w));
const quarter = (y: number) => Math.min(10, Math.max(0.25, Math.round(y * 4) / 4));

/** Years in quarter steps (0.25–10): "2 years", "18 months", "6 mo". */
function parseYears(text: string): number | null {
  const n = "(\\d+(?:\\.\\d+)?|" + WORDS.join("|") + ")";
  const months = text.match(new RegExp(`${n}\\s*(?:-|\\s)?(?:months?|mos?)\\b`, "i"));
  if (months) return quarter(toNum(months[1]) / 12);
  const years = text.match(new RegExp(`${n}\\s*(?:-|\\s)?(?:years?|yrs?|y)\\b`, "i"));
  return years ? quarter(toNum(years[1])) : null;
}

/** Very small stand-in for the backend's preference matcher: "no X", "include TICKER", "only X". */
function parsePreferences(text: string, prev: Preferences | null): Preferences | null {
  const p: Preferences = { mustInclude: [...(prev?.mustInclude ?? [])], exclude: [...(prev?.exclude ?? [])], onlyInclude: [...(prev?.onlyInclude ?? [])] };
  for (const m of text.matchAll(/\b(?:no|avoid|exclude|without)\s+([a-z][a-z &-]{1,30}?)(?=[,.;!]|\s+(?:and|but|or)\b|$)/gi)) p.exclude!.push(m[1].trim());
  for (const m of text.matchAll(/\b(?:include|add|must have)\s+([A-Z]{1,5}(?:-USD)?)\b/g)) p.mustInclude!.push(m[1]);
  for (const m of text.matchAll(/\bonly\s+([a-z][a-z &-]{1,30}?)(?=[,.;!]|$)/gi)) p.onlyInclude!.push(m[1].trim());
  const dedupe = (xs?: string[]) => [...new Set(xs)];
  const out = { ...p, mustInclude: dedupe(p.mustInclude), exclude: dedupe(p.exclude), onlyInclude: dedupe(p.onlyInclude) };
  return out.mustInclude.length || out.exclude.length || out.onlyInclude.length ? out : prev;
}

function parseRisk(text: string): RiskTolerance | null {
  if (/\b(aggressive|high[- ]risk|yolo|max(imum)? growth)\b/i.test(text)) return "AGGRESSIVE";
  if (/\b(conservative|low[- ]risk|safe|preserve|can'?t afford to lose)\b/i.test(text)) return "CONSERVATIVE";
  if (/\b(moderate|balanced|medium|growth|can'?t stomach)\b/i.test(text)) return "MODERATE";
  return null;
}

function parseTheme(text: string): string | null {
  const m = text.match(/\b(?:into|interested in|focus(?:ed)? on|theme(?: is)?|like|love)\s+([a-z][a-z0-9 &/-]{2,60}?)(?:[.!,;]|$)/i);
  return m ? m[1].trim() : null;
}

// Same wording as lib/format horizonLabel; duplicated so this file has no runtime imports (Node type stripping).
const horizonText = (y: number) => (y === 1 ? "1 year" : y < 1 || (y < 2 && !Number.isInteger(y)) ? `${Math.round(y * 12)} months` : `${y} years`);

const RISK_LABEL: Record<RiskTolerance, string> = {
  CONSERVATIVE: "conservative",
  MODERATE: "moderate",
  AGGRESSIVE: "aggressive",
};

export function mockIntakeTurn(messages: ChatMessage[]): IntakeTurn {
  const draft: Draft = { amountUsd: null, horizonYears: null, riskTolerance: null, theme: null, preferences: null };
  for (const m of messages) {
    if (m.role !== "user") continue;
    draft.amountUsd = parseAmount(m.content) ?? draft.amountUsd;
    draft.horizonYears = parseYears(m.content) ?? draft.horizonYears;
    draft.riskTolerance = parseRisk(m.content) ?? draft.riskTolerance;
    draft.theme = parseTheme(m.content) ?? draft.theme;
    draft.preferences = parsePreferences(m.content, draft.preferences);
  }

  const isComplete = draft.amountUsd !== null && draft.horizonYears !== null && draft.riskTolerance !== null;
  const known = [
    draft.amountUsd !== null && `$${draft.amountUsd.toLocaleString("en-US")}`,
    draft.horizonYears !== null && horizonText(draft.horizonYears),
    draft.riskTolerance !== null && `${RISK_LABEL[draft.riskTolerance]} risk`,
  ].filter(Boolean);
  const got = known.length ? `Got it: ${known.join(", ")}. ` : "";

  let reply: string;
  if (draft.amountUsd === null) reply = `${got}How much would you like to invest, roughly?`;
  else if (draft.horizonYears === null) reply = `${got}How long can you leave it invested, 1 year or more like 5?`;
  else if (draft.riskTolerance === null)
    reply = `${got}How would you feel about a 30% drop along the way: fine, uneasy, or not at all?`;
  else
    reply = `Great: **${known.join(", ")}**${draft.theme ? `, focused on ${draft.theme}` : ""}. Check the card and press **Run** when it looks right.`;

  const p = draft.preferences;
  const saved = p
    ? [
        ...(p.onlyInclude ?? []).map((x) => `only ${x}`),
        ...(p.exclude ?? []).map((x) => `no ${x}`),
        ...(p.mustInclude?.length ? [`include ${p.mustInclude.join(", ")}`] : []),
      ]
    : [];
  if (saved.length) reply = `${reply} Saved: ${saved.join(" · ")}`;

  return { reply, intake: draft, isComplete };
}

/** Mirrors the backend's 422 rules. Returns a {field: message} map, empty when valid. */
export function validateIntake(input: Partial<IntakeInput>): Record<string, string> {
  const errors: Record<string, string> = {};
  const { amountUsd, horizonYears, riskTolerance, theme, preferences: p } = input;
  if (typeof amountUsd !== "number" || !Number.isFinite(amountUsd) || amountUsd < 100 || amountUsd > 10_000_000)
    errors.amountUsd = "Must be between 100 and 10,000,000";
  if (typeof horizonYears !== "number" || !Number.isInteger(horizonYears * 4) || horizonYears < 0.25 || horizonYears > 10)
    errors.horizonYears = "Pick 3 months to 10 years, in steps of 3 months";
  if (riskTolerance !== "CONSERVATIVE" && riskTolerance !== "MODERATE" && riskTolerance !== "AGGRESSIVE")
    errors.riskTolerance = "Pick Conservative, Moderate or Aggressive";
  if (theme !== undefined && (typeof theme !== "string" || theme.length > 200)) errors.theme = "At most 200 characters";
  if (p) {
    const t = Object.values(p.assetClassTargets ?? {}) as number[];
    const words = [...(p.mustInclude ?? []), ...(p.exclude ?? []), ...(p.onlyInclude ?? [])];
    if ((p.mustInclude?.length ?? 0) > 10) errors.preferences = "At most 10 must-include symbols";
    else if ((p.exclude?.length ?? 0) > 20) errors.preferences = "At most 20 exclusions";
    else if ((p.onlyInclude?.length ?? 0) > 10) errors.preferences = "At most 10 only-include entries";
    else if (t.some((v) => v < 0 || v > 100) || t.reduce((a, b) => a + b, 0) > 100)
      errors.preferences = "assetClassTargets must each be 0–100 and sum to at most 100";
    else if (words.some((w) => w.length > 40)) errors.preferences = "symbols/words must be at most 40 chars";
    else if ((p.notes?.length ?? 0) > 500) errors.preferences = "Notes are at most 500 characters";
  }
  return errors;
}

/** Mirrors the backend's 422 rules for POST /api/intake/messages. */
export function validateMessages(messages: unknown): string | null {
  if (!Array.isArray(messages) || messages.length === 0) return "messages must be a non-empty array";
  if (messages.length > 20) return "At most 20 messages per request";
  for (const m of messages as ChatMessage[]) {
    if ((m?.role !== "user" && m?.role !== "assistant") || typeof m.content !== "string") return "Each message needs role and content";
    if (m.content.length > 2000) return "Each message is at most 2,000 characters";
  }
  return null;
}
