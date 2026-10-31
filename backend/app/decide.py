"""ORCHESTRATOR + VERIFIER: propose allocation -> backtest -> hard checks -> (rebalance once)."""
import asyncio
import ast
import inspect
import json
import math
import re
import logging
import time

import pandas as pd
from pydantic import BaseModel

from backtest.risk_xray import compute_risk_xray

from . import config, data, engine, llm
from .models import (Allocation, AssetClass, Backtest, Candidate, Check, Finding, IntakeInput, Preferences,
                     Round, Verification)

SANDBOX_TIMEOUT_S = 45
SANDBOX_STYLE = " Write one short script; do not iterate or explain; print only the JSON."
log = logging.getLogger("app.decide")
ORDER = ["STOCKS", "BONDS", "CRYPTO", "GOLD"]
RISKY = {"STOCKS", "CRYPTO"}  # only these can be capped by the drawdown limit; bonds/gold only help
BOND_RANGE = {"CONSERVATIVE": "30-50%", "MODERATE": "10-30%", "AGGRESSIVE": "0-10%"}


# ---------- LLM schemas ----------
class ClassOut(BaseModel):
    assetClass: AssetClass
    weightPct: float
    rationale: str


class HoldingOut(BaseModel):
    symbol: str
    weightPct: float
    rationale: str
    findingIds: list[str]


class OrchestratorOut(BaseModel):
    assetClasses: list[ClassOut]
    holdings: list[HoldingOut]


class VerifierOut(BaseModel):
    changesRequested: list[str]


ORCH_PROMPT = """You are a senior cross-asset allocator building ONE portfolio from the research below.
Risk tolerance: {risk}. Horizon: {years} years. Amount: ${amount:,.0f}. Theme: {theme}.

HARD CONSTRAINTS (the Verifier rejects the portfolio if any is violated):
- Holding weights are % of the whole portfolio and sum to exactly 100.
- Max single holding: {maxh}% (no exceptions, gold included).
- Effective number of positions (1 / sum of squared weights) >= 4: use at least 5 holdings, none dominant.
- Backtested max drawdown no worse than {dd}% ({risk} limit); crypto is the main drawdown driver.
- At least 1 holding in every asset class you use (STOCKS/BONDS/CRYPTO/GOLD).
- Bonds are ballast. Typical BONDS weight for a {risk} profile: {bonds} (AGGRESSIVE: 0-10% unless the user
  targets bonds). Short-duration treasuries (e.g. SHY) cushion drawdowns best; long bonds fell together with
  stocks in 2022. The {maxh}% per-holding cap still applies, so 50% bonds needs at least 2 bond ETFs.
  If the drawdown check fails, add short-duration bonds before cutting anything else.
- Gold is a hedge: typically 5-20% (above 25% is never allowed).
- Every non-gold holding needs a documented bear case (a NEGATIVE finding); never cite a finding whose
  numbersVerified is False.

Output: per asset class a weight and a rationale; per holding a weight, a 1-3 sentence rationale weighing
bull AND bear evidence, and findingIds (only ids from the list below). Use only the candidate symbols. Start
from the risk-parity baseline and tilt using financial/sentiment scores.

Candidates (financialScore 1-10, sentimentScore -1..1):
{cands}

Findings (id | symbol | sentiment | numbersVerified | claim):
{findings}

Risk-parity baseline weights (%): {baseline}
{prefs}{feedback}"""

PREFS_PROMPT = """
User preferences (verbatim): {prefs}
Preference rules (risk limits WIN over preferences, but never silently - explain in rationales):
- every mustInclude symbol gets at least 1% weight;
- never hold a symbol listed in exclude;
- keep each asset class within +/-5 percentage points of assetClassTargets, UNLESS that would break the
  drawdown limit above; then get as close to the target as the limit allows;
- follow the styles and notes.
"""

VERIFIER_PROMPT = """You are the Head of Risk. A proposed portfolio failed these deterministic checks:
{failed}

Backtest metrics: {metrics}
Portfolio (weight %): {weights}
{extra}
Write 1-4 concrete, actionable changesRequested for the allocator, each naming the asset class or symbol to
change, the direction and a target, and quoting the number that failed
(e.g. "Cut crypto below 25%: max drawdown -41.3% exceeds MODERATE limit -35%")."""


# ---------- helpers ----------
def normalise(raw: list[float]) -> list[float]:
    """Largest-remainder rounding to 0.1 so the result sums to exactly 100.0."""
    total = sum(raw)
    units = [max(x, 0) / total * 1000 for x in raw]
    out = [math.floor(u) for u in units]
    for i in sorted(range(len(raw)), key=lambda i: units[i] - out[i], reverse=True)[: 1000 - sum(out)]:
        out[i] += 1
    return [o / 10 for o in out]


def project(w: list[float], cap: float | None = None) -> list[float]:
    """Cap each weight at `cap`, redistributing the excess proportionally to the uncapped (water-filling)."""
    cap = config.MAX_HOLDING_PCT if cap is None else cap
    if len(w) * cap < 100 or max(w) <= cap:
        return w  # nothing to do, or infeasible (fewer than 100/cap holdings); the Verifier flags it
    w = list(w)
    for _ in range(len(w)):
        over = [i for i, x in enumerate(w) if x > cap]
        if not over:
            break
        excess = sum(w[i] - cap for i in over)
        free = [i for i, x in enumerate(w) if x < cap]
        base = sum(w[i] for i in free)
        for i in over:
            w[i] = cap
        for i in free:
            w[i] += excess * (w[i] / base if base > 0 else 1 / len(free))
    return normalise(w)


def _enforce_prefs(hs: list["HoldingOut"], cands: dict[str, Candidate], prefs: Preferences | None
                   ) -> tuple[list["HoldingOut"], list[float]]:
    """Drop excluded symbols, force every resolvable mustInclude to >= 1%, re-cap, re-round."""
    excl = {x.upper() for x in prefs.exclude} if prefs else set()
    hs = [h for h in hs if h.symbol.upper() not in excl]
    must = [m for m in (prefs.mustInclude if prefs else []) if m in cands and m.upper() not in excl]
    for m in must:
        if all(h.symbol != m for h in hs):
            hs.append(HoldingOut(symbol=m, weightPct=0, rationale="Included at the user's request.", findingIds=[]))
    w = normalise([h.weightPct for h in hs])
    low = [i for i, h in enumerate(hs) if h.symbol in must and w[i] < 1.0]
    if low:
        rest = [i for i in range(len(hs)) if i not in low]
        scale = (100 - 1.05 * len(low)) / (sum(w[i] for i in rest) or 1)
        w = [1.05 if i in low else (w[i] * scale if rest else w[i]) for i in range(len(hs))]
        # ponytail: forced weight is 1.05 so it still rounds to >= 1.0 after largest-remainder rounding
    return hs, project(normalise(w))


def _build_allocation(out: OrchestratorOut, cands: dict[str, Candidate], findings: list[Finding],
                      baseline: dict[str, float], amount: float, prefs: Preferences | None = None) -> Allocation:
    ids = {f.id for f in findings}
    seen: dict[str, HoldingOut] = {}
    for h in out.holdings:
        if h.symbol in cands and h.weightPct > 0 and h.symbol not in seen:
            seen[h.symbol] = h
    if not seen:  # ponytail: LLM returned nothing usable -> fall back to the risk-parity baseline
        seen = {s: HoldingOut(symbol=s, weightPct=w, rationale="Risk-parity baseline.", findingIds=[])
                for s, w in baseline.items()}
    # ponytail: constraint projection (preferences + max-holding cap) so a bad LLM proposal still satisfies
    # the hard limits; the Verifier still checks everything, this only removes the most common failures.
    hs, weights = _enforce_prefs(list(seen.values()), cands, prefs)
    cls_rat = {c.assetClass: c.rationale for c in out.assetClasses}
    slices = []
    for ac in ORDER:
        rows = [(h, w) for h, w in zip(hs, weights) if cands[h.symbol].assetClass == ac and w > 0]
        if not rows:
            continue
        holdings = [{
            "symbol": h.symbol, "name": cands[h.symbol].name, "weightPct": w,
            "amountUsd": round(w * amount / 100, 2), "rationale": h.rationale,
            "findingIds": [i for i in h.findingIds if i in ids],
            "financialScore": cands[h.symbol].financialScore, "sentimentScore": cands[h.symbol].sentimentScore,
            "isUserRequested": bool(getattr(cands[h.symbol], "isUserRequested", False)),
        } for h, w in rows]
        slices.append({"assetClass": ac, "weightPct": round(sum(w for _, w in rows), 1),
                       "rationale": cls_rat.get(ac, ""), "holdings": holdings})
    return Allocation.model_validate({"slices": slices})


def _class_weights(alloc: Allocation) -> dict[str, float]:
    return {s.assetClass: s.weightPct for s in alloc.slices}


def _pref_check(prefs: Preferences | None, alloc: Allocation, cands: dict[str, Candidate], dd: float,
                limit: float) -> Check:
    name = "User preferences respected"
    if not prefs or not (prefs.mustInclude or prefs.exclude or prefs.assetClassTargets):
        return Check(name=name, isPassed=True, detail="No hard preferences set")
    held = {h.symbol.upper(): h.weightPct for s in alloc.slices for h in s.holdings}
    ok, bad = [], []
    for m in (m for m in prefs.mustInclude if m in cands):
        (ok if held.get(m.upper(), 0) >= 1.0 else bad).append(
            f"{m} held at {held[m.upper()]}%" if held.get(m.upper(), 0) >= 1.0 else f"{m} missing")
    for x in prefs.exclude:
        if x.upper() in held:
            bad.append(f"excluded {x} is held")
    cw = _class_weights(alloc)
    near = dd <= limit + 5  # drawdown at or near its limit: a lower risky-class weight is explained by it
    for ac, t in (prefs.assetClassTargets or {}).items():
        a = cw.get(ac, 0.0)
        if abs(a - t) <= 5:
            ok.append(f"{ac} {a:g}% within 5pp of {t:g}% target")
        elif a < t and near and ac in RISKY:
            ok.append(f"{ac} capped at {a:g}% vs {t:g}% target by the drawdown limit "
                      f"({dd:.1f}% vs {limit:g}%)")
        else:
            bad.append(f"{ac} {a:g}% vs {t:g}% target (outside 5pp)")
    return Check(name=name, isPassed=not bad,
                 detail="; ".join(bad + ok) if bad else ("Honoured: " + "; ".join(ok)) if ok else "Nothing to check")


def preference_notes(intake: IntakeInput, rounds: list[Round]) -> list[str]:
    """Human-readable account of how each preference was honoured or capped (for Report.preferenceNotes)."""
    prefs = intake.preferences
    if not prefs or not rounds:
        return []
    last = rounds[-1]
    held = {h.symbol: h.weightPct for s in last.allocation.slices for h in s.holdings}
    limit = config.DRAWDOWN_LIMITS[intake.riskTolerance]
    risk = intake.riskTolerance.title()
    notes = []
    for m in prefs.mustInclude:
        notes.append(f"{m} included at {held[m]}% as requested" if m in held
                     else f"{m} could not be included (no usable price history)")
    for x in prefs.exclude:
        notes.append(f"{x} excluded as requested" if x not in held else f"{x} could not be excluded")
    cw = _class_weights(last.allocation)
    for ac, t in (prefs.assetClassTargets or {}).items():
        a = cw.get(ac, 0.0)
        label = ac.title()
        if abs(a - t) <= 5:
            notes.append(f"{label} target {t:g}% met at {a:g}%")
            continue
        tried = next((r for r in rounds if _class_weights(r.allocation).get(ac, 0.0) > a
                      and r.backtest.metrics["PORTFOLIO"].maxDrawdownPct < limit), None)
        if a < t and ac in RISKY:
            why = f"a higher {ac.lower()} weight breaks your {risk} {limit:g}% drawdown limit"
            if tried:
                why += (f" (round {tried.round} tried {_class_weights(tried.allocation)[ac]:g}%: "
                        f"{tried.backtest.metrics['PORTFOLIO'].maxDrawdownPct:g}%)")
            notes.append(f"{label} target {t:g}% → {a:g}%: {why}")
        else:
            notes.append(f"{label} target {t:g}% → {a:g}%: " + ("above target" if a > t else "below target"))
    bonds = [h for s in last.allocation.slices if s.assetClass == "BONDS" for h in s.holdings]
    if bonds:
        notes.append(f"Bonds {cw.get('BONDS', 0.0):g}% ({', '.join(h.symbol for h in bonds)}): added as ballast to "
                     f"keep your {risk} drawdown within {limit:g}%")
    return notes


def _find_json(blob: str) -> dict | None:
    for m in reversed(re.findall(r"\{[^{}]*totalReturnPct[^{}]*\}", blob)):
        for parse in (json.loads, ast.literal_eval):
            try:
                d = parse(m)
                return {"totalReturnPct": float(d["totalReturnPct"]), "maxDrawdownPct": float(d["maxDrawdownPct"])}
            except Exception:
                pass
    return None


def _parse_sandbox(res: llm.AskResult) -> dict:
    """Trust executed-code output first; model prose only counts if code actually ran."""
    ran = any(s.get("type") == "code_execution_call" or s.get("code") for s in res.steps)
    for s in res.steps:
        if s.get("type") == "code_execution_result" and (d := _find_json(str(s.get("result", "")))):
            return d
    if not ran:
        raise ValueError("sandbox did not execute code")
    if d := _find_json(res.text):
        return d
    raise ValueError("no {totalReturnPct, maxDrawdownPct} JSON in sandbox output")


def _sandbox_progress(emit, clock=time.monotonic):
    """on_step callback: human messages, one per distinct message, at most one per second."""
    last = {"msg": None, "t": -1e9}

    async def on_step(step: dict) -> None:
        typ = step.get("type")
        msg = ("Sandbox: executing Python" if typ == "code_execution_call"
               else "Sandbox: result received" if typ == "code_execution_result"
               else "Sandbox: writing backtest code")
        now = clock()
        if msg == last["msg"] or now - last["t"] < 1.0:
            return
        last.update(msg=msg, t=now)
        await _emit(emit, {"type": "agent.progress", "agent": "VERIFIER", "message": msg})

    return on_step


def _checks(alloc: Allocation, bt: Backtest, closes: pd.DataFrame, findings: list[Finding],
            risk: str, sandbox_note: str = "", prefs: Preferences | None = None,
            cands: dict[str, Candidate] | None = None) -> list[Check]:
    hs = [h for s in alloc.slices for h in s.holdings]
    limit = config.DRAWDOWN_LIMITS[risk]
    dd = bt.metrics["PORTFOLIO"].maxDrawdownPct
    total = round(sum(h.weightPct for h in hs), 1)
    top = max(hs, key=lambda h: h.weightPct)
    cc, ci = bt.crossCheck, bt.sharpeCi
    w = engine._weights(alloc)
    en = compute_risk_xray(closes, w, periods_per_year=engine.PPY, min_history=2)["concentration"]["effective_n"]
    bears = {f.symbol for f in findings if f.sentiment == "NEGATIVE"}
    no_bear = [h.symbol for h in hs if h.symbol != "GLD" and h.symbol not in bears]
    by_id = {f.id: f for f in findings}
    unverified = sorted({i for h in hs for i in h.findingIds if by_id[i].numbersVerified is False})
    C = Check
    return [
        C(name=f"Max drawdown within {risk} limit ({limit:g}%)", isPassed=dd >= limit,
          detail=f"Backtest max drawdown {dd:.1f}% vs limit {limit:g}%"),
        C(name="Weights sum to 100%", isPassed=total == 100.0, detail=f"Holdings sum to {total}%"),
        C(name=f"No holding above {config.MAX_HOLDING_PCT:g}%", isPassed=top.weightPct <= config.MAX_HOLDING_PCT,
          detail=f"Largest holding is {top.symbol} at {top.weightPct}%"),
        C(name="Sandbox cross-check matches", isPassed=cc.isMatch,
          detail=f"Sandbox return {cc.sandboxTotalReturnPct}% vs independent {cc.independentTotalReturnPct}%"
                 + ("" if cc.isMatch else f" ({sandbox_note or 'mismatch or sandbox unavailable'})")),
        C(name="Sharpe 95% CI above zero", isPassed=bool(ci and ci.lower > 0),
          detail=f"Sharpe CI lower bound {ci.lower}" if ci else "Sharpe CI unavailable"),
        C(name="Diversification (effective N >= 4)", isPassed=bool(en and en >= 4),
          detail=f"Effective number of positions {en:.2f}" if en else "Effective N unavailable"),
        C(name="Bear case documented for every holding", isPassed=not no_bear,
          detail="Every holding has a negative finding" if not no_bear else f"No bear case for: {', '.join(no_bear)}"),
        C(name="No holding cites unverified numbers", isPassed=not unverified,
          detail="All cited numbers verified" if not unverified else f"Unverified findings cited: {', '.join(unverified)}"),
        _pref_check(prefs, alloc, cands or {}, dd, limit),
    ]


def _crypto_cap(weights: dict[str, float], cands: dict[str, Candidate], closes: pd.DataFrame,
                limit: float, hi: float | None = None) -> tuple[float, float] | None:
    """Largest total CRYPTO % (crypto holdings scaled together, rest scaled up, cap applied) whose backtested
    max drawdown still meets `limit`. Deterministic bisection on the engine; None if even 0% crypto fails."""
    syms = list(weights)
    crypto = [s for s in syms if cands[s].assetClass == "CRYPTO"]
    cur = sum(weights[s] for s in crypto)
    if not crypto or cur <= 0 or len(crypto) == len(syms):
        return None

    def dd_at(c: float) -> tuple[list[float], float]:
        w = [weights[s] * (c / cur if s in crypto else (100 - c) / (100 - cur)) for s in syms]
        w = project(w)
        cw = {s: x / 100 for s, x in zip(syms, w)}
        curve = engine.equity_curve(closes, cw)
        return w, float((curve / curve.cummax() - 1).min() * 100)

    if dd_at(0.0)[1] < limit:
        return None
    lo, hi = 0.0, max(cur, hi or 0.0)
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dd_at(mid)[1] >= limit else (lo, mid)
    w, dd = dd_at(lo)
    return round(sum(x for s, x in zip(syms, w) if s in crypto), 1), round(dd, 1)


def _gen(level: str) -> dict:
    ok = "generation_config" in inspect.signature(llm.ask).parameters
    return {"generation_config": {"thinking_level": level}} if ok else {}


async def _ask(**kw):
    t = time.perf_counter()
    r = await llm.ask(**kw)
    log.info("call agent=%s model=%s %.1fs", kw["agent"], kw["model"], time.perf_counter() - t)
    return r


async def _emit(emit, event: dict) -> None:
    r = emit(event)
    if inspect.isawaitable(r):
        await r


# ---------- public API ----------
async def decide(intake: IntakeInput, candidates: list[Candidate], findings: list[Finding], emit) -> list[Round]:
    syms = [c.symbol for c in candidates]
    closes, missing = await asyncio.to_thread(data.prices, syms + ["SPY", "AGG"])
    cands = {c.symbol: c for c in candidates if c.symbol not in missing and c.symbol in closes}
    if not cands:
        raise ValueError("no candidate has price history")
    baseline = engine.risk_parity(closes, list(cands))
    ids = {f.id for f in findings}
    risk = intake.riskTolerance
    fmt = lambda x: f"{x:.1f}" if isinstance(x, float) else "n/a"
    cand_txt = "\n".join(f"{c.symbol} | {c.name} | {c.assetClass} | financial {fmt(c.financialScore)} | "
                         f"sentiment {fmt(c.sentimentScore)}" for c in cands.values())
    find_txt = "\n".join(f"{f.id} | {f.symbol} | {f.sentiment} | {f.numbersVerified} | {f.claim}" for f in findings)
    base_txt = ", ".join(f"{s} {w * 100:.1f}" for s, w in baseline.items())

    prefs = intake.preferences
    prefs_txt = PREFS_PROMPT.format(prefs=prefs.model_dump_json(exclude_none=True)) if prefs else ""
    rounds: list[Round] = []
    feedback = ""
    for n in range(1, config.MAX_ROUNDS + 1):
        # --- ORCHESTRATOR ---
        t0 = time.perf_counter()
        await _emit(emit, {"type": "agent.started", "agent": "ORCHESTRATOR", "model": config.MODEL_ARGON,
                           "task": f"Build allocation (round {n})"})
        prompt = ORCH_PROMPT.format(risk=risk, years=intake.horizonYears, amount=intake.amountUsd,
                                    theme=intake.theme or "none", dd=config.DRAWDOWN_LIMITS[risk],
                                    maxh=config.MAX_HOLDING_PCT, bonds=BOND_RANGE[risk], cands=cand_txt, findings=find_txt or "(none)",
                                    baseline=base_txt, prefs=prefs_txt, feedback=feedback)
        res = await _ask(model=config.MODEL_ARGON, input=prompt, schema=OrchestratorOut, agent="ORCHESTRATOR",
                         **_gen("medium"))
        alloc = _build_allocation(res.data, cands, findings, {s: w * 100 for s, w in baseline.items()},
                                  intake.amountUsd, prefs)
        await _emit(emit, {"type": "allocation.proposed", "round": n, "allocation": alloc.model_dump()})
        await _emit(emit, {"type": "agent.completed", "agent": "ORCHESTRATOR", "summary": f"Round {n} allocation",
                           "durationMs": int((time.perf_counter() - t0) * 1000)})

        # --- VERIFIER ---
        t0 = time.perf_counter()
        await _emit(emit, {"type": "agent.started", "agent": "VERIFIER", "model": config.MODEL_ARGON,
                           "task": f"Backtest and verify (round {n})"})

        on_step = _sandbox_progress(emit)
        note = ""

        async def sandbox(df: pd.DataFrame, w: dict) -> dict:
            nonlocal note
            await on_step({"type": "writing"})
            sig4 = df.apply(lambda col: col.map(lambda x: float(f"{x:.4g}")))  # cheaper prompt, same numbers to ~0.05%
            try:
                r = await asyncio.wait_for(
                    _ask(model=config.MODEL_FLASH,
                         input=[llm.text_part(engine.sandbox_prompt(w) + SANDBOX_STYLE), llm.csv_part(sig4)],
                         tools=["code_execution"], agent="VERIFIER", on_step=on_step, **_gen("low")),
                    SANDBOX_TIMEOUT_S)
                return _parse_sandbox(r)
            except asyncio.TimeoutError:
                note = f"sandbox timed out after {SANDBOX_TIMEOUT_S:g}s; independent numbers shown"
                await _emit(emit, {"type": "agent.progress", "agent": "VERIFIER", "message": note.capitalize()})
                raise
            except Exception as e:
                note = str(e) if isinstance(e, ValueError) else f"sandbox failed: {e!r}"
                raise

        await _emit(emit, {"type": "agent.progress", "agent": "VERIFIER", "message": "Running backtest in sandbox"})
        bt = await engine.run_backtest(alloc, closes, sandbox)
        await _emit(emit, {"type": "backtest.completed", "round": n, "backtest": bt.model_dump()})

        checks = _checks(alloc, bt, closes, findings, risk, note, prefs, cands)
        failed = [c for c in checks if not c.isPassed]
        changes: list[str] = []
        if failed:
            await _emit(emit, {"type": "agent.progress", "agent": "VERIFIER",
                               "message": f"{len(failed)} check(s) failed; drafting requested changes"})
            weights = {h.symbol: h.weightPct for s in alloc.slices for h in s.holdings}
            t_cry = (prefs.assetClassTargets or {}).get("CRYPTO") if prefs else None
            cur_cry = _class_weights(alloc).get("CRYPTO", 0.0)
            limit_dd = config.DRAWDOWN_LIMITS[risk]
            cap, up_hint = None, ""
            if not checks[0].isPassed:
                cap = _crypto_cap(weights, cands, closes, limit_dd)
            elif t_cry is not None and cur_cry < t_cry - 5:  # drawdown headroom: the target can be approached
                up = _crypto_cap(weights, cands, closes, limit_dd, hi=t_cry)
                if up and up[0] > cur_cry:
                    up_hint = (f"Computed by the backtest engine: you have drawdown headroom. Total CRYPTO can go up "
                               f"to about {up[0]}% (max drawdown ≈ {up[1]}%, limit {limit_dd:g}%) toward the user's "
                               f"{t_cry:g}% target; move to about {up[0]}%, not beyond.\n")
            target_msg = ""
            if cap and t_cry is not None and cur_cry > cap[0]:
                target_msg = (f"Your {t_cry:g}% CRYPTO target exceeds what a {risk.title()} profile allows; "
                              f"reduce to ~{cap[0]:g}%.")
            try:
                v = await _ask(
                    model=config.MODEL_ARGON, schema=VerifierOut, agent="VERIFIER", **_gen("low"),
                    input=VERIFIER_PROMPT.format(failed="\n".join(f"- {c.name}: {c.detail}" for c in failed),
                                                 metrics=bt.metrics["PORTFOLIO"].model_dump(), weights=weights,
                                                 extra=up_hint + (f"User preferences: {prefs.model_dump_json(exclude_none=True)}\n"
                                                        "If the drawdown failure is caused by a user preference "
                                                        "(e.g. a class target), say so explicitly.\n") if prefs else ""))
                changes = v.data.changesRequested
            except Exception:
                pass
            changes = changes or [f"{c.name}: {c.detail}" for c in failed]  # LLM only explains; never blocks
            if target_msg:
                changes = [target_msg] + [c for c in changes if c != target_msg]
        verification = Verification(isApproved=not failed, checks=checks, changesRequested=changes)
        await _emit(emit, {"type": "verification.completed", "round": n, "verification": verification.model_dump()})
        await _emit(emit, {"type": "agent.completed", "agent": "VERIFIER",
                           "summary": "approved" if not failed else f"rejected: {len(failed)} check(s) failed",
                           "durationMs": int((time.perf_counter() - t0) * 1000)})

        rounds.append(Round(round=n, allocation=alloc, backtest=bt, verification=verification))
        if verification.isApproved:
            break
        m = bt.metrics["PORTFOLIO"]
        hint = up_hint
        if cap:
            hint += (f"Computed by the backtest engine: with total CRYPTO at or below {cap[0]}% (crypto holdings "
                    f"scaled down together, the rest scaled up), max drawdown ≈ {cap[1]}%. Keep total "
                    f"CRYPTO at or below {cap[0]}%.\n")
        feedback = (
            f"\nROUND {n + 1}. The Verifier rejected your previous proposal.\n"
            f"Previous allocation (% weights): {', '.join(f'{h.symbol} {h.weightPct}%' for s in alloc.slices for h in s.holdings)}\n"
            f"Previous backtest: return {m.totalReturnPct}%, max drawdown {m.maxDrawdownPct}%, "
            f"volatility {m.volatilityPct}%, sharpe {m.sharpe}\n"
            f"{hint}Failed checks:\n" + "\n".join(f"- {c.name}: {c.detail}" for c in failed) + "\n"
            "changesRequested:\n" + "\n".join(f"- {c}" for c in changes) + "\n"
            "Fix exactly the failed checks listed; keep every other constraint satisfied; make the smallest "
            "changes that fix them (adjust the previous allocation, do not rewrite it).")
    return rounds


async def summarize(intake: IntakeInput, rounds: list[Round], findings: list[Finding]) -> str:
    last = rounds[-1]
    pos = [f"{h.symbol} {h.weightPct}%" for s in last.allocation.slices for h in s.holdings]
    m, spy = last.backtest.metrics["PORTFOLIO"], last.backtest.metrics["SPY"]
    prompt = (
        "Write a 3-5 sentence executive summary (markdown ok, no headings) of this portfolio for the investor. "
        "Mention the main positions, backtested return/drawdown vs SPY, and whether the Verifier approved it "
        "(and if round 1 was rejected, what changed). Be honest about risks; no financial-advice promises.\n"
        f"Intake: {intake.model_dump()}\nHoldings: {', '.join(pos)}\n"
        f"Portfolio: {m.model_dump()}\nSPY: {spy.model_dump()}\n"
        f"Rounds: {[(r.round, r.verification.isApproved, r.verification.changesRequested) for r in rounds]}\n"
        f"Findings: {len(findings)} collected")
    res = await _ask(model=config.MODEL_FLASH, input=prompt, agent="ORCHESTRATOR", **_gen("low"))
    return res.text.strip()
