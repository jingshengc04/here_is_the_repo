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
from .models import (Allocation, AssetClass, Backtest, Candidate, Check, Finding, IntakeInput, Round,
                     Verification)

log = logging.getLogger("app.decide")
ORDER = ["STOCKS", "CRYPTO", "GOLD"]


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
- At least 1 holding in every asset class you use (STOCKS/CRYPTO/GOLD).
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
{feedback}"""

VERIFIER_PROMPT = """You are the Head of Risk. A proposed portfolio failed these deterministic checks:
{failed}

Backtest metrics: {metrics}
Portfolio (weight %): {weights}

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
            w[i] += excess * w[i] / base
    return normalise(w)


def _build_allocation(out: OrchestratorOut, cands: dict[str, Candidate], findings: list[Finding],
                      baseline: dict[str, float], amount: float) -> Allocation:
    ids = {f.id for f in findings}
    seen: dict[str, HoldingOut] = {}
    for h in out.holdings:
        if h.symbol in cands and h.weightPct > 0 and h.symbol not in seen:
            seen[h.symbol] = h
    if not seen:  # ponytail: LLM returned nothing usable -> fall back to the risk-parity baseline
        seen = {s: HoldingOut(symbol=s, weightPct=w, rationale="Risk-parity baseline.", findingIds=[])
                for s, w in baseline.items()}
    hs = list(seen.values())
    # ponytail: constraint projection (max-holding cap) so a bad LLM proposal still satisfies the hard limit;
    # the Verifier still checks everything, this only removes the most common failure.
    weights = project(normalise([h.weightPct for h in hs]))
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
        } for h, w in rows]
        slices.append({"assetClass": ac, "weightPct": round(sum(w for _, w in rows), 1),
                       "rationale": cls_rat.get(ac, ""), "holdings": holdings})
    return Allocation.model_validate({"slices": slices})


def _parse_sandbox(res: llm.AskResult) -> dict:
    blobs = [res.text] + [str(s.get("result", "")) for s in res.steps]
    for blob in blobs:
        for m in reversed(re.findall(r"\{[^{}]*totalReturnPct[^{}]*\}", blob)):
            for parse in (json.loads, ast.literal_eval):
                try:
                    d = parse(m)
                    return {"totalReturnPct": float(d["totalReturnPct"]), "maxDrawdownPct": float(d["maxDrawdownPct"])}
                except Exception:
                    pass
    raise ValueError("no {totalReturnPct, maxDrawdownPct} JSON in sandbox output")


def _checks(alloc: Allocation, bt: Backtest, closes: pd.DataFrame, findings: list[Finding],
            risk: str) -> list[Check]:
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
                 + ("" if cc.isMatch else " (mismatch or sandbox unavailable)")),
        C(name="Sharpe 95% CI above zero", isPassed=bool(ci and ci.lower > 0),
          detail=f"Sharpe CI lower bound {ci.lower}" if ci else "Sharpe CI unavailable"),
        C(name="Diversification (effective N >= 4)", isPassed=bool(en and en >= 4),
          detail=f"Effective number of positions {en:.2f}" if en else "Effective N unavailable"),
        C(name="Bear case documented for every holding", isPassed=not no_bear,
          detail="Every holding has a negative finding" if not no_bear else f"No bear case for: {', '.join(no_bear)}"),
        C(name="No holding cites unverified numbers", isPassed=not unverified,
          detail="All cited numbers verified" if not unverified else f"Unverified findings cited: {', '.join(unverified)}"),
    ]


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

    rounds: list[Round] = []
    feedback = ""
    for n in range(1, config.MAX_ROUNDS + 1):
        # --- ORCHESTRATOR ---
        t0 = time.perf_counter()
        await _emit(emit, {"type": "agent.started", "agent": "ORCHESTRATOR", "model": config.MODEL_ARGON,
                           "task": f"Build allocation (round {n})"})
        prompt = ORCH_PROMPT.format(risk=risk, years=intake.horizonYears, amount=intake.amountUsd,
                                    theme=intake.theme or "none", dd=config.DRAWDOWN_LIMITS[risk],
                                    maxh=config.MAX_HOLDING_PCT, cands=cand_txt, findings=find_txt or "(none)",
                                    baseline=base_txt, feedback=feedback)
        res = await _ask(model=config.MODEL_ARGON, input=prompt, schema=OrchestratorOut, agent="ORCHESTRATOR",
                         **_gen("medium"))
        alloc = _build_allocation(res.data, cands, findings, {s: w * 100 for s, w in baseline.items()},
                                  intake.amountUsd)
        await _emit(emit, {"type": "allocation.proposed", "round": n, "allocation": alloc.model_dump()})
        await _emit(emit, {"type": "agent.completed", "agent": "ORCHESTRATOR", "summary": f"Round {n} allocation",
                           "durationMs": int((time.perf_counter() - t0) * 1000)})

        # --- VERIFIER ---
        t0 = time.perf_counter()
        await _emit(emit, {"type": "agent.started", "agent": "VERIFIER", "model": config.MODEL_ARGON,
                           "task": f"Backtest and verify (round {n})"})

        async def on_step(step: dict) -> None:
            await _emit(emit, {"type": "agent.progress", "agent": "VERIFIER",
                               "message": "Sandbox: running portfolio simulation" if step.get("code")
                               else f"Sandbox step: {step.get('type')}"})

        async def sandbox(df: pd.DataFrame, w: dict) -> dict:
            r = await _ask(model=config.MODEL_FLASH,
                           input=[llm.text_part(engine.sandbox_prompt(w)), llm.csv_part(df)],
                           tools=["code_execution"], agent="VERIFIER", on_step=on_step)
            return _parse_sandbox(r)

        await _emit(emit, {"type": "agent.progress", "agent": "VERIFIER", "message": "Running backtest in sandbox"})
        bt = await engine.run_backtest(alloc, closes, sandbox)
        await _emit(emit, {"type": "backtest.completed", "round": n, "backtest": bt.model_dump()})

        checks = _checks(alloc, bt, closes, findings, risk)
        failed = [c for c in checks if not c.isPassed]
        changes: list[str] = []
        if failed:
            await _emit(emit, {"type": "agent.progress", "agent": "VERIFIER",
                               "message": f"{len(failed)} check(s) failed; drafting requested changes"})
            weights = {h.symbol: h.weightPct for s in alloc.slices for h in s.holdings}
            try:
                v = await _ask(
                    model=config.MODEL_ARGON, schema=VerifierOut, agent="VERIFIER", **_gen("low"),
                    input=VERIFIER_PROMPT.format(failed="\n".join(f"- {c.name}: {c.detail}" for c in failed),
                                                 metrics=bt.metrics["PORTFOLIO"].model_dump(), weights=weights))
                changes = v.data.changesRequested
            except Exception:
                pass
            changes = changes or [f"{c.name}: {c.detail}" for c in failed]  # LLM only explains; never blocks
        verification = Verification(isApproved=not failed, checks=checks, changesRequested=changes)
        await _emit(emit, {"type": "verification.completed", "round": n, "verification": verification.model_dump()})
        await _emit(emit, {"type": "agent.completed", "agent": "VERIFIER",
                           "summary": "approved" if not failed else f"rejected: {len(failed)} check(s) failed",
                           "durationMs": int((time.perf_counter() - t0) * 1000)})

        rounds.append(Round(round=n, allocation=alloc, backtest=bt, verification=verification))
        if verification.isApproved:
            break
        m = bt.metrics["PORTFOLIO"]
        feedback = (
            f"\nROUND {n + 1}. The Verifier rejected your previous proposal.\n"
            f"Previous allocation (% weights): {', '.join(f'{h.symbol} {h.weightPct}%' for s in alloc.slices for h in s.holdings)}\n"
            f"Previous backtest: return {m.totalReturnPct}%, max drawdown {m.maxDrawdownPct}%, "
            f"volatility {m.volatilityPct}%, sharpe {m.sharpe}\n"
            "Failed checks:\n" + "\n".join(f"- {c.name}: {c.detail}" for c in failed) + "\n"
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
