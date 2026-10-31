"""Pipeline interface. `run_pipeline` is what the real Gemini pipeline will implement later.

This is the FAKE: it replays app/fixtures/report.example.json as a realistic event sequence.
"""
import asyncio
import importlib
import re
import inspect
import json
import os
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

from itertools import count

from . import agents, config, preferences
from .models import Candidate, IntakeInput, Preferences, Report, ResolvedPreferences, Round

Emit = Callable[[dict], Awaitable[None]]
FIXTURE = Path(__file__).parent / "fixtures" / "report.example.json"
FLASH, ARGON = "gemini-3.8-flash", "gemini-4-argon"
PARALLEL = ["FINANCIALS", "LANDSCAPE", "NEWS", "SOCIAL"]
TASKS = {
    "DISCOVERY": "Find candidate stocks and crypto",
    "FINANCIALS": "Score each stock from financial statements",
    "LANDSCAPE": "Map competitors and market position",
    "NEWS": "Scan recent news per holding",
    "SOCIAL": "Gauge Reddit/X sentiment",
}


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds * float(os.environ.get("FAKE_DELAY_SCALE", "1.0")))


async def _agent(emit: Emit, agent: str, findings: list[dict], model: str = FLASH, start_delay: float = 0) -> None:
    await _sleep(start_delay)
    await emit({"type": "agent.started", "agent": agent, "model": model, "task": TASKS[agent]})
    await _sleep(0.6)
    await emit({"type": "agent.progress", "agent": agent, "message": f"Searching: {TASKS[agent].lower()}"})
    for f in findings:
        await _sleep(0.7)
        await emit({"type": "agent.finding", "finding": f})
    await _sleep(0.5)
    await emit({"type": "agent.completed", "agent": agent, "summary": f"{len(findings)} findings",
                "durationMs": int(1800 + 700 * len(findings))})


async def run_fake(run_id: str, intake: IntakeInput, emit: Emit) -> Report:
    fx = json.loads(FIXTURE.read_text())["report"]
    fx.update(runId=run_id, intake=intake.model_dump(), generatedAt=datetime.now(timezone.utc).isoformat())
    by_agent: dict[str, list[dict]] = {}
    for f in fx["findings"]:
        by_agent.setdefault(f["agent"], []).append(f)

    await emit({"type": "run.started", "agents": ["DISCOVERY", *PARALLEL, "ORCHESTRATOR", "VERIFIER"]})
    await _agent(emit, "DISCOVERY", by_agent.get("DISCOVERY", []))
    await asyncio.gather(*(_agent(emit, a, by_agent.get(a, []), start_delay=0.2 * i) for i, a in enumerate(PARALLEL)))

    for rnd in fx["rounds"]:
        n = rnd["round"]
        await emit({"type": "agent.started", "agent": "ORCHESTRATOR", "model": ARGON, "task": f"Build allocation (round {n})"})
        await _sleep(1.0)
        await emit({"type": "allocation.proposed", "round": n, "allocation": rnd["allocation"]})
        await emit({"type": "agent.completed", "agent": "ORCHESTRATOR", "summary": f"Round {n} allocation", "durationMs": 1000})
        await emit({"type": "agent.started", "agent": "VERIFIER", "model": ARGON, "task": f"Backtest and verify (round {n})"})
        await _sleep(1.0)
        await emit({"type": "agent.progress", "agent": "VERIFIER", "message": "Running backtest in sandbox"})
        await _sleep(1.0)
        await emit({"type": "backtest.completed", "round": n, "backtest": rnd["backtest"]})
        await _sleep(0.6)
        await emit({"type": "verification.completed", "round": n, "verification": rnd["verification"]})
        await emit({"type": "agent.completed", "agent": "VERIFIER", "summary": "approved" if rnd["verification"]["isApproved"] else "rejected", "durationMs": 2600})
    return Report.model_validate(fx)


DISCLAIMER = ("Educational demo only. Not financial advice. Backtests apply today's weights to historical "
              "prices and do not predict future returns.")
_SENT = {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}


def _step_cb(emit: Emit, agent: str):
    async def cb(step: dict) -> None:
        msg = step.get("message") or ((step["query"] if step["query"].lower().startswith("search") else f"Searching: {step['query']}") if step.get("query")
                                      else "Running code analysis" if step.get("code") else None)
        if msg:
            await emit({"type": "agent.progress", "agent": agent, "message": msg})
    return cb


async def _run_agent(emit: Emit, agent: str, model: str, coro_fn):
    """Wrap one agent: started -> (progress via callback) -> findings -> completed. Raises on failure after agent.failed."""
    t0 = time.monotonic()
    await emit({"type": "agent.started", "agent": agent, "model": model, "task": TASKS[agent]})
    try:
        result, findings = await coro_fn(_step_cb(emit, agent))
    except Exception as e:
        await emit({"type": "agent.failed", "agent": agent, "message": f"{type(e).__name__}: {str(e)[:200]}"})
        raise
    for f in findings:
        await emit({"type": "agent.finding", "finding": f.model_dump()})
    await emit({"type": "agent.completed", "agent": agent, "summary": f"{len(findings)} findings",
                "durationMs": int((time.monotonic() - t0) * 1000)})
    return result, findings


async def research(intake: IntakeInput, emit: Emit, notes_out: list[str] | None = None,
                   rp: ResolvedPreferences | None = None) -> tuple[list[Candidate], list]:
    """Research phase: DISCOVERY, then FINANCIALS/LANDSCAPE/NEWS/SOCIAL in parallel. Returns (candidates, findings)."""
    ids = count(1)
    cands, findings = await _run_agent(emit, "DISCOVERY", FLASH, lambda cb: agents.discovery(intake, ids, cb, notes_out=notes_out, rp=rp))
    # every research agent sees the investor's notes AND what their words resolved to (so e.g. 'ethical' gets relevant bear cases)
    notes = " | ".join(x for x in (intake.preferences.notes if intake.preferences else None,
                                   preferences.prompt_text(rp) if rp else None,
                                   agents.horizon_context(intake.horizonYears)) if x) or None
    stocks = [c for c in cands if c.assetClass == "STOCKS"]
    syms = [c.symbol for c in cands]

    async def pack(coro):  # normalise to (result, findings)
        r = await coro
        return (None, r) if isinstance(r, list) else r

    results = await asyncio.gather(
        _run_agent(emit, "FINANCIALS", FLASH, lambda cb: agents.financials(stocks, ids, cb, notes=notes)),
        _run_agent(emit, "LANDSCAPE", FLASH, lambda cb: pack(agents.landscape(syms, ids, cb, notes=notes))),
        _run_agent(emit, "NEWS", FLASH, lambda cb: pack(agents.news(syms, ids, cb, notes=notes))),
        _run_agent(emit, "SOCIAL", FLASH, lambda cb: agents.social(syms, ids, cb, notes=notes)),
        return_exceptions=True)  # a failing agent already emitted agent.failed; the run continues
    ok = [None if isinstance(r, Exception) else r for r in results]
    scores = ok[0][0] if ok[0] else {}
    social = ok[3][0] if ok[3] else {}
    for r in ok:
        if r:
            findings = findings + r[1]
    news_f = ok[2][1] if ok[2] else []
    for c in cands:
        if c.symbol in scores:
            c.financialScore = scores[c.symbol][0]
        parts = [v for v in (social.get(c.symbol),) if v is not None]
        nf = [_SENT[f.sentiment] for f in news_f if f.symbol == c.symbol]
        if nf:
            parts.append(sum(nf) / len(nf))
        if parts:
            c.sentimentScore = round(sum(parts) / len(parts), 2)
    return cands, findings


_FORMAT_WORDS = {"excluded", "requested", "target", "included", "the", "and", "not", "could", "for", "with", "held", "met",
                 "from", "was", "were", "has", "have", "are", "understood"}


def _restates(note: str, interpretations: list[str]) -> bool:
    """True if `note` only repeats an interpretation (same wording, or its content words are a subset and it adds no number)."""
    norm = lambda t: re.sub(r"^understood:\s*", "", t.strip().lower())  # noqa: E731
    if norm(note) in {norm(i) for i in interpretations}:
        return True
    if [n for n in re.findall(r"\d+(?:\.\d+)?", note) if float(n) != 0]:
        return False  # numbers (weights, drawdowns) carry new information
    words = set(re.findall(r"[a-z]{3,}", note.lower())) - _FORMAT_WORDS
    return bool(words) and any(words <= set(re.findall(r"[a-z]{3,}", i.lower())) for i in interpretations)


def dedupe_notes(interpretations: list[str], *groups: list[str]) -> list[str]:
    """Interpretations first, then the other groups in order; drop restatements and case-insensitive duplicates."""
    out, seen = [], set()
    for group_i, notes in enumerate([interpretations, *groups]):
        for n in notes:
            key = n.strip().lower()
            if not key or key in seen or (group_i and _restates(n, interpretations)):
                continue
            seen.add(key)
            out.append(n)
    return out


def not_approved_notes(intake: IntakeInput, rounds: list[Round], rp: ResolvedPreferences) -> list[str]:
    """Plain-English reason per failed check of the final round (empty when approved). Generic: keyed on check wording only."""
    last = rounds[-1]
    if last.verification.isApproved:
        return []
    limit = abs(config.DRAWDOWN_LIMITS[intake.riskTolerance])
    risk = intake.riskTolerance.lower()
    short = [re.split(r"→|->", i)[-1].strip() for i in rp.interpretations][:2]
    universe = "your chosen universe" + (f" ({'; '.join(short)})" if short else f" ({intake.theme})" if intake.theme else "")
    out = []
    for c in last.verification.checks:
        if c.isPassed:
            continue
        name = c.name.lower()
        ci = last.backtest.sharpeCi
        if "sharpe" in name and ci:
            out.append(f"{universe[0].upper() + universe[1:]} hasn't shown a statistically reliable positive risk-adjusted return over the "
                       f"last 5 years (Sharpe 95% CI {ci.lower:g} to {ci.upper:g}), so this portfolio is not approved. Consider broadening it.")
        elif "drawdown" in name:
            dd = abs(last.backtest.metrics["PORTFOLIO"].maxDrawdownPct)
            out.append(f"Even after rebalancing, this mix fell {dd:g}% at its worst in the last 5 years, beyond the {limit:g}% your "
                       f"{risk} profile allows. Consider a higher risk level or fewer volatile assets.")
        elif "preference" in name:
            out.append(c.detail)
        else:
            out.append(f"{c.name}: {c.detail}")
    return out


async def run_real(run_id: str, intake: IntakeInput, emit: Emit) -> Report:
    # import_module (not `from . import`) so it honours sys.modules patches even after the real module was imported
    decide = importlib.import_module("app.decide")

    await emit({"type": "run.started", "agents": ["DISCOVERY", *PARALLEL, "ORCHESTRATOR", "VERIFIER"]})
    try:
        rp = await preferences.resolve(intake)
    except Exception as e:  # never fail the run over personalisation, but never pretend it was applied
        rp = ResolvedPreferences(interpretations=[f"Could not interpret your preferences automatically ({type(e).__name__}), so they were not applied."])
    if rp.interpretations:
        await emit({"type": "agent.progress", "agent": "DISCOVERY",
                    "message": "Understanding your preferences: " + "; ".join(rp.interpretations)[:400]})
    pref_notes: list[str] = []
    cands, findings = await research(intake, emit, pref_notes, rp)
    kw = {"prefs": rp} if "prefs" in inspect.signature(decide.decide).parameters else {}
    rounds = await decide.decide(intake, cands, findings, emit, **kw)
    rounds_m = [r if isinstance(r, Round) else Round.model_validate(r) for r in rounds]
    why_not = not_approved_notes(intake, rounds_m, rp)
    s_intake = intake
    if why_not:  # decide.summarize has no slot for this, so ride along in the intake copy it dumps into its prompt
        pr = intake.preferences or Preferences()
        extra = "Not approved because: " + " ".join(why_not)
        s_intake = intake.model_copy(update={"preferences": pr.model_copy(update={"notes": f"{pr.notes or ''} {extra}".strip()})})
    summary = await decide.summarize(s_intake, rounds, findings)
    notes_fn = getattr(decide, "preference_notes", None)
    extra = notes_fn(intake, rounds, rp) if notes_fn else []
    extra = await extra if inspect.isawaitable(extra) else extra or []
    pref_notes = [*why_not, *dedupe_notes(rp.interpretations, extra, pref_notes)]  # failures first, then understood, decide's, Discovery's
    rounds = [r if isinstance(r, dict) else r.model_dump() for r in rounds]
    return Report.model_validate({
        "runId": run_id, "generatedAt": datetime.now(timezone.utc).isoformat(), "intake": intake.model_dump(),
        "summary": summary, "finalAllocation": rounds[-1]["allocation"], "rounds": rounds,
        "findings": [f.model_dump() for f in findings], "disclaimer": DISCLAIMER,
        "preferenceNotes": pref_notes})


async def run_pipeline(run_id: str, intake: IntakeInput, emit: Emit) -> Report:
    impl = run_real if os.environ.get("PIPELINE", "fake") == "real" else run_fake
    return await impl(run_id, intake, emit)
