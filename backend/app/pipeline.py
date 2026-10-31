"""Pipeline interface. `run_pipeline` is what the real Gemini pipeline will implement later.

This is the FAKE: it replays app/fixtures/report.example.json as a realistic event sequence.
"""
import asyncio
import importlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

from itertools import count

from . import agents
from .models import Candidate, IntakeInput, Report

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
        msg = step.get("message") or (f"Searching: {step['query']}" if step.get("query")
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


async def research(intake: IntakeInput, emit: Emit) -> tuple[list[Candidate], list]:
    """Research phase: DISCOVERY, then FINANCIALS/LANDSCAPE/NEWS/SOCIAL in parallel. Returns (candidates, findings)."""
    ids = count(1)
    cands, findings = await _run_agent(emit, "DISCOVERY", FLASH, lambda cb: agents.discovery(intake, ids, cb))
    stocks = [c for c in cands if c.assetClass == "STOCKS"]
    syms = [c.symbol for c in cands]

    async def pack(coro):  # normalise to (result, findings)
        r = await coro
        return (None, r) if isinstance(r, list) else r

    results = await asyncio.gather(
        _run_agent(emit, "FINANCIALS", FLASH, lambda cb: agents.financials(stocks, ids, cb)),
        _run_agent(emit, "LANDSCAPE", FLASH, lambda cb: pack(agents.landscape(syms, ids, cb))),
        _run_agent(emit, "NEWS", FLASH, lambda cb: pack(agents.news(syms, ids, cb))),
        _run_agent(emit, "SOCIAL", FLASH, lambda cb: agents.social(syms, ids, cb)),
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


async def run_real(run_id: str, intake: IntakeInput, emit: Emit) -> Report:
    # import_module (not `from . import`) so it honours sys.modules patches even after the real module was imported
    decide = importlib.import_module("app.decide")

    await emit({"type": "run.started", "agents": ["DISCOVERY", *PARALLEL, "ORCHESTRATOR", "VERIFIER"]})
    cands, findings = await research(intake, emit)
    rounds = await decide.decide(intake, cands, findings, emit)
    summary = await decide.summarize(intake, rounds, findings)
    rounds = [r if isinstance(r, dict) else r.model_dump() for r in rounds]
    return Report.model_validate({
        "runId": run_id, "generatedAt": datetime.now(timezone.utc).isoformat(), "intake": intake.model_dump(),
        "summary": summary, "finalAllocation": rounds[-1]["allocation"], "rounds": rounds,
        "findings": [f.model_dump() for f in findings], "disclaimer": DISCLAIMER})


async def run_pipeline(run_id: str, intake: IntakeInput, emit: Emit) -> Report:
    impl = run_real if os.environ.get("PIPELINE", "fake") == "real" else run_fake
    return await impl(run_id, intake, emit)
