"""Pipeline interface. `run_pipeline` is what the real Gemini pipeline will implement later.

This is the FAKE: it replays app/fixtures/report.example.json as a realistic event sequence.
"""
import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

from .models import IntakeInput, Report

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


async def run_pipeline(run_id: str, intake: IntakeInput, emit: Emit) -> Report:
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
