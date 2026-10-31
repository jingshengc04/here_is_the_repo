import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from app import agents, pipeline
from app.models import Candidate, Finding, IntakeInput, Report

INTAKE = IntakeInput(amountUsd=10000, horizonYears=2, riskTolerance="MODERATE", theme="AI")
ROUNDS = json.loads(pipeline.FIXTURE.read_text())["report"]["rounds"]


def F(ids, agent, sym, sent):
    return Finding(id=f"f_{next(ids)}", agent=agent, symbol=sym, claim="c", sentiment=sent, sources=[])


@pytest.fixture
def fakes(monkeypatch):
    seen = {}

    async def discovery(intake, ids, on_step=None, **kw):
        await on_step({"query": "AI stocks"})
        return [Candidate(symbol="NVDA", name="Nvidia", assetClass="STOCKS"),
                Candidate(symbol="GLD", name="Gold", assetClass="GOLD")], [F(ids, "DISCOVERY", "NVDA", "POSITIVE")]

    async def financials(stocks, ids, on_step=None, **kw):
        return {"NVDA": (8.0, "good")}, [F(ids, "FINANCIALS", "NVDA", "POSITIVE")]

    async def landscape(syms, ids, on_step=None, **kw):
        return [F(ids, "LANDSCAPE", "NVDA", "NEUTRAL")]

    async def news(syms, ids, on_step=None, **kw):
        return [F(ids, "NEWS", "NVDA", "NEGATIVE")]

    async def social(syms, ids, on_step=None, **kw):
        return {"NVDA": 0.5}, [F(ids, "SOCIAL", "NVDA", "POSITIVE")]

    for fn in (discovery, financials, landscape, news, social):
        monkeypatch.setattr(agents, fn.__name__, fn)

    async def decide(intake, cands, findings, emit):
        seen["cands"] = cands
        await emit({"type": "allocation.proposed", "round": 1, "allocation": ROUNDS[0]["allocation"]})
        return ROUNDS

    async def summarize(intake, rounds, findings):
        return "summary"

    monkeypatch.setitem(sys.modules, "app.decide", SimpleNamespace(decide=decide, summarize=summarize))
    monkeypatch.setenv("PIPELINE", "real")
    return seen


def go():
    events = []

    async def emit(ev):
        events.append(ev)

    report = asyncio.run(pipeline.run_pipeline("run_x", INTAKE, emit))
    return report, events


def test_real_flow_order_and_report(fakes):
    report, events = go()
    assert isinstance(report, Report) and report.runId == "run_x" and report.summary == "summary"
    assert report.finalAllocation == report.rounds[-1].allocation and len(report.findings) == 5
    types = [(e["type"], e.get("agent")) for e in events]
    assert types[0] == ("run.started", None)
    assert types.index(("agent.completed", "DISCOVERY")) < types.index(("agent.started", "FINANCIALS"))
    assert types.index(("agent.completed", "SOCIAL")) < types.index(("allocation.proposed", None))
    assert ("agent.progress", "DISCOVERY") in types
    nvda = next(c for c in fakes["cands"] if c.symbol == "NVDA")
    assert nvda.financialScore == 8.0
    assert nvda.sentimentScore == -0.25  # mean(social 0.5, news NEGATIVE -1.0)


def test_failing_agent_does_not_stop_run(fakes, monkeypatch):
    async def bad(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(agents, "news", bad)
    report, events = go()
    assert ("agent.failed", "NEWS") in [(e["type"], e.get("agent")) for e in events]
    assert isinstance(report, Report) and all(f.agent != "NEWS" for f in report.findings)


def test_preference_notes_from_decide_plus_discovery_notes(fakes, monkeypatch):
    async def discovery(intake, ids, on_step=None, notes_out=None, **kw):
        notes_out.append("Requested ZZZ could not be resolved")
        return [Candidate(symbol="NVDA", name="Nvidia", assetClass="STOCKS", isUserRequested=True)], []

    monkeypatch.setattr(agents, "discovery", discovery)
    sys.modules["app.decide"].preference_notes = lambda intake, rounds: ["Honoured MEME style"]
    report, _ = go()
    assert report.preferenceNotes == ["Honoured MEME style", "Requested ZZZ could not be resolved"]
    assert fakes["cands"][0].isUserRequested is True


def test_preference_notes_default_empty_when_decide_has_none(fakes):
    report, _ = go()
    assert report.preferenceNotes == []


def test_resolver_runs_once_prefs_flow_to_agents_and_decide(fakes, monkeypatch):
    from app import preferences
    from app.models import Filter, Preferences, ResolvedPreferences
    rp = ResolvedPreferences(exclude=Filter(sectors=["Technology"]), interpretations=["'no tech' -> excluding sector Technology"])
    calls, seen = [], {}

    async def resolve(intake):
        calls.append(1)
        return rp
    monkeypatch.setattr(preferences, "resolve", resolve)

    async def discovery(intake, ids, on_step=None, notes_out=None, rp=None, **kw):
        seen["discovery_rp"] = rp
        return [Candidate(symbol="NVDA", name="Nvidia", assetClass="STOCKS")], []

    async def news(syms, ids, on_step=None, notes=None, **kw):
        seen["news_notes"] = notes
        return []
    monkeypatch.setattr(agents, "discovery", discovery)
    monkeypatch.setattr(agents, "news", news)

    async def decide(intake, cands, findings, emit, prefs=None):
        seen["decide_prefs"] = prefs
        return ROUNDS
    sys.modules["app.decide"].decide = decide
    sys.modules["app.decide"].preference_notes = lambda intake, rounds: ["decide note"]

    events = []

    async def emit(ev):
        events.append(ev)
    intake = INTAKE.model_copy(update={"preferences": Preferences(exclude=["tech"], notes="my note")})
    report = asyncio.run(pipeline.run_pipeline("run_x", intake, emit))
    assert calls == [1] and seen["discovery_rp"] is rp and seen["decide_prefs"] is rp
    assert "my note" in seen["news_notes"] and "Technology" in seen["news_notes"]
    assert report.preferenceNotes[:2] == ["'no tech' -> excluding sector Technology", "decide note"]
    prog = [e for e in events if e["type"] == "agent.progress" and "Understanding your preferences" in e["message"]]
    assert prog and prog[0]["agent"] == "DISCOVERY" and events[0]["type"] == "run.started" and events.index(prog[0]) == 1


def test_resolver_failure_never_fails_the_run_but_is_disclosed(fakes, monkeypatch):
    from app import preferences

    async def boom(intake):
        raise RuntimeError("gemini down")
    monkeypatch.setattr(preferences, "resolve", boom)
    report, _ = go()
    assert report.preferenceNotes and "not applied" in report.preferenceNotes[0]
