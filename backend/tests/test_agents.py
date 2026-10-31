import asyncio
from itertools import count
from types import SimpleNamespace

from app import agents, data, llm
from app.models import Candidate, IntakeInput

INTAKE = IntakeInput(amountUsd=10000, horizonYears=2, riskTolerance="MODERATE", theme="AI")
SRC = [{"title": f"t{i}", "url": f"http://x/{i}"} for i in range(5)]


def fake_ask(payload):
    async def ask(**kw):
        return SimpleNamespace(data=payload, sources=SRC)
    return ask


def run(c):
    return asyncio.run(c)


def test_discovery_drops_unresolved_and_adds_gld(monkeypatch):
    out = agents.DiscoveryOut(picks=[
        agents.Pick(symbol="NVDA", name="Nvidia", assetClass="STOCKS", thesis="t"),
        agents.Pick(symbol="FAKE", name="Fake", assetClass="STOCKS", thesis="t"),
        agents.Pick(symbol="BTC-USD", name="Bitcoin", assetClass="CRYPTO", thesis="t")])
    monkeypatch.setattr(llm, "ask", fake_ask(out))
    monkeypatch.setattr(data, "resolve", lambda s: s != "FAKE")
    notes = []
    cands, findings = run(agents.discovery(INTAKE.model_copy(update={"riskTolerance": "AGGRESSIVE"}), count(1), on_step=notes.append))
    assert [c.symbol for c in cands] == ["NVDA", "BTC-USD", "GLD"]
    assert cands[-1].assetClass == "GOLD"
    assert any("FAKE" in n["message"] for n in notes)
    assert len({f.id for f in findings}) == len(findings)
    assert all(len(f.sources) <= 3 for f in findings)
    assert [f.symbol for f in findings] == ["NVDA", "BTC-USD", "GLD"]  # exactly 1 per symbol


def test_number_gate():
    f = {"revenue": 302.969e9, "net_income": 75e9, "trailingPE": 28.5}
    assert agents.number_gate("Revenue: $303B", f) is True
    assert agents.number_gate("Strong. Revenue: $250B, P/E: 28.5x", f) is False
    assert agents.number_gate("Healthy balance sheet", f) is None


def test_financials_gate_flags_bad_revenue(monkeypatch):
    out = agents.FinancialsOut(stocks=[
        agents.StockScore(symbol="AAPL", score=8, verdict="Solid.", keyFigures="Revenue: $250B", risk="Slowing."),
        agents.StockScore(symbol="MSFT", score=6, verdict="Fine.", keyFigures="Revenue: $303B", risk="Capex.")])
    monkeypatch.setattr(llm, "ask", fake_ask(out))
    monkeypatch.setattr(data, "fundamentals", lambda s: {"revenue": 302.969e9})
    cands = [Candidate(symbol=s, name=s, assetClass="STOCKS") for s in ("AAPL", "MSFT")]
    scores, fs = run(agents.financials(cands, count(1)))
    assert scores["AAPL"][0] == 8.0
    by = {(f.symbol, f.sentiment): f for f in fs}
    assert by[("AAPL", "POSITIVE")].numbersVerified is False
    assert by[("MSFT", "NEUTRAL")].numbersVerified is True
    assert by[("MSFT", "NEGATIVE")].numbersVerified is None


def test_research_agents_unique_ids(monkeypatch):
    raws = [agents.RawFinding(symbol="NVDA", claim="c", sentiment="NEGATIVE")]
    ids = count(1)
    monkeypatch.setattr(llm, "ask", fake_ask(agents.ResearchOut(findings=raws)))
    a, b = run(agents.landscape(["NVDA"], ids)), run(agents.news(["NVDA"], ids))
    monkeypatch.setattr(llm, "ask", fake_ask(agents.SocialOut(findings=raws, scores=[agents.SymbolSentiment(symbol="nvda", score=0.4)])))
    sc, c = run(agents.social(["NVDA"], ids))
    assert sc == {"NVDA": 0.4}
    all_ids = [f.id for f in a + b + c]
    assert all_ids == ["f_1", "f_2", "f_3"] and a[0].agent == "LANDSCAPE" and c[0].agent == "SOCIAL"


def test_cap_per_symbol_keeps_one_negative_and_two_max():
    R = agents.RawFinding
    raws = [R(symbol="A", claim="1", sentiment="POSITIVE"), R(symbol="A", claim="2", sentiment="NEGATIVE"),
            R(symbol="A", claim="3", sentiment="NEGATIVE"), R(symbol="A", claim="4", sentiment="NEUTRAL"),
            R(symbol="B", claim="5", sentiment="POSITIVE")]
    out = agents.cap_per_symbol(raws)
    assert [r.claim for r in out] == ["1", "2", "5"]
    assert sum(r.sentiment == "NEGATIVE" for r in out if r.symbol == "A") == 1
    assert len(agents.cap_per_symbol(raws, 1)) == 2


def test_agent_enforces_cap_trims_claims_and_ranks_sources(monkeypatch):
    R = agents.RawFinding
    long_claim = "risks " + " ".join(["word"] * 40)
    raws = [R(symbol="NVDA", claim=long_claim, sentiment="NEGATIVE")] + \
           [R(symbol="NVDA", claim=f"c{i}", sentiment="POSITIVE") for i in range(4)]
    cites = [{"title": "Unrelated", "url": "http://u/1"}, {"title": "Nvidia risks", "url": "http://u/2"}]
    res = SimpleNamespace(data=agents.ResearchOut(findings=raws), sources=cites, citations=cites)

    async def ask(**kw):
        assert kw["generation_config"] == {"thinking_level": "low"}
        return res
    monkeypatch.setattr(llm, "ask", ask)
    fs = run(agents.news(["NVDA"], count(1)))
    assert len(fs) == 2 and len(fs[0].claim.split()) <= 26
    assert fs[0].sources[0].url.endswith("/2") and len(fs[0].sources) <= 3


def _prices(monkeypatch):
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2021-01-01", periods=100, freq="W-FRI")
    rng = np.random.default_rng(1)
    vols = {"BRK-B": .03, "JNJ": .01, "PG": .015, "KO": .02, "PEP": .03, "WMT": .04, "COST": .05, "MCD": .06, "V": .07, "MRK": .08}
    df = pd.DataFrame({k: 100 * np.cumprod(1 + rng.normal(0, v, 100)) for k, v in vols.items()}, index=idx)
    monkeypatch.setattr(data, "prices", lambda syms, years=5: (df, []))
    monkeypatch.setattr(data, "resolve", lambda s: True)


def _discover(monkeypatch, risk, picks=("NVDA",)):
    out = agents.DiscoveryOut(picks=[agents.Pick(symbol=s, name=s, assetClass="STOCKS", thesis="t") for s in picks])
    monkeypatch.setattr(llm, "ask", fake_ask(out))
    return run(agents.discovery(INTAKE.model_copy(update={"riskTolerance": risk}), count(1)))


def test_moderate_adds_two_lowest_vol_defensives(monkeypatch):
    _prices(monkeypatch)
    cands, fs = _discover(monkeypatch, "MODERATE")
    assert [c.symbol for c in cands] == ["NVDA", "JNJ", "PG", "GLD"]
    assert all(c.assetClass == "STOCKS" for c in cands[1:3])
    ballast = [f for f in fs if f.symbol == "JNJ"][0]
    assert "defensive ballast" in ballast.claim and "MODERATE" in ballast.claim and "vol" in ballast.claim


def test_aggressive_adds_no_defensives(monkeypatch):
    _prices(monkeypatch)
    cands, _ = _discover(monkeypatch, "AGGRESSIVE")
    assert [c.symbol for c in cands] == ["NVDA", "GLD"]


def test_defensive_not_duplicated_when_already_picked(monkeypatch):
    _prices(monkeypatch)
    cands, _ = _discover(monkeypatch, "CONSERVATIVE", picks=("NVDA", "JNJ"))
    syms = [c.symbol for c in cands]
    assert syms.count("JNJ") == 1 and syms == ["NVDA", "JNJ", "PG", "KO", "GLD"]
