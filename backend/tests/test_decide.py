import ast
import asyncio
import re

import numpy as np
import pandas as pd
import pytest

from app import decide, engine, llm
from app.models import Candidate, Finding, IntakeInput

SYMS = ["NVDA", "MSFT", "AAPL", "BTC-USD", "ETH-USD", "GLD"]
CANDS = [
    Candidate(symbol="NVDA", name="NVIDIA", assetClass="STOCKS", financialScore=8, sentimentScore=0.3),
    Candidate(symbol="MSFT", name="Microsoft", assetClass="STOCKS", financialScore=9, sentimentScore=0.2),
    Candidate(symbol="AAPL", name="Apple", assetClass="STOCKS", financialScore=8, sentimentScore=0.1),
    Candidate(symbol="BTC-USD", name="Bitcoin", assetClass="CRYPTO", sentimentScore=0.4),
    Candidate(symbol="ETH-USD", name="Ethereum", assetClass="CRYPTO", sentimentScore=0.2),
    Candidate(symbol="GLD", name="Gold", assetClass="GOLD"),
]
INTAKE = IntakeInput(amountUsd=10000, horizonYears=2, riskTolerance="MODERATE")


def finding(i, sym, sentiment="NEGATIVE", verified=None):
    return Finding(id=f"f_{i}", agent="NEWS", symbol=sym, claim="c", sentiment=sentiment, sources=[],
                   numbersVerified=verified)


FINDINGS = [finding(i, s) for i, s in enumerate(["NVDA", "MSFT", "AAPL", "BTC-USD", "ETH-USD"])]


def closes():
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2022-01-07", periods=n, freq="W-FRI")
    vol = {"NVDA": .03, "MSFT": .025, "AAPL": .025, "BTC-USD": .16, "ETH-USD": .16, "GLD": .01, "SPY": .02, "AGG": .005}
    cols = {s: 100 * np.cumprod(1 + rng.normal(.006 if v < .1 else .01, v, n)) for s, v in vol.items()}
    return pd.DataFrame(cols, index=idx)


@pytest.fixture
def fake(monkeypatch):
    cl = closes()
    monkeypatch.setattr(decide.data, "prices", lambda syms, years=5: (cl[[s for s in syms if s in cl]], []))
    state = {"orch_inputs": [], "calls": 0}

    def holdings(w):
        return {"assetClasses": [{"assetClass": "STOCKS", "weightPct": 0, "rationale": "s"}],
                "holdings": [{"symbol": s, "weightPct": p, "rationale": "r", "findingIds": ["f_0", "bogus"]}
                             for s, p in w.items()]}

    R1 = {"NVDA": 10, "MSFT": 5, "AAPL": 5, "BTC-USD": 25, "ETH-USD": 25, "GLD": 25, "SPY": 0}
    R2 = {"NVDA": 20, "MSFT": 20, "AAPL": 20, "BTC-USD": 8, "ETH-USD": 7, "GLD": 25}

    async def ask(*, model, input, schema=None, tools=(), agent, on_step=None):
        state["calls"] += 1
        if schema is decide.OrchestratorOut:
            text = input if isinstance(input, str) else input[0]["text"]
            state["orch_inputs"].append(text)
            data = decide.OrchestratorOut.model_validate(holdings(R2 if "changesRequested" in text and "Cut" in text else R1))
        elif schema is decide.VerifierOut:
            data = decide.VerifierOut(changesRequested=["Cut crypto below 25%: drawdown too deep"])
        else:  # sandbox
            w = ast.literal_eval(re.search(r"\{'.*?\}", input[0]["text"]).group(0))
            c = cl[list(w)]
            m = engine.metrics(engine.equity_curve(c, w), c, w)
            if state.get("sandbox_broken"):
                return llm.AskResult("", "no json here", [], model, 1, [])
            return llm.AskResult("", f'```json\n{{"totalReturnPct": {m["totalReturnPct"]}, '
                                 f'"maxDrawdownPct": {m["maxDrawdownPct"]}}}\n```', [], model, 1, [])
        return llm.AskResult(data, "", [], model, 1, [])

    monkeypatch.setattr(decide.llm, "ask", ask)
    return state


def run(cands=CANDS, findings=FINDINGS):
    events = []

    async def emit(e):
        events.append(e)

    rounds = asyncio.run(decide.decide(INTAKE, cands, findings, emit))
    return rounds, events


def test_largest_remainder_sums_to_100():
    rng = np.random.default_rng(0)
    for _ in range(200):
        raw = list(rng.random(rng.integers(1, 9)) * 100 + 0.01)
        out = decide.normalise(raw)
        assert round(sum(out), 1) == 100.0 and all(round(x, 1) == x for x in out)


def test_round1_rejected_on_drawdown_round2_approved(fake):
    rounds, _ = run()
    assert [r.round for r in rounds] == [1, 2]
    v1, v2 = rounds[0].verification, rounds[1].verification
    assert not v1.isApproved and not next(c for c in v1.checks if c.name.startswith("Max drawdown")).isPassed
    assert v1.changesRequested
    assert "changesRequested" in fake["orch_inputs"][1] and "Cut crypto" in fake["orch_inputs"][1]
    assert v2.isApproved, [c for c in v2.checks if not c.isPassed]
    assert fake["calls"] <= 5  # r1: orch+sandbox+verifier, r2: orch+sandbox (verifier skipped when approved)


def test_unknown_finding_ids_dropped_and_scores_copied(fake):
    rounds, _ = run()
    nv = next(h for s in rounds[0].allocation.slices for h in s.holdings if h.symbol == "NVDA")
    assert nv.findingIds == ["f_0"] and nv.financialScore == 8 and nv.amountUsd == nv.weightPct * 100
    assert all(sum(h.weightPct for h in s.holdings) == pytest.approx(s.weightPct) for s in rounds[0].allocation.slices)


def test_events_in_order_with_round(fake):
    _, events = run()
    types = [(e["type"], e.get("agent"), e.get("round")) for e in events]
    order = [t for t in types if t[0] in ("allocation.proposed", "backtest.completed", "verification.completed")]
    assert order == [(t, None, r) for r in (1, 2) for t in
                     ("allocation.proposed", "backtest.completed", "verification.completed")]
    assert types[0][:2] == ("agent.started", "ORCHESTRATOR")
    assert ("agent.completed", "VERIFIER", None) in types


def test_sandbox_failure_gives_failed_check_without_crash(fake):
    fake["sandbox_broken"] = True
    rounds, _ = run()
    bt, v = rounds[0].backtest, rounds[0].verification
    assert bt.crossCheck.isMatch is False
    assert not next(c for c in v.checks if "cross" in c.name.lower()).isPassed


def test_missing_bear_case_fails_check(fake):
    rounds, _ = run(findings=[f for f in FINDINGS if f.symbol != "NVDA"])
    c = next(c for c in rounds[0].verification.checks if "bear" in c.name.lower())
    assert not c.isPassed and "NVDA" in c.detail


def test_cap_projection_54pct_gld():
    out = decide.project([54.0, 25.0, 12.0, 5.0, 2.0, 2.0])
    assert max(out) <= 25.0 and round(sum(out), 1) == 100.0 and out[1] == 25.0


def test_cap_projection_infeasible_leaves_weights():
    assert decide.project([60.0, 40.0]) == [60.0, 40.0]  # < 4 holdings: cap cannot sum to 100


def test_round2_prompt_has_failed_checks_and_constraints(fake):
    run()
    p1, p2 = fake["orch_inputs"]
    for p in (p1, p2):
        assert "25" in p and "effective" in p.lower() and "-35" in p
    assert "Max drawdown" in p2 and "Backtest max drawdown" in p2 and "smallest changes" in p2
    assert "BTC-USD 25.0%" in p2  # round-1 allocation included


def test_round2_prompt_has_computed_crypto_cap(fake):
    run()
    p2 = fake["orch_inputs"][1]
    m = re.search(r"total CRYPTO at or below ([\d.]+)%.*?drawdown (?:of )?≈ (-[\d.]+)%", p2)
    assert m and float(m.group(1)) < 75 and float(m.group(2)) >= -35


def test_cap_projection_zero_weight_free_slots():
    out = decide.project([60.0, 40.0, 0.0, 0.0])
    assert max(out) <= 25.0 and round(sum(out), 1) == 100.0
