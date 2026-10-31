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
            if state.get("sandbox_sleep"):
                await asyncio.sleep(state["sandbox_sleep"])
            out = f'{{"totalReturnPct": {m["totalReturnPct"]}, "maxDrawdownPct": {m["maxDrawdownPct"]}}}'
            mode = state.get("sandbox_mode", "steps")
            if state.get("sandbox_broken"):
                return llm.AskResult("", "no json here", [{"type": "code_execution_call", "code": "x"}], model, 1, [])
            if mode == "text_only":  # hallucinated JSON, no code ran
                return llm.AskResult("", out, [], model, 1, [])
            steps = [{"type": "code_execution_call", "code": "print(1)"},
                     {"type": "code_execution_result", "result": out}]
            return llm.AskResult("", "done", steps, model, 1, [])
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


def _res(text="", steps=()):
    return llm.AskResult("", text, list(steps), "m", 1, [])


J = '{"totalReturnPct": 12.5, "maxDrawdownPct": -3.0}'


def test_parse_accepts_json_in_code_execution_result():
    r = _res("prose", [{"type": "code_execution_call", "code": "c"}, {"type": "code_execution_result", "result": J}])
    assert decide._parse_sandbox(r) == {"totalReturnPct": 12.5, "maxDrawdownPct": -3.0}


def test_parse_result_wins_over_prose():
    fake_json = J.replace("12.5", "99.9")
    r = _res(fake_json, [{"type": "code_execution_call", "code": "c"}, {"type": "code_execution_result", "result": J}])
    assert decide._parse_sandbox(r)["totalReturnPct"] == 12.5


def test_parse_text_ok_only_if_code_call_exists():
    ok = _res(J, [{"type": "code_execution_call", "code": "c"}])
    assert decide._parse_sandbox(ok)["totalReturnPct"] == 12.5


def test_parse_text_only_without_code_rejected():
    with pytest.raises(ValueError, match="did not execute code"):
        decide._parse_sandbox(_res(J))


def test_text_only_json_fails_cross_check_in_run(fake):
    fake["sandbox_mode"] = "text_only"
    rounds, _ = run()
    c = next(c for c in rounds[0].verification.checks if "cross" in c.name.lower())
    assert not c.isPassed and "did not execute code" in c.detail


def test_sandbox_timeout_gives_detail_and_run_continues(fake, monkeypatch):
    monkeypatch.setattr(decide, "SANDBOX_TIMEOUT_S", 0.05)
    fake["sandbox_sleep"] = 0.5
    rounds, events = run()
    assert len(rounds) == 2 and rounds[0].backtest.crossCheck.isMatch is False
    c = next(c for c in rounds[0].verification.checks if "cross" in c.name.lower())
    assert "sandbox timed out after 0.05s; independent numbers shown" in c.detail
    assert sum("timed out" in e.get("message", "") for e in events if e["type"] == "agent.progress") == 2


def test_progress_throttle_dedupes_and_rate_limits():
    sent, t = [], [0.0]

    async def emit(e):
        sent.append(e["message"])

    on_step = decide._sandbox_progress(emit, clock=lambda: t[0])

    async def go():
        for typ in ["code_execution_call", "code_execution_call", "code_execution_result"]:
            await on_step({"type": typ})  # same instant: only the first passes
        t[0] = 2.0
        await on_step({"type": "code_execution_result"})
        await on_step({"type": "code_execution_result"})  # duplicate message

    asyncio.run(go())
    assert sent == ["Sandbox: executing Python", "Sandbox: result received"]


# ---------- generic preferences ----------
from app import preferences  # noqa: E402
from app.models import Filter, ResolvedPreferences  # noqa: E402

PROFILES = {  # lower-cased Yahoo-style profile data, injected instead of hitting the network
    "NVDA": ("technology", "semiconductors", "united states"), "MSFT": ("technology", "software", "united states"),
    "AAPL": ("technology", "consumer electronics", "united states"),
    "MRNA": ("healthcare", "biotechnology", "united states"), "VRTX": ("healthcare", "biotechnology", "united states"),
    "REGN": ("healthcare", "biotechnology", "united states"), "GILD": ("healthcare", "biotechnology", "united states"),
    "BABA": ("consumer cyclical", "internet retail", "china"), "KO": ("consumer defensive", "beverages", "united states"),
}


@pytest.fixture(autouse=True)
def fake_profiles(monkeypatch):
    def profile(symbol):
        sec, ind, cty = PROFILES.get(symbol.upper(), ("", "", ""))
        return {"symbol": symbol.upper(), "assetClass": preferences.asset_class(symbol), "sector": sec,
                "industry": ind, "country": cty, "longName": ""}
    monkeypatch.setattr(preferences, "profile", profile)


def C(sym, ac="STOCKS"):
    return Candidate(symbol=sym, name=sym, assetClass=ac)


EXTRA = [C("MRNA"), C("VRTX"), C("REGN"), C("GILD"), C("BABA"), C("KO"), C("SHY", "BONDS"), C("TIP", "BONDS")]
ALLC = {c.symbol: c for c in CANDS + EXTRA}


def _out(weights):
    return decide.OrchestratorOut(
        assetClasses=[],
        holdings=[decide.HoldingOut(symbol=s, weightPct=w, rationale="r", findingIds=[]) for s, w in weights.items()])


def _alloc(weights, prefs):
    return decide._build_allocation(_out(weights), ALLC, FINDINGS, {}, 10000, prefs), ALLC


def _w(alloc):
    return {h.symbol: h.weightPct for s in alloc.slices for h in s.holdings}


def _sum100(w):
    return round(sum(w.values()), 1) == 100.0 and max(w.values()) <= 25.0


def test_no_gold_class_target_zero_drops_class_and_check_passes():
    prefs = ResolvedPreferences(classTargets={"GOLD": 0})
    a, cands = _alloc({"NVDA": 25, "MSFT": 25, "KO": 20, "BTC-USD": 10, "GLD": 20}, prefs)
    w = _w(a)
    assert "GLD" not in w and _sum100(w)
    assert decide._pref_check(prefs, a, cands, -10, -35.0).isPassed


def test_no_gold_violation_is_reported_with_reason():
    prefs = ResolvedPreferences(classTargets={"GOLD": 0})
    held, cands = decide._build_allocation(_out({"NVDA": 50, "GLD": 50}), ALLC, FINDINGS, {}, 1e4, None), ALLC
    c = decide._pref_check(prefs, held, cands, -10, -35.0)
    assert not c.isPassed and "GLD" in c.detail and "no gold" in c.detail


def test_only_biotech_drops_other_stocks_keeps_other_classes():
    prefs = ResolvedPreferences(only=Filter(industries=["Biotechnology"]))
    a, cands = _alloc({"MRNA": 20, "VRTX": 20, "REGN": 15, "NVDA": 20, "SHY": 15, "GLD": 10}, prefs)
    w = _w(a)
    assert "NVDA" not in w and {"MRNA", "VRTX", "REGN", "SHY", "GLD"} <= set(w) and _sum100(w)
    assert decide._pref_check(prefs, a, cands, -10, -35.0).isPassed


def test_no_tech_drops_every_tech_stock():
    prefs = ResolvedPreferences(exclude=Filter(sectors=["Technology"]))
    a, _ = _alloc({"NVDA": 20, "MSFT": 20, "AAPL": 10, "KO": 20, "GLD": 10, "SHY": 10, "BTC-USD": 10}, prefs)
    w = _w(a)
    assert not {"NVDA", "MSFT", "AAPL"} & set(w) and _sum100(w)


def test_no_china_drops_chinese_company():
    prefs = ResolvedPreferences(exclude=Filter(countries=["China"]))
    a, cands = _alloc({"BABA": 25, "NVDA": 25, "KO": 20, "GLD": 15, "SHY": 15}, prefs)
    w = _w(a)
    assert "BABA" not in w and _sum100(w)
    held = decide._build_allocation(_out({"BABA": 50, "KO": 50}), ALLC, FINDINGS, {}, 1e4, None)
    c = decide._pref_check(prefs, held, cands, -10, -35.0)
    assert not c.isPassed and "BABA" in c.detail and "country 'China'" in c.detail


def test_must_include_forced_at_least_1pct_and_sum_exact():
    a, _ = _alloc({"NVDA": 25, "MSFT": 25, "BTC-USD": 25, "GLD": 25}, ResolvedPreferences(mustInclude=["AAPL"]))
    w = _w(a)
    assert w["AAPL"] >= 1.0 and _sum100(w)


def test_must_include_beats_exclude_filter():
    prefs = ResolvedPreferences(mustInclude=["NVDA"], exclude=Filter(sectors=["Technology"]))
    a, _ = _alloc({"NVDA": 20, "MSFT": 20, "KO": 20, "GLD": 20, "SHY": 20}, prefs)
    w = _w(a)
    assert "NVDA" in w and "MSFT" not in w and _sum100(w)


def test_unresolvable_must_include_is_ignored():
    a, _ = _alloc({"NVDA": 50, "MSFT": 50}, ResolvedPreferences(mustInclude=["NOPE"]))
    assert set(_w(a)) == {"NVDA", "MSFT"}


def test_is_user_requested_copied_from_candidate():
    cands = dict(ALLC)
    cands["NVDA"] = cands["NVDA"].model_copy()
    object.__setattr__(cands["NVDA"], "isUserRequested", True)
    a = decide._build_allocation(_out({"NVDA": 50, "MSFT": 50}), cands, FINDINGS, {}, 10000, None)
    flags = {h.symbol: h.isUserRequested for s in a.slices for h in s.holdings}
    assert flags == {"NVDA": True, "MSFT": False}


def _pref_check(weights, prefs, dd):
    a, cands = _alloc(weights, prefs)
    return decide._pref_check(prefs, a, cands, dd, -35.0)


def test_class_target_passes_within_5pp():
    c = _pref_check({"NVDA": 25, "MSFT": 25, "AAPL": 10, "BTC-USD": 22, "GLD": 18},
                    ResolvedPreferences(classTargets={"CRYPTO": 25}), -10)
    assert c.isPassed, c.detail


def test_class_target_fails_outside_5pp_when_drawdown_has_headroom():
    c = _pref_check({"NVDA": 25, "MSFT": 25, "AAPL": 20, "BTC-USD": 12, "GLD": 18},
                    ResolvedPreferences(classTargets={"CRYPTO": 30}), -10)
    assert not c.isPassed and "CRYPTO" in c.detail


def test_class_target_underweight_explained_by_drawdown_limit_passes():
    c = _pref_check({"NVDA": 25, "MSFT": 25, "AAPL": 20, "BTC-USD": 12, "GLD": 18},
                    ResolvedPreferences(classTargets={"CRYPTO": 30}), -33.0)
    assert c.isPassed and "drawdown limit" in c.detail and "CRYPTO" in c.detail


def test_overweight_never_explained_by_drawdown():
    c = _pref_check({"NVDA": 20, "MSFT": 20, "AAPL": 10, "BTC-USD": 25, "ETH-USD": 5, "GLD": 20},
                    ResolvedPreferences(classTargets={"CRYPTO": 10}), -34.0)
    assert not c.isPassed


def test_preference_notes_empty_without_prefs(fake):
    rounds, _ = run()
    assert decide.preference_notes(INTAKE, rounds) == [] and decide.preference_notes(INTAKE, rounds, None) == []


def test_candidates_not_allowed_never_reach_orchestrator(fake, monkeypatch):
    prefs = ResolvedPreferences(exclude=Filter(sectors=["Technology"]), classTargets={"GOLD": 0},
                                interpretations=["'no tech' → sector Technology"])
    cands = CANDS + [C("KO"), C("SHY", "BONDS")]
    fake_closes = closes()
    fake_closes["KO"] = fake_closes["MSFT"] * 0.9
    fake_closes["SHY"] = 100 + np.arange(len(fake_closes)) * 0.01
    monkeypatch.setattr(decide.data, "prices", lambda syms, years=5: (fake_closes[[s for s in syms if s in fake_closes]], []))
    rounds = asyncio.run(decide.decide(INTAKE, cands, FINDINGS, lambda e: None, prefs))
    p = fake["orch_inputs"][0]
    assert "NVDA |" not in p and "GLD |" not in p and "KO |" in p
    assert "'no tech' → sector Technology" in p and "preferences beat our defaults" in p
    held = {h.symbol for r in rounds for s in r.allocation.slices for h in s.holdings}
    assert not held & {"NVDA", "MSFT", "AAPL", "GLD"}


def test_full_run_notes_and_ninth_check(fake):
    prefs = ResolvedPreferences(mustInclude=["AAPL"], classTargets={"CRYPTO": 30, "GOLD": 20},
                                interpretations=["'some crypto' → CRYPTO 30%"])
    rounds = asyncio.run(decide.decide(INTAKE, CANDS, FINDINGS, lambda e: None, prefs))
    assert all(any(c.name == "User preferences respected" for c in r.verification.checks) for r in rounds)
    assert "preferences (verbatim)" in fake["orch_inputs"][0] and "mustInclude" in fake["orch_inputs"][0]
    notes = decide.preference_notes(INTAKE, rounds, prefs)
    assert "Understood: 'some crypto' → CRYPTO 30%" in notes
    assert any(n.startswith("AAPL included at") and n.endswith("as requested") for n in notes), notes
    assert any(n.startswith("Crypto target 30% → ") and "drawdown limit" in n and "round 1 tried" in n for n in notes), notes
    assert any(n.startswith("Gold target 20%") for n in notes), notes


def test_gold_zero_note_says_excluded_as_requested(fake):
    prefs = ResolvedPreferences(classTargets={"GOLD": 0})
    cands = [c for c in CANDS if c.symbol != "GLD"]
    rounds = asyncio.run(decide.decide(INTAKE, cands, FINDINGS, lambda e: None, prefs))
    assert "Gold excluded as requested (0% target)" in decide.preference_notes(INTAKE, rounds, prefs)


def test_drawdown_failure_from_class_target_is_named_in_changes(fake):
    prefs = ResolvedPreferences(classTargets={"CRYPTO": 50})
    rounds = asyncio.run(decide.decide(INTAKE, CANDS, FINDINGS, lambda e: None, prefs))
    assert any("Your 50% CRYPTO target exceeds what a Moderate profile allows; reduce to ~" in c
               for c in rounds[0].verification.changesRequested)


# ---------- BONDS ----------
BOND_CANDS = CANDS + [Candidate(symbol="SHY", name="Short Treasuries", assetClass="BONDS"),
                      Candidate(symbol="TIP", name="TIPS", assetClass="BONDS")]


def test_four_class_allocation_order_and_exact_sum():
    cands = {c.symbol: c for c in BOND_CANDS}
    out = _out({"GLD": 7, "BTC-USD": 11, "SHY": 23, "TIP": 17, "NVDA": 13, "MSFT": 29})
    a = decide._build_allocation(out, cands, FINDINGS, {}, 10000, None)
    assert [s.assetClass for s in a.slices] == ["STOCKS", "BONDS", "CRYPTO", "GOLD"]
    assert round(sum(h.weightPct for s in a.slices for h in s.holdings), 1) == 100.0
    assert max(h.weightPct for s in a.slices for h in s.holdings) <= 25.0


def test_prompt_states_bond_ranges_per_risk(fake):
    run()
    p = fake["orch_inputs"][0]
    assert "BONDS" in p and "10-30%" in p and "short-duration" in p.lower() and "2022" in p


def test_bonds_underweight_not_explained_by_drawdown():
    prefs = ResolvedPreferences(classTargets={"BONDS": 40})
    cands = {c.symbol: c for c in BOND_CANDS}
    a = decide._build_allocation(_out({"NVDA": 25, "MSFT": 25, "AAPL": 20, "SHY": 10, "GLD": 20}), cands, FINDINGS, {}, 1e4, prefs)
    assert not decide._pref_check(prefs, a, cands, -34.0, -35.0).isPassed  # near the limit, but bonds only help


def test_notes_mention_bonds(fake, monkeypatch):
    cl = closes()
    cl["SHY"] = 100 * np.cumprod(1 + np.random.default_rng(5).normal(.0005, .002, len(cl)))
    cl["TIP"] = 100 * np.cumprod(1 + np.random.default_rng(6).normal(.0008, .004, len(cl)))
    monkeypatch.setattr(decide.data, "prices", lambda syms, years=5: (cl[[s for s in syms if s in cl]], []))
    prefs = ResolvedPreferences(classTargets={"BONDS": 20})
    state = fake
    orig = decide.llm.ask

    async def ask(**kw):
        if kw.get("schema") is decide.OrchestratorOut:
            w = {"NVDA": 15, "MSFT": 15, "AAPL": 15, "SHY": 20, "TIP": 15, "GLD": 20}
            data = decide.OrchestratorOut.model_validate({"assetClasses": [], "holdings": [
                {"symbol": s, "weightPct": p, "rationale": "r", "findingIds": []} for s, p in w.items()]})
            return llm.AskResult(data, "", [], "m", 1, [])
        return await orig(**kw)

    monkeypatch.setattr(decide.llm, "ask", ask)
    rounds = asyncio.run(decide.decide(INTAKE, BOND_CANDS, FINDINGS, lambda e: None, prefs))
    notes = decide.preference_notes(INTAKE, rounds, prefs)
    assert any(n.startswith("Bonds target 20% met at 35%") for n in notes) or any("Bonds" in n for n in notes), notes
    assert any(n.startswith("Bonds 35%") and "SHY" in n and "drawdown within -35%" in n for n in notes), notes


def test_class_cap_can_search_upward_toward_a_target():
    cands = {c.symbol: c for c in CANDS}
    w = {"NVDA": 25, "MSFT": 25, "AAPL": 20, "BTC-USD": 5, "GLD": 25}
    up = decide._class_cap(w, cands, closes(), -60.0, "CRYPTO", hi=40.0)
    assert up and 5 < up[0] <= 40 and up[1] >= -60.0


# ---------- horizons ----------
def test_horizon_rendering():
    assert [decide.horizon_text(h) for h in (0.25, 0.5, 1, 2, 10)] == ["3-month", "6-month", "1-year", "2-year", "10-year"]
    assert decide.horizon_text(0.99) == "12-month" or decide.horizon_text(0.99) == "1-year"


def test_short_horizon_in_orchestrator_prompt(fake):
    intake = INTAKE.model_copy(update={"horizonYears": 0.25})
    asyncio.run(decide.decide(intake, CANDS, FINDINGS, lambda e: None))
    p = fake["orch_inputs"][0]
    assert "3-month" in p and "liquidity" in p and "capital preservation" in p


def test_long_horizon_has_no_short_guidance(fake):
    run()
    p = fake["orch_inputs"][0]
    assert "2-year" in p and "capital preservation" not in p


def test_summary_prompt_mentions_horizon(fake, monkeypatch):
    rounds, _ = run()
    seen = []

    async def ask(**kw):
        seen.append(kw["input"])
        return llm.AskResult("", "ok", [], "m", 1, [])

    monkeypatch.setattr(decide.llm, "ask", ask)
    intake = INTAKE.model_copy(update={"horizonYears": 0.25})
    asyncio.run(decide.summarize(intake, rounds, FINDINGS))
    assert "Horizon: 3-month" in seen[0]
