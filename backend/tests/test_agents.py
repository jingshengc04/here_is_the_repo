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
    monkeypatch.setattr(data, "prices", lambda syms, years=5: (df[[c for c in syms if c in df]], []))
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


# ---------------- Wave 4A ----------------
import datetime as _dt

from app.models import Preferences, ResolvedPreferences


def test_gate_missing_or_zero_data_is_never_verified():
    assert agents.number_gate("Revenue: $0.0B", {"revenue": None}) is None
    assert agents.number_gate("Revenue: $90B, Net income: $30B", {"revenue": 0, "net_income": 0}) is None
    # one figure verified, one unverifiable -> still not a tick
    assert agents.number_gate("Revenue: $100B, Net income: $30B", {"revenue": 100e9, "net_income": None}) is None
    assert agents.number_gate("Revenue: $100B, Net income: $30B", {"revenue": 100e9, "net_income": 30e9}) is True
    assert agents.number_gate("Revenue: $50B, Net income: $30B", {"revenue": 100e9, "net_income": None}) is False


def test_gate_covers_cash_flow_debt_cash_and_margins():
    f = {"revenue": 100e9, "gross_profit": 60e9, "net_income": 20e9, "cfo": 25e9, "capex": 5e9, "total_debt": 40e9, "cash": 10e9}
    assert agents.number_gate("Operating cash flow: $25B, Total debt: $40B, Cash: $10B", f) is True
    assert agents.number_gate("Operating cash flow: $-3B", f) is False  # the 'KO negative cfo' kind of error
    assert agents.number_gate("Free cash flow: $20B", f) is True
    assert agents.number_gate("Gross margin: 60%", f) is True
    assert agents.number_gate("Gross margin: 45%", f) is False
    assert agents.number_gate("Total debt: $90B", f) is False


def _capture(monkeypatch, payload):
    prompts = []

    async def ask(**kw):
        prompts.append(kw["input"])
        return SimpleNamespace(data=payload, sources=SRC, citations=[])
    monkeypatch.setattr(llm, "ask", ask)
    return prompts


def test_todays_date_is_in_every_research_prompt(monkeypatch):
    today = _dt.date.today().isoformat()
    monkeypatch.setattr(data, "fundamentals", lambda s: {})
    monkeypatch.setattr(data, "resolve", lambda s: True)
    monkeypatch.setattr(data, "prices", lambda syms, years=5: (__import__("pandas").DataFrame(), []))
    prompts = _capture(monkeypatch, agents.DiscoveryOut(picks=[]))
    run(agents.discovery(INTAKE.model_copy(update={"riskTolerance": "AGGRESSIVE"}), count(1)))
    _capture_more = _capture(monkeypatch, agents.FinancialsOut(stocks=[]))
    run(agents.financials([Candidate(symbol="AAPL", name="A", assetClass="STOCKS")], count(1)))
    p2 = _capture(monkeypatch, agents.ResearchOut(findings=[]))
    run(agents.landscape(["A"], count(1)))
    run(agents.news(["A"], count(1)))
    p3 = _capture(monkeypatch, agents.SocialOut(findings=[], scores=[]))
    run(agents.social(["A"], count(1)))
    for p in prompts + _capture_more + p2 + p3:
        assert f"Today is {today}" in p and "last 90 days" in p
    assert len(prompts + _capture_more + p2 + p3) == 5


def test_sources_are_mapped_per_claim_not_shared(monkeypatch):
    R = agents.RawFinding
    cites = [{"title": "Vertiv liquid cooling orders surge", "url": "http://a.com/vertiv-cooling"},
             {"title": "Arista Networks switch demand", "url": "http://b.com/arista-switches"},
             {"title": "Gold hits record", "url": "http://c.com/gold"},
             {"title": "Misc market wrap", "url": "http://d.com/wrap"}]
    res = SimpleNamespace(data=agents.ResearchOut(findings=[
        R(symbol="VRT", claim="Vertiv cooling orders surged this quarter", sentiment="POSITIVE"),
        R(symbol="ANET", claim="Arista switch demand is slowing", sentiment="NEGATIVE")]), sources=cites, citations=cites)

    async def ask(**kw):
        return res
    monkeypatch.setattr(llm, "ask", ask)
    a, b = run(agents.news(["VRT", "ANET"], count(1)))
    assert [s.url for s in a.sources] == ["http://a.com/vertiv-cooling"]
    assert [s.url for s in b.sources] == ["http://b.com/arista-switches"]


def test_sources_fall_back_to_call_level_when_nothing_matches():
    res = SimpleNamespace(sources=[{"title": "x", "url": "http://u/1"}], citations=[{"title": "zzz", "url": "http://u/1"}])
    assert [s["url"] for s in agents._pick_sources("totally unrelated words here", "QQQ", res)] == ["http://u/1"]


def _intake(**prefs):
    return INTAKE.model_copy(update={"riskTolerance": "AGGRESSIVE", "preferences": Preferences(**prefs)})


def _discover_prefs(monkeypatch, picks, **prefs):
    out = agents.DiscoveryOut(picks=picks)
    prompts = _capture(monkeypatch, out)
    notes, msgs = [], []
    cands, fs = run(agents.discovery(_intake(**prefs), count(1), on_step=msgs.append, notes_out=notes))
    return cands, fs, prompts[0], notes, msgs


def P(sym, cls="STOCKS"):
    return agents.Pick(symbol=sym, name=sym, assetClass=cls, thesis="t")


def test_must_include_added_marked_and_unresolvable_noted(monkeypatch):
    monkeypatch.setattr(data, "resolve", lambda s: s != "NOPE")
    cands, fs, prompt, notes, msgs = _discover_prefs(
        monkeypatch, [P("AVGO")], mustInclude=["DOGE-USD", "NVDA", "NOPE"], notes="I love dogs")
    by = {c.symbol: c for c in cands}
    assert by["DOGE-USD"].isUserRequested and by["DOGE-USD"].assetClass == "CRYPTO"
    assert by["NVDA"].isUserRequested and not by["AVGO"].isUserRequested
    assert "NOPE" not in by and any("NOPE" in n for n in notes) and any("NOPE" in m["message"] for m in msgs)
    assert "I love dogs" in prompt  # notes verbatim
    assert [f.symbol for f in fs if f.symbol == "DOGE-USD"] == ["DOGE-USD"]


def test_must_include_already_picked_by_model_is_marked_not_duplicated(monkeypatch):
    monkeypatch.setattr(data, "resolve", lambda s: True)
    cands, *_ = _discover_prefs(monkeypatch, [P("NVDA")], mustInclude=["NVDA"])
    assert [c.symbol for c in cands].count("NVDA") == 1 and cands[0].isUserRequested




def test_asset_class_targets_drive_candidate_counts_and_notes(monkeypatch):
    monkeypatch.setattr(data, "resolve", lambda s: True)
    cands, _, prompt, notes, _ = _discover_prefs(monkeypatch, [P("NVDA"), P("BTC-USD", "CRYPTO")], assetClassTargets={"CRYPTO": 30})
    assert "at least 4 cryptocurrencies" in prompt
    assert any("CRYPTO" in n and "only 1" in n for n in notes)


def test_defensive_style_adds_three_ballast_even_when_aggressive(monkeypatch):
    _prices(monkeypatch)
    cands, *_ = _discover_prefs(monkeypatch, [P("NVDA")], styles=["DEFENSIVE"])
    assert [c.symbol for c in cands] == ["NVDA", "JNJ", "PG", "KO", "GLD"]


# ---------------- Wave 5A: bonds ----------------
def _bond_prices(monkeypatch):
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2021-01-01", periods=100, freq="W-FRI")
    # drawdown depth: SHY smallest, then BIL, TIP, BND, AGG, IEF, VGSH deepest; defensive stocks flat-ish
    depth = {"SHY": .02, "BIL": .03, "TIP": .06, "BND": .10, "AGG": .12, "IEF": .15, "VGSH": .20}
    cols = {}
    for k, d in depth.items():
        v = np.ones(100)
        v[50:] = 1 - d  # one step down = max drawdown d
        cols[k] = 100 * v
    for k in agents.DEFENSIVE:
        cols[k] = 100 * np.cumprod(1 + np.random.default_rng(len(k)).normal(0, .01, 100))
    df = pd.DataFrame(cols, index=idx)
    monkeypatch.setattr(data, "prices", lambda syms, years=5: (df[[c for c in syms if c in df]], []))
    monkeypatch.setattr(data, "resolve", lambda s: True)
    monkeypatch.setattr(data, "company_info", lambda s: {})


def _bonds(monkeypatch, risk, **prefs):
    _bond_prices(monkeypatch)
    intake = INTAKE.model_copy(update={"riskTolerance": risk, "preferences": Preferences(**prefs) if prefs else None})
    _capture(monkeypatch, agents.DiscoveryOut(picks=[P("NVDA")]))
    notes = []
    cands, fs = run(agents.discovery(intake, count(1), notes_out=notes))
    return [c for c in cands if c.assetClass == "BONDS"], fs, notes


def test_asset_class_marks_bond_etfs():
    from app.preferences import asset_class
    assert asset_class("SHY") == "BONDS" and asset_class("TLT") == "BONDS"
    assert asset_class("GLD") == "GOLD" and asset_class("NVDA") == "STOCKS"


def test_conservative_gets_two_lowest_drawdown_bonds(monkeypatch):
    bonds, fs, _ = _bonds(monkeypatch, "CONSERVATIVE")
    assert [b.symbol for b in bonds] == ["SHY", "BIL"]
    f = next(f for f in fs if f.symbol == "SHY")
    assert "bond ballast" in f.claim and "CONSERVATIVE" in f.claim and "-2.0%" in f.claim


def test_moderate_gets_one_bond_and_aggressive_none(monkeypatch):
    assert [b.symbol for b in _bonds(monkeypatch, "MODERATE")[0]] == ["SHY"]
    assert _bonds(monkeypatch, "AGGRESSIVE")[0] == []


def test_aggressive_with_bonds_target_or_defensive_style_adds_bonds(monkeypatch):
    assert [b.symbol for b in _bonds(monkeypatch, "AGGRESSIVE", assetClassTargets={"BONDS": 20})[0]] == ["SHY"]
    assert [b.symbol for b in _bonds(monkeypatch, "AGGRESSIVE", assetClassTargets={"BONDS": 40})[0]] == ["SHY", "BIL"]
    assert [b.symbol for b in _bonds(monkeypatch, "AGGRESSIVE", styles=["DEFENSIVE"])[0]] == ["SHY"]



def test_requested_bond_counts_toward_quota(monkeypatch):
    bonds, *_ = _bonds(monkeypatch, "CONSERVATIVE", mustInclude=["BND"])
    assert sorted(b.symbol for b in bonds) == ["BND", "SHY"] and next(b for b in bonds if b.symbol == "BND").isUserRequested


# ---------------- Wave 6A: general personalisation (5 profiles, zero case-specific code) ----------------
from app import preferences
from app.models import Filter

INFO = {
    "NVDA": {"sector": "technology", "industry": "semiconductors", "country": "united states", "longName": "nvidia"},
    "MRNA": {"sector": "healthcare", "industry": "biotechnology", "country": "united states", "longName": "moderna"},
    "VRTX": {"sector": "healthcare", "industry": "biotechnology", "country": "united states", "longName": "vertex"},
    "BABA": {"sector": "consumer cyclical", "industry": "internet retail", "country": "china", "longName": "alibaba"},
    "MO": {"sector": "consumer defensive", "industry": "tobacco", "country": "united states", "longName": "altria"},
    "LMT": {"sector": "industrials", "industry": "aerospace & defense", "country": "united states", "longName": "lockheed"},
    "XOM": {"sector": "energy", "industry": "oil & gas e&p", "country": "united states", "longName": "exxon"},
    "KO": {"sector": "consumer defensive", "industry": "beverages - non-alcoholic", "country": "united states", "longName": "coca-cola"},
}


def _profile(monkeypatch, risk, prefs, resolved, picks):
    """resolve() then discovery() with ONE fake llm.ask that serves both; returns (cands, notes, rp, llm agents called)."""
    _bond_prices(monkeypatch)
    monkeypatch.setattr(data, "company_info", lambda s: INFO.get(s, {}))
    called = []

    async def ask(**kw):
        called.append(kw["agent"])
        if kw["agent"] == "PREFERENCES":
            return SimpleNamespace(data=resolved)
        return SimpleNamespace(data=agents.DiscoveryOut(picks=[P(x) if not isinstance(x, tuple) else P(*x) for x in picks]),
                               sources=SRC, citations=[])
    monkeypatch.setattr(llm, "ask", ask)
    intake = INTAKE.model_copy(update={"riskTolerance": risk, "preferences": Preferences(**prefs)})
    rp = run(preferences.resolve(intake))
    notes = []
    cands, _ = run(agents.discovery(intake, count(1), notes_out=notes, rp=rp))
    return cands, notes, rp, called


def _syms(cands):
    return [c.symbol for c in cands]


def test_profile_no_gold(monkeypatch):
    out = preferences.ResolveOut(exclude=Filter(assetClasses=["GOLD"]), interpretations=["'gold' -> excluding asset class GOLD"])
    cands, notes, rp, called = _profile(monkeypatch, "AGGRESSIVE", {"exclude": ["gold"]}, out, ["NVDA"])
    assert "GLD" not in _syms(cands) and any("Skipped gold" in n and "excluded" in n for n in notes)
    assert called.count("PREFERENCES") == 1


def test_profile_only_bio_stocks_skips_defensive_ballast_with_note(monkeypatch):
    out = preferences.ResolveOut(only=Filter(industries=["Biotechnology"]), interpretations=["'bio stocks' -> only industry Biotechnology"])
    cands, notes, rp, _ = _profile(monkeypatch, "MODERATE", {"onlyInclude": ["bio stocks"]}, out, ["MRNA", "VRTX", "NVDA"])
    stocks = [c.symbol for c in cands if c.assetClass == "STOCKS"]
    assert stocks == ["MRNA", "VRTX"]                                   # NVDA dropped, KO/BRK-B/... ballast skipped
    assert any("defensive ballast" in n and "outside the universe you asked for" in n and "risk limit" in n for n in notes)
    assert "SHY" in _syms(cands)                                        # bonds are still allowed ballast


def test_profile_no_tech(monkeypatch):
    out = preferences.ResolveOut(exclude=Filter(sectors=["Technology"]))
    cands, _, _, _ = _profile(monkeypatch, "AGGRESSIVE", {"exclude": ["tech"]}, out, ["NVDA", "MRNA"])
    assert _syms(cands)[:1] == ["MRNA"] and all(INFO.get(c.symbol, {}).get("sector") != "technology" for c in cands)


def test_profile_no_chinese_companies(monkeypatch):
    out = preferences.ResolveOut(exclude=Filter(countries=["China"]))
    cands, _, _, _ = _profile(monkeypatch, "AGGRESSIVE", {"exclude": ["Chinese companies"]}, out, ["BABA", "NVDA"])
    assert "BABA" not in _syms(cands) and "NVDA" in _syms(cands)


def test_profile_ethical_excludes_industries(monkeypatch):
    out = preferences.ResolveOut(exclude=Filter(industries=["Tobacco", "Aerospace & Defense", "Oil & Gas E&P"]))
    cands, _, rp, _ = _profile(monkeypatch, "AGGRESSIVE", {"exclude": ["weapons", "tobacco", "oil"]}, out, ["MO", "LMT", "XOM", "MRNA"])
    assert [c.symbol for c in cands if c.assetClass == "STOCKS"] == ["MRNA"]
    assert "Tobacco" in preferences.prompt_text(rp)


def test_profile_no_bonds_via_resolved_filter(monkeypatch):
    out = preferences.ResolveOut(exclude=Filter(assetClasses=["BONDS"]))
    cands, notes, _, _ = _profile(monkeypatch, "CONSERVATIVE", {"exclude": ["treasuries"]}, out, ["MRNA"])
    assert not [c for c in cands if c.assetClass == "BONDS"] and any("bond ballast" in n for n in notes)


def test_discovery_prompt_carries_resolved_preferences(monkeypatch):
    _bond_prices(monkeypatch)
    prompts = _capture(monkeypatch, agents.DiscoveryOut(picks=[]))
    rp = ResolvedPreferences(only=Filter(industries=["Biotechnology"]), exclude=Filter(countries=["China"]),
                             classTargets={"GOLD": 0})
    run(agents.discovery(INTAKE.model_copy(update={"preferences": Preferences(onlyInclude=["bio"])}), count(1), rp=rp))
    p = prompts[0]
    assert "Biotechnology" in p and "China" in p and "NO GOLD" in p and "at least 5" in p


def test_horizon_rendering_and_short_horizon_hint(monkeypatch):
    assert [agents.horizon_text(h) for h in (0.25, 0.5, 1, 2, 1.5)] == ["3-month", "6-month", "1-year", "2-year", "1.5-year"]
    _bond_prices(monkeypatch)
    prompts = _capture(monkeypatch, agents.DiscoveryOut(picks=[]))
    run(agents.discovery(INTAKE.model_copy(update={"horizonYears": 0.25}), count(1)))
    run(agents.discovery(INTAKE.model_copy(update={"horizonYears": 2}), count(1)))
    assert "3-month horizon" in prompts[0] and "capital preservation" in prompts[0]
    assert "2-year horizon" in prompts[1] and "capital preservation" not in prompts[1]
