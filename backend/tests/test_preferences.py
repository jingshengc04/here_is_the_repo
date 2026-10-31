from app.models import Filter, ResolvedPreferences
from app.preferences import allowed, asset_class, match

P = {
    "MRNA": {"symbol": "MRNA", "assetClass": "STOCKS", "sector": "healthcare", "industry": "biotechnology", "country": "united states", "longName": "moderna, inc."},
    "NVDA": {"symbol": "NVDA", "assetClass": "STOCKS", "sector": "technology", "industry": "semiconductors", "country": "united states", "longName": "nvidia corporation"},
    "BABA": {"symbol": "BABA", "assetClass": "STOCKS", "sector": "consumer cyclical", "industry": "internet retail", "country": "china", "longName": "alibaba group"},
    "GLD": {"symbol": "GLD", "assetClass": "GOLD", "sector": "", "industry": "", "country": "", "longName": ""},
    "BTC-USD": {"symbol": "BTC-USD", "assetClass": "CRYPTO", "sector": "", "industry": "", "country": "", "longName": ""},
    "KO": {"symbol": "KO", "assetClass": "STOCKS", "sector": "consumer defensive", "industry": "beverages - non-alcoholic", "country": "united states", "longName": "the coca-cola company"},
}


def ok(sym, rp):
    return allowed(sym, rp, P[sym])[0]


def test_asset_class():
    assert [asset_class(s) for s in ["GLD", "SHY", "DOGE-USD", "MRNA"]] == ["GOLD", "BONDS", "CRYPTO", "STOCKS"]


def test_no_tech():
    rp = ResolvedPreferences(exclude=Filter(sectors=["Technology"]))
    assert not ok("NVDA", rp) and ok("MRNA", rp) and ok("GLD", rp)


def test_only_bio_stocks_restricts_stocks_not_other_classes():
    rp = ResolvedPreferences(only=Filter(industries=["Biotechnology"]))
    assert ok("MRNA", rp) and not ok("NVDA", rp) and not ok("KO", rp)   # defensive ballast KO also blocked
    assert ok("GLD", rp) and ok("BTC-USD", rp)                          # other classes untouched unless targeted 0


def test_no_gold_via_class_target_zero_and_via_exclude():
    assert not ok("GLD", ResolvedPreferences(classTargets={"GOLD": 0}))
    assert not ok("GLD", ResolvedPreferences(exclude=Filter(assetClasses=["GOLD"])))


def test_no_chinese_companies():
    rp = ResolvedPreferences(exclude=Filter(countries=["China"]))
    assert not ok("BABA", rp) and ok("NVDA", rp)


def test_must_include_beats_exclude():
    rp = ResolvedPreferences(mustInclude=["NVDA"], exclude=Filter(sectors=["Technology"]))
    assert ok("NVDA", rp)


def test_match_reason_is_human_readable():
    assert match(P["MRNA"], Filter(industries=["Biotechnology"])) == "industry 'Biotechnology'"


# ---------------- Wave 6A: resolve() ----------------
import asyncio
import datetime
from types import SimpleNamespace

import pytest

from app import data, llm, preferences
from app.models import IntakeInput, Preferences


def _intake(**prefs):
    return IntakeInput(amountUsd=10000, horizonYears=2, riskTolerance="MODERATE",
                       preferences=Preferences(**prefs) if prefs else None)


def _fake_ask(monkeypatch, out):
    calls = []

    async def ask(**kw):
        calls.append(kw)
        return SimpleNamespace(data=out)
    monkeypatch.setattr(llm, "ask", ask)
    return calls


def test_no_preferences_means_no_llm_call(monkeypatch):
    calls = _fake_ask(monkeypatch, None)
    for intake in (_intake(), IntakeInput(amountUsd=10000, horizonYears=2, riskTolerance="MODERATE", preferences=Preferences())):
        rp = asyncio.run(preferences.resolve(intake))
        assert rp == ResolvedPreferences() and calls == []


def test_resolve_one_call_validates_sectors_and_tickers(monkeypatch):
    out = preferences.ResolveOut(
        mustInclude=["NVDA", "NOPE"], interpretations=["'bio stocks' -> only industry Biotechnology"],
        only=Filter(industries=["Biotechnology"]),
        exclude=Filter(sectors=["Technology", "Crypto Stuff"], symbols=["TSLA", "ZZZZ"], assetClasses=["GOLD"]),
        classTargets=[preferences.ClassTarget(assetClass="CRYPTO", pct=0)])
    calls = _fake_ask(monkeypatch, out)
    monkeypatch.setattr(data, "resolve", lambda s: s not in ("NOPE", "ZZZZ"))
    rp = asyncio.run(preferences.resolve(_intake(mustInclude=["DOGE-USD"], exclude=["tech"], onlyInclude=["bio stocks"],
                                                 assetClassTargets={"BONDS": 20})))
    assert len(calls) == 1 and calls[0]["generation_config"] == {"thinking_level": "low"} and calls[0]["tools"] == ()
    assert datetime.date.today().isoformat() in calls[0]["input"] and "bio stocks" in calls[0]["input"]
    assert rp.mustInclude == ["NVDA", "DOGE-USD"]                      # NOPE dropped, user's own ticker kept
    assert rp.exclude.sectors == ["Technology"] and rp.exclude.symbols == ["TSLA"]
    assert rp.classTargets == {"CRYPTO": 0, "BONDS": 20}               # model's 0 kept, user's number merged in
    assert any("Crypto Stuff" in i for i in rp.interpretations) and any("NOPE" in i for i in rp.interpretations)
    assert any("ZZZZ" in i for i in rp.interpretations) and rp.interpretations[0].startswith("'bio stocks'")


def test_describe_lists_only_non_empty_fields():
    assert preferences.describe(Filter()) == ""
    d = preferences.describe(Filter(industries=["Biotechnology"], countries=["China"]))
    assert "Biotechnology" in d and "China" in d and "sector" not in d.lower()


def test_only_filter_drops_redundant_stocks_class(monkeypatch):
    out = preferences.ResolveOut(only=Filter(industries=["Biotechnology"], assetClasses=["STOCKS"]))
    _fake_ask(monkeypatch, out)
    rp = asyncio.run(preferences.resolve(_intake(onlyInclude=["bio stocks"])))
    assert rp.only.assetClasses == [] and rp.only.industries == ["Biotechnology"]
    P_KO = {"symbol": "KO", "assetClass": "STOCKS", "sector": "consumer defensive", "industry": "beverages", "country": "", "longName": ""}
    assert not allowed("KO", rp, P_KO)[0]   # with STOCKS left in `only`, KO would have slipped through
