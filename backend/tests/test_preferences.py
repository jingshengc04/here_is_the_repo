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
