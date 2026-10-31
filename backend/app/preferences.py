"""Personalisation, generalised: user words → Yahoo taxonomy filters → one matcher used everywhere.

Flow: resolve(intake) [one Flash call, Wave 6A] → ResolvedPreferences → allowed(symbol, rp) is called by
Discovery, by our defaults (gold / bonds / defensive ballast), by the Orchestrator enforcement and by the
Verifier's "User preferences respected" check. Nothing here is specific to any one preference.
Rule: user preferences beat our defaults; only risk limits push back, and that is explained, never silent.
"""
from __future__ import annotations

from .models import AssetClass, Filter, IntakeInput, ResolvedPreferences

# Yahoo Finance's 11 sectors: the closed vocabulary the resolver must map sector words onto.
SECTORS = ["Technology", "Healthcare", "Financial Services", "Consumer Cyclical", "Consumer Defensive",
           "Industrials", "Energy", "Utilities", "Real Estate", "Basic Materials", "Communication Services"]
GOLD_ETFS = {"GLD", "IAU", "SGOL", "GLDM"}
BOND_ETFS = {"SHY", "BND", "AGG", "TIP", "IEF", "VGSH", "SCHO", "BIL", "GOVT", "TLT"}


def asset_class(symbol: str) -> AssetClass:
    s = symbol.upper()
    if s.endswith("-USD"):
        return "CRYPTO"
    return "GOLD" if s in GOLD_ETFS else "BONDS" if s in BOND_ETFS else "STOCKS"


def profile(symbol: str) -> dict:
    """Everything a filter can match on, lower-cased. Stock fields come from Yahoo (cached in data)."""
    from . import data
    p = {"symbol": symbol.upper(), "assetClass": asset_class(symbol), "sector": "", "industry": "",
         "country": "", "longName": ""}
    if p["assetClass"] == "STOCKS":
        p.update({k: v for k, v in data.company_info(symbol).items() if k in p})
    return p


def match(p: dict, f: Filter) -> str | None:
    """Return why profile p matches filter f (e.g. "industry 'biotechnology'"), or None."""
    if p["symbol"] in {s.upper() for s in f.symbols}:
        return f"ticker {p['symbol']}"
    if p["assetClass"] in f.assetClasses:
        return f"asset class {p['assetClass']}"
    for field, wanted in (("sector", f.sectors), ("industry", f.industries), ("country", f.countries),
                          ("longName", f.keywords)):
        for w in wanted:
            if w and w.lower() in (p.get(field) or ""):
                return f"{field} '{w}'"
    return None


def _only_covers(only: Filter, ac: AssetClass) -> bool:
    """An `only` filter restricts the classes it talks about: STOCKS if it has stock fields, plus named classes."""
    stock_terms = only.sectors or only.industries or only.countries or only.keywords or \
        any(asset_class(s) == "STOCKS" for s in only.symbols)
    return ac in only.assetClasses or (ac == "STOCKS" and bool(stock_terms)) or \
        any(asset_class(s) == ac for s in only.symbols)


def allowed(symbol: str, rp: ResolvedPreferences | None, p: dict | None = None) -> tuple[bool, str]:
    """(ok, reason). mustInclude always wins; then exclude; then class target 0; then `only`."""
    if rp is None:
        return True, ""
    p = p or profile(symbol)
    if p["symbol"] in {s.upper() for s in rp.mustInclude}:
        return True, "requested by you"
    if why := match(p, rp.exclude):
        return False, f"excluded by your preference ({why})"
    if rp.classTargets.get(p["assetClass"]) == 0:
        return False, f"you asked for no {p['assetClass'].lower()}"
    if not rp.only.is_empty() and _only_covers(rp.only, p["assetClass"]) and not match(p, rp.only):
        return False, "outside the universe you asked for"
    return True, ""


async def resolve(intake: IntakeInput) -> ResolvedPreferences:
    """Map intake.preferences (+ theme/notes) to ResolvedPreferences with ONE Flash call. Implemented in Wave 6A."""
    raise NotImplementedError
