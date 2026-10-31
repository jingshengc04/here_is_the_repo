"""Personalisation, generalised: user words → Yahoo taxonomy filters → one matcher used everywhere.

Flow: resolve(intake) [one Flash call, Wave 6A] → ResolvedPreferences → allowed(symbol, rp) is called by
Discovery, by our defaults (gold / bonds / defensive ballast), by the Orchestrator enforcement and by the
Verifier's "User preferences respected" check. Nothing here is specific to any one preference.
Rule: user preferences beat our defaults; only risk limits push back, and that is explained, never silent.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

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


class ClassTarget(BaseModel):
    assetClass: AssetClass
    pct: float = Field(ge=0, le=100)


class ResolveOut(BaseModel):
    """Flat schema for the resolver call (dict fields are awkward for Gemini schemas); converted to ResolvedPreferences."""
    mustInclude: list[str] = Field(default_factory=list)
    prefer: Filter = Field(default_factory=Filter)
    only: Filter = Field(default_factory=Filter)
    exclude: Filter = Field(default_factory=Filter)
    classTargets: list[ClassTarget] = Field(default_factory=list)
    interpretations: list[str] = Field(default_factory=list)


def describe(f: Filter) -> str:
    """Human/LLM-readable one-liner of a filter's non-empty fields, e.g. "industries: Biotechnology; countries: China"."""
    parts = [f"{name}: {', '.join(vals)}" for name, vals in (
        ("tickers", f.symbols), ("sectors", f.sectors), ("industries", f.industries), ("countries", f.countries),
        ("asset classes", f.assetClasses), ("name keywords", f.keywords)) if vals]
    return "; ".join(parts)


def prompt_text(rp: ResolvedPreferences) -> str:
    """The resolved preferences as prompt text for the research agents ('' when there are none)."""
    lines = []
    if rp.mustInclude:
        lines.append(f"Must include: {', '.join(rp.mustInclude)}.")
    for label, f in (("Prefer", rp.prefer), ("ONLY allow (restricts stocks to)", rp.only), ("NEVER include", rp.exclude)):
        if d := describe(f):
            lines.append(f"{label} {d}.")
    if zero := [c for c, v in rp.classTargets.items() if v == 0]:
        lines.append(f"The investor wants NO {', '.join(zero)}.")
    return " ".join(lines)


_RESOLVE_PROMPT = """You translate an investor's personal preferences into data filters. Today is {today}.
Investor: risk {risk}, theme: {theme}.
Preferences (JSON): {prefs}

Map EVERY phrase onto Yahoo Finance's taxonomy, whatever it is (never refuse or special-case a phrase):
- sectors: ONLY from this list: {sectors}. Map a vaguer word to the closest sector(s), or leave it out and say so in interpretations.
- industries: Yahoo industry names, e.g. "Biotechnology", "Drug Manufacturers - General", "Semiconductors", "Banks - Regional", "Oil & Gas E&P", "Tobacco", "Aerospace & Defense".
- countries: Yahoo country names (e.g. "China", "Germany"). A region ("European") means a list of its countries.
- assetClasses: STOCKS, CRYPTO, GOLD, BONDS. symbols: exact tickers (stocks like NVDA, crypto like DOGE-USD).
- keywords: last resort only, substring of a company name.
Fields: mustInclude = tickers the investor insists on. only = restrict the universe to this (e.g. "only biotech" -> industries ["Biotechnology"]). exclude = never include (e.g. "no tech" -> sectors ["Technology"]; "ethical" -> industries like Tobacco, Aerospace & Defense, Oil & Gas E&P, Oil & Gas Integrated, Thermal Coal; "no gold" -> assetClasses ["GOLD"]). prefer = soft preference (styles or likes). classTargets = asset-class percentages, with pct 0 for a class the investor does not want at all.
interpretations: ONE short line per investor phrase, e.g. "'bio stocks' -> only stocks in industry Biotechnology", "'no tech' -> excluding sector Technology".
Only encode what the investor actually said. Leave everything else empty."""


def _has_prefs(pr) -> bool:
    return pr is not None and any((pr.mustInclude, pr.exclude, pr.onlyInclude, pr.styles, pr.assetClassTargets, pr.notes))


async def resolve(intake: IntakeInput) -> ResolvedPreferences:
    """Map the user's preferences onto Yahoo's taxonomy with ONE Flash call (none at all when there are no preferences)."""
    pr = intake.preferences
    if not _has_prefs(pr):
        return ResolvedPreferences()
    import asyncio
    import datetime

    from . import config, data, llm

    prompt = _RESOLVE_PROMPT.format(
        today=datetime.date.today().isoformat(), risk=intake.riskTolerance, theme=intake.theme or "none",
        prefs=pr.model_dump_json(exclude_defaults=True), sectors=", ".join(SECTORS))
    out: ResolveOut = (await llm.ask(model=config.MODEL_FLASH, input=prompt, schema=ResolveOut, tools=(),
                                     agent="PREFERENCES", generation_config={"thinking_level": "low"})).data
    notes = list(out.interpretations)

    async def valid(syms: list[str]) -> list[str]:
        syms = list(dict.fromkeys(x.strip().upper() for x in syms if x.strip()))
        ok = await asyncio.gather(*(asyncio.to_thread(data.resolve, x) for x in syms))
        for x, good in zip(syms, ok):
            if not good:
                notes.append(f"Dropped ticker {x}: no market data found.")
        return [x for x, good in zip(syms, ok) if good]

    by_lower = {x.lower(): x for x in SECTORS}

    async def clean(f: Filter) -> Filter:
        keep = [by_lower[x.lower()] for x in f.sectors if x.lower() in by_lower]
        notes.extend(f"Ignored unknown sector '{x}'." for x in f.sectors if x.lower() not in by_lower)
        return f.model_copy(update={"sectors": keep, "symbols": await valid(f.symbols)})

    must = await valid([*out.mustInclude, *(pr.mustInclude if pr else [])])
    targets = {t.assetClass: t.pct for t in out.classTargets}
    targets.update(pr.assetClassTargets or {})  # the user's own numbers are authoritative
    return ResolvedPreferences(mustInclude=must, prefer=await clean(out.prefer), only=await clean(out.only),
                               exclude=await clean(out.exclude), classTargets=targets, interpretations=notes)
