"""Research agents: each is ONE llm.ask call covering all symbols. Returns (data, findings)."""
import asyncio
import datetime
import inspect
import math
import re
from collections import Counter
from itertools import count
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from . import config, data, llm
from .models import AssetClass, Candidate, Finding, IntakeInput
from src.tools.report_audit_tool import extract_data_points

Sentiment = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
FLASH = config.MODEL_FLASH

# Adapted from Vibe-Trading research-discipline (bias checklist) - shared by every research prompt.
DISCIPLINE = (
    "Rules: include small/mid caps and suppliers, not only mega-caps; judge the business, not the concept label; "
    "date every figure and flag data older than a year; never invent numbers or sources. "
    "Output at most 2 findings per symbol: exactly 1 NEGATIVE (bear case) and 1 other. "
    "Each claim is ONE sentence of at most 25 words."
)
LOW = {"thinking_level": "low"}  # ponytail: faster Flash for search agents; Financials keeps default thinking
MAX_WORDS = 25


class RawFinding(BaseModel):
    symbol: str | None = None
    claim: str
    sentiment: Sentiment


class Pick(BaseModel):
    symbol: str = Field(description="yfinance ticker, e.g. NVDA or BTC-USD")
    name: str
    assetClass: Literal["STOCKS", "CRYPTO"]
    thesis: str = Field(description="one sentence, at most 25 words")


class DiscoveryOut(BaseModel):
    picks: list[Pick]


class StockScore(BaseModel):
    symbol: str
    score: int = Field(ge=1, le=10)
    verdict: str
    keyFigures: str = Field(description="one line like 'Revenue: $302.9B, Net income: $75B, P/E: 28.5'")
    risk: str


class FinancialsOut(BaseModel):
    stocks: list[StockScore]


class ResearchOut(BaseModel):
    findings: list[RawFinding]


class SymbolSentiment(BaseModel):
    symbol: str
    score: float = Field(ge=-1, le=1)


class SocialOut(BaseModel):
    findings: list[RawFinding]
    scores: list[SymbolSentiment]


async def _notify(on_step, message: str) -> None:
    if on_step:
        r = on_step({"message": message})
        if inspect.isawaitable(r):
            await r


def _trim(claim: str) -> str:
    w = claim.split()
    return claim.strip() if len(w) <= MAX_WORDS else " ".join(w[:MAX_WORDS]).rstrip(",;:") + "..."


def cap_per_symbol(raws: list[RawFinding], limit: int = 2) -> list[RawFinding]:
    """<= limit per symbol; when limit > 1 keep exactly one NEGATIVE (if any) plus the first non-negatives."""
    out, by_sym = [], {}
    for r in raws:
        by_sym.setdefault(r.symbol, []).append(r)
    for rs in by_sym.values():
        neg = [r for r in rs if r.sentiment == "NEGATIVE"][:1] if limit > 1 else []
        rest = [r for r in rs if r.sentiment != "NEGATIVE"]
        keep = neg + rest[: limit - len(neg)]
        out += [r for r in rs if r in keep]  # preserve original order
    return out


_STOP = {"with", "that", "this", "from", "have", "will", "stock", "stocks", "shares", "share", "their", "about", "which",
         "after", "into", "than", "more", "also", "over", "high", "while", "being", "year", "price", "market"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", text.lower())} - _STOP


def _pick_sources(claim: str, symbol: str | None, res, used: Counter | None = None) -> list[dict]:
    """Up to 3 sources for ONE claim: citations whose title/url share keywords with it (best first, least-reused
    first on ties). Only when nothing matches do we fall back to the call-level sources."""
    used = used if used is not None else Counter()
    words = _tokens(f"{claim} {symbol or ''}")
    cites = list(getattr(res, "citations", None) or []) or list(res.sources or [])
    scored = []
    for c in cites:
        score = len(words & _tokens(f"{c.get('title', '')} {c['url']}"))
        if score:
            scored.append((-score, used[c["url"]], c))
    scored.sort(key=lambda t: t[:2])
    pool = [c for *_, c in scored] or sorted(res.sources or [], key=lambda c: used[c["url"]])
    out, seen = [], set()
    for c in pool:
        if c["url"] not in seen and len(out) < 3:
            seen.add(c["url"])
            used[c["url"]] += 1
            out.append({"title": c.get("title", ""), "url": c["url"]})
    return out


def _findings(raws, agent, ids: Iterator[int], res=None, verified=None) -> list[Finding]:
    used: Counter = Counter()
    return [Finding(id=f"f_{next(ids)}", agent=agent, symbol=r.symbol, claim=_trim(r.claim), sentiment=r.sentiment,
                    sources=_pick_sources(r.claim, r.symbol, res, used) if res is not None else [],
                    numbersVerified=(verified or {}).get(i)) for i, r in enumerate(raws)]


def _context(notes: str | None = None) -> str:
    """Shared prompt preamble: today's date (searches kept stale years) + the investor's free-text notes, verbatim."""
    out = f"Today is {datetime.date.today().isoformat()}. Prefer sources from the last 90 days; date every figure."
    return out + (f" Investor notes (verbatim): {notes}" if notes else "")


DEFENSIVE = {"BRK-B": "Berkshire Hathaway", "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble", "KO": "Coca-Cola",
             "PEP": "PepsiCo", "WMT": "Walmart", "COST": "Costco", "MCD": "McDonald's", "V": "Visa", "MRK": "Merck"}


async def defensive_picks(exclude: set[str], n: int = 2) -> list[tuple[str, str, float]]:
    """The n lowest-weekly-volatility (5y) names from DEFENSIVE not in `exclude`: [(symbol, name, annualised vol %)]."""
    closes, _missing = await asyncio.to_thread(data.prices, list(DEFENSIVE), 5)  # caches parquet per symbol
    if closes is None or closes.empty:
        return []
    vol = (closes.pct_change().std() * 52 ** 0.5 * 100).dropna().sort_values()
    out = []
    for sym, v in vol.items():
        if sym in exclude or not await asyncio.to_thread(data.resolve, sym):
            continue
        out.append((sym, DEFENSIVE[sym], round(float(v), 1)))
        if len(out) == n:
            break
    return out


STYLE_HINTS = {
    "MEME": "meme coins (e.g. DOGE, SHIB, PEPE, WIF, BONK) and meme-popular stocks",
    "DIVIDEND": "reliable dividend payers",
    "VALUE": "low P/E value stocks",
    "GROWTH": "high-growth companies",
    "MOMENTUM": "6-month price-momentum leaders",
    "ESG": "ESG-friendly names (avoid fossil fuels, tobacco, weapons)",
    "DEFENSIVE": "defensive, low-volatility names",
}
ESG_INDUSTRIES = ("oil & gas", "coal", "tobacco", "aerospace & defense", "uranium")


def target_count(pct: float) -> int:
    return min(8, max(2, math.ceil(pct / 10) + 1))


def _asset_class(sym: str) -> str:
    return "CRYPTO" if data.is_crypto(sym) else "GOLD" if sym == "GLD" else "STOCKS"


async def _is_excluded(c: Candidate, exclude: list[str], esg: bool) -> bool:
    syms = {x.strip().upper() for x in exclude}
    if c.symbol in syms or c.symbol.removesuffix("-USD") in syms:
        return True
    words = [x.strip().lower() for x in exclude if len(x.strip()) > 2]
    if not words and not esg:
        return False
    info = {} if c.assetClass == "CRYPTO" else await asyncio.to_thread(data.company_info, c.symbol)
    hay = " ".join([info.get("sector", ""), info.get("industry", ""), info.get("longName", ""), c.name.lower()])
    return any(w in hay for w in words) or (esg and any(w in f"{info.get('sector', '')} {info.get('industry', '')}" for w in ESG_INDUSTRIES))


async def discovery(intake: IntakeInput, ids: Iterator[int], on_step=None, notes_out: list[str] | None = None
                    ) -> tuple[list[Candidate], list[Finding]]:
    pr = intake.preferences
    must = [m.strip().upper() for m in (pr.mustInclude if pr else []) if m.strip()]
    exclude = list(pr.exclude) if pr else []
    styles = list(pr.styles) if pr else []
    targets = (pr.assetClassTargets or {}) if pr else {}
    notes = pr.notes if pr else None
    note = (lambda m: notes_out.append(m)) if notes_out is not None else (lambda m: None)

    def count_txt(cls: str, default: str) -> str:
        if cls in targets:
            return "no" if targets[cls] == 0 else f"at least {target_count(targets[cls])}"
        return default

    prompt = (
        f"You are the Discovery analyst. {_context(notes)}\n"
        f"Investor: ${intake.amountUsd:,.0f}, {intake.horizonYears}-year horizon, "
        f"{intake.riskTolerance} risk, theme: {intake.theme or 'none (broad diversified growth)'}.\n"
        f"Run 2-3 Google searches, then pick {count_txt('STOCKS', '4-6')} stocks (mix of leaders plus small/mid caps or suppliers) and "
        f"{count_txt('CRYPTO', '3-5' if 'MEME' in styles else '2-3')} cryptocurrencies suited to the theme and risk level. "
        "Symbols in yfinance form (NVDA, BTC-USD). "
        + (f"Investment styles to favour: {'; '.join(STYLE_HINTS[x] for x in styles)}. " if styles else "")
        + (f"NEVER pick any of these symbols/sectors/words: {', '.join(exclude)}. " if exclude else "")
        + (f"The investor insists on {', '.join(must)}; they are added automatically, do not spend picks on them. " if must else "")
        + "One-sentence thesis per pick (at most 25 words, dated figures only). No bear-case research. "
        "Defensive low-volatility large caps (e.g. JNJ, PG) and gold are added automatically afterwards, so focus on the theme."
    )
    res = await llm.ask(model=FLASH, input=prompt, schema=DiscoveryOut, tools=["google_search"], agent="DISCOVERY",
                        on_step=on_step, generation_config=LOW)
    cands, raws = [], []
    for p in res.data.picks:
        sym = p.symbol.strip().upper()
        if any(c.symbol == sym for c in cands):
            continue
        if not await asyncio.to_thread(data.resolve, sym):
            await _notify(on_step, f"Dropped {sym}: no market data found")
            continue
        cands.append(Candidate(symbol=sym, name=p.name, assetClass=p.assetClass, isUserRequested=sym in must))
        raws.append(RawFinding(symbol=sym, claim=p.thesis, sentiment="POSITIVE"))
    for sym in must:  # user-requested symbols are always candidates, if they exist
        if any(c.symbol == sym for c in cands):
            continue
        if not await asyncio.to_thread(data.resolve, sym):
            await _notify(on_step, f"Could not find market data for requested symbol {sym}")
            note(f"Requested {sym} could not be resolved (no market data), so it was left out.")
            continue
        cands.append(Candidate(symbol=sym, name=sym, assetClass=_asset_class(sym), isUserRequested=True))
        raws.append(RawFinding(symbol=sym, claim="Included at the investor's request.", sentiment="NEUTRAL"))
    if intake.riskTolerance in ("CONSERVATIVE", "MODERATE") or "DEFENSIVE" in styles:
        for sym, name, v in await defensive_picks({c.symbol for c in cands} | {x.upper() for x in exclude},
                                                  3 if "DEFENSIVE" in styles else 2):
            cands.append(Candidate(symbol=sym, name=name, assetClass="STOCKS"))
            raws.append(RawFinding(symbol=sym, sentiment="NEUTRAL",
                                   claim=f"Added as defensive ballast for a {intake.riskTolerance} profile (5y weekly vol {v}%)."))
    if not any(c.symbol == "GLD" for c in cands):
        cands.append(Candidate(symbol="GLD", name="SPDR Gold Shares", assetClass="GOLD"))
        raws.append(RawFinding(symbol="GLD", claim="Gold is added as a diversifying hedge against equity and crypto drawdowns.", sentiment="NEUTRAL"))

    if exclude or "ESG" in styles:  # Python post-filter: never return excluded symbols/words
        flags = await asyncio.gather(*(_is_excluded(c, exclude, "ESG" in styles) for c in cands))
        for c, bad in zip(cands, flags):
            if bad:
                await _notify(on_step, f"Dropped {c.symbol}: matches your exclusions")
                if c.isUserRequested:
                    note(f"{c.symbol} was requested but also matches your exclusions, so it was left out.")
        keep = {c.symbol for c, bad in zip(cands, flags) if not bad}
        cands = [c for c in cands if c.symbol in keep]
        raws = [r for r in raws if r.symbol in keep]
    for cls in ("STOCKS", "CRYPTO"):
        have = sum(1 for c in cands if c.assetClass == cls)
        if targets.get(cls) and have < target_count(targets[cls]):
            note(f"Target {cls} {targets[cls]:g}%: only {have} candidates found, so it may be capped by diversification limits.")
    return cands, _findings(cap_per_symbol(raws, 1), "DISCOVERY", ids, res)


_SCALE = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_FIELD = [  # (label regex, fundamentals key); order matters: most specific first
    (r"free cash flow|\bfcf\b", "fcf"), (r"operating cash flow|cash flow from op|cash flow|\bcfo\b", "cfo"),
    (r"gross margin", "gross_margin"), (r"operating margin", "operating_margin"), (r"net margin|profit margin", "net_margin"),
    (r"net income", "net_income"), (r"revenue", "revenue"), (r"\bp/?e\b", "trailingPE"),
    (r"debt", "total_debt"), (r"\bcash\b", "cash"),
]
_MARGINS = {"gross_margin": "gross_profit", "operating_margin": "operating_income", "net_margin": "net_income"}
MARGIN_TOL_PP = 2.0


def _actual(key: str, f: dict) -> float | None:
    """Fetched value for a gate key (derived for fcf/margins); None when missing or zero = unverifiable."""
    if key == "fcf":
        v = None if f.get("cfo") is None or f.get("capex") is None else f["cfo"] - abs(f["capex"])
    elif key in _MARGINS:
        num, rev = f.get(_MARGINS[key]), f.get("revenue")
        v = None if num is None or not rev else num / rev * 100
    else:
        v = f.get(key)
    return v if v else None


def number_gate(text: str, fundamentals: dict) -> bool | None:
    """False = a figure contradicts the data (>2%, margins >2pp); None = nothing checkable OR any figure is unverifiable
    (fetched value missing/zero); True = every figure was checked against real data and matched. Never True on missing data."""
    lines = re.sub(r"[;,]|\.\s", "\n", text)
    results: list[bool | None] = []
    for p in extract_data_points(lines):
        label = p["label"].lower()
        key = next((k for pat, k in _FIELD if re.search(pat, label)), None)
        if key is None:
            continue
        actual = _actual(key, fundamentals)
        if actual is None:
            results.append(None)
            continue
        if key in _MARGINS:
            results.append(abs(p["reported_value"] - actual) <= MARGIN_TOL_PP)
        else:
            claimed = p["reported_value"] if key == "trailingPE" else p["reported_value"] * _SCALE.get(p["unit"].upper()[:1], 1.0)
            results.append(abs(claimed - actual) / abs(actual) * 100 <= config.NUMBER_TOL_PCT)
    if False in results or not results or None in results:
        return False if False in results else None
    return True


async def financials(stocks: list[Candidate], ids: Iterator[int], on_step=None, notes: str | None = None):
    funda = dict(zip([c.symbol for c in stocks],
                     await asyncio.gather(*(asyncio.to_thread(data.fundamentals, c.symbol) for c in stocks))))
    prompt = (
        f"You are the Financials analyst. {_context(notes)} " "Below is fundamentals JSON per stock (TTM from SEC filings plus yfinance P/E; "
        "values in USD). Use code execution ONCE (a single script covering all stocks) to compute margins (gross/operating/net), leverage (debt vs cash), "
        "free cash flow (cfo - capex) and valuation (P/E, market cap), then score each stock 1-10 across "
        "financial health, valuation and earnings quality. Verdict = 1 sentence (at most 25 words); keyFigures = one line "
        "'Revenue: $X B, Net income: $Y B, P/E: Z' using ONLY numbers from the data; risk = one bear-case sentence (at most 25 words). "
        "Date figures with asOf. " + DISCIPLINE + "\n\n" + llm_json(funda)
    )
    res = await llm.ask(model=FLASH, input=prompt, schema=FinancialsOut, tools=["code_execution"], agent="FINANCIALS", on_step=on_step, generation_config=LOW)
    scores, findings = {}, []
    for s in res.data.stocks:
        sym = s.symbol.strip().upper()
        scores[sym] = (float(s.score), s.verdict)
        f = funda.get(sym, {})
        ok = number_gate(f"{s.verdict}\n{s.keyFigures}", f)
        sent = "POSITIVE" if s.score >= 7 else "NEGATIVE" if s.score <= 4 else "NEUTRAL"
        findings += _findings([RawFinding(symbol=sym, claim=f"{s.verdict} ({s.keyFigures})", sentiment=sent)],
                              "FINANCIALS", ids, None, {0: ok})
        findings += _findings([RawFinding(symbol=sym, claim=s.risk, sentiment="NEGATIVE")], "FINANCIALS", ids, None,
                              {0: number_gate(s.risk, f)})
    return scores, findings


def llm_json(d: dict) -> str:
    import json
    return json.dumps(d, indent=1)


async def _search_agent(agent: str, role: str, symbols: list[str], ids, on_step, schema=ResearchOut, notes=None):
    prompt = (f"You are the {role}. {_context(notes)} Symbols: {', '.join(symbols)}. Use at most 3 Google searches, each "
              f"covering several symbols. {DISCIPLINE}")
    res = await llm.ask(model=FLASH, input=prompt, schema=schema, tools=["google_search"], agent=agent,
                        on_step=on_step, generation_config=LOW)
    return res, _findings(cap_per_symbol(res.data.findings), agent, ids, res)


async def landscape(symbols: list[str], ids: Iterator[int], on_step=None, notes: str | None = None):
    _, f = await _search_agent(
        "LANDSCAPE",
        "Landscape analyst: for each symbol give 1-2 findings on competitors, market position/moat and "
        "supply-chain exposure (cover small/mid-cap rivals and suppliers), plus one bear-case finding", symbols, ids, on_step, notes=notes)
    return f


async def news(symbols: list[str], ids: Iterator[int], on_step=None, notes: str | None = None):
    _, f = await _search_agent(
        "NEWS",
        "News analyst: for each symbol find the most material news of the last 30 days (earnings, regulation, "
        "deals, incidents), classify each POSITIVE/NEGATIVE/NEUTRAL, and include at least one NEGATIVE item per symbol", symbols, ids, on_step, notes=notes)
    return f


async def social(symbols: list[str], ids: Iterator[int], on_step=None, notes: str | None = None):
    res, f = await _search_agent(
        "SOCIAL",
        "Social-sentiment analyst: read Reddit and X/FinTwit discussion per symbol. Findings: dominant narrative, "
        "euphoria/capitulation extremes and crowd bear case. scores: net sentiment per symbol from -1 (very bearish) to 1 (very bullish); "
        "extreme unanimity is a contrarian warning", symbols, ids, on_step, schema=SocialOut, notes=notes)
    return {s.symbol.strip().upper(): s.score for s in res.data.scores}, f
