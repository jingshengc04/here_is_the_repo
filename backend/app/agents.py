"""Research agents: each is ONE llm.ask call covering all symbols. Returns (data, findings)."""
import asyncio
import inspect
import re
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


def _pick_sources(claim: str, symbol: str | None, res) -> list[dict]:
    """Up to 3 sources: citations whose title shares words with the claim first, then call-level sources."""
    words = {w for w in re.findall(r"[a-z0-9]{4,}", f"{claim} {symbol or ''}".lower())}
    cites = getattr(res, "citations", None) or []
    scored = sorted(cites, key=lambda c: -len(words & set(re.findall(r"[a-z0-9]{4,}", c.get("title", "").lower()))))
    out, seen = [], set()
    for c in [*scored, *(res.sources or [])]:
        if c["url"] not in seen and len(out) < 3:
            seen.add(c["url"])
            out.append({"title": c.get("title", ""), "url": c["url"]})
    return out


def _findings(raws, agent, ids: Iterator[int], res=None, verified=None) -> list[Finding]:
    return [Finding(id=f"f_{next(ids)}", agent=agent, symbol=r.symbol, claim=_trim(r.claim), sentiment=r.sentiment,
                    sources=_pick_sources(r.claim, r.symbol, res) if res is not None else [],
                    numbersVerified=(verified or {}).get(i)) for i, r in enumerate(raws)]


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


async def discovery(intake: IntakeInput, ids: Iterator[int], on_step=None) -> tuple[list[Candidate], list[Finding]]:
    prompt = (
        f"You are the Discovery analyst. Investor: ${intake.amountUsd:,.0f}, {intake.horizonYears}-year horizon, "
        f"{intake.riskTolerance} risk, theme: {intake.theme or 'none (broad diversified growth)'}.\n"
        "Run 2-3 Google searches, then pick 4-6 stocks (mix of leaders plus small/mid caps or suppliers) and 2-3 "
        "cryptocurrencies suited to the theme and risk level. Symbols in yfinance form (NVDA, BTC-USD). "
        "One-sentence thesis per pick (at most 25 words, dated figures only). No bear-case research. "
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
        cands.append(Candidate(symbol=sym, name=p.name, assetClass=p.assetClass))
        raws.append(RawFinding(symbol=sym, claim=p.thesis, sentiment="POSITIVE"))
    if intake.riskTolerance in ("CONSERVATIVE", "MODERATE"):
        for sym, name, v in await defensive_picks({c.symbol for c in cands}):
            cands.append(Candidate(symbol=sym, name=name, assetClass="STOCKS"))
            raws.append(RawFinding(symbol=sym, sentiment="NEUTRAL",
                                   claim=f"Added as defensive ballast for a {intake.riskTolerance} profile (5y weekly vol {v}%)."))
    if not any(c.symbol == "GLD" for c in cands):
        cands.append(Candidate(symbol="GLD", name="SPDR Gold Shares", assetClass="GOLD"))
        raws.append(RawFinding(symbol="GLD", claim="Gold is added as a diversifying hedge against equity and crypto drawdowns.", sentiment="NEUTRAL"))
    return cands, _findings(cap_per_symbol(raws, 1), "DISCOVERY", ids, res)


_SCALE = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_FIELD = [("net income", "net_income"), ("revenue", "revenue"), ("p/e", "trailingPE"), ("pe", "trailingPE")]


def number_gate(text: str, fundamentals: dict) -> bool | None:
    """True = every checked number within NUMBER_TOL_PCT of SEC/yfinance data, False = a mismatch, None = nothing checkable."""
    lines = re.sub(r"[;,]|\.\s", "\n", text)
    checked = []
    for p in extract_data_points(lines):
        label = p["label"].lower()
        key = next((k for needle, k in _FIELD if needle in label), None)
        actual = fundamentals.get(key) if key else None
        if not actual:
            continue
        claimed = p["reported_value"] * _SCALE.get(p["unit"].upper()[:1], 1.0) if key != "trailingPE" else p["reported_value"]
        checked.append(abs(claimed - actual) / abs(actual) * 100 <= config.NUMBER_TOL_PCT)
    return all(checked) if checked else None


async def financials(stocks: list[Candidate], ids: Iterator[int], on_step=None):
    funda = dict(zip([c.symbol for c in stocks],
                     await asyncio.gather(*(asyncio.to_thread(data.fundamentals, c.symbol) for c in stocks))))
    prompt = (
        "You are the Financials analyst. Below is fundamentals JSON per stock (TTM from SEC filings plus yfinance P/E; "
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


async def _search_agent(agent: str, role: str, symbols: list[str], ids, on_step, schema=ResearchOut):
    prompt = (f"You are the {role}. Symbols: {', '.join(symbols)}. Use at most 3 Google searches, each covering several "
              f"symbols. {DISCIPLINE}")
    res = await llm.ask(model=FLASH, input=prompt, schema=schema, tools=["google_search"], agent=agent,
                        on_step=on_step, generation_config=LOW)
    return res, _findings(cap_per_symbol(res.data.findings), agent, ids, res)


async def landscape(symbols: list[str], ids: Iterator[int], on_step=None):
    _, f = await _search_agent(
        "LANDSCAPE",
        "Landscape analyst: for each symbol give 1-2 findings on competitors, market position/moat and "
        "supply-chain exposure (cover small/mid-cap rivals and suppliers), plus one bear-case finding", symbols, ids, on_step)
    return f


async def news(symbols: list[str], ids: Iterator[int], on_step=None):
    _, f = await _search_agent(
        "NEWS",
        "News analyst: for each symbol find the most material news of the last 30 days (earnings, regulation, "
        "deals, incidents), classify each POSITIVE/NEGATIVE/NEUTRAL, and include at least one NEGATIVE item per symbol", symbols, ids, on_step)
    return f


async def social(symbols: list[str], ids: Iterator[int], on_step=None):
    res, f = await _search_agent(
        "SOCIAL",
        "Social-sentiment analyst: read Reddit and X/FinTwit discussion per symbol. Findings: dominant narrative, "
        "euphoria/capitulation extremes and crowd bear case. scores: net sentiment per symbol from -1 (very bearish) to 1 (very bullish); "
        "extreme unanimity is a contrarian warning", symbols, ids, on_step, schema=SocialOut)
    return {s.symbol.strip().upper(): s.score for s in res.data.scores}, f
