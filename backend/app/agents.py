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
    "Research discipline: (1) don't only name mega-caps - include small/mid caps and suppliers; "
    "(2) ignore concept labels, judge the actual business; (3) for EVERY symbol give at least one "
    "NEGATIVE (bear-case) finding - search 'risks / problems / bear case'; (4) date every figure "
    "(e.g. 'Q2 2026') and flag anything older than a year as possibly stale. "
    "Each claim is ONE sentence. Never invent numbers or sources."
)


class RawFinding(BaseModel):
    symbol: str | None = None
    claim: str
    sentiment: Sentiment


class Pick(BaseModel):
    symbol: str = Field(description="yfinance ticker, e.g. NVDA or BTC-USD")
    name: str
    assetClass: Literal["STOCKS", "CRYPTO"]
    thesis: str
    risk: str = Field(description="bear case for this symbol")


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


def _findings(raws, agent, ids: Iterator[int], sources, verified=None) -> list[Finding]:
    # ponytail: call-level source attribution (every finding gets the call's top 3); citation spans later
    src = [{"title": s["title"], "url": s["url"]} for s in sources[:3]]
    return [Finding(id=f"f_{next(ids)}", agent=agent, symbol=r.symbol, claim=r.claim, sentiment=r.sentiment,
                    sources=src, numbersVerified=(verified or {}).get(i)) for i, r in enumerate(raws)]


async def discovery(intake: IntakeInput, ids: Iterator[int], on_step=None) -> tuple[list[Candidate], list[Finding]]:
    prompt = (
        f"You are the Discovery analyst. Investor: ${intake.amountUsd:,.0f}, {intake.horizonYears}-year horizon, "
        f"{intake.riskTolerance} risk, theme: {intake.theme or 'none (broad diversified growth)'}.\n"
        "Use Google Search to pick 4-6 stocks and 2-3 cryptocurrencies suited to this theme and risk level. "
        "Symbols in yfinance form (NVDA, BTC-USD, ETH-USD). Per pick: a one-sentence thesis and one-sentence bear-case risk. "
        + DISCIPLINE
    )
    res = await llm.ask(model=FLASH, input=prompt, schema=DiscoveryOut, tools=["google_search"], agent="DISCOVERY", on_step=on_step)
    cands, raws = [], []
    for p in res.data.picks:
        sym = p.symbol.strip().upper()
        if any(c.symbol == sym for c in cands):
            continue
        if not await asyncio.to_thread(data.resolve, sym):
            await _notify(on_step, f"Dropped {sym}: no market data found")
            continue
        cands.append(Candidate(symbol=sym, name=p.name, assetClass=p.assetClass))
        raws += [RawFinding(symbol=sym, claim=p.thesis, sentiment="POSITIVE"),
                 RawFinding(symbol=sym, claim=p.risk, sentiment="NEGATIVE")]
    if not any(c.symbol == "GLD" for c in cands):
        cands.append(Candidate(symbol="GLD", name="SPDR Gold Shares", assetClass="GOLD"))
        raws += [RawFinding(symbol="GLD", claim="Gold is added as a diversifying hedge against equity and crypto drawdowns.", sentiment="NEUTRAL"),
                 RawFinding(symbol="GLD", claim="Gold pays no yield and can fall when real interest rates rise.", sentiment="NEGATIVE")]
    return cands, _findings(raws, "DISCOVERY", ids, res.sources)


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
    funda = {}
    for c in stocks:
        funda[c.symbol] = await asyncio.to_thread(data.fundamentals, c.symbol)
    prompt = (
        "You are the Financials analyst. Below is fundamentals JSON per stock (TTM from SEC filings plus yfinance P/E; "
        "values in USD). Use code execution to compute margins (gross/operating/net), leverage (debt vs cash), "
        "free cash flow (cfo - capex) and valuation (P/E, market cap), then score each stock 1-10 across "
        "financial health, valuation and earnings quality. Verdict = 1-2 sentences; keyFigures = one line "
        "'Revenue: $X B, Net income: $Y B, P/E: Z' using ONLY numbers from the data; risk = one bear-case sentence. "
        "Date figures with asOf. " + DISCIPLINE + "\n\n" + llm_json(funda)
    )
    res = await llm.ask(model=FLASH, input=prompt, schema=FinancialsOut, tools=["code_execution"], agent="FINANCIALS", on_step=on_step)
    scores, findings = {}, []
    for s in res.data.stocks:
        sym = s.symbol.strip().upper()
        scores[sym] = (float(s.score), s.verdict)
        f = funda.get(sym, {})
        ok = number_gate(f"{s.verdict}\n{s.keyFigures}", f)
        sent = "POSITIVE" if s.score >= 7 else "NEGATIVE" if s.score <= 4 else "NEUTRAL"
        findings += _findings([RawFinding(symbol=sym, claim=f"{s.verdict} ({s.keyFigures})", sentiment=sent)],
                              "FINANCIALS", ids, [], {0: ok})
        findings += _findings([RawFinding(symbol=sym, claim=s.risk, sentiment="NEGATIVE")], "FINANCIALS", ids, [],
                              {0: number_gate(s.risk, f)})
    return scores, findings


def llm_json(d: dict) -> str:
    import json
    return json.dumps(d, indent=1)


async def _search_agent(agent: str, role: str, symbols: list[str], ids, on_step, schema=ResearchOut):
    prompt = f"You are the {role}. Symbols: {', '.join(symbols)}. Use Google Search. {DISCIPLINE}"
    res = await llm.ask(model=FLASH, input=prompt, schema=schema, tools=["google_search"], agent=agent, on_step=on_step)
    return res, _findings(res.data.findings, agent, ids, res.sources)


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
