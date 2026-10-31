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

from . import config, data, llm, preferences
from .models import AssetClass, Candidate, Finding, IntakeInput, ResolvedPreferences
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


SPAN_THRESHOLD = 0.45
SOURCE_STATS: Counter = Counter()  # (agent, "span" | "title" | "fallback" | "none") -> findings; lets a run report how sources were found
_NUM = re.compile(r"\d[\d,]*\.?\d*")
_ENT = re.compile(r"\b[A-Z][A-Za-z0-9&.\-]{1,}\b")


def _numbers(text: str) -> set[str]:
    """Figures in the text, minus calendar years (they appear in every sentence and say nothing about which one a claim came from)."""
    nums = {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}
    return {n for n in nums if not re.fullmatch(r"(19|20)\d\d", n)}


def _entities(text: str) -> set[str]:
    return {e.lower() for e in _ENT.findall(text)} - _STOP


def _segments(res) -> list[tuple[str, list[dict]]]:
    """[(grounded text covered by a citation span, [citations sharing that span])]; [] when offsets/text are missing.
    Gemini's citation offsets are UTF-8 BYTE offsets into the grounded text (verified live: char slicing drifts after
    any non-ASCII character such as an en dash), so slice the encoded bytes."""
    text = (getattr(res, "grounded_text", "") or "").encode("utf-8")
    by_span: dict[tuple[int, int], list[dict]] = {}
    for c in getattr(res, "citations", None) or []:
        st, en = c.get("start"), c.get("end")
        if text and isinstance(st, int) and isinstance(en, int) and 0 <= st < en <= len(text):
            by_span.setdefault((st, en), []).append(c)
    return [(text[st:en].decode("utf-8", errors="ignore"), cs) for (st, en), cs in by_span.items()]


def _span_score(claim: str, seg: str) -> float:
    """Token recall of the claim in the segment, plus extra credit for shared numbers and named entities."""
    ct, st = _tokens(claim), _tokens(seg)
    base = len(ct & st) / len(ct) if ct else 0.0
    return base + 0.5 * min(2, len(_numbers(claim) & _numbers(seg))) + 0.15 * min(3, len(_entities(claim) & _entities(seg)))


def _pick(claim: str, symbol: str | None, res, used: Counter) -> tuple[list[dict], str]:
    """(sources, how) for ONE claim. how = 'span' (matched a cited text span), 'title' (keyword match on citation titles),
    'fallback' (call-level sources, nothing matched) or 'none'."""
    scored = sorted(((_span_score(claim, seg), cs) for seg, cs in _segments(res)), key=lambda t: -t[0])
    out, seen = [], set()
    best = scored[0][0] if scored else 0.0
    for score, cs in scored:
        if score < max(SPAN_THRESHOLD, 0.75 * best):  # the best segment, plus near-ties only
            break
        for c in sorted(cs, key=lambda c: used[c["url"]]):
            if c["url"] not in seen and len(out) < 3:
                seen.add(c["url"])
                out.append({"title": c.get("title", ""), "url": c["url"]})  # publisher title kept verbatim, never invented
    if out:
        for c in out:
            used[c["url"]] += 1
        return out, "span"
    words = _tokens(f"{claim} {symbol or ''}")  # second tier: citation title/url keywords (publisher names like vertiv.com)
    titled = [c for c in (getattr(res, "citations", None) or []) if words & _tokens(f"{c.get('title', '')} {c['url']}")]
    titled.sort(key=lambda c: (-len(words & _tokens(f"{c.get('title', '')} {c['url']}")), used[c["url"]]))
    if titled:
        picks, seen_t = [], set()
        for c in titled:
            if c["url"] not in seen_t and len(picks) < 3:
                seen_t.add(c["url"]); picks.append(c); used[c["url"]] += 1
        return [{"title": c.get("title", ""), "url": c["url"]} for c in picks], "title"
    pool = sorted(res.sources or [], key=lambda c: used[c["url"]])  # call-level, least-reused first
    for c in pool[:3]:
        used[c["url"]] += 1
    return [{"title": c.get("title", ""), "url": c["url"]} for c in pool[:3]], ("fallback" if pool else "none")


def _pick_sources(claim: str, symbol: str | None, res, used: Counter | None = None) -> list[dict]:
    return _pick(claim, symbol, res, used if used is not None else Counter())[0]


def _findings(raws, agent, ids: Iterator[int], res=None, verified=None) -> list[Finding]:
    used: Counter = Counter()
    out = []
    for i, r in enumerate(raws):
        sources = []
        if res is not None:
            sources, how = _pick(r.claim, r.symbol, res, used)
            SOURCE_STATS[(agent, how)] += 1
        out.append(Finding(id=f"f_{next(ids)}", agent=agent, symbol=r.symbol, claim=_trim(r.claim), sentiment=r.sentiment,
                           sources=sources, numbersVerified=(verified or {}).get(i)))
    return out


def horizon_text(years: float) -> str:
    """0.25 -> '3-month', 2 -> '2-year', 1.5 -> '1.5-year' (a horizon under a year is read in months)."""
    return f"{round(years * 12)}-month" if years < 1 else f"{years:g}-year"


SHORT_HORIZON_HINT = "Short horizon: favour liquidity and capital preservation; a deep drawdown may not recover in time."


def horizon_context(years: float) -> str:
    return f"{horizon_text(years)} horizon." + (f" {SHORT_HORIZON_HINT}" if years <= 1 else "")


def _context(notes: str | None = None) -> str:
    """Shared prompt preamble: today's date (searches kept stale years) + the investor's free-text notes, verbatim."""
    out = f"Today is {datetime.date.today().isoformat()}. Prefer sources from the last 90 days; date every figure."
    return out + (f" Investor notes (verbatim): {notes}" if notes else "")


DEFENSIVE = {"BRK-B": "Berkshire Hathaway", "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble", "KO": "Coca-Cola",
             "PEP": "PepsiCo", "WMT": "Walmart", "COST": "Costco", "MCD": "McDonald's", "V": "Visa", "MRK": "Merck"}


async def defensive_picks(exclude: set[str], n: int = 2, ok=None, skipped: list | None = None) -> list[tuple[str, str, float]]:
    """The n lowest-weekly-volatility (5y) names from DEFENSIVE: [(symbol, name, annualised vol %)].
    `ok(sym) -> (bool, reason)` is the preference matcher; rejected names are appended to `skipped` as (sym, reason)."""
    closes, _missing = await asyncio.to_thread(data.prices, list(DEFENSIVE), 5)  # caches parquet per symbol
    if closes is None or closes.empty:
        return []
    vol = (closes.pct_change().std() * 52 ** 0.5 * 100).dropna().sort_values()
    out = []
    for sym, v in vol.items():
        if sym in exclude or not await asyncio.to_thread(data.resolve, sym):
            continue
        good, why = await ok(sym) if ok else (True, "")
        if not good:
            if skipped is not None:
                skipped.append((sym, why))
            continue
        out.append((sym, DEFENSIVE[sym], round(float(v), 1)))
        if len(out) == n:
            break
    return out


# Bond ballast is chosen by DATA (lowest 5y max drawdown): in 2022 only short treasuries cushioned stocks.
BOND_POOL = {"SHY": "iShares 1-3 Year Treasury Bond ETF", "BIL": "SPDR 1-3 Month T-Bill ETF",
             "VGSH": "Vanguard Short-Term Treasury ETF", "TIP": "iShares TIPS Bond ETF",
             "BND": "Vanguard Total Bond Market ETF", "AGG": "iShares Core US Aggregate Bond ETF",
             "IEF": "iShares 7-10 Year Treasury Bond ETF"}


async def bond_picks(exclude: set[str], n: int, ok=None, skipped: list | None = None) -> list[tuple[str, str, float]]:
    """The n BOND_POOL ETFs with the smallest 5y max drawdown: [(symbol, name, max drawdown %)] (negative numbers)."""
    if n <= 0:
        return []
    closes, _missing = await asyncio.to_thread(data.prices, list(BOND_POOL), 5)  # caches parquet per symbol
    if closes is None or closes.empty:
        return []
    dd = ((closes / closes.cummax() - 1).min() * 100).dropna().sort_values(ascending=False)
    out = []
    for sym, v in dd.items():
        if sym in exclude or not await asyncio.to_thread(data.resolve, sym):
            continue
        good, why = await ok(sym) if ok else (True, "")
        if not good:
            if skipped is not None:
                skipped.append((sym, why))
            continue
        out.append((sym, BOND_POOL[sym], round(float(v), 1)))
        if len(out) == n:
            break
    return out


MIN_STOCKS = 6   # fewer allowed stock candidates than this triggers one extra Discovery pass
MIN_HOLDINGS = 4  # 25% max per holding => at least 4 candidates so the cap is feasible
MIN_THIN = 4     # still fewer than this after the pass: tell the user the universe is thin
STYLE_HINTS = {
    "MEME": "meme coins (e.g. DOGE, SHIB, PEPE, WIF, BONK) and meme-popular stocks",
    "DIVIDEND": "reliable dividend payers",
    "VALUE": "low P/E value stocks",
    "GROWTH": "high-growth companies",
    "MOMENTUM": "6-month price-momentum leaders",
    "ESG": "ESG-friendly names (avoid fossil fuels, tobacco, weapons)",
    "DEFENSIVE": "defensive, low-volatility names",
}


def target_count(pct: float) -> int:
    return min(8, max(2, math.ceil(pct / 10) + 1))


async def _ok(sym: str, rp: ResolvedPreferences, name: str | None = None) -> tuple[bool, str]:
    """THE preference check: the one matcher (preferences.allowed) on the symbol's Yahoo profile."""
    p = await asyncio.to_thread(preferences.profile, sym)
    if name and not p["longName"]:
        p["longName"] = name.lower()
    return preferences.allowed(sym, rp, p)


def _skip_note(kind: str, skipped: list[tuple[str, str]], got: int, wanted: int) -> str | None:
    if not skipped:
        return None
    names = ", ".join(s for s, _ in skipped[:3]) + (f" and {len(skipped) - 3} more" if len(skipped) > 3 else "")
    msg = f"Skipped {names} as {kind}: {skipped[0][1]}."
    if got < wanted:
        msg += f" No allowed {kind} was available, so the portfolio has less cushioning and the risk limit may be harder to meet."
    return msg


async def discovery(intake: IntakeInput, ids: Iterator[int], on_step=None, notes_out: list[str] | None = None,
                    rp: ResolvedPreferences | None = None) -> tuple[list[Candidate], list[Finding]]:
    pr = intake.preferences
    rp = rp or ResolvedPreferences(mustInclude=[m.strip().upper() for m in (pr.mustInclude if pr else []) if m.strip()])
    must = [m.upper() for m in rp.mustInclude]
    styles = list(pr.styles) if pr else []
    targets = {**((pr.assetClassTargets or {}) if pr else {}), **rp.classTargets}
    notes = pr.notes if pr else None
    note = (lambda m: notes_out.append(m)) if notes_out is not None else (lambda m: None)

    def count_txt(cls: str, default: str) -> str:
        if cls in targets:
            return "no" if targets[cls] == 0 else f"at least {target_count(targets[cls])}"
        if cls == "STOCKS" and not rp.only.is_empty() and preferences._only_covers(rp.only, "STOCKS"):
            return "at least 5 (all matching the ONLY filter)"
        return default

    restricted = not rp.only.is_empty() and preferences._only_covers(rp.only, "STOCKS")
    theme_line = ((f"optional context (do NOT narrow the ONLY rule with it): also related to {intake.theme} if possible." if intake.theme
                   else "no theme.") if restricted else f"theme: {intake.theme or 'none (broad diversified growth)'}.")
    prompt = (
        f"You are the Discovery analyst. {_context(notes)}\n"
        f"Investor: ${intake.amountUsd:,.0f}, {horizon_text(intake.horizonYears)} horizon, "
        f"{intake.riskTolerance} risk, " + theme_line + "\n"
        + (f"{SHORT_HORIZON_HINT}\n" if intake.horizonYears <= 1 else "")
        + (f"Resolved preferences (hard rules): {preferences.prompt_text(rp)}\n" if preferences.prompt_text(rp) else "")
        + f"Run 2-3 Google searches, then pick {count_txt('STOCKS', '4-6')} stocks (mix of leaders plus small/mid caps or suppliers) and "
        f"{count_txt('CRYPTO', '3-5' if 'MEME' in styles else '2-3')} cryptocurrencies suited to the theme and risk level. "
        "Symbols in yfinance form (NVDA, BTC-USD). "
        + (f"Investment styles to favour: {'; '.join(STYLE_HINTS[x] for x in styles)}. " if styles else "")
        + (f"The investor insists on {', '.join(must)}; they are added automatically, do not spend picks on them. " if must else "")
        + "One-sentence thesis per pick (at most 25 words, dated figures only). No bear-case research. "
        "Defensive low-volatility large caps, bond ETFs and gold may be added automatically afterwards (only if the "
        "investor's preferences allow), so focus on the theme."
    )
    res = await llm.ask(model=FLASH, input=prompt, schema=DiscoveryOut, tools=["google_search"], agent="DISCOVERY",
                        on_step=on_step, generation_config=LOW)
    cands, raws = [], []

    async def take(picks) -> None:
        """Resolve + preference-filter picks and append them to cands/raws (deduped)."""
        for p in picks:
            sym = p.symbol.strip().upper()
            if any(c.symbol == sym for c in cands):
                continue
            if not await asyncio.to_thread(data.resolve, sym):
                await _notify(on_step, f"Dropped {sym}: no market data found")
                continue
            good, why = await _ok(sym, rp, p.name)
            if not good:
                await _notify(on_step, f"Dropped {sym}: {why}")
                continue
            cands.append(Candidate(symbol=sym, name=p.name, assetClass=preferences.asset_class(sym), isUserRequested=sym in must))
            raws.append(RawFinding(symbol=sym, claim=p.thesis, sentiment="POSITIVE"))

    await take(res.data.picks)
    for sym in must:  # user-requested symbols are always candidates, if they exist
        if any(c.symbol == sym for c in cands):
            continue
        if not await asyncio.to_thread(data.resolve, sym):
            await _notify(on_step, f"Could not find market data for requested symbol {sym}")
            note(f"Requested {sym} could not be resolved (no market data), so it was left out.")
            continue
        cands.append(Candidate(symbol=sym, name=sym, assetClass=preferences.asset_class(sym), isUserRequested=True))
        raws.append(RawFinding(symbol=sym, claim="Included at the investor's request.", sentiment="NEUTRAL"))

    ok = lambda sym: _ok(sym, rp)  # noqa: E731  (our defaults must pass the same matcher as everything else)
    taken = {c.symbol for c in cands}
    if (intake.riskTolerance in ("CONSERVATIVE", "MODERATE") or "DEFENSIVE" in styles) and targets.get("STOCKS") != 0:
        want, skipped = (3 if "DEFENSIVE" in styles else 2), []
        picks = await defensive_picks(taken, want, ok, skipped)
        for sym, name, v in picks:
            cands.append(Candidate(symbol=sym, name=name, assetClass="STOCKS"))
            raws.append(RawFinding(symbol=sym, sentiment="NEUTRAL",
                                   claim=f"Added as defensive ballast for a {intake.riskTolerance} profile (5y weekly vol {v}%)."))
        if msg := _skip_note("defensive ballast", skipped, len(picks), want):
            note(msg)

    n_stocks = lambda: sum(1 for c in cands if c.assetClass == "STOCKS")  # noqa: E731
    if n_stocks() < MIN_STOCKS and targets.get("STOCKS") != 0 and "STOCKS" not in rp.exclude.assetClasses:
        # thin universe: ONE focused extra pass for liquid, established names that still pass the same filters
        criteria = (preferences.describe(rp.only) or "; ".join(rp.interpretations) or intake.theme or
                    f"a {intake.riskTolerance.lower()}-risk investor")
        extra_prompt = (
            f"You are the Discovery analyst. {_context(notes)}\n"
            f"List 8-10 liquid, established (prefer large/mid-cap, profitable where possible) stocks matching: {criteria}. "
            f"Already chosen (do not repeat): {', '.join(c.symbol for c in cands) or 'none'}. "
            + (f"Resolved preferences (hard rules): {preferences.prompt_text(rp)} " if preferences.prompt_text(rp) else "")
            + (f"Optional context: also related to {intake.theme} if possible. " if intake.theme else "")
            + "Symbols in yfinance form. One-sentence thesis per pick (at most 25 words, dated figures only)."
        )
        await _notify(on_step, f"Only {n_stocks()} matching stocks so far; searching for more")
        res2 = await llm.ask(model=FLASH, input=extra_prompt, schema=DiscoveryOut, tools=["google_search"], agent="DISCOVERY",
                             on_step=on_step, generation_config=LOW)
        await take([p for p in res2.data.picks if p.assetClass == "STOCKS"])
        if n_stocks() < MIN_THIN:
            note(f"Your preferred universe is thin: only {n_stocks()} matching stock(s) could be found, so the portfolio leans on "
                 "bonds and other allowed assets and may be less diversified.")

    bond_target = targets.get("BONDS")
    n_bonds = {"CONSERVATIVE": 2, "MODERATE": 1}.get(intake.riskTolerance, 0)
    if "DEFENSIVE" in styles:
        n_bonds = max(n_bonds, 1)
    if bond_target is not None and bond_target > 0:
        n_bonds = max(n_bonds, 2 if bond_target > 25 else 1)  # 25% max per holding => need 2+ for targets above 25
    elif bond_target == 0:
        n_bonds = 0
    have = sum(1 for c in cands if c.assetClass == "BONDS")  # e.g. user already requested one
    if n_bonds - have > 0:
        skipped = []
        picks = await bond_picks({c.symbol for c in cands}, n_bonds - have, ok, skipped)
        for sym, name, dd in picks:
            cands.append(Candidate(symbol=sym, name=name, assetClass="BONDS"))
            raws.append(RawFinding(symbol=sym, sentiment="NEUTRAL",
                                   claim=f"Added as bond ballast for a {intake.riskTolerance} profile (5y max drawdown {dd}%)."))
        if msg := _skip_note("bond ballast", skipped, len(picks), n_bonds - have):
            note(msg)

    # 25% max per holding => need >= 4 holdings; a thin stock universe gets a 2nd bond ETF so the cap stays feasible
    bonds_now = sum(1 for c in cands if c.assetClass == "BONDS")
    want_bonds = max(2 if n_stocks() < MIN_THIN else 0, MIN_HOLDINGS - len(cands) - 1 + bonds_now)  # -1: gold may still be added
    if targets.get("BONDS") != 0 and want_bonds > bonds_now:
        skipped = []
        picks = await bond_picks({c.symbol for c in cands}, want_bonds - bonds_now, ok, skipped)
        for sym, name, dd in picks:
            cands.append(Candidate(symbol=sym, name=name, assetClass="BONDS"))
            raws.append(RawFinding(symbol=sym, sentiment="NEUTRAL",
                                   claim=f"Added as bond ballast so no holding needs to exceed 25% (5y max drawdown {dd}%)."))
        if msg := _skip_note("bond ballast", skipped, len(picks), want_bonds - bonds_now):
            note(msg)

    if not any(c.assetClass == "GOLD" for c in cands):
        good, why = await _ok("GLD", rp)
        if good:
            cands.append(Candidate(symbol="GLD", name="SPDR Gold Shares", assetClass="GOLD"))
            raws.append(RawFinding(symbol="GLD", claim="Gold is added as a diversifying hedge against equity and crypto drawdowns.", sentiment="NEUTRAL"))
        else:
            note(f"Skipped gold (GLD): {why}.")

    for cls in ("STOCKS", "CRYPTO"):
        have = sum(1 for c in cands if c.assetClass == cls)
        if targets.get(cls) and have < target_count(targets[cls]):
            note(f"Target {cls} {targets[cls]:g}%: only {have} candidates found, so it may be capped by diversification limits.")
    if (targets.get("BONDS") or 0) > 25 and sum(1 for c in cands if c.assetClass == "BONDS") < 2:
        note("Target BONDS above 25% needs 2+ bond ETFs (25% max per holding), but fewer were available.")
    # only the model's own picks cite the search; ballast/gold/requested findings are ours and carry no sources
    sourced = [r for r in raws if r.sentiment == "POSITIVE"]
    ours = [r for r in raws if r.sentiment != "POSITIVE"]
    return cands, _findings(cap_per_symbol(sourced, 1), "DISCOVERY", ids, res) + _findings(cap_per_symbol(ours, 1), "DISCOVERY", ids)


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
