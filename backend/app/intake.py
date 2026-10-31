"""Intake chat agent for portfolio onboarding."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app import config, llm
from app.models import (
    AssetClass,
    IntakeDraft,
    IntakeInput,
    IntakeTurn,
    IntakeTurnRequest,
    Preferences,
    RiskTolerance,
    Style,
)

logger = logging.getLogger("app.intake")

INTAKE_SYSTEM_INSTRUCTION = """\
You are a friendly, encouraging onboarding assistant for a portfolio-research demo.
Your job is to collect 3 required fields and any personalisation preferences from the user to build their investment portfolio:
1. amountUsd: Total amount to invest in USD (must be between 100 and 10,000,000).
2. horizonYears: Investment time horizon in years (integer between 1 and 10 years).
3. riskTolerance: One of CONSERVATIVE, MODERATE, or AGGRESSIVE. Map plain language like "safe/steady/can't stomach crashes/preservation/fairly cautious" to CONSERVATIVE, "balanced/growth but careful" to MODERATE, and "high growth/crypto/moonshots/max returns" to AGGRESSIVE.
Optional:
4. theme: An investment theme or focus if mentioned (e.g., "AI infrastructure", "clean energy"), max 200 characters.
5. preferences: Personalisation for ANY preference (e.g. "I don't like gold", "only bio stocks", "no tech", "no Chinese companies", "companies I use every day", "ethical only"):
   - onlyInclude: Restrict-the-universe phrases or words in user's own words, without over-normalising (e.g., "only bio stocks" -> ["bio stocks"], "only biotech" -> ["biotech"], "ethical investing only" -> ["ethical investing"]). Max 10.
   - exclude: Symbols or plain words to avoid/exclude in user's own words (e.g., "no tech" -> ["tech"], "no Chinese companies" -> ["Chinese companies"], "no gold" -> ["gold"]). Max 20.
   - mustInclude: Specific tickers user insists on (e.g. "NVDA", "BTC-USD", "DOGE-USD"). Max 10.
   - assetClassTargets: Specific target percentages (e.g. {"CRYPTO": 30}). For "no gold" (or no crypto / bonds / stocks), set {"GOLD": 0} as well as exclude: ["gold"].
   - styles: Styles if mentioned: GROWTH, VALUE, DIVIDEND, MOMENTUM, MEME, ESG, DEFENSIVE (e.g. "dividends" -> DIVIDEND, "ethical" -> ESG).
   - notes: "I want X" when X is not a ticker (e.g., "companies I use every day"), or anything else that doesn't fit other fields, <= 500 chars.

RULES:
- Ask ONE short, conversational question at a time for whatever information is missing or unclear.
- Keep your replies concise: at most 2 sentences.
- When all 3 required fields (amount, horizon 1-10 years, risk tolerance) are known:
  - If no preferences exist yet, ask ONE short optional question: "Anything you specifically want or want to avoid, like certain coins, companies, dividends, or ethical exclusions?"
  - If preferences already exist or the user replies with "no"/"nothing", summarize the intake and tell the user they can review and run.
- Conflict warning: Judge whether ANY stated preference may clash with the risk level (concentration from "only X", high-volatility assets, removing hedges like gold or bonds for a conservative user). If it does, mention the trade-off once in <= 1 sentence and offer an alternative (in `conflictNote` and/or in reply). Never repeat it if already mentioned in earlier assistant messages.
- Ambiguity: If an "only" phrase is ambiguous about other asset classes (e.g. "only bio stocks": does the user also want no crypto, gold or bonds?), ask ONE short clarifying question in `clarifyingQuestion` or reply: "Do you also want no crypto, gold, or bonds, or should those asset classes still be included?"
- Merge across turns: never drop preferences gathered earlier.
- EXECUTION GUARD: If the user asks to place, buy, or execute a trade (e.g., "buy NVDA now"), politely state that this tool researches and backtests portfolios but does not place trades or give financial advice, and continue the intake.
- If the user provides a horizon outside 1 to 10 years (e.g. 15 years), explain that the research model supports horizons from 1 to 10 years, and ask them for a horizon within that range.
"""

KNOWN_CRYPTO_MAP = {
    "doge": "DOGE-USD",
    "dogecoin": "DOGE-USD",
    "pepe": "PEPE-USD",
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
    "shiba": "SHIB-USD",
    "shib": "SHIB-USD",
    "shiba inu": "SHIB-USD",
    "cardano": "ADA-USD",
    "ada": "ADA-USD",
    "ripple": "XRP-USD",
    "xrp": "XRP-USD",
    "bonk": "BONK-USD",
    "floki": "FLOKI-USD",
    "wif": "WIF-USD",
    "dogwifhat": "WIF-USD",
}

KNOWN_COMPANY_MAP = {
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "googl": "GOOGL",
    "goog": "GOOGL",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "meta": "META",
    "facebook": "META",
    "palantir": "PLTR",
    "pltr": "PLTR",
    "amd": "AMD",
}

KNOWN_SECTOR_WORDS = {
    "oil", "tobacco", "weapons", "defense", "fossil fuels", "gambling", "alcohol",
    "cannabis", "adult", "guns", "coal", "mining", "oil companies", "gas",
}

STYLE_MAP = {
    "growth": "GROWTH",
    "value": "VALUE",
    "dividend": "DIVIDEND",
    "dividends": "DIVIDEND",
    "momentum": "MOMENTUM",
    "meme": "MEME",
    "memes": "MEME",
    "meme coin": "MEME",
    "meme coins": "MEME",
    "memecoins": "MEME",
    "esg": "ESG",
    "ethical": "ESG",
    "green": "ESG",
    "sustainable": "ESG",
    "defensive": "DEFENSIVE",
    "safe": "DEFENSIVE",
    "cautious": "DEFENSIVE",
}

ASSET_CLASS_MAP = {
    "crypto": "CRYPTO",
    "cryptocurrency": "CRYPTO",
    "cryptos": "CRYPTO",
    "coins": "CRYPTO",
    "stocks": "STOCKS",
    "stock": "STOCKS",
    "equities": "STOCKS",
    "shares": "STOCKS",
    "gold": "GOLD",
    "bonds": "BONDS",
    "bond": "BONDS",
}

MEME_COIN_SYMBOLS = {
    "DOGE-USD",
    "PEPE-USD",
    "SHIB-USD",
    "BONK-USD",
    "FLOKI-USD",
    "WIF-USD",
    "MEME-USD",
}


def normalize_symbol_or_entity(name: str, is_exclude: bool = False) -> str:
    cleaned = name.strip()
    lower = cleaned.lower()

    if is_exclude:
        # Check sectors first if is_exclude
        if lower in KNOWN_SECTOR_WORDS:
            # Clean "oil companies" -> "oil"
            if "oil" in lower:
                return "oil"
            return lower

    if lower in KNOWN_CRYPTO_MAP:
        return KNOWN_CRYPTO_MAP[lower]

    if lower in KNOWN_COMPANY_MAP:
        return KNOWN_COMPANY_MAP[lower]

    if is_exclude and lower in KNOWN_SECTOR_WORDS:
        return lower

    # If it's crypto-like (e.g. DOGE-USD)
    if re.match(r"^[A-Z0-9]+-USD$", cleaned, re.IGNORECASE):
        return cleaned.upper()

    # If already an uppercase ticker (e.g. NVDA, AAPL, TSLA)
    if re.match(r"^[A-Z]{1,5}$", cleaned):
        return cleaned

    # For exclusion, keep word if not a company
    if is_exclude:
        return cleaned

    return cleaned.upper()


class _PreferencesExtraction(BaseModel):
    mustInclude: list[str] = Field(
        default_factory=list,
        description="Symbols user insists on including (e.g. DOGE-USD, PEPE-USD, NVDA, BTC-USD). Max 10.",
    )
    exclude: list[str] = Field(
        default_factory=list,
        description="Symbols or plain words to avoid/exclude in user's words (e.g. TSLA, tech, Chinese companies, gold). Max 20.",
    )
    assetClassTargets: dict[str, float] | None = Field(
        default=None,
        description="Target % per asset class (STOCKS, CRYPTO, GOLD, BONDS), each 0-100. If 'no gold', include {'GOLD': 0}.",
    )
    styles: list[str] = Field(
        default_factory=list,
        description="Investment styles: MEME, DIVIDEND, VALUE, GROWTH, MOMENTUM, ESG, DEFENSIVE.",
    )
    notes: str | None = Field(
        default=None,
        description="Free text notes or specific requests, max 500 chars.",
    )
    onlyInclude: list[str] = Field(
        default_factory=list,
        description="Restrict-the-universe phrases or words in user's words (e.g. 'bio stocks', 'biotech', 'ethical investing'). Max 10.",
    )


class _IntakeExtraction(BaseModel):
    reply: str = Field(description="Assistant reply to the user, <= 2 sentences.")
    amountUsd: float | None = Field(default=None, description="Amount in USD, or null if unknown.")
    horizonYears: int | None = Field(default=None, description="Horizon in years, 1-10, or null.")
    riskTolerance: Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"] | None = Field(
        default=None, description="Risk tolerance level, or null if unknown."
    )
    theme: str | None = Field(default=None, description="Optional investment theme or focus.")
    preferences: _PreferencesExtraction | None = Field(
        default=None, description="Personalisation preferences extracted from user."
    )
    conflictNote: str | None = Field(
        default=None,
        description="If any preference clashes with risk level, state trade-off in <= 1 sentence and offer an alternative. Null if no clash.",
    )
    clarifyingQuestion: str | None = Field(
        default=None,
        description="If an 'only' phrase is ambiguous about other asset classes ('only bio stocks': do they also want no crypto, gold or bonds?), ONE short clarifying question. Null if not ambiguous.",
    )


def validate_and_sanitize_preferences(
    raw: _PreferencesExtraction | Preferences | dict | None,
) -> tuple[Preferences | None, list[str]]:
    """Validates preferences against Pydantic model. Invalid parts are dropped. Never crashes."""
    if raw is None:
        return None, []

    if isinstance(raw, Preferences):
        return raw, []

    if isinstance(raw, _PreferencesExtraction):
        raw_dict = raw.model_dump()
    elif isinstance(raw, dict):
        raw_dict = dict(raw)
    else:
        return None, []

    dropped_warnings = []

    # 1. mustInclude (normalize and filter <= 40 chars, max 10)
    raw_must = raw_dict.get("mustInclude") or []
    must_include = []
    for s in raw_must:
        if isinstance(s, str) and s.strip():
            norm = normalize_symbol_or_entity(s, is_exclude=False)
            if len(norm) <= 40 and norm not in must_include:
                must_include.append(norm)
    must_include = must_include[:10]

    # 2. exclude (normalize and filter <= 40 chars, max 20)
    raw_ex = raw_dict.get("exclude") or []
    exclude = []
    for s in raw_ex:
        if isinstance(s, str) and s.strip():
            norm = normalize_symbol_or_entity(s, is_exclude=True)
            if len(norm) <= 40 and norm not in exclude:
                exclude.append(norm)
    exclude = exclude[:20]

    # 3. onlyInclude (filter <= 40 chars, max 10, keep user words)
    raw_only = raw_dict.get("onlyInclude") or []
    only_include = []
    for s in raw_only:
        if isinstance(s, str) and s.strip():
            norm = s.strip()
            if len(norm) <= 40 and norm not in only_include:
                only_include.append(norm)
    only_include = only_include[:10]

    # 4. styles
    raw_styles = raw_dict.get("styles") or []
    styles = []
    for st in raw_styles:
        if isinstance(st, str):
            norm_st = STYLE_MAP.get(st.strip().lower(), st.strip().upper())
            if norm_st in ("GROWTH", "VALUE", "DIVIDEND", "MOMENTUM", "MEME", "ESG", "DEFENSIVE"):
                if norm_st not in styles:
                    styles.append(norm_st)
    styles = styles[:7]

    # 5. assetClassTargets
    raw_targets = raw_dict.get("assetClassTargets")
    targets = None
    if raw_targets and isinstance(raw_targets, dict):
        parsed_targets = {}
        for k, v in raw_targets.items():
            norm_k = ASSET_CLASS_MAP.get(str(k).strip().lower(), str(k).strip().upper())
            if norm_k in ("STOCKS", "CRYPTO", "GOLD", "BONDS"):
                try:
                    val = float(v)
                    parsed_targets[norm_k] = val
                except (ValueError, TypeError):
                    pass
        # Check sum <= 100 and each 0..100
        if any(not 0 <= v <= 100 for v in parsed_targets.values()) or sum(parsed_targets.values()) > 100:
            dropped_warnings.append("targets_over_100")
            targets = None
        else:
            targets = parsed_targets if parsed_targets else None

    # Sync asset class targets if "no gold" etc. in exclude
    for item in exclude:
        lower_item = item.lower()
        if lower_item in ("gold", "crypto", "bonds", "stocks"):
            norm_k = ASSET_CLASS_MAP.get(lower_item, lower_item.upper())
            if targets is None:
                targets = {}
            if norm_k not in targets:
                targets[norm_k] = 0.0

    # 6. notes
    notes = raw_dict.get("notes")
    if notes is not None:
        notes = str(notes).strip()[:500] if str(notes).strip() else None

    if not must_include and not exclude and not styles and targets is None and notes is None and not only_include:
        return None, dropped_warnings

    try:
        pref = Preferences(
            mustInclude=must_include,
            exclude=exclude,
            assetClassTargets=targets,
            styles=styles,
            notes=notes,
            onlyInclude=only_include,
        )
        return pref, dropped_warnings
    except Exception as e:
        logger.warning(f"Preferences validation error: {e}")
        return None, dropped_warnings


def extract_explicit_preferences(text: str) -> dict:
    """Extract explicit preferences from user text keeping user's words, without case-specific logic."""
    pref: dict[str, Any] = {
        "mustInclude": [],
        "exclude": [],
        "assetClassTargets": {},
        "styles": [],
        "notes": None,
        "onlyInclude": [],
    }
    text_lower = text.lower()

    # 1. Asset class targets: "about a third in crypto", "30% crypto", "30% in crypto"
    m_third = re.search(r"\b(?:about\s+)?a\s+third\s+(?:in\s+)?(crypto|stocks?|gold|bonds?)\b", text_lower)
    if m_third:
        cls_key = ASSET_CLASS_MAP.get(m_third.group(1), "CRYPTO")
        pref["assetClassTargets"][cls_key] = 33.0

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%\s*(?:in\s+)?(crypto|stocks?|gold|bonds?)\b", text_lower):
        val = float(m.group(1))
        cls_key = ASSET_CLASS_MAP.get(m.group(2), "CRYPTO")
        pref["assetClassTargets"][cls_key] = val

    # 2. Restrict-the-universe: "only bio stocks", "only biotech", "ethical investing only", "ethical only"
    for m in re.finditer(r"\bonly\s+([a-zA-Z0-9_\-\s]{2,30})\b", text, re.I):
        phrase = m.group(1).strip()
        phrase = re.split(r"[,.;]|and|or|\bwith\b|\bfor\b", phrase)[0].strip()
        if phrase and phrase.lower() not in ("preferences", "preference", "thanks", "problem", "usd", "dollars"):
            if phrase not in pref["onlyInclude"]:
                pref["onlyInclude"].append(phrase)

    for m in re.finditer(r"\b([a-zA-Z0-9_\-\s]{2,30})\s+only\b", text, re.I):
        phrase = m.group(1).strip()
        phrase = re.split(r"[,.;]|and|or|\bwith\b|\bfor\b", phrase)[-1].strip()
        if phrase and phrase.lower() not in ("preferences", "preference", "thanks", "problem", "usd", "dollars"):
            if phrase not in pref["onlyInclude"]:
                pref["onlyInclude"].append(phrase)

    # 3. Dislikes / Exclusions: "no gold", "no tech", "no Chinese companies", "I don't like gold", "I hate tech", "avoid oil"
    for m in re.finditer(r"\b(?:no|avoid|exclude|without|hate|dislike|don't like|do not like)\s+([a-zA-Z0-9_\-\s]{2,30})\b", text, re.I):
        phrase = m.group(1).strip()
        phrase = re.split(r"[,.;]|and|or|\bwith\b|\bfor\b|\bbut\b", phrase)[0].strip()
        if phrase and phrase.lower() not in ("preferences", "preference", "thanks", "problem", "crashes", "loss"):
            norm = normalize_symbol_or_entity(phrase, is_exclude=True)
            if norm not in pref["exclude"]:
                pref["exclude"].append(norm)
            if norm.lower() in ("gold", "crypto", "bonds", "stocks"):
                ac_key = ASSET_CLASS_MAP.get(norm.lower(), norm.upper())
                pref["assetClassTargets"][ac_key] = 0.0

    # 4. Styles: e.g. "dividend", "dividends", "ethical", "growth", "value", "meme"
    for word, st in STYLE_MAP.items():
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
            if st not in pref["styles"]:
                pref["styles"].append(st)

    # 5. "I want X" -> mustInclude (tickers) or notes
    for kw, sym in KNOWN_CRYPTO_MAP.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            if not any(kw in ex.lower() or sym.lower() in ex.lower() for ex in pref["exclude"]):
                if sym not in pref["mustInclude"]:
                    pref["mustInclude"].append(sym)

    for kw, sym in KNOWN_COMPANY_MAP.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            if not any(kw in ex.lower() or sym.lower() in ex.lower() for ex in pref["exclude"]):
                if sym not in pref["mustInclude"]:
                    pref["mustInclude"].append(sym)

    for m in re.finditer(r"\b(?:want|prefer|include|invest in)\s+([a-zA-Z0-9_\-\s]{2,40})\b", text, re.I):
        phrase = m.group(1).strip()
        phrase = re.split(r"[,.;]|and|or|\bwith\b|\bfor\b|\bbut\b", phrase)[0].strip()
        if re.match(r"^[A-Z]{1,5}$", phrase) or re.match(r"^[A-Z0-9]+-USD$", phrase, re.I):
            norm = phrase.upper()
            if norm not in pref["mustInclude"]:
                pref["mustInclude"].append(norm)
        elif len(phrase) > 3 and phrase.lower() not in ("growth", "safe", "cautious", "moderate", "aggressive", "returns", "money"):
            if not any(st_word in phrase.lower() for st_word in STYLE_MAP):
                pref["notes"] = phrase

    if not pref["assetClassTargets"]:
        pref["assetClassTargets"] = None

    return pref


def merge_preferences(base: Preferences | None, incoming: Preferences | dict | None) -> Preferences | None:
    """Merge incoming preferences into base preferences without dropping earlier gathered items."""
    if incoming is None:
        return base
    if base is None:
        if isinstance(incoming, Preferences):
            return incoming
        val, _ = validate_and_sanitize_preferences(incoming)
        return val

    inc_dict = incoming.model_dump() if isinstance(incoming, (Preferences, BaseModel)) else incoming

    must = list(base.mustInclude)
    for s in inc_dict.get("mustInclude") or []:
        if s not in must:
            must.append(s)

    ex = list(base.exclude)
    for s in inc_dict.get("exclude") or []:
        if s not in ex:
            ex.append(s)

    only = list(base.onlyInclude)
    for s in inc_dict.get("onlyInclude") or []:
        if s not in only:
            only.append(s)

    styles = list(base.styles)
    for s in inc_dict.get("styles") or []:
        if s not in styles:
            styles.append(s)

    targets = dict(base.assetClassTargets) if base.assetClassTargets else {}
    if inc_dict.get("assetClassTargets"):
        targets.update(inc_dict["assetClassTargets"])

    notes = base.notes
    if inc_dict.get("notes"):
        if notes and inc_dict["notes"] not in notes:
            notes = f"{notes}; {inc_dict['notes']}"[:500]
        else:
            notes = inc_dict["notes"][:500]

    merged_pref, _ = validate_and_sanitize_preferences({
        "mustInclude": must,
        "exclude": ex,
        "onlyInclude": only,
        "assetClassTargets": targets if targets else None,
        "styles": styles,
        "notes": notes,
    })
    return merged_pref or base



def parse_explicit_amount(text: str) -> float | None:
    """Robustly parse amounts like '10k', '$10,000', 'ten thousand', '1.5m'."""
    # 1. Suffix forms: 10k, $10k, 10.5k, 10m, $1m, 10 thousand
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([kmKM]|thousand|million)\b", text)
    if m:
        num = float(m.group(1))
        unit = m.group(2).lower()
        if unit in ("k", "thousand"):
            return num * 1000.0
        elif unit in ("m", "million"):
            return num * 1_000_000.0

    text_lower = text.lower()
    words_map = {
        "ten thousand": 10000.0,
        "twenty thousand": 20000.0,
        "twenty-five thousand": 25000.0,
        "thirty thousand": 30000.0,
        "forty thousand": 40000.0,
        "fifty thousand": 50000.0,
        "one hundred thousand": 100000.0,
        "two hundred thousand": 200000.0,
        "five hundred thousand": 500000.0,
        "one million": 1000000.0,
        "five thousand": 5000.0,
        "two thousand": 2000.0,
        "one thousand": 1000.0,
    }
    for phrase, val in words_map.items():
        if phrase in text_lower:
            return val

    # 2. Currency format: $10,000 or $10000
    m = re.search(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))

    return None


async def intake_turn(req: IntakeTurnRequest) -> IntakeTurn:
    """Process one intake chat turn."""
    formatted_conversation = "\n".join(
        f"{msg.role.capitalize()}: {msg.content}" for msg in req.messages
    )

    prompt = f"""\
{INTAKE_SYSTEM_INSTRUCTION}

CONVERSATION HISTORY:
{formatted_conversation}

Analyze the full conversation history. Extract the current known intake values and formulate your next concise reply (<= 2 sentences).
"""

    res = await llm.ask(
        model=config.MODEL_FLASH,
        input=prompt,
        schema=_IntakeExtraction,
        agent="INTAKE",
    )

    data = res.data
    reply = getattr(data, "reply", "") or res.text or ""
    amount = getattr(data, "amountUsd", None)
    horizon = getattr(data, "horizonYears", None)
    risk = getattr(data, "riskTolerance", None)
    theme = getattr(data, "theme", None)
    raw_pref = getattr(data, "preferences", None)

    # 1. Preferences extraction and validation
    pref, dropped_warnings = validate_and_sanitize_preferences(raw_pref)

    # Extract explicit preferences from all user messages and merge
    for msg in req.messages:
        if msg.role == "user":
            explicit_p = extract_explicit_preferences(msg.content)
            pref = merge_preferences(pref, explicit_p)

    # 2. Parse explicit amounts from user messages (don't trust model blindly)
    for msg in reversed(req.messages):
        if msg.role == "user":
            parsed_amt = parse_explicit_amount(msg.content)
            if parsed_amt is not None:
                amount = parsed_amt
                break

    # 3. Python validation: clamp nothing silently. Out of range -> None
    if amount is not None:
        if not (100.0 <= amount <= 10_000_000.0):
            amount = None

    had_out_of_range_horizon = False
    for msg in reversed(req.messages):
        if msg.role == "user":
            m_years = re.search(r"\b(\d+)\s*(?:years?|yrs?)\b", msg.content, re.IGNORECASE)
            if m_years:
                val_y = int(m_years.group(1))
                if not (1 <= val_y <= 10):
                    horizon = None
                    had_out_of_range_horizon = True
                elif horizon is None:
                    horizon = val_y
                break

    if horizon is not None and not (1 <= horizon <= 10):
        horizon = None
        had_out_of_range_horizon = True

    if had_out_of_range_horizon:
        if not ("horizon" in reply.lower() or "year" in reply.lower()):
            reply = "We support investment horizons between 1 and 10 years. How many years (1–10) would you like to invest for?"

    if risk not in ("CONSERVATIVE", "MODERATE", "AGGRESSIVE"):
        risk = None

    # Check explicit risk in latest user message if not set or if user requested switch
    latest_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if re.search(r"\bswitch to (aggressive|moderate|conservative)\b", latest_user_msg, re.I):
        m_switch = re.search(r"\bswitch to (aggressive|moderate|conservative)\b", latest_user_msg, re.I)
        risk = m_switch.group(1).upper()
    elif risk is None:
        if re.search(r"\b(cautious|conservative|safe|preservation)\b", latest_user_msg, re.I):
            risk = "CONSERVATIVE"
        elif re.search(r"\b(moderate|balanced)\b", latest_user_msg, re.I):
            risk = "MODERATE"
        elif re.search(r"\b(aggressive|growth|moonshot|high risk)\b", latest_user_msg, re.I):
            risk = "AGGRESSIVE"

    if theme and len(theme) > 200:
        theme = theme[:200]

    # 4. Execution guard: check if user asked to place or execute a trade
    if re.search(r"\b(buy|sell|trade|purchase|execute)\b", latest_user_msg, re.IGNORECASE):
        reply_lower = reply.lower()
        if "trade" not in reply_lower and "advice" not in reply_lower:
            reply = "This tool researches and backtests portfolios, but does not place trades or give financial advice. " + reply

    # 5. Conflict warning: judged by model, ensuring it's mentioned once and not repeated
    conflict_note = getattr(data, "conflictNote", None)
    if conflict_note and not conflict_note.strip():
        conflict_note = None

    already_warned = False
    for m in req.messages:
        if m.role == "assistant":
            c_low = m.content.lower()
            if conflict_note and conflict_note.lower() in c_low:
                already_warned = True
                break
            if any(phrase in c_low for phrase in (
                "switch to aggressive",
                "meme coins can drop",
                "removes a portfolio hedge",
                "removes a hedge",
                "hedge for a conservative",
                "volatile for a",
                "concentration risk",
                "kept small",
            )):
                already_warned = True
                break

    conflict_active = False
    if conflict_note:
        if not already_warned:
            if conflict_note.lower() not in reply.lower():
                reply = f"{conflict_note} {reply}".strip()
            conflict_active = True
        else:
            # Strip repeated conflict note if present in reply
            if conflict_note.lower() in reply.lower():
                idx = reply.lower().find(conflict_note.lower())
                reply = (reply[:idx] + reply[idx + len(conflict_note):]).strip()

    # 6. Ambiguity: if an "only" phrase is ambiguous about other asset classes, ask ONE clarifying question
    clarifying_active = False
    has_only = pref is not None and bool(pref.onlyInclude)
    targets_or_excludes_other_classes = False
    if pref:
        targets_or_excludes_other_classes = (
            (pref.assetClassTargets is not None and len(pref.assetClassTargets) > 1)
            or any(norm.lower() in ("crypto", "gold", "bonds") for norm in pref.exclude)
        )

    already_clarified = any(
        any(w in m.content.lower() for w in ("crypto", "gold", "bonds", "asset class", "asset classes"))
        for m in req.messages
        if m.role == "assistant"
    )

    if has_only and not targets_or_excludes_other_classes and not already_clarified:
        clarifying_q = getattr(data, "clarifyingQuestion", None)
        if not clarifying_q or not clarifying_q.strip():
            clarifying_q = "Do you also want no crypto, gold, or bonds, or should those asset classes still be included?"
        if not any(w in reply.lower() for w in ("crypto", "gold", "bonds", "asset class")):
            reply = f"{clarifying_q} {reply}".strip() if reply else clarifying_q
        clarifying_active = True

    # 7. Ask about preferences if amount, horizon, risk known but no preferences exist yet
    has_preferences = pref is not None and (
        bool(pref.mustInclude)
        or bool(pref.exclude)
        or bool(pref.onlyInclude)
        or (pref.assetClassTargets is not None and len(pref.assetClassTargets) > 0)
        or bool(pref.styles)
        or (pref.notes is not None and len(pref.notes.strip()) > 0)
    )

    already_asked_preferences = any(
        "specifically want or want to avoid" in m.content.lower()
        or "ethical exclusions" in m.content.lower()
        or "like certain coins" in m.content.lower()
        for m in req.messages
        if m.role == "assistant"
    )

    declined_preferences = any(
        re.search(p, latest_user_msg.lower())
        for p in [
            r"\bno\b",
            r"\bnothing\b",
            r"\bnope\b",
            r"\bnone\b",
            r"\bno preferences?\b",
            r"\bno thanks\b",
            r"\bskip\b",
            r"\bpass\b",
            r"\ball good\b",
            r"\bno exclusions?\b",
        ]
    )

    asked_preferences_now = False
    if amount is not None and horizon is not None and risk is not None and not had_out_of_range_horizon:
        if not has_preferences and not already_asked_preferences and not declined_preferences and not conflict_active and not clarifying_active:
            reply = "Anything you specifically want or want to avoid, like certain coins, companies, dividends, or ethical exclusions?"
            asked_preferences_now = True

    # 8. Check if dropped invalid parts (e.g. targets over 100)
    if "targets_over_100" in dropped_warnings:
        reply = "Target allocations must sum to at most 100%. How would you like to divide your portfolio between stocks, crypto, and gold?"

    # Ensure replies are concise (<= 2 sentences)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reply) if s.strip()]
    if len(sentences) > 2:
        reply = " ".join(sentences[:2])

    if not reply.strip():
        if amount is None:
            reply = "How much are you looking to invest?"
        elif horizon is None:
            reply = "How long do you plan to invest (1 to 10 years)?"
        elif risk is None:
            reply = "What is your risk tolerance: conservative, moderate, or aggressive?"
        else:
            reply = "Great, your intake is complete! Review your details and click Run to begin."

    draft = IntakeDraft(
        amountUsd=amount,
        horizonYears=horizon,
        riskTolerance=risk,
        theme=theme,
        preferences=pref,
    )

    is_complete = False
    if (
        draft.amountUsd is not None
        and draft.horizonYears is not None
        and draft.riskTolerance is not None
        and not had_out_of_range_horizon
        and not conflict_active
        and not clarifying_active
        and not asked_preferences_now
        and "targets_over_100" not in dropped_warnings
    ):
        try:
            IntakeInput(
                amountUsd=draft.amountUsd,
                horizonYears=draft.horizonYears,
                riskTolerance=draft.riskTolerance,
                theme=draft.theme,
                preferences=draft.preferences,
            )
            is_complete = True
        except Exception:
            is_complete = False

    return IntakeTurn(
        reply=reply,
        intake=draft,
        isComplete=is_complete,
    )
