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
5. preferences: Personalisation preferences:
   - mustInclude: Symbols user insists on (normalise: "doge"/"dogecoin" -> "DOGE-USD", "pepe" -> "PEPE-USD", "nvidia" -> "NVDA", "bitcoin" -> "BTC-USD"). Max 10.
   - exclude: Symbols or plain words to avoid/exclude (keep sector word e.g. "oil", "tobacco"; symbol for company e.g. "TSLA" for Tesla). Max 20.
   - assetClassTargets: Specific target percentages (e.g. "about a third in crypto" -> {"CRYPTO": 33}, "30% crypto" -> {"CRYPTO": 30}).
   - styles: Styles from plain language: MEME, DIVIDEND, VALUE, GROWTH, MOMENTUM, ESG, DEFENSIVE.
   - notes: Anything else that matters, <= 500 chars.

RULES:
- Ask ONE short, conversational question at a time for whatever information is missing or unclear.
- Keep your replies concise: at most 2 sentences.
- When all 3 required fields (amount, horizon 1-10 years, risk tolerance) are known:
  - If no preferences exist yet, ask ONE short optional question: "Anything you specifically want or want to avoid, like certain coins, companies, dividends, or ethical exclusions?"
  - If preferences already exist or the user replies with "no"/"nothing", summarize the intake and tell the user they can review and run.
- Conflict warning: If preferences clash with risk (MEME style, or crypto target > 15%, or meme coins in mustInclude, with CONSERVATIVE/MODERATE), say it once, briefly: "Meme coins can drop 80%+; with a Conservative profile they'll be kept small (a few %). Want to switch to Aggressive?" Then respect the user's answer.
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
        description="Symbols the user insists on including, normalized (e.g. DOGE-USD, PEPE-USD, NVDA, BTC-USD). Max 10.",
    )
    exclude: list[str] = Field(
        default_factory=list,
        description="Symbols or plain words to avoid/exclude, e.g. TSLA, oil, tobacco. Max 20.",
    )
    assetClassTargets: dict[str, float] | None = Field(
        default=None,
        description="Target % per asset class (STOCKS, CRYPTO, GOLD), each 0-100, e.g. {'CRYPTO': 33}.",
    )
    styles: list[str] = Field(
        default_factory=list,
        description="Investment styles: MEME, DIVIDEND, VALUE, GROWTH, MOMENTUM, ESG, DEFENSIVE.",
    )
    notes: str | None = Field(
        default=None,
        description="Free text notes or specific requests, max 500 chars.",
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

    # 3. styles
    raw_styles = raw_dict.get("styles") or []
    styles = []
    for st in raw_styles:
        if isinstance(st, str):
            norm_st = STYLE_MAP.get(st.strip().lower(), st.strip().upper())
            if norm_st in ("GROWTH", "VALUE", "DIVIDEND", "MOMENTUM", "MEME", "ESG", "DEFENSIVE"):
                if norm_st not in styles:
                    styles.append(norm_st)
    styles = styles[:7]

    # 4. assetClassTargets
    raw_targets = raw_dict.get("assetClassTargets")
    targets = None
    if raw_targets and isinstance(raw_targets, dict):
        parsed_targets = {}
        for k, v in raw_targets.items():
            norm_k = ASSET_CLASS_MAP.get(str(k).strip().lower(), str(k).strip().upper())
            if norm_k in ("STOCKS", "CRYPTO", "GOLD"):
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

    # 5. notes
    notes = raw_dict.get("notes")
    if notes is not None:
        notes = str(notes).strip()[:500] if str(notes).strip() else None

    if not must_include and not exclude and not styles and targets is None and notes is None:
        return None, dropped_warnings

    try:
        pref = Preferences(
            mustInclude=must_include,
            exclude=exclude,
            assetClassTargets=targets,
            styles=styles,
            notes=notes,
        )
        return pref, dropped_warnings
    except Exception as e:
        logger.warning(f"Preferences validation error: {e}")
        return None, dropped_warnings


def extract_explicit_preferences(text: str) -> dict:
    """Extract explicit preferences from user text using patterns and keywords."""
    pref: dict[str, Any] = {
        "mustInclude": [],
        "exclude": [],
        "assetClassTargets": {},
        "styles": [],
        "notes": None,
    }
    text_lower = text.lower()

    # Targets: "about a third in crypto", "30% crypto", "30% in crypto"
    m_third = re.search(r"\b(?:about\s+)?a\s+third\s+(?:in\s+)?(crypto|stocks?|gold)\b", text_lower)
    if m_third:
        cls_key = ASSET_CLASS_MAP.get(m_third.group(1), "CRYPTO")
        pref["assetClassTargets"][cls_key] = 33.0

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%\s*(?:in\s+)?(crypto|stocks?|gold)\b", text_lower):
        val = float(m.group(1))
        cls_key = ASSET_CLASS_MAP.get(m.group(2), "CRYPTO")
        pref["assetClassTargets"][cls_key] = val

    # Exclude: "no Tesla", "no oil", "no oil companies", "exclude tobacco", "no tobacco"
    for m in re.finditer(r"\b(?:no|avoid|exclude|without)\s+([a-zA-Z0-9_\-\s]{2,30})\b", text, re.I):
        phrase = m.group(1).strip()
        phrase = re.split(r"[,.;]|and|or", phrase)[0].strip()
        if phrase and phrase.lower() not in ("preferences", "preference", "thanks", "problem"):
            norm = normalize_symbol_or_entity(phrase, is_exclude=True)
            if norm not in pref["exclude"]:
                pref["exclude"].append(norm)

    # Styles
    for word, st in STYLE_MAP.items():
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
            if st not in pref["styles"]:
                pref["styles"].append(st)

    # Must include
    for kw, sym in KNOWN_CRYPTO_MAP.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            if not any(kw in ex.lower() or sym.lower() in ex.lower() for ex in pref["exclude"]):
                if sym not in pref["mustInclude"]:
                    pref["mustInclude"].append(sym)

    for kw, sym in KNOWN_COMPANY_MAP.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            if not any(kw in ex.lower() or sym.lower() in ex.lower() for ex in pref["exclude"]):
                # Check positive intent
                if re.search(r"\b(?:want|include|invest in|especially|like|some|buy)\s+[^,.]*" + re.escape(kw), text_lower):
                    if sym not in pref["mustInclude"]:
                        pref["mustInclude"].append(sym)

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
        "assetClassTargets": targets if targets else None,
        "styles": styles,
        "notes": notes,
    })
    return merged_pref or base


def check_risk_conflict(risk: str | None, preferences: Preferences | None) -> bool:
    """Check if preferences clash with risk (MEME style, crypto target > 15%, or meme coins in mustInclude with CONSERVATIVE/MODERATE)."""
    if not risk or not preferences:
        return False
    if risk not in ("CONSERVATIVE", "MODERATE"):
        return False

    # 1. MEME style
    if "MEME" in preferences.styles:
        return True

    # 2. Crypto target > 15%
    if preferences.assetClassTargets:
        if preferences.assetClassTargets.get("CRYPTO", 0) > 15:
            return True

    # 3. Meme coins in mustInclude
    for sym in preferences.mustInclude:
        if sym.upper() in MEME_COIN_SYMBOLS:
            return True

    return False



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

    # Check explicit risk in latest user message if not set
    latest_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if risk is None:
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

    # 5. Check conflict warning (MEME style, crypto target > 15%, or meme coins in mustInclude with CONSERVATIVE/MODERATE)
    has_conflict = check_risk_conflict(risk, pref)
    already_warned = any(
        "switch to aggressive" in m.content.lower() or "meme coins can drop" in m.content.lower()
        for m in req.messages
        if m.role == "assistant"
    )

    conflict_active = False
    if already_warned:
        # Check user's answer to the conflict warning
        if re.search(r"\b(yes|switch|switch to aggressive|aggressive|sure|yep|yeah)\b", latest_user_msg, re.I):
            risk = "AGGRESSIVE"
            has_conflict = False
            reply = "Updated your risk profile to Aggressive. Your intake is complete! Review your details and click Run to begin."
        elif re.search(r"\b(no|keep|stay|conservative|moderate|cautious|fairly cautious)\b", latest_user_msg, re.I):
            # User chose to keep conservative/moderate
            has_conflict = False
            reply = f"Meme coins will be kept small with your {risk.capitalize()} profile. Your intake is complete! Review your details and click Run to begin."
    elif has_conflict:
        profile_str = "Conservative" if risk == "CONSERVATIVE" else "Moderate"
        reply = f"Meme coins can drop 80%+; with a {profile_str} profile they'll be kept small (a few %). Want to switch to Aggressive?"
        conflict_active = True

    # 6. Ask about preferences if amount, horizon, risk known but no preferences exist yet
    has_preferences = pref is not None and (
        bool(pref.mustInclude)
        or bool(pref.exclude)
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
        if not has_preferences and not already_asked_preferences and not declined_preferences and not conflict_active:
            reply = "Anything you specifically want or want to avoid, like certain coins, companies, dividends, or ethical exclusions?"
            asked_preferences_now = True

    # 7. Check if dropped invalid parts (e.g. targets over 100)
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
