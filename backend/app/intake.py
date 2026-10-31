"""Intake chat agent for portfolio onboarding."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app import config, llm
from app.models import IntakeDraft, IntakeInput, IntakeTurn, IntakeTurnRequest

logger = logging.getLogger("app.intake")

INTAKE_SYSTEM_INSTRUCTION = """\
You are a friendly, encouraging onboarding assistant for a portfolio-research demo.
Your job is to collect 3 required preferences from the user to build their investment portfolio:
1. amountUsd: Total amount to invest in USD (must be between 100 and 10,000,000).
2. horizonYears: Investment time horizon in years (integer between 1 and 10 years).
3. riskTolerance: One of CONSERVATIVE, MODERATE, or AGGRESSIVE. Map plain language like "safe/steady/can't stomach crashes/preservation" to CONSERVATIVE, "balanced/growth but careful" to MODERATE, and "high growth/crypto/moonshots/max returns" to AGGRESSIVE.
Optional:
4. theme: An investment theme or focus if mentioned (e.g., "AI infrastructure", "clean energy"), max 200 characters.

RULES:
- Ask ONE short, conversational question at a time for whatever information is missing or unclear.
- Keep your replies concise: at most 2 sentences.
- When all 3 required fields (amount, horizon 1-10 years, risk tolerance) are known, summarize the intake and tell the user they can review and run.
- EXECUTION GUARD: If the user asks to place, buy, or execute a trade (e.g., "buy NVDA now"), politely state that this tool researches and backtests portfolios but does not place trades or give financial advice, and continue the intake.
- If the user provides a horizon outside 1 to 10 years (e.g. 15 years), explain that the research model supports horizons from 1 to 10 years, and ask them for a horizon within that range.
"""


class _IntakeExtraction(BaseModel):
    reply: str = Field(description="Assistant reply to the user, <= 2 sentences.")
    amountUsd: float | None = Field(default=None, description="Amount in USD, or null if unknown.")
    horizonYears: int | None = Field(default=None, description="Horizon in years, 1-10, or null.")
    riskTolerance: Literal["CONSERVATIVE", "MODERATE", "AGGRESSIVE"] | None = Field(
        default=None, description="Risk tolerance level, or null if unknown."
    )
    theme: str | None = Field(default=None, description="Optional investment theme or focus.")


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

    # 1. Parse explicit amounts from user messages (don't trust model blindly)
    for msg in reversed(req.messages):
        if msg.role == "user":
            parsed_amt = parse_explicit_amount(msg.content)
            if parsed_amt is not None:
                amount = parsed_amt
                break

    # 2. Python validation: clamp nothing silently. Out of range -> None
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

    if theme and len(theme) > 200:
        theme = theme[:200]

    # 3. Execution guard: check if user asked to place or execute a trade
    latest_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if re.search(r"\b(buy|sell|trade|purchase|execute)\b", latest_user_msg, re.IGNORECASE):
        reply_lower = reply.lower()
        if not ("place trade" in reply_lower or "place trades" in reply_lower or "doesn't place trades" in reply_lower or "does not place trades" in reply_lower or "cannot place trades" in reply_lower):
            reply = "This tool researches and backtests portfolios, but does not place trades or give financial advice. " + reply

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
    )

    is_complete = False
    if draft.amountUsd is not None and draft.horizonYears is not None and draft.riskTolerance is not None:
        try:
            IntakeInput(
                amountUsd=draft.amountUsd,
                horizonYears=draft.horizonYears,
                riskTolerance=draft.riskTolerance,
                theme=draft.theme,
            )
            is_complete = True
        except Exception:
            is_complete = False

    return IntakeTurn(
        reply=reply,
        intake=draft,
        isComplete=is_complete,
    )
