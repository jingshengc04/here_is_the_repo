"""Data layer over vendored Vibe-Trading loaders.

Provides prices, fundamentals, existence resolution, and caching.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
if not os.getenv("SEC_USER_AGENT"):
    tetra_env = os.path.expanduser("~/orca/workspaces/here_is_the_repo/tetra/backend/.env")
    if os.path.exists(tetra_env):
        load_dotenv(tetra_env)

logger = logging.getLogger(__name__)

ETF_SYMBOLS = {"GLD", "SPY", "AGG"}


def is_crypto(symbol: str) -> bool:
    """Return True if symbol is crypto (ends with -USD and isn't an equity)."""
    return symbol.strip().upper().endswith("-USD")


def is_etf(symbol: str) -> bool:
    """Return True if symbol is an ETF (GLD, SPY, AGG)."""
    return symbol.strip().upper() in ETF_SYMBOLS


def fundamentals(symbol: str) -> dict[str, Any]:
    """Return fundamental metrics for symbol.

    Returns {} immediately for crypto and ETFs (GLD/SPY/AGG).
    """
    sym = symbol.strip().upper()
    if is_crypto(sym) or is_etf(sym):
        return {}
    raise NotImplementedError
