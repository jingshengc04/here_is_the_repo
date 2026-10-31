"""Prefetch and warm price and fundamental caches for demo safety net."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("SEC_USER_AGENT"):
    tetra_env = os.path.expanduser("~/orca/workspaces/here_is_the_repo/tetra/backend/.env")
    if os.path.exists(tetra_env):
        load_dotenv(tetra_env)

from app.data import (
    fundamentals,
    is_crypto,
    is_etf,
    load_daily_price,
)

PREFETCH_SYMBOLS = [
    "NVDA", "MSFT", "AVGO", "TSM", "AMD", "GOOGL", "AMZN", "META", "AAPL",
    "ANET", "LLY", "NVO", "JPM", "V", "XOM", "COST", "ASML", "PLTR", "ORCL", "CRWD",
    "GLD", "SPY", "AGG",
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "ADA-USD", "DOGE-USD",
]


def prefetch() -> None:
    end_dt = pd.Timestamp.now().normalize()
    start_dt = (end_dt - pd.DateOffset(years=5) - pd.Timedelta(14, unit="D")).normalize()
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    for sym in PREFETCH_SYMBOLS:
        try:
            df, source = load_daily_price(sym, start_date, end_date, force_refresh=True)
            rows = len(df)
            last_date = df.index[-1].strftime("%Y-%m-%d") if not df.empty else "N/A"
            print(f"{sym}: {rows} rows, last date {last_date}, source {source}")
        except Exception as exc:
            print(f"{sym}: 0 rows, last date N/A, source failed ({exc})")

        # Also warm fundamentals cache for equities
        if not is_crypto(sym) and not is_etf(sym):
            try:
                fundamentals(sym, force_refresh=True)
            except Exception:
                pass


if __name__ == "__main__":
    prefetch()
