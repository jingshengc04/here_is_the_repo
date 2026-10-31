"""Data layer over vendored Vibe-Trading loaders.

Provides prices, fundamentals, existence resolution, and caching.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

import backtest.loaders.okx as okx_loader_mod
import backtest.loaders.yfinance_loader as yf_loader_mod

load_dotenv()
if not os.getenv("SEC_USER_AGENT"):
    tetra_env = os.path.expanduser("~/orca/workspaces/here_is_the_repo/tetra/backend/.env")
    if os.path.exists(tetra_env):
        load_dotenv(tetra_env)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
PRICES_CACHE_DIR = CACHE_DIR / "prices"
FUNDAMENTALS_CACHE_DIR = CACHE_DIR / "fundamentals"

PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FUNDAMENTALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRICES_CACHE_FRESH_SECONDS = 12 * 3600
FUNDAMENTALS_CACHE_FRESH_SECONDS = 24 * 3600

ETF_SYMBOLS = {"GLD", "SPY", "AGG"}


SEC_FIELDS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "total_debt",
    "cash",
    "cfo",
    "capex",
    "shares_diluted",
]


def is_crypto(symbol: str) -> bool:
    """Return True if symbol is crypto (ends with -USD and isn't an equity)."""
    return symbol.strip().upper().endswith("-USD")


def is_etf(symbol: str) -> bool:
    """Return True if symbol is an ETF (GLD, SPY, AGG)."""
    return symbol.strip().upper() in ETF_SYMBOLS


def _map_stock_or_etf_code(symbol: str) -> str:
    sym = symbol.strip().upper()
    return sym if sym.endswith(".US") else f"{sym}.US"


def _map_crypto_okx_code(symbol: str) -> str:
    sym = symbol.strip().upper()
    if sym.endswith("-USD"):
        return sym[:-4] + "-USDT"
    return sym


def _fetch_daily_prices(symbol: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    """Fetch daily price bars for a single symbol using vendored loaders.

    Returns:
        tuple of (DataFrame with trade_date index and close column, source_used)
    """
    sym = symbol.strip().upper()
    if is_crypto(sym):
        okx_code = _map_crypto_okx_code(sym)
        try:
            loader = okx_loader_mod.DataLoader()
            res = loader.fetch([okx_code], start_date, end_date, interval="1D")
            df = res.get(okx_code)
            if df is not None and not df.empty and "close" in df.columns:
                return df, "okx"
            logger.info("OKX returned empty data for %s, falling back to yfinance", okx_code)
        except Exception as exc:
            logger.warning("OKX loader failed for %s (%s), falling back to yfinance", okx_code, exc)

        # Fallback to yfinance with BTC-USD (original contract symbol)
        yf_loader = yf_loader_mod.DataLoader()
        res = yf_loader.fetch([sym], start_date, end_date, interval="1D")
        df = res.get(sym)
        if df is not None and not df.empty and "close" in df.columns:
            return df, "yfinance"
        return pd.DataFrame(), "none"

    # Equities and ETFs
    code = _map_stock_or_etf_code(sym)
    yf_loader = yf_loader_mod.DataLoader()
    res = yf_loader.fetch([code], start_date, end_date, interval="1D")
    df = res.get(code)
    if df is not None and not df.empty and "close" in df.columns:
        return df, "yfinance"
    return pd.DataFrame(), "none"


def load_daily_price(
    symbol: str,
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load daily closes from cache if fresh (<12h), otherwise live fetch.

    On live fetch failure, falls back to stale cache with a warning.
    """
    sym = symbol.strip().upper()
    cache_path = PRICES_CACHE_DIR / f"{sym}.parquet"

    if not force_refresh and cache_path.exists():
        try:
            age = time.time() - os.path.getmtime(cache_path)
            if age < PRICES_CACHE_FRESH_SECONDS:
                df = pd.read_parquet(cache_path)
                if not df.empty and "close" in df.columns:
                    return df, "cache"
        except Exception as exc:
            logger.warning("Failed reading price cache for %s: %s", sym, exc)

    # Attempt live fetch
    try:
        df, source = _fetch_daily_prices(sym, start_date, end_date)
        if not df.empty and "close" in df.columns:
            to_cache = df[["close"]].copy()
            # Normalize index to DatetimeIndex without timezone
            to_cache.index = pd.to_datetime(to_cache.index)
            if getattr(to_cache.index, "tz", None) is not None:
                to_cache.index = to_cache.index.tz_localize(None)
            to_cache.index.name = "trade_date"
            to_cache.to_parquet(cache_path)
            return to_cache, source
        raise ValueError(f"No price data returned for {sym}")
    except Exception as exc:
        if cache_path.exists():
            logger.warning("Live fetch failed for %s (%s); using stale cache", sym, exc)
            try:
                stale_df = pd.read_parquet(cache_path)
                return stale_df, "stale_cache"
            except Exception as read_exc:
                logger.error("Failed to read stale cache for %s: %s", sym, read_exc)
        else:
            logger.warning("Live fetch failed for %s (%s) and no cache exists", sym, exc)
    return pd.DataFrame(), "none"


def prices(symbols: list[str], years: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Return weekly closes (resampled W-FRI last) on a common weekly index.

    Symbols with fewer than 52 weekly points go into the returned missing list.
    """
    end_dt = pd.Timestamp.now().normalize()
    start_dt = (end_dt - pd.DateOffset(years=years) - pd.Timedelta(14, unit="D")).normalize()
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    missing: list[str] = []
    valid_series: dict[str, pd.Series] = {}

    for s in symbols:
        sym = s.strip().upper()
        df_daily, _ = load_daily_price(sym, start_date, end_date)
        if df_daily.empty or "close" not in df_daily.columns:
            missing.append(sym)
            continue

        df_daily.index = pd.to_datetime(df_daily.index)
        if getattr(df_daily.index, "tz", None) is not None:
            df_daily.index = df_daily.index.tz_localize(None)

        # Resample W-FRI taking the last available observation in each week
        weekly = df_daily["close"].resample("W-FRI").last().dropna()
        if len(weekly) < 52:
            missing.append(sym)
        else:
            valid_series[sym] = weekly

    if not valid_series:
        return pd.DataFrame(), missing

    combined = pd.DataFrame(valid_series).sort_index()
    # Align crypto to the same grid; forward fill any interior holiday gaps
    combined = combined.ffill().bfill()
    return combined, missing


def resolve(symbol: str) -> bool:
    """Cheap existence check for Discovery (uses cache when possible)."""
    sym = symbol.strip().upper()
    parquet_path = PRICES_CACHE_DIR / f"{sym}.parquet"
    json_path = FUNDAMENTALS_CACHE_DIR / f"{sym}.json"
    if parquet_path.exists() or json_path.exists():
        return True

    end_d = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_d = (pd.Timestamp.now() - pd.Timedelta(30, unit="D")).strftime("%Y-%m-%d")
    try:
        df, _ = _fetch_daily_prices(sym, start_d, end_d)
        return not df.empty and "close" in df.columns
    except Exception:
        return False


def _fetch_fundamentals(symbol: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    clean_sym = sym[:-3] if sym.endswith(".US") else sym

    end_d = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_d = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    from backtest.loaders.fundamentals_loader import load_fundamental_panel
    import yfinance as yf

    panel = load_fundamental_panel([clean_sym], SEC_FIELDS, start_d, end_d, freq="ttm")

    result: dict[str, Any] = {field: None for field in SEC_FIELDS}
    as_of = None
    for field in SEC_FIELDS:
        df_field = panel.get(field)
        if df_field is not None and clean_sym in df_field.columns:
            series = df_field[clean_sym].dropna()
            if not series.empty:
                val = series.iloc[-1]
                if pd.notna(val):
                    result[field] = float(val)
                as_of = series.index[-1].strftime("%Y-%m-%d")

    result["asOf"] = as_of or pd.Timestamp.now().strftime("%Y-%m-%d")

    # yfinance valuation fields
    try:
        ticker = yf.Ticker(clean_sym)
        info = getattr(ticker, "info", None) or {}
        pe = info.get("trailingPE")
        result["trailingPE"] = float(pe) if pe is not None and pd.notna(pe) else None
        mc = info.get("marketCap")
        result["marketCap"] = int(mc) if mc is not None and pd.notna(mc) else None
        cp = info.get("currentPrice")
        result["currentPrice"] = float(cp) if cp is not None and pd.notna(cp) else None
    except Exception as exc:
        logger.warning("Failed loading yfinance info for %s: %s", clean_sym, exc)
        result["trailingPE"] = None
        result["marketCap"] = None
        result["currentPrice"] = None

    return result


def fundamentals(symbol: str, force_refresh: bool = False) -> dict[str, Any]:
    """Return fundamental metrics for symbol.

    Returns {} immediately for crypto and ETFs (GLD/SPY/AGG).
    Caches to backend/data/cache/fundamentals/{symbol}.json (valid for 24h).
    """
    sym = symbol.strip().upper()
    if is_crypto(sym) or is_etf(sym):
        return {}

    cache_path = FUNDAMENTALS_CACHE_DIR / f"{sym}.json"

    if not force_refresh and cache_path.exists():
        try:
            age = time.time() - os.path.getmtime(cache_path)
            if age < FUNDAMENTALS_CACHE_FRESH_SECONDS:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            logger.warning("Failed reading fundamentals cache for %s: %s", sym, exc)

    # Attempt live fetch
    try:
        data = _fetch_fundamentals(sym)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as write_exc:
            logger.warning("Failed writing fundamentals cache for %s: %s", sym, write_exc)
        return data
    except Exception as exc:
        if cache_path.exists():
            logger.warning("Live fundamentals fetch failed for %s (%s); using stale cache", sym, exc)
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as read_exc:
                logger.error("Failed reading stale fundamentals cache for %s: %s", sym, read_exc)
        logger.warning("Live fundamentals fetch failed for %s (%s) and no cache exists", sym, exc)
        return {
            "revenue": None,
            "gross_profit": None,
            "operating_income": None,
            "net_income": None,
            "total_debt": None,
            "cash": None,
            "cfo": None,
            "capex": None,
            "shares_diluted": None,
            "trailingPE": None,
            "marketCap": None,
            "currentPrice": None,
            "asOf": pd.Timestamp.now().strftime("%Y-%m-%d"),
        }
