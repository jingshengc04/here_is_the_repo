import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.data import (
    FUNDAMENTALS_CACHE_DIR,
    PRICES_CACHE_DIR,
    fundamentals,
    is_crypto,
    prices,
    resolve,
)


def _make_daily_df(dates, close_values):
    return pd.DataFrame({"close": close_values}, index=pd.DatetimeIndex(dates, name="trade_date"))


def test_is_crypto():
    assert is_crypto("BTC-USD") is True
    assert is_crypto("ETH-USD") is True
    assert is_crypto("sol-usd") is True
    assert is_crypto("NVDA") is False
    assert is_crypto("GLD") is False
    assert is_crypto("SPY") is False
    assert is_crypto("AGG") is False


def test_fundamentals_crypto_and_etfs_return_empty():
    assert fundamentals("BTC-USD") == {}
    assert fundamentals("ETH-USD") == {}
    assert fundamentals("GLD") == {}
    assert fundamentals("SPY") == {}
    assert fundamentals("AGG") == {}
    assert fundamentals("gld") == {}


def test_symbol_mapping_and_fetching(tmp_path, monkeypatch):
    """Verify NVDA maps to NVDA.US and BTC-USD maps to BTC-USDT on OKX."""
    monkeypatch.setattr("app.data.PRICES_CACHE_DIR", tmp_path / "prices")
    (tmp_path / "prices").mkdir(parents=True, exist_ok=True)

    # >= 52 weeks (300 business days = 60 weeks, 400 calendar days = 57 weeks)
    dates = pd.date_range("2023-01-01", periods=350, freq="B")
    mock_yf_df = _make_daily_df(dates, np.linspace(100, 200, 350))
    mock_okx_df = _make_daily_df(pd.date_range("2023-01-01", periods=450, freq="D"), np.linspace(40000, 50000, 450))

    yf_mock_fetch = MagicMock(return_value={"NVDA.US": mock_yf_df})
    okx_mock_fetch = MagicMock(return_value={"BTC-USDT": mock_okx_df})

    with patch("backtest.loaders.yfinance_loader.DataLoader.fetch", yf_mock_fetch), \
         patch("backtest.loaders.okx.DataLoader.fetch", okx_mock_fetch):

        df, missing = prices(["NVDA", "BTC-USD"], years=1)

        # Check yf called with NVDA.US
        yf_args, _ = yf_mock_fetch.call_args
        assert yf_args[0] == ["NVDA.US"]

        # Check okx called with BTC-USDT
        okx_args, _ = okx_mock_fetch.call_args
        assert okx_args[0] == ["BTC-USDT"]

        assert missing == []
        assert "NVDA" in df.columns
        assert "BTC-USD" in df.columns


def test_okx_failure_falls_back_to_yfinance(tmp_path, monkeypatch):
    """When OKX fails or returns empty, falls back to yfinance with BTC-USD."""
    monkeypatch.setattr("app.data.PRICES_CACHE_DIR", tmp_path / "prices")
    (tmp_path / "prices").mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2023-01-01", periods=450, freq="D")
    mock_fallback_df = _make_daily_df(dates, np.linspace(40000, 50000, 450))

    # OKX raises exception
    okx_fail = MagicMock(side_effect=RuntimeError("OKX timeout"))
    yf_mock = MagicMock(return_value={"BTC-USD": mock_fallback_df})

    with patch("backtest.loaders.okx.DataLoader.fetch", okx_fail), \
         patch("backtest.loaders.yfinance_loader.DataLoader.fetch", yf_mock):

        df, missing = prices(["BTC-USD"], years=1)
        assert missing == []
        assert "BTC-USD" in df.columns
        yf_args, _ = yf_mock.call_args
        assert yf_args[0] == ["BTC-USD"]


def test_weekly_resample_and_alignment(tmp_path, monkeypatch):
    """Weekly resampling uses W-FRI last and aligns series on common Fridays."""
    monkeypatch.setattr("app.data.PRICES_CACHE_DIR", tmp_path / "prices")
    (tmp_path / "prices").mkdir(parents=True, exist_ok=True)

    stock_dates = pd.date_range("2023-01-01", periods=400, freq="B")
    crypto_dates = pd.date_range("2023-01-01", periods=560, freq="D")

    stock_df = _make_daily_df(stock_dates, np.arange(len(stock_dates)))
    crypto_df = _make_daily_df(crypto_dates, np.arange(len(crypto_dates)) * 10)

    yf_mock = MagicMock(return_value={"NVDA.US": stock_df})
    okx_mock = MagicMock(return_value={"BTC-USDT": crypto_df})

    with patch("backtest.loaders.yfinance_loader.DataLoader.fetch", yf_mock), \
         patch("backtest.loaders.okx.DataLoader.fetch", okx_mock):

        df, missing = prices(["NVDA", "BTC-USD"], years=1)
        assert missing == []
        # Check all index points are Friday
        assert (df.index.dayofweek == 4).all()
        assert "NVDA" in df.columns
        assert "BTC-USD" in df.columns
        # Check that crypto and stock share the exact same index
        assert not df["NVDA"].isna().any()
        assert not df["BTC-USD"].isna().any()


def test_missing_list_for_short_histories(tmp_path, monkeypatch):
    """Symbols with fewer than 52 weekly points go into missing."""
    monkeypatch.setattr("app.data.PRICES_CACHE_DIR", tmp_path / "prices")
    (tmp_path / "prices").mkdir(parents=True, exist_ok=True)

    # Long history (60 weeks)
    long_dates = pd.date_range("2023-01-01", periods=450, freq="B")
    long_df = _make_daily_df(long_dates, np.linspace(100, 200, 450))

    # Short history (20 weeks)
    short_dates = pd.date_range("2024-06-01", periods=100, freq="B")
    short_df = _make_daily_df(short_dates, np.linspace(50, 60, 100))

    def mock_fetch(codes, *args, **kwargs):
        res = {}
        for c in codes:
            if "SHORT" in c:
                res[c] = short_df
            else:
                res[c] = long_df
        return res

    with patch("backtest.loaders.yfinance_loader.DataLoader.fetch", side_effect=mock_fetch):
        df, missing = prices(["NVDA", "SHORT"], years=2)
        assert "SHORT" in missing
        assert "NVDA" not in missing
        assert "NVDA" in df.columns
        assert "SHORT" not in df.columns


def test_stale_cache_used_when_live_fetch_raises(tmp_path, monkeypatch):
    """When cache is older than 12h and live fetch raises, use stale cache with warning."""
    cache_dir = tmp_path / "prices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.data.PRICES_CACHE_DIR", cache_dir)

    dates = pd.date_range("2023-01-01", periods=400, freq="B")
    cached_df = _make_daily_df(dates, np.linspace(100, 150, 400))
    parquet_path = cache_dir / "NVDA.parquet"
    cached_df.to_parquet(parquet_path)

    # Set mtime to 20 hours ago (stale)
    stale_time = time.time() - (20 * 3600)
    os.utime(parquet_path, (stale_time, stale_time))

    yf_fail = MagicMock(side_effect=ConnectionError("Yahoo finance unreachable"))

    with patch("backtest.loaders.yfinance_loader.DataLoader.fetch", yf_fail):
        df, missing = prices(["NVDA"], years=1)
        assert missing == []
        assert "NVDA" in df.columns
        assert len(df) > 0


def test_resolve(tmp_path, monkeypatch):
    """resolve() returns True on cache hit or successful fetch, False otherwise."""
    cache_dir = tmp_path / "prices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.data.PRICES_CACHE_DIR", cache_dir)

    # 1. Cache hit
    (cache_dir / "CACHED.parquet").touch()
    assert resolve("CACHED") is True

    # 2. Live fetch succeeds
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    mock_df = _make_daily_df(dates, [100.0] * 10)
    with patch("backtest.loaders.yfinance_loader.DataLoader.fetch", return_value={"LIVE.US": mock_df}):
        assert resolve("LIVE") is True

    # 3. Live fetch fails / empty
    with patch("backtest.loaders.yfinance_loader.DataLoader.fetch", return_value={}):
        assert resolve("UNKNOWN") is False


def test_fundamentals_stock_fields(tmp_path, monkeypatch):
    """fundamentals(symbol) returns SEC TTM fields + yfinance valuation fields, caching to JSON."""
    cache_dir = tmp_path / "fundamentals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.data.FUNDAMENTALS_CACHE_DIR", cache_dir)

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    mock_panel = {
        "revenue": pd.DataFrame({"NVDA": [300.0e9] * 10}, index=dates),
        "gross_profit": pd.DataFrame({"NVDA": [220.0e9] * 10}, index=dates),
        "operating_income": pd.DataFrame({"NVDA": [180.0e9] * 10}, index=dates),
        "net_income": pd.DataFrame({"NVDA": [150.0e9] * 10}, index=dates),
        "total_debt": pd.DataFrame({"NVDA": [10.0e9] * 10}, index=dates),
        "cash": pd.DataFrame({"NVDA": [35.0e9] * 10}, index=dates),
        "cfo": pd.DataFrame({"NVDA": [160.0e9] * 10}, index=dates),
        "capex": pd.DataFrame({"NVDA": [15.0e9] * 10}, index=dates),
        "shares_diluted": pd.DataFrame({"NVDA": [24.5e9] * 10}, index=dates),
    }

    mock_ticker = MagicMock()
    mock_ticker.info = {
        "trailingPE": 25.87,
        "marketCap": 4941441662976,
        "currentPrice": 204.64,
    }

    with patch("backtest.loaders.fundamentals_loader.load_fundamental_panel", return_value=mock_panel), \
         patch("yfinance.Ticker", return_value=mock_ticker):

        res = fundamentals("NVDA")

        # Verify SEC fields
        assert res["revenue"] == 300.0e9
        assert res["gross_profit"] == 220.0e9
        assert res["operating_income"] == 180.0e9
        assert res["net_income"] == 150.0e9
        assert res["total_debt"] == 10.0e9
        assert res["cash"] == 35.0e9
        assert res["cfo"] == 160.0e9
        assert res["capex"] == 15.0e9
        assert res["shares_diluted"] == 24.5e9

        # Verify valuation fields
        assert res["trailingPE"] == 25.87
        assert res["marketCap"] == 4941441662976
        assert res["currentPrice"] == 204.64
        assert "asOf" in res
        assert res["asOf"] is not None

        # Verify JSON cache exists
        json_path = cache_dir / "NVDA.json"
        assert json_path.exists()


def test_fundamentals_stale_cache_used_when_live_fetch_raises(tmp_path, monkeypatch):
    """When fundamentals cache is older than 24h and live fetch raises, use stale cache."""
    import json

    cache_dir = tmp_path / "fundamentals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("app.data.FUNDAMENTALS_CACHE_DIR", cache_dir)

    cached_data = {
        "revenue": 300.0e9,
        "gross_profit": 220.0e9,
        "operating_income": 180.0e9,
        "net_income": 150.0e9,
        "total_debt": 10.0e9,
        "cash": 35.0e9,
        "cfo": 160.0e9,
        "capex": 15.0e9,
        "shares_diluted": 24.5e9,
        "trailingPE": 25.87,
        "marketCap": 4941441662976,
        "currentPrice": 204.64,
        "asOf": "2024-01-12",
    }
    json_path = cache_dir / "NVDA.json"
    with open(json_path, "w") as f:
        json.dump(cached_data, f)

    # Set mtime to 30 hours ago (stale)
    stale_time = time.time() - (30 * 3600)
    os.utime(json_path, (stale_time, stale_time))

    fail_panel = MagicMock(side_effect=RuntimeError("SEC EDGAR unreachable"))

    with patch("backtest.loaders.fundamentals_loader.load_fundamental_panel", fail_panel):
        res = fundamentals("NVDA")
        assert res["revenue"] == 300.0e9
        assert res["trailingPE"] == 25.87


@pytest.mark.skipif(os.getenv("LIVE") != "1", reason="Live fetch test requires LIVE=1")
def test_live_fetches_nvda_gld_btc():
    """Live integration test: fetches NVDA, GLD, and BTC-USD prices and fundamentals."""
    df, missing = prices(["NVDA", "GLD", "BTC-USD"], years=5)
    assert missing == []
    assert set(df.columns) == {"NVDA", "GLD", "BTC-USD"}
    assert len(df) >= 250
    assert (df.index.dayofweek == 4).all()
    # Check no missing prices in common grid
    assert not df["NVDA"].isna().any()
    assert not df["GLD"].isna().any()
    assert not df["BTC-USD"].isna().any()

    # Fundamentals
    fund_nvda = fundamentals("NVDA")
    assert fund_nvda["revenue"] is not None and fund_nvda["revenue"] > 1e10
    assert fund_nvda["trailingPE"] is not None
    assert fund_nvda["currentPrice"] is not None
    assert fund_nvda["asOf"] is not None

    # ETFs and crypto return empty dict
    assert fundamentals("GLD") == {}
    assert fundamentals("BTC-USD") == {}


