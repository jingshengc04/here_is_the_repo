import asyncio

import numpy as np
import pandas as pd
import pytest

from app import engine
from app.models import Backtest

# Hand-built stand-in for the PLAN's 4-row CSV (not in the repo): weekly returns
# A: -5%, +5%, +8%   B: +15%, +10%, +15%  => 60/40 weekly-rebalanced total return 22.1%.
IDX4 = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
FOUR = pd.DataFrame({"A": [100, 95, 99.75, 107.73], "B": [100, 115, 126.5, 145.475]}, index=IDX4)


def synthetic(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-07", periods=n, freq="W-FRI")
    cols = ["AAA", "BBB", "SPY", "AGG"]
    r = rng.normal(0.003, 0.02, size=(n, 4))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=cols)


ALLOC = {"slices": [
    {"assetClass": "STOCKS", "weightPct": 70, "rationale": "x", "holdings": [
        {"symbol": "AAA", "name": "A", "weightPct": 70, "amountUsd": 7000, "rationale": "r",
         "findingIds": [], "financialScore": 7, "sentimentScore": 0.1}]},
    {"assetClass": "CRYPTO", "weightPct": 30, "rationale": "x", "holdings": [
        {"symbol": "BBB", "name": "B", "weightPct": 30, "amountUsd": 3000, "rationale": "r",
         "findingIds": [], "financialScore": None, "sentimentScore": None}]},
]}


def test_drawdown_and_total_return():
    idx = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    closes = pd.DataFrame({"X": [100, 120, 90, 130, 65, 140]}, index=idx)
    w = {"X": 1.0}
    m = engine.metrics(engine.equity_curve(closes, w), closes, w)
    assert m["maxDrawdownPct"] == -50.0 and m["totalReturnPct"] == 40.0


def test_sixty_forty_fixture():
    w = {"A": 0.6, "B": 0.4}
    m = engine.metrics(engine.equity_curve(FOUR, w), FOUR, w)
    assert m["totalReturnPct"] == 22.1


def test_run_backtest_shape_and_ci():
    bt = asyncio.run(engine.run_backtest(ALLOC, synthetic(), None))
    assert isinstance(bt, Backtest)
    n = len(bt.equityCurve.dates)
    assert set(bt.equityCurve.series) == {"PORTFOLIO", "SPY", "SIXTY_FORTY", "EQUAL_WEIGHT"}
    assert all(len(s) == n for s in bt.equityCurve.series.values())
    assert bt.equityCurve.dates[0] == bt.startDate and bt.equityCurve.dates[-1] == bt.endDate
    m = bt.metrics["PORTFOLIO"]
    assert bt.sharpeCi.lower <= m.sharpe <= bt.sharpeCi.upper
    assert "sharpe_samples" not in engine.sharpe_ci(engine.equity_curve(synthetic(), {"AAA": 1.0}))


def test_deterministic():
    a = asyncio.run(engine.run_backtest(ALLOC, synthetic(), None))
    b = asyncio.run(engine.run_backtest(ALLOC, synthetic(), None))
    assert a == b


def test_risk_parity_sums_to_one():
    w = engine.risk_parity(synthetic(), ["AAA", "BBB", "SPY"])
    assert set(w) == {"AAA", "BBB", "SPY"} and sum(w.values()) == pytest.approx(1.0)


def _ours():
    bt = asyncio.run(engine.run_backtest(ALLOC, synthetic(), None))
    return bt.metrics["PORTFOLIO"]


def _sandbox(dr, dd):
    ours = _ours()

    async def sb(closes, weights):
        return {"totalReturnPct": ours.totalReturnPct + dr, "maxDrawdownPct": ours.maxDrawdownPct + dd}
    return sb


def test_cross_check_match_mismatch_and_failure():
    run = lambda sb: asyncio.run(engine.run_backtest(ALLOC, synthetic(), sb)).crossCheck
    assert run(_sandbox(0.2, -0.2)).isMatch is True
    assert run(_sandbox(3.0, 0)).isMatch is False
    assert run(_sandbox(0, 3.0)).isMatch is False

    async def boom(c, w):
        raise RuntimeError("sandbox down")
    cc = run(boom)
    assert cc.isMatch is False and cc.sandboxTotalReturnPct == cc.independentTotalReturnPct


def test_sandbox_prompt():
    p = engine.sandbox_prompt({"AAA": 0.7, "BBB": 0.3})
    for s in ("pandas", "AAA", "0.7", "weekly", "totalReturnPct", "maxDrawdownPct", "JSON"):
        assert s in p
