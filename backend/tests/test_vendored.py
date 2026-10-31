"""Offline smoke test: vendored Vibe-Trading modules import and compute on synthetic data."""
import numpy as np
import pandas as pd

from backtest.risk_xray import compute_risk_xray
from backtest.validation import bootstrap_sharpe_ci
from backtest.optimizers.risk_parity import RiskParityOptimizer
from src.tools.financial_rigor_tool import cross_validate, verify_valuation
from src.tools.report_audit_tool import extract_data_points
import backtest.loaders.yfinance_loader, backtest.loaders.okx, backtest.loaders.fundamentals_loader  # noqa: F401


def _closes():
    idx = pd.date_range("2021-01-01", periods=260, freq="W-FRI")
    rng = np.random.default_rng(0)
    return pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(m, v, 260)) for s, m, v in
                         [("A", .003, .03), ("B", .002, .05), ("C", .001, .01)]}, index=idx)


def test_risk_xray_and_ci():
    c = _closes()
    x = compute_risk_xray(c, {"A": .5, "B": .3, "C": .2}, periods_per_year=52)
    assert x["drawdown"]["max_drawdown"] < 0
    eq = (c.pct_change().fillna(0) @ pd.Series({"A": .5, "B": .3, "C": .2}) + 1).cumprod()
    ci = bootstrap_sharpe_ci(eq, bars_per_year=52)
    assert ci["ci_lower"] <= ci["observed_sharpe"] <= ci["ci_upper"]


def test_risk_parity_sums_to_one():
    cov = _closes().pct_change().dropna().cov().values * 52
    w = RiskParityOptimizer()._calc_weights({"cov": cov})
    assert abs(w.sum() - 1) < 1e-6 and (w > 0).all()


def test_rigor_tools():
    assert cross_validate("rev", {"sec": 303.0, "yf": 302.9, "llm": 310.0})["all_consistent"] is False
    assert abs(verify_valuation(price=204.64, eps=192.879 / 24.4)["metrics"]["PE"] - 25.89) < 0.01
    assert [p["reported_value"] for p in extract_data_points("- Revenue: $303.0B\n- P/E: 25.9x\n")] == [303.0, 25.9]
