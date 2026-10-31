"""Deterministic backtest: Allocation + weekly closes -> contract `Backtest`.

Weekly bars everywhere (52 periods/year). The Gemini sandbox step is an injected async callable.
"""
import logging
from collections.abc import Awaitable, Callable, Mapping

import numpy as np
import pandas as pd

from backtest.optimizers.risk_parity import RiskParityOptimizer
from backtest.risk_xray import compute_risk_xray
from backtest.validation import bootstrap_sharpe_ci

from .models import Allocation, Backtest

log = logging.getLogger("app.engine")
PPY = 52
TOLERANCE_PP = 0.5
Sandbox = Callable[[pd.DataFrame, dict], Awaitable[dict]]


def _returns(closes: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    cols = list(weights)
    w = np.array([weights[c] for c in cols], dtype=float)
    return closes[cols].pct_change().dropna() @ w  # weekly rebalance == fixed weights every bar


def equity_curve(closes: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Growth of $1, weekly-rebalanced; starts at 1.0 on the first date (same length as closes)."""
    growth = (1 + _returns(closes, weights)).cumprod()
    return pd.concat([pd.Series([1.0], index=closes.index[:1]), growth])


def metrics(curve: pd.Series, closes: pd.DataFrame, weights: Mapping[str, float]) -> dict:
    total = curve.iloc[-1] - 1
    years = (len(curve) - 1) / PPY
    cagr = curve.iloc[-1] ** (1 / years) - 1 if years > 0 and curve.iloc[-1] > 0 else float("nan")
    xray = compute_risk_xray(closes, weights, periods_per_year=PPY, min_history=2)
    rets = curve.pct_change().dropna()
    std = rets.std(ddof=1)
    sharpe = rets.mean() / std * np.sqrt(PPY) if len(rets) > 1 and std > 1e-12 else None
    # ponytail: worst rolling 4-week return stands in for "worst month"
    worst = curve.pct_change(min(4, len(curve) - 1)).min()
    r = lambda x, d=1: None if x is None or not np.isfinite(x) else round(float(x), d)
    return {
        "totalReturnPct": r(total * 100),
        "cagrPct": r(cagr * 100),
        "volatilityPct": r(xray["volatility"]["annualized_vol"] * 100),
        "sharpe": r(sharpe, 2),
        "maxDrawdownPct": r(xray["drawdown"]["max_drawdown"] * 100),
        "worstMonthPct": r(worst * 100),
    }


def sharpe_ci(curve: pd.Series) -> dict | None:
    res = bootstrap_sharpe_ci(curve, bars_per_year=PPY)  # seeded (seed=42) => deterministic
    if "error" in res:
        return None
    return {"lower": round(res["ci_lower"], 2), "upper": round(res["ci_upper"], 2),
            "probPositive": round(res["prob_positive"], 2)}


def risk_parity(closes: pd.DataFrame, symbols: list[str]) -> dict[str, float]:
    cov = closes[symbols].pct_change().dropna().cov().to_numpy() * PPY
    w = RiskParityOptimizer()._calc_weights({"cov": cov})
    return {s: float(x) for s, x in zip(symbols, w)}


def sandbox_prompt(weights: Mapping[str, float]) -> str:
    return (
        "Load the attached CSV with pandas (first column is the date, the other columns are weekly "
        "closing prices). Simulate a portfolio with these target weights, rebalanced back to the "
        f"targets every week: {dict(weights)}. Compute the weekly portfolio returns as the "
        "weight-weighted sum of each asset's weekly percentage change, compound them into an equity "
        "curve starting at 1.0, then compute the total return and the maximum drawdown (the most "
        "negative peak-to-trough decline) of that curve. Print only a JSON object "
        '{"totalReturnPct": <number>, "maxDrawdownPct": <negative number>} with both values as '
        "percentages (e.g. 22.1, not 0.221)."
    )


def _weights(allocation: dict | Allocation) -> dict[str, float]:
    alloc = allocation if isinstance(allocation, Allocation) else Allocation.model_validate(allocation)
    w: dict[str, float] = {}
    for s in alloc.slices:
        for h in s.holdings:
            w[h.symbol] = w.get(h.symbol, 0.0) + h.weightPct / 100
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


async def run_backtest(allocation: dict | Allocation, closes: pd.DataFrame, sandbox: Sandbox | None) -> Backtest:
    weights = _weights(allocation)
    closes = closes[[*weights, *(c for c in ("SPY", "AGG") if c not in weights)]].dropna()
    series_w = {
        "PORTFOLIO": weights,
        "SPY": {"SPY": 1.0},
        "SIXTY_FORTY": {"SPY": 0.6, "AGG": 0.4},
        "EQUAL_WEIGHT": {s: 1 / len(weights) for s in weights},
    }
    curves = {k: equity_curve(closes, w) for k, w in series_w.items()}
    mets = {k: metrics(curves[k], closes, w) for k, w in series_w.items()}
    ours = mets["PORTFOLIO"]

    cross = {"sandboxTotalReturnPct": ours["totalReturnPct"], "independentTotalReturnPct": ours["totalReturnPct"],
             "isMatch": False}
    if sandbox is not None:
        try:
            sb = await sandbox(closes[list(weights)], weights)
            sb_ret, sb_dd = float(sb["totalReturnPct"]), float(sb["maxDrawdownPct"])
            cross = {
                "sandboxTotalReturnPct": round(sb_ret, 1),
                "independentTotalReturnPct": ours["totalReturnPct"],
                "isMatch": abs(sb_ret - ours["totalReturnPct"]) <= TOLERANCE_PP
                and abs(sb_dd - ours["maxDrawdownPct"]) <= TOLERANCE_PP,
            }
        except Exception as e:
            log.warning("sandbox cross-check failed, using independent numbers only: %r", e)
    else:
        log.warning("no sandbox provided; crossCheck.isMatch=False")

    dates = [pd.Timestamp(d).date().isoformat() for d in closes.index]
    return Backtest.model_validate({
        "startDate": dates[0], "endDate": dates[-1], "metrics": mets,
        "equityCurve": {"dates": dates, "series": {k: [round(float(v), 4) for v in c] for k, c in curves.items()}},
        "sharpeCi": sharpe_ci(curves["PORTFOLIO"]), "crossCheck": cross,
    })
