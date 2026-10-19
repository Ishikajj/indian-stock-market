"""
Performance and risk metrics for investment analysis.

All functions accept pd.Series of periodic (usually monthly) returns unless
stated otherwise.  Annualisation assumes the number of periods per year
passed as `periods_per_year` (default 12 for monthly data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Core return metrics ───────────────────────────────────────────────────────

def cagr(prices: pd.Series) -> float:
    """Compound Annual Growth Rate from a price level series."""
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1


def annualised_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    total = (1 + returns).prod()
    n = len(returns)
    return total ** (periods_per_year / n) - 1


def annualised_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    return returns.std() * np.sqrt(periods_per_year)


# ── Risk-adjusted metrics ─────────────────────────────────────────────────────

def sharpe_ratio(
    returns: pd.Series,
    rf: pd.Series | float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """Annualised Sharpe ratio.  rf can be a scalar or a matching-index Series."""
    excess = returns - (rf / periods_per_year if isinstance(rf, float) else rf.reindex(returns.index).ffill() / periods_per_year)
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    rf: pd.Series | float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """Annualised Sortino ratio (downside deviation in denominator)."""
    excess = returns - (rf / periods_per_year if isinstance(rf, float) else rf.reindex(returns.index).ffill() / periods_per_year)
    downside = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return (excess.mean() / downside) * np.sqrt(periods_per_year)


def max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    roll_max = prices.cummax()
    drawdown = prices / roll_max - 1
    return drawdown.min()


def drawdown_series(prices: pd.Series) -> pd.Series:
    roll_max = prices.cummax()
    return (prices / roll_max - 1)


def calmar_ratio(prices: pd.Series) -> float:
    mdd = abs(max_drawdown(prices))
    if mdd == 0:
        return np.nan
    ann_ret = cagr(prices)
    return ann_ret / mdd


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (positive = loss)."""
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall)."""
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= -var]
    return -tail.mean() if len(tail) > 0 else var


# ── SIP simulation ────────────────────────────────────────────────────────────

def simulate_sip(
    prices: pd.Series,
    monthly_amount: float = 10_000,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """
    Simulate a Systematic Investment Plan (SIP).

    Invests `monthly_amount` on the first trading day of each month.
    Returns a dict with portfolio value series, total invested, XIRR proxy (CAGR
    on invested capital), and final corpus.
    """
    p = prices.copy()
    if start:
        p = p[p.index >= pd.to_datetime(start)]
    if end:
        p = p[p.index <= pd.to_datetime(end)]

    # Resample to monthly (last price of each month)
    monthly = p.resample("ME").last().dropna()

    units = 0.0
    invested = 0.0
    history = []

    for date, price in monthly.items():
        units_bought = monthly_amount / price
        units += units_bought
        invested += monthly_amount
        portfolio_value = units * price
        history.append({"date": date, "invested": invested, "value": portfolio_value, "units": units})

    hist_df = pd.DataFrame(history).set_index("date")

    n_years = (hist_df.index[-1] - hist_df.index[0]).days / 365.25
    final_value = hist_df["value"].iloc[-1]
    total_invested = hist_df["invested"].iloc[-1]

    # XIRR approximation: solve for r in sum(CF/(1+r)^t) = 0
    # Using CAGR on average invested capital as practical proxy
    abs_return = (final_value - total_invested) / total_invested
    xirr_approx = (final_value / total_invested) ** (1 / n_years) - 1

    return {
        "history": hist_df,
        "total_invested": total_invested,
        "final_value": final_value,
        "absolute_return_pct": abs_return * 100,
        "cagr_approx": xirr_approx,
        "n_years": n_years,
    }


# ── Rolling metrics ───────────────────────────────────────────────────────────

def rolling_cagr(prices: pd.Series, window_years: int = 3) -> pd.Series:
    """Rolling CAGR over `window_years`-year windows (monthly data)."""
    w = window_years * 12
    result = []
    for i in range(w, len(prices) + 1):
        segment = prices.iloc[i - w : i]
        result.append(cagr(segment))
    idx = prices.index[w - 1 :]
    return pd.Series(result, index=idx, name=f"{window_years}Y Rolling CAGR")


def rolling_sharpe(
    returns: pd.Series,
    rf_annual: float = 0.06,
    window_months: int = 36,
) -> pd.Series:
    rf_monthly = rf_annual / 12
    excess = returns - rf_monthly
    roll_mean = excess.rolling(window_months).mean()
    roll_std = returns.rolling(window_months).std()
    return (roll_mean / roll_std * np.sqrt(12)).rename(f"{window_months//12}Y Rolling Sharpe")


# ── Summary table ─────────────────────────────────────────────────────────────

def summary_table(
    price_dict: dict[str, pd.Series],
    returns_dict: dict[str, pd.Series],
    rf_annual: float = 0.065,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Build a combined performance summary DataFrame."""
    rows = []
    for name in price_dict:
        if name not in returns_dict:
            continue
        p = price_dict[name].dropna()
        r = returns_dict[name].dropna()
        if len(p) < 13 or len(r) < 13:
            continue
        rows.append({
            "Asset": name,
            "CAGR (%)": round(cagr(p) * 100, 2),
            "Ann. Vol (%)": round(annualised_volatility(r) * 100, 2),
            "Sharpe": round(sharpe_ratio(r, rf=rf_annual), 2),
            "Sortino": round(sortino_ratio(r, rf=rf_annual), 2),
            "Max DD (%)": round(max_drawdown(p) * 100, 2),
            "Calmar": round(calmar_ratio(p), 2),
            "VaR 95% (monthly, %)": round(value_at_risk(r) * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Asset")
