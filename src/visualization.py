"""
Reusable plotting functions for the Indian stock market analysis notebook.
All functions return matplotlib Figure objects so callers can save or display them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Global style
sns.set_theme(style="whitegrid", palette="tab10")
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)
FIGSIZE_TALL = (12, 8)


def _pct_formatter(x, _):
    return f"{x:.0f}%"


def plot_cumulative_returns(
    price_df: pd.DataFrame,
    title: str = "Cumulative Returns (base = 100)",
    log_scale: bool = False,
) -> plt.Figure:
    """Normalised cumulative return chart — each series rebased to 100 at start."""
    normalised = price_df.dropna(how="all").apply(lambda s: s / s.dropna().iloc[0] * 100)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for col in normalised.columns:
        ax.plot(normalised.index, normalised[col], label=col, linewidth=1.5)

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Indexed Value (start = 100)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def plot_rolling_cagr(
    rolling_dict: dict[str, pd.Series],
    window_years: int = 3,
    title: str | None = None,
) -> plt.Figure:
    if title is None:
        title = f"{window_years}-Year Rolling CAGR"
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for name, series in rolling_dict.items():
        ax.plot(series.index, series * 100, label=name, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def plot_drawdowns(
    drawdown_df: pd.DataFrame,
    title: str = "Underwater Chart (Drawdown from Peak)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for col in drawdown_df.columns:
        ax.fill_between(drawdown_df.index, drawdown_df[col] * 100, 0, alpha=0.3, label=col)
        ax.plot(drawdown_df.index, drawdown_df[col] * 100, linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def plot_metrics_bar(
    summary_df: pd.DataFrame,
    metric: str = "Sharpe",
    title: str | None = None,
    color_map: bool = True,
) -> plt.Figure:
    if title is None:
        title = f"{metric} by Asset"
    col = [c for c in summary_df.columns if metric.lower() in c.lower()]
    if not col:
        raise ValueError(f"Column containing '{metric}' not found in summary_df")
    vals = summary_df[col[0]].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(vals) * 0.45)))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(vals))) if color_map else None
    bars = ax.barh(vals.index, vals.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(col[0])
    for bar, v in zip(bars, vals.values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_sip_growth(sip_results: dict[str, dict], title: str = "SIP Portfolio Growth") -> plt.Figure:
    """Plot portfolio value and invested amount over time for each fund/index."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = plt.cm.tab10.colors
    for i, (name, res) in enumerate(sip_results.items()):
        hist = res["history"]
        ax.plot(hist.index, hist["value"] / 1e5, label=name, color=colors[i % 10], linewidth=1.8)
    # Plot invested amount once (same for all)
    first_hist = next(iter(sip_results.values()))["history"]
    ax.plot(first_hist.index, first_hist["invested"] / 1e5,
            color="black", linewidth=1.5, linestyle="--", label="Invested Amount")
    ax.set_ylabel("₹ Lakhs")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(returns_df: pd.DataFrame, title: str = "Return Correlations") -> plt.Figure:
    corr = returns_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_factor_alphas(
    factor_table: pd.DataFrame,
    title: str = "Jensen's Alpha by Fund (Fama-French 3-Factor)",
) -> plt.Figure:
    if "Alpha (ann %)" not in factor_table.columns:
        raise ValueError("factor_table must contain 'Alpha (ann %)' column")
    alphas = factor_table["Alpha (ann %)"].sort_values()
    tstats = factor_table.get("Alpha t-stat", pd.Series(dtype=float))

    fig, ax = plt.subplots(figsize=(10, max(4, len(alphas) * 0.5)))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in alphas]
    bars = ax.barh(alphas.index, alphas.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Annualised Alpha (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    for bar, v in zip(bars, alphas.values):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}%", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_efficient_frontier(
    returns_df: pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free: float = 0.065,
    title: str = "Efficient Frontier (Indian Assets)",
) -> plt.Figure:
    """Monte-Carlo efficient frontier for the assets in returns_df."""
    ann_ret = returns_df.mean() * 12
    ann_cov = returns_df.cov() * 12
    n = len(ann_ret)

    results = {"vol": [], "ret": [], "sharpe": [], "weights": []}
    rng = np.random.default_rng(42)

    for _ in range(n_portfolios):
        w = rng.dirichlet(np.ones(n))
        ret = w @ ann_ret.values
        vol = np.sqrt(w @ ann_cov.values @ w)
        sr = (ret - risk_free) / vol if vol > 0 else 0
        results["vol"].append(vol * 100)
        results["ret"].append(ret * 100)
        results["sharpe"].append(sr)
        results["weights"].append(w)

    vols = np.array(results["vol"])
    rets = np.array(results["ret"])
    srs = np.array(results["sharpe"])

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sc = ax.scatter(vols, rets, c=srs, cmap="viridis", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Individual assets
    for name in returns_df.columns:
        a_vol = returns_df[name].std() * np.sqrt(12) * 100
        a_ret = returns_df[name].mean() * 12 * 100
        ax.scatter(a_vol, a_ret, marker="*", s=200, zorder=5)
        ax.annotate(name, (a_vol, a_ret), fontsize=7, textcoords="offset points", xytext=(5, 3))

    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_expense_drag(cost_table: pd.DataFrame) -> plt.Figure:
    """Grouped bar: gross vs net ₹ value after 10 years."""
    gross_col = [c for c in cost_table.columns if "Gross ₹" in c][0]
    net_col = [c for c in cost_table.columns if "Net ₹" in c][0]

    x = np.arange(len(cost_table))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, cost_table[gross_col] / 1e5, width, label="Gross (before fees)", color="#3498db", alpha=0.85)
    ax.bar(x + width / 2, cost_table[net_col] / 1e5, width, label="Net (after fees)", color="#e67e22", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cost_table.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("₹ Lakhs (per ₹1L invested)")
    ax.set_title("Expense Ratio Drag: Gross vs Net Wealth after 10 Years", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig
