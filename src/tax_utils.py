"""
Indian tax calculations for investment returns (FY 2024-25 onwards).

Key rules
  Equity (stocks, equity MFs, equity ETFs)
    STCG  < 12 months : 20 %   (raised from 15% in Budget 2024)
    LTCG  ≥ 12 months : 12.5 % on gains > ₹1.25 L / year (raised from 10% / ₹1L)
    Grandfathering     : cost basis set at 31-Jan-2018 price for pre-2018 holdings

  Debt MFs / Gold MFs (purchased after 1-Apr-2023)
    Taxed at slab rate regardless of holding period (no indexation benefit)

  Physical Gold / Gold ETFs (≥ 24 months = LTCG)
    LTCG  ≥ 24 months : 12.5 % (Budget 2024; previously 20% with indexation)
    STCG  < 24 months : slab rate

  Government Securities / Bonds
    Interest income    : added to income, taxed at slab rate
    Capital gains      : same as debt MF rules above

Reference: Finance Act 2024, Budget speech July 2024.
"""

from __future__ import annotations

import pandas as pd

# ── Tax constants ─────────────────────────────────────────────────────────────

EQUITY_LTCG_RATE = 0.125  # 12.5 % (post Budget 2024)
EQUITY_STCG_RATE = 0.20  # 20 %  (post Budget 2024)
EQUITY_LTCG_EXEMPTION = 125_000  # ₹1.25 lakh per year

GOLD_LTCG_RATE = 0.125  # 12.5 % (≥ 24 months)
GOLD_HOLDING_MONTHS = 24

DEBT_TAX_RATE_SLAB = 0.30  # worst-case 30 % slab

SURCHARGE = 0.10  # 10% surcharge on tax for income > ₹50L
CESS = 0.04  # 4% health & education cess


def effective_rate(base_rate: float, apply_surcharge: bool = False) -> float:
    """Final effective tax rate after surcharge and cess."""
    rate = base_rate
    if apply_surcharge:
        rate += base_rate * SURCHARGE
    rate += rate * CESS
    return rate


# ── Per-asset after-tax return ────────────────────────────────────────────────


def equity_after_tax_return(
    gross_return: float,
    holding_years: float,
    invested_amount: float = 100_000,
    high_income: bool = False,
) -> float:
    """
    Net return after Indian equity capital gains tax.
    gross_return : total (not annualised) return, e.g. 1.50 for 150% gain
    holding_years: e.g. 10
    Returns net annualised return.
    """
    gain = invested_amount * gross_return
    final = invested_amount + gain

    if holding_years >= 1:  # LTCG
        taxable_gain = max(0, gain - EQUITY_LTCG_EXEMPTION)
        rate = effective_rate(EQUITY_LTCG_RATE, apply_surcharge=high_income)
        tax = taxable_gain * rate
    else:  # STCG
        rate = effective_rate(EQUITY_STCG_RATE, apply_surcharge=high_income)
        tax = gain * rate

    net_final = final - tax
    net_total_return = (net_final / invested_amount) - 1
    cagr = (1 + net_total_return) ** (1 / holding_years) - 1
    return cagr


def gold_after_tax_return(
    gross_return: float,
    holding_years: float,
    invested_amount: float = 100_000,
    high_income: bool = False,
) -> float:
    """Net CAGR after gold capital gains tax (Budget 2024 rules)."""
    gain = invested_amount * gross_return
    final = invested_amount + gain

    holding_months = holding_years * 12
    if holding_months >= GOLD_HOLDING_MONTHS:
        rate = effective_rate(GOLD_LTCG_RATE, apply_surcharge=high_income)
        tax = gain * rate
    else:
        rate = effective_rate(DEBT_TAX_RATE_SLAB, apply_surcharge=high_income)
        tax = gain * rate

    net_final = final - tax
    return (
        ((net_final / invested_amount) - 1) ** (1 / holding_years) - 1
        if holding_years > 0
        else 0.0
    )


def debt_after_tax_return(
    gross_annual_yield: float,
    holding_years: float,
    high_income: bool = False,
) -> float:
    """Approximate net-of-tax return for debt instruments (G-Secs, debt MFs).
    Post Apr-2023: all gains taxed at slab rate."""
    rate = effective_rate(DEBT_TAX_RATE_SLAB, apply_surcharge=high_income)
    return gross_annual_yield * (1 - rate)


# ── Expense-ratio drag ────────────────────────────────────────────────────────


def expense_ratio_drag(
    gross_cagr: float,
    expense_ratio: float,
    years: int = 10,
    invested: float = 100_000,
) -> dict:
    """
    Compute the cumulative wealth lost to an expense ratio over `years`.
    Both gross and net final values are on an invested lump-sum basis.
    """
    gross_final = invested * (1 + gross_cagr) ** years
    net_cagr = gross_cagr - expense_ratio
    net_final = invested * (1 + net_cagr) ** years
    cost_drag = gross_final - net_final
    return {
        "gross_cagr": gross_cagr,
        "net_cagr": net_cagr,
        "expense_ratio": expense_ratio,
        "gross_final": gross_final,
        "net_final": net_final,
        "wealth_lost_to_fees": cost_drag,
        "wealth_lost_pct": cost_drag / gross_final * 100,
        "years": years,
    }


def build_cost_comparison_table(
    asset_returns: dict[str, float],
    expense_ratios: dict[str, float],
    years: int = 10,
    invested: float = 100_000,
) -> pd.DataFrame:
    """Side-by-side table of gross vs net returns for all assets."""
    rows = []
    for name, gross in asset_returns.items():
        er = expense_ratios.get(name, 0.0)
        drag = expense_ratio_drag(gross, er, years, invested)
        rows.append(
            {
                "Asset": name,
                "Gross CAGR (%)": round(gross * 100, 2),
                "Expense Ratio (%)": round(er * 100, 2),
                "Net CAGR (%)": round(drag["net_cagr"] * 100, 2),
                f"Gross ₹ (after {years}Y)": round(drag["gross_final"]),
                f"Net ₹ (after {years}Y)": round(drag["net_final"]),
                "₹ Lost to Fees": round(drag["wealth_lost_to_fees"]),
            }
        )
    return pd.DataFrame(rows).set_index("Asset")
