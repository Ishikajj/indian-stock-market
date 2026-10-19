"""
Fama-French factor construction and regression analysis for Indian markets.

Factor proxies (all derived from freely available NSE index ETF prices):
  Mkt-Rf : Nifty 500 (broad market) excess return over RBI repo rate
  SMB     : Nifty Next 50 / small-cap ETF minus Nifty 50 (size)
  HML     : Nifty 50 Value 20 ETF minus Nifty 50 (book-to-market / value)
  MOM     : Nifty 200 Momentum 30 ETF minus Nifty 50 (momentum / Carhart)
  QMJ     : Quality ETF (Nifty 200 Quality 30) minus Nifty 50

Literature references
  Fama & French (1993) — "Common risk factors in the returns on stocks and bonds"
  Carhart (1997)       — "On persistence in mutual fund performance"
  Mohanty (2002)       — "Evidence of size effect on Indian stock returns"
  Wahal & Yezdi (2018) — "Fama-French factors in India"
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import NamedTuple


class FactorReturns(NamedTuple):
    mkt_rf: pd.Series
    smb: pd.Series
    hml: pd.Series
    mom: pd.Series
    qmj: pd.Series | None
    rf: pd.Series


class RegressionResult(NamedTuple):
    alpha_annual: float
    alpha_tstat: float
    beta_mkt: float
    beta_smb: float | None
    beta_hml: float | None
    beta_mom: float | None
    r_squared: float
    information_ratio: float


def build_factor_returns(
    price_dict: dict[str, pd.Series],
    rf_series: pd.Series,
) -> FactorReturns:
    """
    Construct monthly factor returns from index/ETF price series.

    Expected keys in price_dict (use whatever subset is available):
      'Nifty 50' or 'Nifty 50 ETF (NIFTYBEES)'   -> large-cap anchor
      'Nifty Next 50 ETF (JUNIORBEES)'             -> size factor
      'Momentum ETF (MOM100)'                      -> momentum factor
      'Quality ETF (QUAL30IETF)'                   -> quality factor
      'Value ETF (NV20BEES)'                       -> value factor
    """

    def _monthly_ret(name: str) -> pd.Series | None:
        for key in price_dict:
            if name.lower() in key.lower():
                s = price_dict[key].resample("ME").last().pct_change().dropna()
                s.name = key
                return s
        return None

    large = _monthly_ret("nifty 50") or _monthly_ret("niftybees")
    next50 = _monthly_ret("junior") or _monthly_ret("next 50")
    mom = _monthly_ret("mom100") or _monthly_ret("momentum")
    quality = _monthly_ret("qual") or _monthly_ret("quality")
    value = _monthly_ret("nv20") or _monthly_ret("value")

    # Monthly risk-free rate
    rf_monthly = rf_series.resample("ME").last() / 12
    rf_monthly.name = "Rf_monthly"

    if large is None:
        raise ValueError("No Nifty 50 price series found in price_dict")

    # Align everything
    def _align(*series) -> list[pd.Series]:
        valid = [s for s in series if s is not None]
        combined = pd.concat(valid, axis=1).dropna()
        return [combined[s.name] if s is not None else None for s in series]

    # Market excess return (Nifty 50 - Rf)
    rf_aligned = rf_monthly.reindex(large.index).ffill()
    mkt_rf = (large - rf_aligned).rename("Mkt-Rf")

    # SMB: Next 50 / small-cap minus large-cap
    smb = None
    if next50 is not None:
        smb = (next50.reindex(mkt_rf.index).ffill() - large.reindex(mkt_rf.index).ffill()).rename("SMB")

    # HML: Value minus large-cap (as growth proxy)
    hml = None
    if value is not None:
        hml = (value.reindex(mkt_rf.index).ffill() - large.reindex(mkt_rf.index).ffill()).rename("HML")

    # MOM: Momentum ETF minus Nifty 50
    mom_factor = None
    if mom is not None:
        mom_factor = (mom.reindex(mkt_rf.index).ffill() - large.reindex(mkt_rf.index).ffill()).rename("MOM")

    # QMJ: Quality minus large-cap
    qmj = None
    if quality is not None:
        qmj = (quality.reindex(mkt_rf.index).ffill() - large.reindex(mkt_rf.index).ffill()).rename("QMJ")

    return FactorReturns(
        mkt_rf=mkt_rf,
        smb=smb,
        hml=hml,
        mom=mom_factor,
        qmj=qmj,
        rf=rf_aligned,
    )


def run_factor_regression(
    asset_returns: pd.Series,
    factors: FactorReturns,
    model: str = "ff3",
) -> RegressionResult:
    """
    Run OLS factor regression and return structured results.

    model options:
      'capm'  : CAPM (Mkt-Rf only)
      'ff3'   : Fama-French 3-factor (+ SMB, HML)
      'ff4'   : Carhart 4-factor (+ MOM)
      'ff5qmj': 5-factor with QMJ instead of FF5 operating profitability
    """
    # Excess return of the asset
    rf_monthly = factors.rf.reindex(asset_returns.index).ffill() / 12
    excess = (asset_returns - rf_monthly).dropna()

    # Build regressor matrix
    X = pd.DataFrame({"Mkt-Rf": factors.mkt_rf.reindex(excess.index).ffill()})

    if model in ("ff3", "ff4", "ff5qmj"):
        if factors.smb is not None:
            X["SMB"] = factors.smb.reindex(excess.index).ffill()
        if factors.hml is not None:
            X["HML"] = factors.hml.reindex(excess.index).ffill()

    if model in ("ff4", "ff5qmj"):
        if factors.mom is not None:
            X["MOM"] = factors.mom.reindex(excess.index).ffill()

    if model == "ff5qmj":
        if factors.qmj is not None:
            X["QMJ"] = factors.qmj.reindex(excess.index).ffill()

    # Drop rows with any NaN
    data = pd.concat([excess.rename("Y"), X], axis=1).dropna()
    Y = data["Y"]
    X_clean = sm.add_constant(data.drop(columns="Y"))

    ols = sm.OLS(Y, X_clean).fit()

    alpha_monthly = ols.params.get("const", 0.0)
    alpha_annual = (1 + alpha_monthly) ** 12 - 1
    alpha_tstat = ols.tvalues.get("const", 0.0)

    # Information ratio: alpha / residual std, annualised
    resid_std = ols.resid.std() * np.sqrt(12)
    ir = (alpha_annual / resid_std) if resid_std > 0 else np.nan

    return RegressionResult(
        alpha_annual=alpha_annual,
        alpha_tstat=alpha_tstat,
        beta_mkt=ols.params.get("Mkt-Rf", np.nan),
        beta_smb=ols.params.get("SMB", None),
        beta_hml=ols.params.get("HML", None),
        beta_mom=ols.params.get("MOM", None),
        r_squared=ols.rsquared,
        information_ratio=ir,
    )


def factor_summary_table(
    fund_returns: dict[str, pd.Series],
    factors: FactorReturns,
    model: str = "ff3",
) -> pd.DataFrame:
    rows = []
    for name, ret in fund_returns.items():
        try:
            res = run_factor_regression(ret, factors, model)
            rows.append({
                "Fund": name,
                "Alpha (ann %)": round(res.alpha_annual * 100, 2),
                "Alpha t-stat": round(res.alpha_tstat, 2),
                "Beta (Mkt)": round(res.beta_mkt, 2),
                "Beta (SMB)": round(res.beta_smb, 2) if res.beta_smb is not None else "—",
                "Beta (HML)": round(res.beta_hml, 2) if res.beta_hml is not None else "—",
                "Beta (MOM)": round(res.beta_mom, 2) if res.beta_mom is not None else "—",
                "R²": round(res.r_squared, 3),
                "Info Ratio": round(res.information_ratio, 2),
            })
        except Exception as exc:
            print(f"  [WARN] regression failed for {name}: {exc}")
    return pd.DataFrame(rows).set_index("Fund")
