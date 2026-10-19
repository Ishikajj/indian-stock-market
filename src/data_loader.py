"""
Data loading utilities for Indian stock market analysis.

Fetches from:
  - yfinance        : market indices and ETFs
  - mftool          : AMFI mutual fund NAVs
  - FRED (optional) : India 10-year G-Sec yield (IRLTLT01INM156N)

All price/NAV series are returned as pd.Series with DatetimeIndex.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Ticker maps ───────────────────────────────────────────────────────────────

INDEX_TICKERS: dict[str, str] = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "S&P 500 (USD)": "^GSPC",
}

# Factor-proxy ETFs on NSE (listed on yfinance with .NS suffix)
FACTOR_ETF_TICKERS: dict[str, str] = {
    "Nifty 50 ETF (NIFTYBEES)": "NIFTYBEES.NS",
    "Nifty Next 50 ETF (JUNIORBEES)": "JUNIORBEES.NS",
    "Gold ETF (GOLDBEES)": "GOLDBEES.NS",
    "Momentum ETF (MOM100)": "MOM100.NS",
    "Quality ETF (QUAL30IETF)": "QUAL30IETF.NS",
    "Value ETF (NV20BEES)": "NV20BEES.NS",
    "Liquid BeES (cash proxy)": "LIQUIDBEES.NS",
}

# AMFI scheme codes for popular actively-managed and passive funds
# Verify at: https://www.amfiindia.com/nav-history-download
MUTUAL_FUND_CODES: dict[str, str] = {
    "Axis Bluechip Direct Growth": "120503",
    "Mirae Asset Large Cap Direct Growth": "118989",
    "HDFC Top 100 Direct Growth": "119533",
    "SBI Bluechip Direct Growth": "119598",
    "Parag Parikh Flexi Cap Direct Growth": "122639",
    "Mirae Asset Emerging Bluechip Direct Growth": "119597",
    "HDFC Midcap Opportunities Direct Growth": "119315",
    "UTI Nifty 50 Index Direct Growth": "120716",
    "Motilal Oswal Nifty 500 Index Direct Growth": "147946",
}

# Typical annual expense ratios (%) — used in cost-drag analysis
EXPENSE_RATIOS: dict[str, float] = {
    "Axis Bluechip Direct Growth": 0.55,
    "Mirae Asset Large Cap Direct Growth": 0.54,
    "HDFC Top 100 Direct Growth": 0.97,
    "SBI Bluechip Direct Growth": 0.79,
    "Parag Parikh Flexi Cap Direct Growth": 0.63,
    "Mirae Asset Emerging Bluechip Direct Growth": 0.69,
    "HDFC Midcap Opportunities Direct Growth": 0.76,
    "UTI Nifty 50 Index Direct Growth": 0.20,
    "Motilal Oswal Nifty 500 Index Direct Growth": 0.18,
    "Nifty 50 (index, no cost)": 0.00,
    "Sensex (index, no cost)": 0.00,
}


# ── Downloaders ───────────────────────────────────────────────────────────────


def download_prices(
    tickers: dict[str, str],
    start: str = "2005-01-01",
    end: str | None = None,
) -> dict[str, pd.Series]:
    """Return {name: pd.Series of adjusted close prices}."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    result: dict[str, pd.Series] = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                print(f"  [SKIP] {name} ({ticker}) — no data returned")
                continue
            close = raw["Close"].squeeze()
            close.name = name
            result[name] = close.dropna()
            print(
                f"  [OK]   {name}  ({len(close)} days, {close.index[0].date()} – {close.index[-1].date()})"
            )
        except Exception as exc:
            print(f"  [ERR]  {name} ({ticker}): {exc}")
    return result


def download_mutual_fund_navs(
    fund_codes: dict[str, str] | None = None,
    start: str = "2005-01-01",
) -> dict[str, pd.Series]:
    """Return {name: pd.Series of NAV} from AMFI via mftool."""
    try:
        from mftool import Mftool
    except ImportError:
        print("mftool not installed — skipping mutual fund download")
        return {}

    if fund_codes is None:
        fund_codes = MUTUAL_FUND_CODES

    mf = Mftool()
    start_dt = pd.to_datetime(start)
    result: dict[str, pd.Series] = {}

    for name, code in fund_codes.items():
        try:
            raw = mf.get_scheme_historical_nav(code, as_Dataframe=True)
            if raw is None or raw.empty:
                print(f"  [SKIP] {name} — empty response")
                continue
            raw.index = pd.to_datetime(raw.index, dayfirst=True)
            nav = raw["nav"].astype(float).sort_index()
            nav = nav[nav.index >= start_dt]
            nav.name = name
            result[name] = nav.dropna()
            print(
                f"  [OK]   {name}  ({len(nav)} days, {nav.index[0].date()} – {nav.index[-1].date()})"
            )
        except Exception as exc:
            print(f"  [ERR]  {name} (code {code}): {exc}")

    return result


# ── Risk-free rate ─────────────────────────────────────────────────────────────


def get_rbi_repo_rate_series(end: str | None = None) -> pd.Series:
    """
    Daily series of the RBI repo rate (annualised, as decimal).
    Compiled from RBI press releases; forward-filled between change dates.
    """
    rate_changes: list[tuple[str, float]] = [
        ("2004-01-01", 4.50),
        ("2004-09-01", 4.75),
        ("2005-04-01", 5.00),
        ("2006-06-01", 5.75),
        ("2006-07-01", 6.00),
        ("2006-10-01", 6.25),
        ("2007-03-01", 7.50),
        ("2008-06-01", 8.00),
        ("2008-07-01", 8.50),
        ("2009-03-01", 5.00),
        ("2009-04-01", 4.75),
        ("2010-03-01", 5.00),
        ("2010-04-01", 5.25),
        ("2010-07-01", 5.75),
        ("2010-09-01", 6.00),
        ("2010-11-01", 6.25),
        ("2011-01-01", 6.50),
        ("2011-03-01", 6.75),
        ("2011-05-01", 7.25),
        ("2011-07-01", 7.50),
        ("2011-10-01", 8.50),
        ("2012-04-01", 8.00),
        ("2013-09-01", 7.50),
        ("2014-01-01", 8.00),
        ("2015-01-01", 7.75),
        ("2015-03-01", 7.50),
        ("2015-06-01", 7.25),
        ("2015-09-01", 6.75),
        ("2016-04-01", 6.50),
        ("2017-08-01", 6.00),
        ("2018-06-01", 6.25),
        ("2018-08-01", 6.50),
        ("2019-02-01", 6.25),
        ("2019-04-01", 6.00),
        ("2019-06-01", 5.75),
        ("2019-08-01", 5.40),
        ("2019-10-01", 5.15),
        ("2020-03-01", 4.40),
        ("2020-05-01", 4.00),
        ("2022-05-01", 4.40),
        ("2022-06-01", 4.90),
        ("2022-08-01", 5.40),
        ("2022-09-01", 5.90),
        ("2022-12-01", 6.25),
        ("2023-02-01", 6.50),
        ("2025-02-01", 6.25),
        ("2025-04-01", 6.00),
    ]

    end_dt = pd.to_datetime(end) if end else datetime.today()
    dates = pd.date_range("2004-01-01", end_dt, freq="D")
    series = pd.Series(np.nan, index=dates, name="RBI Repo Rate")

    for date_str, rate in rate_changes:
        dt = pd.to_datetime(date_str)
        if dt <= end_dt:
            series.loc[dt:] = rate / 100.0

    return series.ffill().dropna()


def get_india_10y_gsec(start: str = "2005-01-01") -> pd.Series:
    """
    Monthly India 10-year G-Sec yield from FRED (series IRLTLT01INM156N).
    Falls back to repo rate + 150 bps term premium if FRED is unavailable.
    """
    try:
        import pandas_datareader.data as web

        raw = web.DataReader("IRLTLT01INM156N", "fred", start=start)
        s = raw.squeeze() / 100.0
        s.name = "India 10Y G-Sec Yield"
        print(f"  [OK]   India 10Y G-Sec from FRED ({len(s)} monthly obs)")
        return s
    except Exception as exc:
        print(f"  [WARN] FRED unavailable ({exc}) — using repo rate + 150 bps proxy")
        rf = get_rbi_repo_rate_series()
        monthly = (rf + 0.015).resample("ME").last()
        monthly.name = "India 10Y G-Sec (proxy)"
        return monthly


# ── Alignment utility ─────────────────────────────────────────────────────────


def get_sp500_inr(start: str = "2005-01-01", end: str | None = None) -> pd.Series | None:
    """
    S&P 500 price series re-expressed in INR using the USD/INR spot rate.

    This is the economically correct return series for an Indian investor buying
    a US-domiciled S&P 500 ETF via the LRS route (Vested, INDmoney, etc.).
    It is NOT what an Indian mutual fund tracking the S&P 500 returns — those
    NAVs are already in INR but are taxed differently (see SIP section).

    USD/INR has moved from ~45 (2005) → ~84 (2025) → ~86+ (2026), meaningfully
    boosting INR returns relative to USD returns.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    sp500 = download_prices({"_sp500": "^GSPC"}, start=start, end=end).get("_sp500")
    usdinr = download_prices({"_usdinr": "USDINR=X"}, start=start, end=end).get("_usdinr")

    if sp500 is None or usdinr is None:
        print("  [WARN] Could not construct S&P 500 (INR) — missing price or FX data")
        return None

    combined = pd.concat([sp500.rename("sp500"), usdinr.rename("usdinr")], axis=1).ffill().dropna()

    result = (combined["sp500"] * combined["usdinr"]).rename("S&P 500 (INR)")
    print(
        f"  [OK]   S&P 500 (INR): {len(result)} daily obs, "
        f"{result.index[0].date()} – {result.index[-1].date()}"
    )
    return result


def to_monthly_prices(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Resample all series to month-end, align on common dates, drop all-NaN rows."""
    resampled = {
        name: s.resample("ME").last().rename(name)
        for name, s in series_dict.items()
        if isinstance(s, pd.Series)
    }
    df = pd.DataFrame(resampled).sort_index()
    return df.dropna(how="all")


def to_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a price DataFrame to simple monthly returns."""
    return price_df.pct_change().dropna(how="all")
